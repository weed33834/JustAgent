"""DAG-based workflow engine — dependency resolution, conditional routing, parallel execution.

Turns a directed acyclic graph (DAG) of :class:`WorkflowNode` objects into an
executable pipeline. Nodes are connected by :class:`WorkflowEdge` arcs which may
carry a ``condition`` expression; the engine resolves execution order with a
topological sort, runs independent nodes concurrently via
:func:`asyncio.gather`, evaluates edge conditions to decide which branches are
taken, and supports pause / resume / cancel through asyncio events.

Design:

* :class:`WorkflowStatus` / :class:`NodeStatus` — typed lifecycle enumerations.
* :class:`NodeType` — the kind of work a node performs (task, decision,
  parallel fan-out, human approval, webhook, agent call, script, ...). Each
  type is backed by an injectable async handler.
* :class:`WorkflowNode` — a single step: inputs, outputs, dependencies,
  status and free-form ``config``.
* :class:`WorkflowEdge` — a directed arc with an optional boolean condition.
* :class:`Workflow` — the graph definition plus shared variables.
* :class:`WorkflowExecution` — a live run: status, results per node, and the
  set of nodes currently executing.
* :class:`WorkflowEngine` — async, thread-safe orchestrator that validates
  the DAG, creates executions, and drives them to completion.

The engine is local-first: node handlers default to in-process behaviour
(simulated work, subprocess scripts, ``httpx`` webhooks) but can be fully
overridden via :meth:`WorkflowEngine.register_node_handler`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("myagent.orchestration.workflow")


class WorkflowError(Exception):
    """Raised for invalid workflow definitions or execution failures."""


class WorkflowStatus(str, Enum):  # noqa: UP042 - match existing codebase style
    """Lifecycle status of a workflow definition or execution."""

    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(str, Enum):  # noqa: UP042
    """Lifecycle status of a single workflow node within an execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class NodeType(str, Enum):  # noqa: UP042
    """The kind of work a :class:`WorkflowNode` performs.

    Each value is dispatched to a dedicated async handler on the
    :class:`WorkflowEngine`. Handlers are overridable via
    :meth:`WorkflowEngine.register_node_handler`.
    """

    TASK = "task"
    DECISION = "decision"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HUMAN_APPROVAL = "human_approval"
    WEBHOOK = "webhook"
    AGENT_CALL = "agent_call"
    SCRIPT = "script"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class WorkflowNode(BaseModel):
    """A single step in a workflow DAG.

    Attributes:
        id: Unique node identifier (auto-generated UUID4 hex when omitted).
        name: Human-readable name.
        node_type: The :class:`NodeType` controlling execution dispatch.
        inputs: Static input values merged with workflow variables at runtime.
        outputs: Populated by the node handler after successful execution.
        dependencies: IDs of nodes that must complete before this one starts
            (supplements incoming :class:`WorkflowEdge` arcs).
        status: Current :class:`NodeStatus` within an execution.
        config: Free-form, type-specific configuration (e.g. ``url`` for a
            webhook node, ``condition`` for a decision node, ``command`` for
            a script node).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    node_type: NodeType = NodeType.TASK
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    config: dict[str, Any] = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    """A directed arc between two nodes, optionally gated by a condition.

    Attributes:
        source: ID of the upstream node.
        target: ID of the downstream node.
        condition: Optional boolean expression evaluated against the source
            node's outputs and the workflow variables. When it evaluates to
            a falsy value the target node is skipped (and the skip cascades
            to its downstream nodes). ``None`` means unconditional.
    """

    source: str
    target: str
    condition: str | None = None


class Workflow(BaseModel):
    """A workflow definition: nodes, edges, shared variables and metadata.

    Attributes:
        id: Unique workflow identifier.
        name: Human-readable name.
        description: Free-form description.
        nodes: The node set (order is not significant; execution order is
            derived from edges/dependencies).
        edges: Directed arcs with optional conditions.
        status: Workflow-level status (DRAFT until validated to READY).
        created_by: User or system that authored the workflow.
        created_at: UTC creation timestamp.
        updated_at: UTC timestamp of the last modification.
        variables: Shared variables accessible to every node handler.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: str = ""
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.DRAFT
    created_by: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    variables: dict[str, Any] = Field(default_factory=dict)

    def node_by_id(self, node_id: str) -> WorkflowNode | None:
        """Return the node with *node_id*, or ``None``."""

        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


class WorkflowExecution(BaseModel):
    """A live run of a :class:`Workflow`.

    Attributes:
        id: Unique execution identifier.
        workflow_id: The workflow being executed.
        status: Current :class:`WorkflowStatus`.
        started_at: UTC timestamp when execution began.
        completed_at: UTC timestamp when execution reached a terminal state.
        node_results: Mapping of ``node_id`` -> result dict. Each entry
            contains the node's ``status``, ``outputs`` and any ``error``.
        current_nodes: IDs of nodes currently executing (empty when idle).
        variables: Execution-specific variable overrides merged over the
            workflow's variables.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.READY
    started_at: datetime | None = None
    completed_at: datetime | None = None
    node_results: dict[str, Any] = Field(default_factory=dict)
    current_nodes: list[str] = Field(default_factory=list)
    variables: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Handler type alias
# ---------------------------------------------------------------------------

#: Signature of an async node handler. It receives the node, the live
#: execution metadata, and the working workflow copy, and must return a
#: dict of outputs to store on the node.
NodeHandler = Callable[[WorkflowNode, WorkflowExecution, Workflow], Awaitable[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WorkflowEngine:
    """Async, thread-safe DAG workflow engine.

    The engine owns a registry of workflow definitions and executions. Node
    execution is dispatched by :class:`NodeType` to injectable async handlers;
    sensible defaults are provided for every type. Independent nodes within a
    dependency level run concurrently via :func:`asyncio.gather`.

    Registry access (workflows, executions, handlers) is guarded by a
    :class:`threading.RLock` so metadata can be queried safely from any
    thread; execution coordination (pause / resume / cancel) uses
    :class:`asyncio.Event` and :class:`asyncio.Lock` instances so long-running
    handlers never block the event loop.

    Example::

        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="deploy",
            nodes=[
                WorkflowNode(id="build", name="Build", node_type=NodeType.SCRIPT,
                             config={"command": "echo building"}),
                WorkflowNode(id="ship", name="Ship", node_type=NodeType.TASK,
                             dependencies=["build"]),
            ],
            edges=[WorkflowEdge(source="build", target="ship")],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        assert result.status is WorkflowStatus.COMPLETED
    """

    def __init__(self) -> None:
        self._workflows: dict[str, Workflow] = {}
        self._executions: dict[str, WorkflowExecution] = {}
        # Per-execution working copy of the workflow (live node statuses).
        self._execution_workflows: dict[str, Workflow] = {}
        # Per-execution asyncio coordination primitives.
        self._pause_events: dict[str, asyncio.Event] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._execution_locks: dict[str, asyncio.Lock] = {}
        self._running_executions: set[str] = set()
        # Per-(execution, node) approval events for HUMAN_APPROVAL nodes.
        self._approval_events: dict[tuple[str, str], asyncio.Event] = {}
        # NodeType -> async handler.
        self._node_handlers: dict[NodeType, NodeHandler] = {}
        # agent_id -> async handler for AGENT_CALL nodes.
        self._agent_handlers: dict[
            str, Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
        ] = {}
        self._lock = threading.RLock()
        self._register_default_handlers()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_node_handler(self, node_type: NodeType, handler: NodeHandler) -> None:
        """Register or replace the async handler for a :class:`NodeType`."""

        with self._lock:
            self._node_handlers[node_type] = handler
        logger.debug("Registered node handler for %s", node_type.value)

    def register_agent_handler(
        self,
        agent_id: str,
        handler: Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> None:
        """Register an async handler invoked by ``AGENT_CALL`` nodes.

        The handler receives the node ``inputs`` and the merged workflow
        ``variables`` and must return a dict of outputs.
        """

        with self._lock:
            self._agent_handlers[agent_id] = handler
        logger.debug("Registered agent handler for %r", agent_id)

    def _register_default_handlers(self) -> None:
        """Wire the built-in default handler for every :class:`NodeType`."""

        defaults: dict[NodeType, NodeHandler] = {
            NodeType.TASK: self._default_task_handler,
            NodeType.DECISION: self._default_decision_handler,
            NodeType.PARALLEL: self._default_parallel_handler,
            NodeType.SEQUENTIAL: self._default_sequential_handler,
            NodeType.HUMAN_APPROVAL: self._default_human_approval_handler,
            NodeType.WEBHOOK: self._default_webhook_handler,
            NodeType.AGENT_CALL: self._default_agent_call_handler,
            NodeType.SCRIPT: self._default_script_handler,
        }
        self._node_handlers.update(defaults)

    # ------------------------------------------------------------------
    # Workflow management
    # ------------------------------------------------------------------

    async def create_workflow(
        self,
        name: str,
        *,
        description: str = "",
        nodes: list[WorkflowNode] | None = None,
        edges: list[WorkflowEdge] | None = None,
        created_by: str = "",
        variables: dict[str, Any] | None = None,
    ) -> Workflow:
        """Validate, store and return a new workflow definition.

        The DAG is checked for cycles (via topological sort) and dangling
        edge references. On success the workflow status is set to
        :attr:`WorkflowStatus.READY`.
        """

        if not name or not name.strip():
            raise WorkflowError("workflow name must not be empty")
        workflow = Workflow(
            name=name,
            description=description,
            nodes=list(nodes) if nodes else [],
            edges=list(edges) if edges else [],
            created_by=created_by,
            variables=dict(variables) if variables else {},
        )
        self._validate_dag(workflow)
        workflow.status = WorkflowStatus.READY
        with self._lock:
            self._workflows[workflow.id] = workflow
        logger.info(
            "Created workflow %s (%s) with %d node(s)",
            workflow.name,
            workflow.id,
            len(workflow.nodes),
        )
        return workflow

    async def get_workflow(self, workflow_id: str) -> Workflow | None:
        """Return the workflow definition by ID, or ``None``."""

        with self._lock:
            return self._workflows.get(workflow_id)

    async def create_execution(
        self,
        workflow_id: str,
        *,
        variables: dict[str, Any] | None = None,
    ) -> WorkflowExecution:
        """Create a fresh execution for *workflow_id*.

        A deep copy of the workflow is taken so node statuses are isolated
        per execution. Returns the :class:`WorkflowExecution` (status READY).
        """

        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise WorkflowError(f"workflow not found: {workflow_id}")
            working = workflow.model_copy(deep=True)
            # Reset all node statuses to PENDING for a clean run.
            for node in working.nodes:
                node.status = NodeStatus.PENDING
                node.outputs = {}
            merged_vars = dict(workflow.variables)
            if variables:
                merged_vars.update(variables)
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                status=WorkflowStatus.READY,
                variables=merged_vars,
            )
            self._executions[execution.id] = execution
            self._execution_workflows[execution.id] = working
            self._pause_events[execution.id] = asyncio.Event()
            self._pause_events[execution.id].set()  # not paused by default
            self._cancel_events[execution.id] = asyncio.Event()
            self._execution_locks[execution.id] = asyncio.Lock()
        logger.info("Created execution %s for workflow %s", execution.id, workflow_id)
        return execution

    # ------------------------------------------------------------------
    # Execution lifecycle
    # ------------------------------------------------------------------

    async def start_execution(self, execution_id: str) -> WorkflowExecution:
        """Run an execution to completion and return its final state.

        The execution drives the DAG level-by-level: at each wave, all nodes
        whose dependencies are satisfied run concurrently. Between waves the
        engine honours pause (waits on the pause event) and cancel (aborts)
        signals. This method awaits the full run; to pause or cancel
        concurrently, schedule it as a task::

            task = asyncio.create_task(engine.start_execution(eid))
            await engine.pause_execution(eid)
            await engine.resume_execution(eid)
            result = await task
        """

        with self._lock:
            execution = self._executions.get(execution_id)
            if execution is None:
                raise WorkflowError(f"execution not found: {execution_id}")
            if execution_id in self._running_executions:
                raise WorkflowError(f"execution {execution_id} is already running")
            if execution.status in (
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED,
            ):
                raise WorkflowError(
                    f"execution {execution_id} is already terminal ({execution.status.value})"
                )
            self._running_executions.add(execution_id)
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.now(UTC)
            working = self._execution_workflows[execution_id]
            pause_event = self._pause_events[execution_id]
            cancel_event = self._cancel_events[execution_id]
            exec_lock = self._execution_locks[execution_id]

        try:
            await self._run_execution_loop(execution, working, pause_event, cancel_event, exec_lock)
        except Exception as exc:  # noqa: BLE001 - record and mark failed
            logger.exception("Execution %s crashed: %s", execution_id, exc)
            with self._lock:
                execution.status = WorkflowStatus.FAILED
                execution.completed_at = datetime.now(UTC)
        finally:
            with self._lock:
                self._running_executions.discard(execution_id)
                if execution.status not in (
                    WorkflowStatus.COMPLETED,
                    WorkflowStatus.FAILED,
                    WorkflowStatus.CANCELLED,
                    WorkflowStatus.PAUSED,
                ):
                    execution.status = WorkflowStatus.FAILED
                    execution.completed_at = datetime.now(UTC)
        return execution

    async def _run_execution_loop(
        self,
        execution: WorkflowExecution,
        working: Workflow,
        pause_event: asyncio.Event,
        cancel_event: asyncio.Event,
        exec_lock: asyncio.Lock,
    ) -> None:
        """Drive the DAG wave-by-wave until completion, pause or cancel."""

        while True:
            if cancel_event.is_set():
                await self._mark_cancelled(execution, working, exec_lock)
                return
            # Block here while paused; returns immediately when running.
            await pause_event.wait()
            if cancel_event.is_set():
                await self._mark_cancelled(execution, working, exec_lock)
                return

            ready, blocked = self._compute_ready_nodes(working)
            # Persist any BLOCKED / SKIPPED transitions.
            async with exec_lock:
                for node in working.nodes:
                    if node.status == NodeStatus.PENDING and node.id in blocked:
                        node.status = NodeStatus.BLOCKED
                        execution.node_results[node.id] = {
                            "status": NodeStatus.BLOCKED.value,
                            "outputs": {},
                            "error": "dependencies unmet",
                        }

            if not ready:
                async with exec_lock:
                    pending = [n for n in working.nodes if n.status == NodeStatus.PENDING]
                if not pending:
                    break  # nothing left to do
                # Truly blocked (no progress possible) -> fail the workflow.
                logger.warning(
                    "Execution %s stalled: %d pending node(s) blocked",
                    execution.id,
                    len(pending),
                )
                async with exec_lock:
                    execution.status = WorkflowStatus.FAILED
                    execution.completed_at = datetime.now(UTC)
                return

            async with exec_lock:
                execution.current_nodes = [n.id for n in ready]
            logger.debug(
                "Execution %s: running wave %s",
                execution.id,
                [n.name for n in ready],
            )

            results = await asyncio.gather(
                *[self._execute_node(n, execution, working) for n in ready],
                return_exceptions=True,
            )

            # Process wave outcomes.
            failed = False
            for node, result in zip(ready, results, strict=True):
                if isinstance(result, BaseException):
                    async with exec_lock:
                        node.status = NodeStatus.FAILED
                        execution.node_results[node.id] = {
                            "status": NodeStatus.FAILED.value,
                            "outputs": {},
                            "error": str(result),
                        }
                    logger.error("Node %s (%s) failed: %s", node.name, node.id, result)
                    failed = True
                else:
                    async with exec_lock:
                        execution.node_results[node.id] = {
                            "status": node.status.value,
                            "outputs": node.outputs,
                            "error": "",
                        }

            if cancel_event.is_set():
                await self._mark_cancelled(execution, working, exec_lock)
                return

            if failed:
                async with exec_lock:
                    execution.status = WorkflowStatus.FAILED
                    execution.completed_at = datetime.now(UTC)
                    execution.current_nodes = []
                logger.warning("Execution %s failed due to node error(s)", execution.id)
                return

        # All nodes processed; determine terminal status.
        async with exec_lock:
            execution.current_nodes = []
            any_failed = any(n.status == NodeStatus.FAILED for n in working.nodes)
            execution.status = WorkflowStatus.FAILED if any_failed else WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now(UTC)
        logger.info(
            "Execution %s finished with status %s",
            execution.id,
            execution.status.value,
        )

    async def _mark_cancelled(
        self,
        execution: WorkflowExecution,
        working: Workflow,
        exec_lock: asyncio.Lock,
    ) -> None:
        """Mark the execution and its in-flight nodes as cancelled."""

        async with exec_lock:
            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.now(UTC)
            execution.current_nodes = []
            for node in working.nodes:
                if node.status in (NodeStatus.PENDING, NodeStatus.RUNNING):
                    node.status = NodeStatus.FAILED
                    execution.node_results[node.id] = {
                        "status": NodeStatus.FAILED.value,
                        "outputs": node.outputs,
                        "error": "cancelled",
                    }
        logger.info("Execution %s cancelled", execution.id)

    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running execution. Returns True if it was running."""

        with self._lock:
            event = self._pause_events.get(execution_id)
            execution = self._executions.get(execution_id)
            if event is None or execution is None:
                return False
            if execution_id not in self._running_executions:
                return False
            event.clear()
            execution.status = WorkflowStatus.PAUSED
        logger.info("Execution %s paused", execution_id)
        return True

    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution. Returns True if it was paused."""

        with self._lock:
            event = self._pause_events.get(execution_id)
            execution = self._executions.get(execution_id)
            if event is None or execution is None:
                return False
            if execution.status is not WorkflowStatus.PAUSED:
                return False
            event.set()
            execution.status = WorkflowStatus.RUNNING
        logger.info("Execution %s resumed", execution_id)
        return True

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running or paused execution. Returns True if active."""

        with self._lock:
            event = self._cancel_events.get(execution_id)
            execution = self._executions.get(execution_id)
            if event is None or execution is None:
                return False
            if execution_id not in self._running_executions:
                return False
            event.set()
            # If paused, unpause so the loop can observe the cancel signal.
            pause_event = self._pause_events.get(execution_id)
            if pause_event is not None and not pause_event.is_set():
                pause_event.set()
        logger.info("Execution %s cancel requested", execution_id)
        return True

    async def approve_node(self, execution_id: str, node_id: str) -> bool:
        """Approve a ``HUMAN_APPROVAL`` node waiting for external sign-off."""

        with self._lock:
            event = self._approval_events.get((execution_id, node_id))
        if event is None:
            return False
        event.set()
        logger.info("Approved node %s in execution %s", node_id, execution_id)
        return True

    async def get_execution_status(self, execution_id: str) -> WorkflowExecution | None:
        """Return a snapshot of the execution by ID, or ``None``."""

        with self._lock:
            execution = self._executions.get(execution_id)
            if execution is None:
                return None
            return execution.model_copy(deep=True)

    async def get_execution_nodes(self, execution_id: str) -> list[WorkflowNode] | None:
        """Return the live node states for an execution, or ``None``."""

        with self._lock:
            working = self._execution_workflows.get(execution_id)
            if working is None:
                return None
            return [node.model_copy(deep=True) for node in working.nodes]

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------

    async def _execute_node(
        self,
        node: WorkflowNode,
        execution: WorkflowExecution,
        working: Workflow,
    ) -> dict[str, Any]:
        """Dispatch *node* to its registered handler and record the output."""

        with self._lock:
            handler = self._node_handlers.get(node.node_type)
        if handler is None:
            raise WorkflowError(f"no handler registered for node type {node.node_type.value}")
        node.status = NodeStatus.RUNNING
        logger.debug(
            "Executing node %s (%s, type=%s) in execution %s",
            node.name,
            node.id,
            node.node_type.value,
            execution.id,
        )
        try:
            outputs = await handler(node, execution, working)
        except Exception:
            node.status = NodeStatus.FAILED
            raise
        node.outputs = outputs or {}
        node.status = NodeStatus.COMPLETED
        # Merge outputs into the shared variables for downstream nodes.
        with self._lock:
            working.variables.update(node.outputs)
        return node.outputs

    # ------------------------------------------------------------------
    # Default node handlers
    # ------------------------------------------------------------------

    async def _default_task_handler(
        self,
        node: WorkflowNode,
        execution: WorkflowExecution,
        working: Workflow,
    ) -> dict[str, Any]:
        """Generic task: merge inputs with variables and produce a result."""

        inputs = {**working.variables, **node.inputs}
        await asyncio.sleep(0)  # yield to the loop
        result = node.config.get("result", "completed")
        return {"task": node.name, "inputs": inputs, "result": result}

    async def _default_decision_handler(
        self,
        node: WorkflowNode,
        execution: WorkflowExecution,
        working: Workflow,
    ) -> dict[str, Any]:
        """Evaluate a boolean condition and record the chosen branch."""

        condition = node.config.get("condition", "true")
        namespace = {**working.variables, **node.inputs}
        result = self._eval_condition(condition, namespace)
        branch = (
            node.config.get("true_branch", "true")
            if result
            else node.config.get("false_branch", "false")
        )
        return {
            "decision": branch,
            "condition": condition,
            "evaluated": result,
            "outputs": dict(node.inputs),
        }

    async def _default_parallel_handler(
        self,
        node: WorkflowNode,
        execution: WorkflowExecution,
        working: Workflow,
    ) -> dict[str, Any]:
        """Fan out across ``config['branches']`` concurrently."""

        branches = node.config.get("branches", [])
        if not branches:
            return {"branches": [], "result": "no branches"}

        async def _run_branch(branch: Any) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"branch": branch, "status": "completed"}

        results = await asyncio.gather(*[_run_branch(b) for b in branches])
        return {"branches": list(results), "result": "completed"}

    async def _default_sequential_handler(
        self,
        node: WorkflowNode,
        execution: WorkflowExecution,
        working: Workflow,
    ) -> dict[str, Any]:
        """Run ``config['steps']`` one after another."""

        steps = node.config.get("steps", [])
        outputs: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            await asyncio.sleep(0)
            outputs.append({"step": index, "value": step, "status": "completed"})
        return {"steps": outputs, "result": "completed"}

    async def _default_human_approval_handler(
        self,
        node: WorkflowNode,
        execution: WorkflowExecution,
        working: Workflow,
    ) -> dict[str, Any]:
        """Block until a human approves the node (or auto-approve / timeout)."""

        if node.config.get("auto_approve"):
            return {"approved": True, "approver": "auto"}
        key = (execution.id, node.id)
        with self._lock:
            event = self._approval_events.get(key)
            if event is None:
                event = asyncio.Event()
                self._approval_events[key] = event
        timeout = node.config.get("timeout")
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return {"approved": True, "approver": "human"}
        except TimeoutError:
            return {"approved": False, "reason": "approval timed out"}

    async def _default_webhook_handler(
        self,
        node: WorkflowNode,
        execution: WorkflowExecution,
        working: Workflow,
    ) -> dict[str, Any]:
        """POST/GET a webhook endpoint via ``httpx`` (lazy import)."""

        url = node.config.get("url")
        if not url:
            return {"status": "skipped", "reason": "no url configured"}
        payload = node.config.get("payload", node.inputs)
        try:
            import httpx  # lazy import for optional dependency
        except ImportError:
            logger.warning("httpx not installed; webhook node simulated")
            return {"status": "simulated", "url": url, "reason": "httpx unavailable"}
        method = str(node.config.get("method", "POST")).upper()
        timeout = float(node.config.get("timeout", 30.0))
        headers = node.config.get("headers", {})
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "GET":
                    resp = await client.get(url, params=payload, headers=headers)
                else:
                    resp = await client.request(method, url, json=payload, headers=headers)
                return {
                    "status_code": resp.status_code,
                    "body": resp.text,
                    "url": url,
                    "method": method,
                }
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "url": url, "error": str(exc)}

    async def _default_agent_call_handler(
        self,
        node: WorkflowNode,
        execution: WorkflowExecution,
        working: Workflow,
    ) -> dict[str, Any]:
        """Invoke a registered agent handler keyed by ``config['agent']``."""

        agent_id = node.config.get("agent", "")
        with self._lock:
            handler = self._agent_handlers.get(agent_id)
        inputs = {**working.variables, **node.inputs}
        if handler is not None:
            try:
                return await handler(inputs, dict(working.variables))
            except Exception as exc:  # noqa: BLE001
                return {"agent": agent_id, "status": "failed", "error": str(exc)}
        await asyncio.sleep(0)
        return {
            "agent": agent_id,
            "status": "simulated",
            "inputs": inputs,
        }

    async def _default_script_handler(
        self,
        node: WorkflowNode,
        execution: WorkflowExecution,
        working: Workflow,
    ) -> dict[str, Any]:
        """Run a shell command via :func:`asyncio.create_subprocess_shell`."""

        command = node.config.get("command") or node.config.get("script")
        if not command:
            return {"status": "skipped", "reason": "no command configured"}
        timeout = float(node.config.get("timeout", 60.0))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return {"exit_code": -1, "error": f"failed to start: {exc}"}
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "exit_code": proc.returncode,
                "stdout": stdout_b.decode(errors="replace") if stdout_b else "",
                "stderr": stderr_b.decode(errors="replace") if stderr_b else "",
                "command": command,
            }
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return {"exit_code": -1, "error": f"timed out after {timeout}s"}

    # ------------------------------------------------------------------
    # DAG analysis
    # ------------------------------------------------------------------

    def _validate_dag(self, workflow: Workflow) -> None:
        """Validate node/edge integrity and detect cycles via topological sort."""

        node_ids = {n.id for n in workflow.nodes}
        if len(node_ids) != len(workflow.nodes):
            raise WorkflowError("workflow contains duplicate node ids")
        for edge in workflow.edges:
            if edge.source not in node_ids:
                raise WorkflowError(f"edge source {edge.source!r} references unknown node")
            if edge.target not in node_ids:
                raise WorkflowError(f"edge target {edge.target!r} references unknown node")
        for node in workflow.nodes:
            for dep in node.dependencies:
                if dep not in node_ids:
                    raise WorkflowError(f"node {node.id!r} depends on unknown node {dep!r}")
        levels = self._topological_levels(workflow)
        visited = {n for level in levels for n in level}
        if visited != node_ids:
            raise WorkflowError("workflow DAG contains a cycle")

    def _topological_levels(self, workflow: Workflow) -> list[list[str]]:
        """Return nodes grouped into dependency levels (Kahn's algorithm).

        Each level contains nodes whose dependencies have all been resolved
        by the preceding levels. Nodes within a level are independent and
        may run concurrently.
        """

        adjacency: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        in_degree: dict[str, int] = {n.id: 0 for n in workflow.nodes}
        for edge in workflow.edges:
            adjacency[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        for node in workflow.nodes:
            for dep in node.dependencies:
                if dep in in_degree and node.id not in adjacency[dep]:
                    adjacency[dep].append(node.id)
                    in_degree[node.id] += 1
        levels: list[list[str]] = []
        current = [nid for nid, deg in in_degree.items() if deg == 0]
        while current:
            levels.append(sorted(current))
            next_level: list[str] = []
            for nid in current:
                for child in adjacency[nid]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_level.append(child)
            current = next_level
        return levels

    def _compute_ready_nodes(self, working: Workflow) -> tuple[list[WorkflowNode], list[str]]:
        """Determine which PENDING nodes are ready to run.

        Returns a tuple of ``(ready_nodes, blocked_node_ids)``. A node is
        ready when every dependency and every incoming-edge source has reached
        a terminal state (COMPLETED or SKIPPED). If any incoming edge's
        condition evaluates to False (or a dependency was SKIPPED), the node
        is marked SKIPPED and the skip cascades to its downstream nodes.
        """

        node_map = {n.id: n for n in working.nodes}
        incoming: dict[str, list[WorkflowEdge]] = {n.id: [] for n in working.nodes}
        for edge in working.edges:
            incoming[edge.target].append(edge)

        terminal = {NodeStatus.COMPLETED, NodeStatus.SKIPPED}
        ready: list[WorkflowNode] = []
        blocked: list[str] = []
        for node in working.nodes:
            if node.status != NodeStatus.PENDING:
                continue
            # Check explicit dependencies.
            deps_ok = True
            skip = False
            for dep_id in node.dependencies:
                dep = node_map.get(dep_id)
                if dep is None or dep.status not in terminal:
                    deps_ok = False
                    break
                if dep.status is NodeStatus.SKIPPED:
                    skip = True
            if not deps_ok:
                blocked.append(node.id)
                continue
            # Check incoming edges.
            edges_ok = True
            for edge in incoming[node.id]:
                source = node_map.get(edge.source)
                if source is None or source.status not in terminal:
                    edges_ok = False
                    break
                if source.status is NodeStatus.SKIPPED:
                    skip = True
                elif edge.condition:
                    namespace = {**working.variables, **source.outputs}
                    if not self._eval_condition(edge.condition, namespace):
                        skip = True
            if not edges_ok:
                blocked.append(node.id)
                continue
            if skip:
                node.status = NodeStatus.SKIPPED
                node.outputs = {"skipped": True}
                working.variables.setdefault("skipped_nodes", []).append(node.id)
                logger.debug("Node %s skipped (condition/dependency)", node.name)
                continue
            ready.append(node)
        return ready, blocked

    def _eval_condition(self, expression: str, namespace: dict[str, Any]) -> bool:
        """Evaluate a boolean condition string against *namespace*.

        Simple literals (``true``/``false``/``1``/``0``) are handled directly.
        Otherwise the expression is evaluated with a restricted builtin
        namespace. This is acceptable for a local-first, trusted-author
        workflow engine. Any evaluation error is treated as False.
        """

        if not expression:
            return True
        expr = expression.strip()
        low = expr.lower()
        if low in ("true", "1", "yes", "ok", "pass"):
            return True
        if low in ("false", "0", "no", "fail"):
            return False
        safe_globals: dict[str, Any] = {"__builtins__": {}}
        safe_locals = dict(namespace)
        try:
            result = eval(expr, safe_globals, safe_locals)  # noqa: S307 - trusted local workflows
        except Exception:  # noqa: BLE001 - treat unparseable as False
            logger.debug("Condition %r evaluated to False (parse error)", expression)
            return False
        return bool(result)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of engine state for dashboards."""

        with self._lock:
            return {
                "workflows": len(self._workflows),
                "executions": len(self._executions),
                "running": len(self._running_executions),
            }


__all__ = [
    "NodeHandler",
    "NodeStatus",
    "NodeType",
    "Workflow",
    "WorkflowEdge",
    "WorkflowEngine",
    "WorkflowError",
    "WorkflowExecution",
    "WorkflowNode",
    "WorkflowStatus",
]
