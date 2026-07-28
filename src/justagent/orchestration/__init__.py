"""Decision execution and orchestration module for the JustAgent platform.

This package provides the multi-agent orchestration, workflow automation and
decision-execution layer that sits above the communication, resources and
knowledge subsystems. It is the "brain" that turns intent into coordinated
action across a fleet of cooperating agents.

Four integrated subsystems:

* **Workflow** (:mod:`justagent.orchestration.workflow`) — a DAG-based workflow
  engine. Workflows are graphs of :class:`WorkflowNode` steps connected by
  :class:`WorkflowEdge` arcs with optional boolean conditions. The
  :class:`WorkflowEngine` resolves execution order with a topological sort,
  runs independent nodes concurrently via :func:`asyncio.gather`, evaluates
  edge conditions to take or skip branches, and supports pause / resume /
  cancel through asyncio events. Eight built-in node types (task, decision,
  parallel, sequential, human-approval, webhook, agent-call, script) are each
  backed by an overridable async handler.

* **Decision** (:mod:`justagent.orchestration.decision`) — a decision execution
  engine that converts natural-language intent into structured actions. The
  :class:`IntentParser` uses regex + keyword heuristics to split a compound
  instruction into typed :class:`DecisionAction` objects (notify, schedule,
  allocate, query, approve, deploy, configure, analyze). The
  :class:`DecisionExecutor` validates, permission-checks and executes each
  action, dispatching to communication/notification, resources/scheduler and
  other subsystems via injectable handlers, with an approval gate for
  sensitive actions.

* **Mesh** (:mod:`justagent.orchestration.mesh`) — an agent mesh networking
  fabric for distributed multi-agent communication. Agents register as
  :class:`MeshNode` peers advertising :class:`AgentCapability` sets and
  exchange :class:`MeshMessage` packets. The :class:`MeshRouter` supports
  direct, broadcast and capability-based routing across STAR / MESH / TREE /
  RING topologies, with heartbeats and stale-node reaping.

* **Coordinator** (:mod:`justagent.orchestration.coordinator`) — a multi-agent
  task coordinator for delegation and result aggregation. The
  :class:`TaskCoordinator` selects the best agent for an :class:`AgentTask`
  using a pluggable :class:`CoordinationStrategy` (round-robin, least-loaded,
  capability-match, priority, random), delegates with timeout and retry
  semantics, tracks :class:`DelegationRecord` audit entries, can cancel
  in-flight tasks, and aggregates multiple :class:`AgentResult` objects.

All subsystems are asyncio-first and thread-safe (``threading.RLock`` for
registry metadata, ``asyncio`` primitives for execution coordination), use
Pydantic v2 for data models, and follow the ``justagent.orchestration.<submodule>``
logging namespace.

Architecture overview::

    Intent (NL)
        |
        v
    IntentParser -----------> DecisionIntent
                                  |
                                  v
                            DecisionExecutor --- dispatches to ---> communication / resources / knowledge
                                  |                                       ^
                                  v                                       |
                            WorkflowEngine --- DAG nodes --- AGENT_CALL --+-- AgentMesh (discovery/routing)
                                  |                                       |
                                  v                                       v
                            TaskCoordinator --- delegate ------------> MeshNode agents (handlers)

Quick start::

    from justagent.orchestration import (
        WorkflowEngine, WorkflowNode, WorkflowEdge, NodeType, WorkflowStatus,
        IntentParser, DecisionExecutor, DecisionType,
        AgentMesh, MeshNode, AgentCapability, MeshTopology,
        TaskCoordinator, AgentTask, CoordinationStrategy,
    )

    # 1. Parse and execute a decision.
    parser = IntentParser()
    executor = DecisionExecutor()
    intent = parser.parse("notify engineering about the deploy")
    result = await executor.execute(intent)

    # 2. Run a workflow DAG.
    engine = WorkflowEngine()
    wf = await engine.create_workflow(
        name="release",
        nodes=[
            WorkflowNode(id="notify", name="Notify", node_type=NodeType.TASK),
            WorkflowNode(id="ship", name="Ship", node_type=NodeType.SCRIPT,
                         dependencies=["notify"], config={"command": "echo shipped"}),
        ],
        edges=[WorkflowEdge(source="notify", target="ship")],
    )
    execution = await engine.create_execution(wf.id)
    await engine.start_execution(execution.id)

    # 3. Coordinate agents over the mesh.
    mesh = AgentMesh(topology=MeshTopology.MESH)
    await mesh.register_node(MeshNode(name="coder",
        capabilities={AgentCapability.CODE_GENERATION}))
    coordinator = TaskCoordinator()
    await coordinator.register_agent("coder",
        capabilities={AgentCapability.CODE_GENERATION})
    res = await coordinator.delegate(AgentTask(
        description="write a fizzbuzz",
        required_capabilities={AgentCapability.CODE_GENERATION},
    ))
"""

from __future__ import annotations

import logging

from justagent.orchestration.coordinator import (
    AgentCapability,
    AgentResult,
    AgentTask,
    AgentTaskHandler,
    CoordinationStrategy,
    CoordinatorConfig,
    CoordinatorError,
    DelegationRecord,
    TaskCoordinator,
)
from justagent.orchestration.decision import (
    ActionHandler,
    Approver,
    DecisionAction,
    DecisionError,
    DecisionExecutor,
    DecisionIntent,
    DecisionStatus,
    DecisionType,
    ExecutionResult,
    IntentParser,
    PermissionChecker,
)
from justagent.orchestration.mesh import (
    AgentMesh,
    MeshError,
    MeshMessage,
    MeshNode,
    MeshNodeStatus,
    MeshRouter,
    MeshTopology,
    MessageHandler,
)
from justagent.orchestration.workflow import (
    NodeHandler,
    NodeStatus,
    NodeType,
    Workflow,
    WorkflowEdge,
    WorkflowEngine,
    WorkflowError,
    WorkflowExecution,
    WorkflowNode,
    WorkflowStatus,
)

logger = logging.getLogger("justagent.orchestration")

__all__ = [
    # workflow.py
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
    # decision.py
    "ActionHandler",
    "Approver",
    "DecisionAction",
    "DecisionError",
    "DecisionExecutor",
    "DecisionIntent",
    "DecisionStatus",
    "DecisionType",
    "ExecutionResult",
    "IntentParser",
    "PermissionChecker",
    # mesh.py
    "AgentMesh",
    "MessageHandler",
    "MeshError",
    "MeshMessage",
    "MeshNode",
    "MeshNodeStatus",
    "MeshRouter",
    "MeshTopology",
    # coordinator.py
    "AgentCapability",
    "AgentResult",
    "AgentTask",
    "AgentTaskHandler",
    "CoordinationStrategy",
    "CoordinatorConfig",
    "CoordinatorError",
    "DelegationRecord",
    "TaskCoordinator",
]
