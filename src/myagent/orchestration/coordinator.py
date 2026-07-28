"""Multi-agent coordinator — task delegation, agent selection and result aggregation.

Brokers work between a caller and a pool of registered agents. Each agent
advertises a set of :class:`AgentCapability` values and an optional async
handler. The :class:`TaskCoordinator` selects the best agent for a task using
a pluggable :class:`CoordinationStrategy`, delegates the task with timeout and
retry semantics, tracks the delegation, and can aggregate results from
multiple agents into a single :class:`AgentResult`.

Design:

* :class:`CoordinationStrategy` — agent-selection policy (round-robin,
  least-loaded, capability-match, priority, random).
* :class:`AgentTask` — a unit of delegatable work with required capabilities,
  priority, deadline and inputs.
* :class:`AgentResult` — the outcome of a delegated task.
* :class:`DelegationRecord` — an audit trail entry for a delegation.
* :class:`CoordinatorConfig` — retry, timeout and fallback settings.
* :class:`TaskCoordinator` — async, thread-safe orchestrator.

The coordinator imports :class:`AgentCapability` from
:mod:`myagent.orchestration.mesh` so that capability taxonomies are shared
across the mesh and the coordinator.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from myagent.orchestration.mesh import AgentCapability

logger = logging.getLogger("myagent.orchestration.coordinator")


class CoordinatorError(Exception):
    """Raised for invalid coordination operations (no capable agent, ...)."""


class CoordinationStrategy(str, Enum):  # noqa: UP042 - match existing codebase style
    """Policy for selecting which agent receives a task."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_MATCH = "capability_match"
    PRIORITY = "priority"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class AgentTask(BaseModel):
    """A unit of work to be delegated to a capable agent.

    Attributes:
        id: Unique task identifier (auto-generated UUID4 hex when omitted).
        description: Human-readable description of the work.
        required_capabilities: Capabilities an agent must advertise to be
            eligible. An agent is eligible when its capabilities are a
            superset of this set.
        priority: Higher priority tasks are preferred (used by the PRIORITY
            strategy and for tie-breaking).
        deadline: Optional Unix timestamp by which the task must finish;
            tasks past their deadline fail fast.
        inputs: Arbitrary structured inputs passed to the agent handler.
        assigned_to: ID of the agent the task is currently assigned to.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    description: str
    required_capabilities: set[AgentCapability] = Field(default_factory=set)
    priority: int = 0
    deadline: float | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    assigned_to: str | None = None

    @property
    def is_expired(self) -> bool:
        """True when a deadline is set and has already passed."""

        return self.deadline is not None and time.time() > self.deadline


class AgentResult(BaseModel):
    """The outcome of a delegated task.

    Attributes:
        task_id: The task that was executed.
        agent_id: The agent that produced the result.
        status: Free-form status string (``"completed"``, ``"failed"``,
            ``"cancelled"``, ``"timeout"``, ``"partial"``).
        output: Human-readable output text.
        data: Structured result data.
        execution_time: Wall-clock seconds the agent spent on the task.
        error: Error message when the task did not complete successfully.
    """

    task_id: str
    agent_id: str = ""
    status: str = "completed"
    output: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True when the task completed without error."""

        return self.status == "completed" and self.error is None


class DelegationRecord(BaseModel):
    """An audit-trail entry for a single delegation.

    Attributes:
        task_id: The delegated task.
        from_agent: The delegating agent (or ``"coordinator"``).
        to_agent: The agent that received the task.
        delegated_at: UTC timestamp of delegation.
        status: Current delegation status (``"running"``, ``"completed"``,
            ``"failed"``, ``"cancelled"``).
    """

    task_id: str
    from_agent: str
    to_agent: str
    delegated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str = "running"


class CoordinatorConfig(BaseModel):
    """Configuration for a :class:`TaskCoordinator`.

    Attributes:
        strategy: Default :class:`CoordinationStrategy` for agent selection.
        max_retries: Maximum retry attempts after the first try (total
            executions = ``max_retries + 1``).
        timeout: Per-attempt execution timeout in seconds (0 = no timeout).
        fallback_enabled: When True, a failed delegation retries on a
            different capable agent before giving up.
    """

    strategy: CoordinationStrategy = CoordinationStrategy.CAPABILITY_MATCH
    max_retries: int = 1
    timeout: float = 60.0
    fallback_enabled: bool = True


# ---------------------------------------------------------------------------
# Internal agent descriptor
# ---------------------------------------------------------------------------


class _AgentDescriptor(BaseModel):
    """In-memory record of a registered agent (excluding its handler)."""

    id: str
    name: str = ""
    capabilities: set[AgentCapability] = Field(default_factory=set)
    priority: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    active_tasks: int = 0
    registered_at: float = Field(default_factory=time.time)

    def can_handle(self, required: set[AgentCapability]) -> bool:
        """True when this agent advertises every required capability."""

        return required.issubset(self.capabilities)


#: Signature of an async agent task handler. It receives the :class:`AgentTask`
#: and must return an :class:`AgentResult` (or a dict that can be coerced into
#: one).
AgentTaskHandler = Callable[[AgentTask], Awaitable[AgentResult | dict[str, Any]]]


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class TaskCoordinator:
    """Async, thread-safe multi-agent task coordinator.

    Agents are registered with :meth:`register_agent` together with their
    capability set and an optional async handler. :meth:`delegate` selects an
    agent via the configured :class:`CoordinationStrategy`, runs the handler
    with timeout and retry, and returns an :class:`AgentResult`. Delegations
    are tracked for audit and can be cancelled mid-flight.

    Example::

        coordinator = TaskCoordinator()
        await coordinator.register_agent(
            "coder",
            capabilities={AgentCapability.CODE_GENERATION},
            handler=my_code_handler,
        )
        task = AgentTask(
            description="generate a fizzbuzz function",
            required_capabilities={AgentCapability.CODE_GENERATION},
        )
        result = await coordinator.delegate(task)
        assert result.succeeded
    """

    def __init__(self, config: CoordinatorConfig | None = None) -> None:
        self._config = config or CoordinatorConfig()
        self._agents: dict[str, _AgentDescriptor] = {}
        self._handlers: dict[str, AgentTaskHandler] = {}
        self._delegations: dict[str, DelegationRecord] = {}
        self._active_tasks: dict[str, asyncio.Task[AgentResult | dict[str, Any]]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._rr_index = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    async def register_agent(
        self,
        agent_id: str,
        *,
        capabilities: set[AgentCapability] | None = None,
        handler: AgentTaskHandler | None = None,
        priority: int = 0,
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> _AgentDescriptor:
        """Register an agent with its capabilities and optional handler."""

        if not agent_id:
            raise CoordinatorError("agent_id must not be empty")
        descriptor = _AgentDescriptor(
            id=agent_id,
            name=name or agent_id,
            capabilities=set(capabilities) if capabilities else set(),
            priority=priority,
            metadata=dict(metadata) if metadata else {},
        )
        with self._lock:
            if agent_id in self._agents:
                raise CoordinatorError(f"agent {agent_id!r} is already registered")
            self._agents[agent_id] = descriptor
            if handler is not None:
                self._handlers[agent_id] = handler
        logger.info(
            "Registered agent %s (capabilities=%s, priority=%d)",
            agent_id,
            sorted(c.value for c in descriptor.capabilities),
            priority,
        )
        return descriptor

    async def deregister_agent(self, agent_id: str) -> _AgentDescriptor | None:
        """Remove an agent by id; return the removed descriptor or ``None``."""

        with self._lock:
            descriptor = self._agents.pop(agent_id, None)
            self._handlers.pop(agent_id, None)
        if descriptor is not None:
            logger.info("Deregistered agent %s", agent_id)
        return descriptor

    async def list_agents(self) -> list[_AgentDescriptor]:
        """Return all registered agent descriptors."""

        with self._lock:
            return sorted(self._agents.values(), key=lambda a: a.name)

    # ------------------------------------------------------------------
    # Agent selection
    # ------------------------------------------------------------------

    async def select_agent(
        self,
        task: AgentTask,
        strategy: CoordinationStrategy | None = None,
    ) -> str:
        """Select the best agent id for *task* using *strategy*.

        Raises :class:`CoordinatorError` when no capable agent is available.
        """

        chosen = strategy or self._config.strategy
        with self._lock:
            eligible = [
                desc
                for desc in self._agents.values()
                if desc.can_handle(task.required_capabilities)
            ]
            if not eligible:
                raise CoordinatorError(
                    f"no agent capable of {sorted(c.value for c in task.required_capabilities)}"
                )
            agent_id = self._apply_strategy(chosen, eligible, task)
        logger.debug(
            "Selected agent %s for task %s via %s",
            agent_id,
            task.id,
            chosen.value,
        )
        return agent_id

    def _apply_strategy(
        self,
        strategy: CoordinationStrategy,
        eligible: list[_AgentDescriptor],
        task: AgentTask,
    ) -> str:
        """Apply *strategy* to the eligible agent list and return an agent id."""

        if strategy is CoordinationStrategy.ROUND_ROBIN:
            return self._select_round_robin(eligible)
        if strategy is CoordinationStrategy.LEAST_LOADED:
            return self._select_least_loaded(eligible)
        if strategy is CoordinationStrategy.CAPABILITY_MATCH:
            return self._select_capability_match(eligible, task)
        if strategy is CoordinationStrategy.PRIORITY:
            return self._select_priority(eligible)
        if strategy is CoordinationStrategy.RANDOM:
            return self._select_random(eligible)
        # Fallback.
        return eligible[0].id

    def _select_round_robin(self, eligible: list[_AgentDescriptor]) -> str:
        ordered = sorted(eligible, key=lambda a: a.name)
        chosen = ordered[self._rr_index % len(ordered)]
        self._rr_index = (self._rr_index + 1) % max(len(ordered), 1)
        return chosen.id

    def _select_least_loaded(self, eligible: list[_AgentDescriptor]) -> str:
        return min(eligible, key=lambda a: (a.active_tasks, a.priority, a.name)).id

    def _select_capability_match(self, eligible: list[_AgentDescriptor], task: AgentTask) -> str:
        """Pick the most specialised agent (fewest surplus capabilities)."""

        def surplus(desc: _AgentDescriptor) -> int:
            return len(desc.capabilities - task.required_capabilities)

        return min(
            eligible,
            key=lambda a: (surplus(a), a.active_tasks, -a.priority, a.name),
        ).id

    def _select_priority(self, eligible: list[_AgentDescriptor]) -> str:
        return max(eligible, key=lambda a: (a.priority, -a.active_tasks, a.name)).id

    def _select_random(self, eligible: list[_AgentDescriptor]) -> str:
        return random.choice(eligible).id  # noqa: S311 - non-cryptographic selection

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    async def delegate(self, task: AgentTask) -> AgentResult:
        """Delegate *task* to a selected agent and return the result.

        Honours the configured timeout (per attempt) and retry policy. When
        ``fallback_enabled`` is True and the chosen agent fails after retries,
        the coordinator retries on a different capable agent.
        """

        if task.is_expired:
            result = AgentResult(
                task_id=task.id,
                status="failed",
                error="task deadline already passed",
            )
            await self._finalize_delegation(task.id, result)
            return result

        tried: set[str] = set()
        last_result: AgentResult | None = None
        attempts_remaining = 1 + self._config.fallback_enabled  # original + fallback

        while attempts_remaining > 0:
            attempts_remaining -= 1
            try:
                agent_id = await self._select_excluding(task, tried)
            except CoordinatorError:
                break  # no more capable agents
            tried.add(agent_id)
            task.assigned_to = agent_id
            await self.track_delegation(task.id, "coordinator", agent_id, "running")
            with self._lock:
                self._agents[agent_id].active_tasks += 1
            try:
                result = await self._run_with_retries(task, agent_id)
            finally:
                with self._lock:
                    if agent_id in self._agents:
                        self._agents[agent_id].active_tasks = max(
                            0, self._agents[agent_id].active_tasks - 1
                        )
            last_result = result
            if result.succeeded:
                await self._finalize_delegation(task.id, result)
                return result
            if attempts_remaining <= 0 or not self._config.fallback_enabled:
                break
            logger.info("Agent %s failed task %s; attempting fallback", agent_id, task.id)

        result = last_result or AgentResult(
            task_id=task.id,
            status="failed",
            error="no capable agent available",
        )
        await self._finalize_delegation(task.id, result)
        return result

    async def _select_excluding(self, task: AgentTask, exclude: set[str]) -> str:
        """Select an agent for *task* excluding ids in *exclude*."""

        with self._lock:
            eligible = [
                desc
                for desc in self._agents.values()
                if desc.id not in exclude and desc.can_handle(task.required_capabilities)
            ]
            if not eligible:
                raise CoordinatorError("no eligible agent after exclusions")
            return self._apply_strategy(self._config.strategy, eligible, task)

    async def _run_with_retries(self, task: AgentTask, agent_id: str) -> AgentResult:
        """Run the agent handler with timeout and retry semantics."""

        handler = self._handlers.get(agent_id)
        max_retries = self._config.max_retries
        attempt = 0
        last_result: AgentResult | None = None
        while attempt <= max_retries:
            attempt += 1
            started = time.monotonic()
            try:
                result = await self._invoke_handler(task, agent_id, handler)
            except TimeoutError:
                last_result = AgentResult(
                    task_id=task.id,
                    agent_id=agent_id,
                    status="timeout",
                    execution_time=time.monotonic() - started,
                    error=f"timed out after {self._config.timeout}s",
                )
                logger.warning(
                    "Agent %s timed out on task %s (attempt %d)",
                    agent_id,
                    task.id,
                    attempt,
                )
            except asyncio.CancelledError:
                last_result = AgentResult(
                    task_id=task.id,
                    agent_id=agent_id,
                    status="cancelled",
                    execution_time=time.monotonic() - started,
                    error="task cancelled",
                )
                raise
            except Exception as exc:  # noqa: BLE001
                last_result = AgentResult(
                    task_id=task.id,
                    agent_id=agent_id,
                    status="failed",
                    execution_time=time.monotonic() - started,
                    error=str(exc),
                )
                logger.warning(
                    "Agent %s raised on task %s (attempt %d): %s",
                    agent_id,
                    task.id,
                    attempt,
                    exc,
                )
            else:
                if not isinstance(result, AgentResult):
                    result = self._coerce_result(task, agent_id, result, started)
                else:
                    result.execution_time = time.monotonic() - started
                    if not result.agent_id:
                        result.agent_id = agent_id
                    if not result.task_id:
                        result.task_id = task.id
                return result

            if attempt <= max_retries:
                await asyncio.sleep(0)  # yield before retry
        return last_result  # type: ignore[return-value]

    async def _invoke_handler(
        self,
        task: AgentTask,
        agent_id: str,
        handler: AgentTaskHandler | None,
    ) -> AgentResult | dict[str, Any]:
        """Run the handler (or simulator) as a cancellable, timeout-bounded task."""

        coro = handler(task) if handler is not None else self._simulate_handler(task)
        exec_task = asyncio.ensure_future(coro)
        with self._lock:
            self._active_tasks[task.id] = exec_task
            self._cancel_events[task.id] = asyncio.Event()
        timeout = self._config.timeout if self._config.timeout > 0 else None
        try:
            if timeout is not None:
                return await asyncio.wait_for(exec_task, timeout=timeout)
            return await exec_task
        finally:
            with self._lock:
                self._active_tasks.pop(task.id, None)
                self._cancel_events.pop(task.id, None)

    async def _simulate_handler(self, task: AgentTask) -> AgentResult:
        """Default handler for agents without an explicit one (simulated)."""

        await asyncio.sleep(0)
        return AgentResult(
            task_id=task.id,
            status="completed",
            output=f"simulated completion of {task.description!r}",
            data={"inputs": task.inputs, "simulated": True},
        )

    @staticmethod
    def _coerce_result(
        task: AgentTask,
        agent_id: str,
        raw: AgentResult | dict[str, Any],
        started: float,
    ) -> AgentResult:
        """Normalise a handler return value into an :class:`AgentResult`."""

        if isinstance(raw, AgentResult):
            raw.execution_time = time.monotonic() - started
            if not raw.agent_id:
                raw.agent_id = agent_id
            if not raw.task_id:
                raw.task_id = task.id
            return raw
        return AgentResult(
            task_id=task.id,
            agent_id=agent_id,
            status=str(raw.get("status", "completed")),
            output=str(raw.get("output", "")),
            data=raw.get("data", {}) if isinstance(raw.get("data"), dict) else {},
            execution_time=time.monotonic() - started,
            error=raw.get("error"),
        )

    # ------------------------------------------------------------------
    # Delegation tracking & cancellation
    # ------------------------------------------------------------------

    async def track_delegation(
        self,
        task_id: str,
        from_agent: str,
        to_agent: str,
        status: str = "running",
    ) -> DelegationRecord:
        """Record or update a delegation record."""

        with self._lock:
            record = DelegationRecord(
                task_id=task_id,
                from_agent=from_agent,
                to_agent=to_agent,
                status=status,
            )
            self._delegations[task_id] = record
            return record

    async def _finalize_delegation(self, task_id: str, result: AgentResult) -> None:
        """Update the delegation record with the terminal result status."""

        with self._lock:
            record = self._delegations.get(task_id)
            if record is not None:
                record.status = result.status

    async def get_active_tasks(self) -> list[str]:
        """Return the ids of tasks currently in flight."""

        with self._lock:
            return list(self._active_tasks.keys())

    async def get_delegation(self, task_id: str) -> DelegationRecord | None:
        """Return the delegation record for *task_id*, or ``None``."""

        with self._lock:
            record = self._delegations.get(task_id)
            return record.model_copy(deep=True) if record else None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an in-flight task. Returns True if it was active."""

        with self._lock:
            exec_task = self._active_tasks.get(task_id)
            event = self._cancel_events.get(task_id)
        if exec_task is None:
            return False
        if event is not None:
            event.set()
        exec_task.cancel()
        logger.info("Cancel requested for task %s", task_id)
        return True

    # ------------------------------------------------------------------
    # Result aggregation
    # ------------------------------------------------------------------

    async def aggregate_results(self, results: list[AgentResult]) -> AgentResult:
        """Combine multiple agent results into a single :class:`AgentResult`.

        Outputs are concatenated, ``data`` dicts are merged (later keys win),
        execution times are summed, and the aggregate status is ``completed``
        only when every input succeeded (otherwise ``partial`` or ``failed``).
        """

        if not results:
            return AgentResult(task_id="", status="failed", error="no results to aggregate")
        if len(results) == 1:
            return results[0]
        task_id = results[0].task_id
        outputs = "\n---\n".join(r.output for r in results if r.output)
        merged_data: dict[str, Any] = {}
        for r in results:
            merged_data.update(r.data)
        total_time = sum(r.execution_time for r in results)
        all_succeeded = all(r.succeeded for r in results)
        any_succeeded = any(r.succeeded for r in results)
        if all_succeeded:
            status = "completed"
        elif any_succeeded:
            status = "partial"
        else:
            status = "failed"
        errors = [r.error for r in results if r.error]
        return AgentResult(
            task_id=task_id,
            agent_id=",".join(sorted({r.agent_id for r in results if r.agent_id})),
            status=status,
            output=outputs,
            data={"partial_results": merged_data, "count": len(results)},
            execution_time=total_time,
            error="; ".join(errors) if errors else None,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of coordinator state for dashboards."""

        with self._lock:
            by_capability: dict[str, int] = {}
            for desc in self._agents.values():
                for cap in desc.capabilities:
                    by_capability[cap.value] = by_capability.get(cap.value, 0) + 1
            return {
                "agents": len(self._agents),
                "active_tasks": len(self._active_tasks),
                "delegations": len(self._delegations),
                "by_capability": by_capability,
                "strategy": self._config.strategy.value,
            }


__all__ = [
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
