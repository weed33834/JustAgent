"""Planning layer — task decomposition, plan management and execution strategy.

This module implements the **Planning** layer of the JustAgent enterprise AI
agent architecture. Research on enterprise agent architectures (e.g. the
canonical Perception / Planning / Memory / Tools / Execution / Orchestration /
Security / Observability / Evaluation layering) shows that "Planning" is the
subsystem responsible for *decomposing* a high-level goal into a dependency
graph of verifiable sub-tasks and *scheduling* them for execution.

The existing :mod:`justagent.agent.plan_act` module only handles **mode
switching** (``plan`` / ``act`` / ``yolo``) — i.e. whether the agent is allowed
to make changes. It does **not** perform task decomposition or build execution
plans. This module fills that gap and is intentionally independent: it owns the
``Plan`` / ``Task`` data model, a dependency-aware scheduler (topological
sort), an LLM-assisted :class:`TaskDecomposer` and a thread-safe
:class:`Planner` registry.

Design:

* :class:`TaskStatus` / :class:`TaskPriority` / :class:`PlanStatus` /
  :class:`ExecutionStrategy` — typed lifecycle enumerations.
* :class:`Task` — a single unit of work: description, status, priority,
  dependencies (task ids), nested ``subtasks``, token estimates and timestamps.
* :class:`Plan` — an ordered collection of :class:`Task` objects plus shared
  metadata and lifecycle status.
* :func:`topological_sort` — Kahn's-algorithm dependency levelling, shared
  with the workflow engine's approach (independent tasks within a level may run
  concurrently).
* :class:`TaskDecomposer` — uses the :class:`ModelGateway` (lazy-imported to
  avoid circular imports) to break a complex task into sub-tasks, with a
  deterministic rule-based fallback when no LLM is available.
* :class:`Planner` — thread-safe (``threading.RLock``) registry that creates
  plans, mutates task state, computes the next runnable task, reports progress
  and can drive execution via an injected async executor.

All data structures are Pydantic v2 models; LLM calls expose both sync and
``async`` variants (the async variants delegate to the sync ones through
:func:`asyncio.to_thread`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("justagent.agent.planning")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PlanningError(Exception):
    """Raised for invalid plan/task operations or scheduling failures."""


class PlanValidationError(PlanningError):
    """Raised when a plan's dependency graph is invalid (cycle / dangling ref)."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):  # noqa: UP042 - match existing codebase style
    """Lifecycle status of a single :class:`Task`."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

    @property
    def is_terminal(self) -> bool:
        """Whether this status is a final state (no further transitions)."""

        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    @property
    def is_runnable(self) -> bool:
        """Whether a task in this status could still be scheduled."""

        return self in (TaskStatus.PENDING, TaskStatus.BLOCKED)


class TaskPriority(str, Enum):  # noqa: UP042 - match existing codebase style
    """Relative importance of a :class:`Task`.

    Higher :attr:`weight` means higher scheduling priority.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> int:
        """Numeric weight used for priority-aware scheduling."""

        return _PRIORITY_WEIGHT[self.value]


_PRIORITY_WEIGHT: dict[str, int] = {
    TaskPriority.LOW.value: 1,
    TaskPriority.MEDIUM.value: 2,
    TaskPriority.HIGH.value: 3,
    TaskPriority.CRITICAL.value: 4,
}


class PlanStatus(str, Enum):  # noqa: UP042 - match existing codebase style
    """Lifecycle status of a :class:`Plan`."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether this status is a final state."""

        return self in (PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED)


class ExecutionStrategy(str, Enum):  # noqa: UP042 - match existing codebase style
    """How a plan's tasks should be scheduled for execution.

    * ``SEQUENTIAL`` — tasks run one at a time, in dependency / plan order.
    * ``PARALLEL`` — independent tasks within a dependency level run
      concurrently via :func:`asyncio.gather`.
    * ``ADAPTIVE`` — parallel where safe, but a failed task aborts the rest of
      its level before proceeding.
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """A single unit of work within a :class:`Plan`.

    Attributes:
        id: Unique task identifier (auto-generated UUID4 hex when omitted).
        description: A clear, actionable description of the work.
        status: Current :class:`TaskStatus`.
        priority: Scheduling :class:`TaskPriority`.
        dependencies: IDs of tasks that must reach ``COMPLETED`` before this
            one can start (defines the plan's DAG).
        subtasks: Nested child tasks produced by decomposition. Subtasks are
            organizational; for scheduling the plan is flattened via
            :meth:`flatten`.
        estimated_tokens: Optional LLM token budget estimate.
        actual_tokens: Accumulated tokens consumed while executing this task.
        created_at: Epoch timestamp of creation.
        completed_at: Epoch timestamp when the task reached a terminal state.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: list[str] = Field(default_factory=list)
    subtasks: list[Task] = Field(default_factory=list)
    estimated_tokens: int | None = None
    actual_tokens: int = 0
    created_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    #: Free-form metadata (e.g. ``{"tool": "write_to_file", "path": "..."}``).
    metadata: dict[str, Any] = Field(default_factory=dict)

    def flatten(self) -> list[Task]:
        """Return this task followed by all nested subtasks (depth-first)."""

        result: list[Task] = [self]
        for sub in self.subtasks:
            result.extend(sub.flatten())
        return result

    def mark(self, status: TaskStatus) -> Task:
        """Transition this task to *status* and update timestamps.

        Sets :attr:`completed_at` when entering a terminal state and clears it
        when leaving one. Returns ``self`` for chaining.
        """

        self.status = status
        if status.is_terminal and self.completed_at is None:
            self.completed_at = time.time()
        elif not status.is_terminal:
            self.completed_at = None
        return self


class Plan(BaseModel):
    """An execution plan: an ordered task list plus lifecycle metadata.

    Attributes:
        id: Unique plan identifier.
        name: Human-readable name.
        description: Free-form description of the plan's goal.
        tasks: Top-level tasks (subtasks are nested on each :class:`Task`).
        status: Current :class:`PlanStatus`.
        created_at: Epoch timestamp of creation.
        updated_at: Epoch timestamp of the last modification.
        metadata: Free-form plan-level metadata.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: str = ""
    tasks: list[Task] = Field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> Plan:
        """Refresh :attr:`updated_at` and return ``self``."""

        self.updated_at = time.time()
        return self

    def flatten_tasks(self) -> list[Task]:
        """Return every task in the plan, including nested subtasks."""

        flat: list[Task] = []
        for task in self.tasks:
            flat.extend(task.flatten())
        return flat

    def find_task(self, task_id: str) -> Task | None:
        """Locate a task (or subtask) by id anywhere in the plan tree."""

        for task in self.tasks:
            found = self._find_in_tree(task, task_id)
            if found is not None:
                return found
        return None

    @staticmethod
    def _find_in_tree(node: Task, task_id: str) -> Task | None:
        """Depth-first search for *task_id* within *node*'s subtree."""

        if node.id == task_id:
            return node
        for sub in node.subtasks:
            found = Plan._find_in_tree(sub, task_id)
            if found is not None:
                return found
        return None


# Resolve the self-referential ``subtasks: list[Task]`` forward reference.
# With ``from __future__ import annotations`` the annotation is a string and
# Pydantic v2 needs an explicit rebuild for recursive models.
Task.model_rebuild()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


#: System instruction shared by all LLM-assisted planning calls.
_PLANNING_SYSTEM_PROMPT = (
    "You are an expert planning assistant for an enterprise AI agent platform. "
    "You decompose complex goals into concrete, verifiable sub-tasks and "
    "refine execution plans. You always respond with STRICT JSON only — no "
    "prose, no markdown code fences."
)


#: Template for breaking a single task into sub-tasks.
TASK_DECOMPOSITION_PROMPT = """\
Decompose the following task into a sequence of concrete, independently \
verifiable sub-tasks suitable for execution by an AI agent.

Task:
{task}

{context_section}

Return ONLY a JSON array (no prose, no code fences) of at most {max_subtasks} \
elements. Each element MUST have:
- "description": a single clear, actionable sentence.
- "priority": one of "low", "medium", "high", "critical".
- "dependencies": a list of 0-based indices of other sub-tasks (within this \
array) that must finish before this one starts. Use [] for the first task.
- "estimated_tokens": an integer estimate of the LLM tokens required, or omit.

Order the array in the suggested execution order. Keep each sub-task small \
enough to complete in a single agent turn.
"""


#: Template for refining an existing plan's task list.
PLAN_REFINEMENT_PROMPT = """\
Review and refine the following execution plan. You may re-order tasks, adjust \
priorities, merge redundant steps, split overly broad ones, or add missing \
prerequisite / verification steps.

Current plan ({plan_name}):
{plan_json}

Return ONLY a JSON array (no prose, no code fences) describing the refined \
task list. Each element MUST have:
- "id": reuse the existing task id when refining; use a new hex string only \
for newly added tasks.
- "description": a single clear, actionable sentence.
- "priority": one of "low", "medium", "high", "critical".
- "dependencies": a list of task ids that must finish first.
- "estimated_tokens": optional integer.

Preserve the plan's intent and do not drop tasks unless they are genuinely \
redundant.
"""


# ---------------------------------------------------------------------------
# Dependency analysis (topological sort)
# ---------------------------------------------------------------------------


def topological_sort(tasks: Sequence[Task]) -> list[list[Task]]:
    """Group *tasks* into dependency levels using Kahn's algorithm.

    Each returned level is a list of tasks whose dependencies have all been
    resolved by the preceding levels. Tasks within a single level are
    independent and may execute concurrently. Tasks are ordered within a level
    by descending priority then ascending creation time so the most urgent,
    oldest work is scheduled first.

    Raises:
        PlanValidationError: if the dependency graph contains a cycle or a
            dependency references an unknown task id.
    """

    task_list = list(tasks)
    if not task_list:
        return []

    # Detect duplicate ids and dangling dependency references.
    ids = [t.id for t in task_list]
    if len(ids) != len(set(ids)):
        raise PlanValidationError("task list contains duplicate ids")
    id_set = set(ids)
    for task in task_list:
        for dep in task.dependencies:
            if dep not in id_set:
                raise PlanValidationError(f"task {task.id!r} depends on unknown task {dep!r}")

    # Build adjacency + in-degree from the dependency edges.
    adjacency: dict[str, list[str]] = {t.id: [] for t in task_list}
    in_degree: dict[str, int] = {t.id: 0 for t in task_list}
    task_map = {t.id: t for t in task_list}
    for task in task_list:
        for dep in task.dependencies:
            # Guard against duplicate dependency entries.
            if task.id not in adjacency[dep]:
                adjacency[dep].append(task.id)
                in_degree[task.id] += 1

    levels: list[list[Task]] = []
    current_ids = [tid for tid, deg in in_degree.items() if deg == 0]
    processed = 0
    while current_ids:
        # Order the current level: highest priority first, then oldest first.
        current_ids.sort(
            key=lambda tid: (
                -task_map[tid].priority.weight,
                task_map[tid].created_at,
            )
        )
        level = [task_map[tid] for tid in current_ids]
        levels.append(level)
        processed += len(current_ids)

        next_ids: list[str] = []
        for tid in current_ids:
            for child in adjacency[tid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_ids.append(child)
        current_ids = next_ids

    if processed != len(task_list):
        raise PlanValidationError(
            "task dependency graph contains a cycle "
            f"(processed {processed} of {len(task_list)} tasks)"
        )
    return levels


def execution_order(
    tasks: Sequence[Task],
    *,
    include_completed: bool = False,
) -> list[list[Task]]:
    """Return dependency levels for *tasks*, optionally skipping done work.

    When ``include_completed`` is False (the default) COMPLETED tasks are
    dropped and references to them are pruned from other tasks' dependencies
    (they are treated as satisfied), so the caller sees only the remaining
    work. Non-completed tasks (including FAILED / BLOCKED) are retained so
    their downstream references stay valid.
    """

    all_tasks = list(tasks)
    if include_completed:
        return topological_sort(all_tasks)

    completed_ids = {t.id for t in all_tasks if t.status is TaskStatus.COMPLETED}
    remaining: list[Task] = []
    for task in all_tasks:
        if task.status is TaskStatus.COMPLETED:
            continue
        # Strip dependencies on completed predecessors (now satisfied) without
        # mutating the caller's task instance.
        pruned = task.model_copy(deep=True)
        pruned.dependencies = [dep for dep in task.dependencies if dep not in completed_ids]
        remaining.append(pruned)
    return topological_sort(remaining)


# ---------------------------------------------------------------------------
# TaskDecomposer
# ---------------------------------------------------------------------------


#: Signature of an async task executor used by :meth:`Planner.execute_plan`.
TaskExecutor = Callable[[Task], Awaitable[bool]]


class TaskDecomposer:
    """Breaks a complex task into sub-tasks, optionally LLM-assisted.

    When a :class:`ModelGateway` is supplied, the decomposer prompts the model
    to produce a structured JSON task list and parses it into :class:`Task`
    objects with wired-up dependencies. When no gateway is available (or the
    LLM call fails) it falls back to a deterministic, rule-based
    decomposition: a linear chain of *analyze → design → implement → verify*
    sub-tasks.

    The :class:`ModelGateway` is imported lazily inside the LLM code paths to
    avoid a circular import between ``justagent.agent`` and
    ``justagent.adapters``.

    Example::

        decomposer = TaskDecomposer(gateway=my_gateway)
        subtasks = decomposer.decompose("Ship version 3.1 to production")
        for t in subtasks:
            print(t.priority, t.description, t.dependencies)
    """

    def __init__(
        self,
        gateway: Any | None = None,
        *,
        max_subtasks: int = 8,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._max_subtasks = max(1, max_subtasks)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._model = model

    @property
    def has_llm(self) -> bool:
        """Whether an LLM gateway is configured for assisted decomposition."""

        return self._gateway is not None

    # -- public API -------------------------------------------------------

    def decompose(
        self,
        task_description: str,
        *,
        context: str = "",
    ) -> list[Task]:
        """Decompose *task_description* into a list of sub-tasks.

        Uses the LLM when available, otherwise the rule-based fallback. The
        fallback is also used if the LLM call raises or returns unparseable
        output, so this method never fails for a transient model error.
        """

        if not task_description or not task_description.strip():
            raise PlanningError("task_description must not be empty")

        if self._gateway is None:
            logger.debug("No LLM gateway; using rule-based decomposition")
            return self._rule_based_decompose(task_description, context)

        try:
            subtasks = self._llm_decompose(task_description, context)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("LLM decomposition failed (%s); falling back to rules", exc)
            return self._rule_based_decompose(task_description, context)

        if not subtasks:
            logger.warning("LLM returned no subtasks; falling back to rules")
            return self._rule_based_decompose(task_description, context)
        return subtasks

    async def adecompose(
        self,
        task_description: str,
        *,
        context: str = "",
    ) -> list[Task]:
        """Async variant of :meth:`decompose`.

        Runs the (synchronous) gateway call in a worker thread via
        :func:`asyncio.to_thread` so the event loop is not blocked.
        """

        return await asyncio.to_thread(self.decompose, task_description, context=context)

    # -- LLM-assisted decomposition --------------------------------------

    def _llm_decompose(self, task_description: str, context: str) -> list[Task]:
        """Call the model gateway and parse the JSON sub-task list."""

        from justagent.adapters.model_gateway import (
            ChatCompletionRequest,
            ChatMessage,
            ModelGateway,
        )

        gateway: ModelGateway = self._gateway  # type: ignore[assignment]
        prompt = TASK_DECOMPOSITION_PROMPT.format(
            task=task_description.strip(),
            context_section=(f"Additional context:\n{context.strip()}" if context.strip() else ""),
            max_subtasks=self._max_subtasks,
        )
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=_PLANNING_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=False,
        )
        logger.debug("Requesting LLM task decomposition for: %s", task_description[:120])
        response = gateway.chat(request)
        return self._parse_decomposition(response.content)

    def _parse_decomposition(self, content: str) -> list[Task]:
        """Parse the model's JSON array into :class:`Task` objects.

        Dependencies arrive as 0-based indices into the returned array; they
        are translated to real task ids once the tasks exist.
        """

        raw_items = _extract_json_array(content)
        if not raw_items:
            return []

        tasks: list[Task] = []
        for item in raw_items[: self._max_subtasks]:
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            priority = _coerce_priority(item.get("priority"))
            estimated = _coerce_int(item.get("estimated_tokens"))
            tasks.append(
                Task(
                    description=description,
                    priority=priority,
                    estimated_tokens=estimated,
                )
            )

        # Resolve index-based dependencies into task ids.
        for index, task in enumerate(tasks):
            raw_deps = item_at(raw_items, index).get("dependencies") or []
            if not isinstance(raw_deps, list):
                continue
            for dep in raw_deps:
                dep_index = _coerce_int(dep)
                if dep_index is None or not (0 <= dep_index < len(tasks)):
                    continue
                dep_id = tasks[dep_index].id
                if dep_id != task.id and dep_id not in task.dependencies:
                    task.dependencies.append(dep_id)
        return tasks

    # -- Rule-based fallback ---------------------------------------------

    def _rule_based_decompose(
        self,
        task_description: str,
        context: str,
    ) -> list[Task]:
        """Deterministic fallback: a linear analyze→verify chain.

        Produces a fixed four-step plan with sequential dependencies so the
        result is always schedulable even without an LLM.
        """

        goal = task_description.strip()
        steps: list[tuple[str, TaskPriority]] = [
            (
                f"Analyze requirements and gather context for: {goal}",
                TaskPriority.HIGH,
            ),
            (
                f"Design the approach and outline the steps for: {goal}",
                TaskPriority.HIGH,
            ),
            (
                f"Implement the core solution for: {goal}",
                TaskPriority.MEDIUM,
            ),
            (
                f"Verify and test the results for: {goal}",
                TaskPriority.MEDIUM,
            ),
        ]
        if context.strip():
            steps.insert(
                0,
                (
                    f"Review provided context before proceeding: {context.strip()}",
                    TaskPriority.CRITICAL,
                ),
            )

        tasks: list[Task] = []
        previous_id: str | None = None
        for description, priority in steps:
            deps = [previous_id] if previous_id else []
            task = Task(description=description, priority=priority, dependencies=deps)
            tasks.append(task)
            previous_id = task.id
        logger.debug("Rule-based decomposition produced %d subtasks", len(tasks))
        return tasks


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """Thread-safe registry and scheduler for execution plans.

    Owns a collection of :class:`Plan` objects guarded by a
    :class:`threading.RLock`. Supports manual task definition, LLM-assisted
    decomposition (via an injectable :class:`TaskDecomposer`), dependency-aware
    scheduling, progress reporting and full plan execution with pluggable
    async executors.

    Example::

        planner = Planner(decomposer=TaskDecomposer(gateway=gw))
        plan = planner.create_plan("release", tasks=[
            Task(description="Cut release branch", priority=TaskPriority.HIGH),
            Task(description="Run CI", dependencies=[<id>]),
        ])
        planner.activate_plan(plan.id)
        while (task := planner.get_next_task(plan.id)) is not None:
            planner.update_task_status(plan.id, task.id, TaskStatus.IN_PROGRESS)
            ...
            planner.update_task_status(plan.id, task.id, TaskStatus.COMPLETED)
        print(planner.get_plan_progress(plan.id))
    """

    def __init__(self, decomposer: TaskDecomposer | None = None) -> None:
        self._lock = threading.RLock()
        self._plans: dict[str, Plan] = {}
        self._decomposer = decomposer or TaskDecomposer()

    # -- properties -------------------------------------------------------

    @property
    def decomposer(self) -> TaskDecomposer:
        """The :class:`TaskDecomposer` used for LLM-assisted planning."""

        return self._decomposer

    # -- plan lifecycle ---------------------------------------------------

    def create_plan(
        self,
        name: str,
        *,
        description: str = "",
        tasks: Sequence[Task] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Plan:
        """Create, validate and store a new :class:`Plan` (status DRAFT).

        Validates the dependency graph (no cycles / dangling references) before
        storing.
        """

        if not name or not name.strip():
            raise PlanningError("plan name must not be empty")
        plan = Plan(
            name=name.strip(),
            description=description,
            tasks=[t.model_copy(deep=True) for t in tasks] if tasks else [],
            metadata=dict(metadata) if metadata else {},
        )
        self._validate_plan(plan)
        with self._lock:
            self._plans[plan.id] = plan
        logger.info("Created plan %s (%s) with %d task(s)", plan.name, plan.id, len(plan.tasks))
        return plan

    def get_plan(self, plan_id: str) -> Plan | None:
        """Return the plan by id, or ``None``."""

        with self._lock:
            plan = self._plans.get(plan_id)
            return plan.model_copy(deep=True) if plan is not None else None

    def list_plans(self) -> list[Plan]:
        """Return a snapshot of all known plans."""

        with self._lock:
            return [p.model_copy(deep=True) for p in self._plans.values()]

    def remove_plan(self, plan_id: str) -> bool:
        """Delete a plan. Returns True if it existed."""

        with self._lock:
            return self._plans.pop(plan_id, None) is not None

    def activate_plan(self, plan_id: str) -> Plan:
        """Transition a plan from DRAFT/PAUSED to ACTIVE."""

        return self._set_plan_status(plan_id, PlanStatus.ACTIVE)

    def pause_plan(self, plan_id: str) -> Plan:
        """Transition a plan to PAUSED."""

        return self._set_plan_status(plan_id, PlanStatus.PAUSED)

    def cancel_plan(self, plan_id: str) -> Plan:
        """Transition a plan to CANCELLED and abort its in-flight tasks."""

        with self._lock:
            plan = self._require_plan(plan_id)
            plan.status = PlanStatus.CANCELLED
            now = time.time()
            for task in plan.flatten_tasks():
                if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED):
                    task.mark(TaskStatus.FAILED)
                    task.metadata["cancel_reason"] = "plan cancelled"
            plan.touch()
            plan.metadata.setdefault("cancelled_at", now)
            return plan.model_copy(deep=True)

    # -- task mutation ----------------------------------------------------

    def add_task(
        self,
        plan_id: str,
        task: Task,
        *,
        parent_id: str | None = None,
    ) -> Task:
        """Add *task* to a plan, optionally nested under *parent_id*.

        The plan is re-validated after insertion. Returns a copy of the
        inserted task (with its id assigned if it was empty).
        """

        if not task.id:
            task.id = uuid.uuid4().hex
        with self._lock:
            plan = self._require_plan(plan_id)
            if plan.status.is_terminal:
                raise PlanningError(
                    f"cannot add tasks to terminal plan {plan_id} ({plan.status.value})"
                )
            if parent_id is not None:
                parent = plan.find_task(parent_id)
                if parent is None:
                    raise PlanningError(f"parent task {parent_id!r} not found")
                parent.subtasks.append(task)
            else:
                plan.tasks.append(task)
            self._validate_plan(plan)
            plan.touch()
            return task.model_copy(deep=True)

    def remove_task(self, plan_id: str, task_id: str) -> bool:
        """Remove a task (and its subtree) from a plan.

        Returns True if the task was found and removed. Tasks that depended on
        the removed task have that dependency pruned.
        """

        with self._lock:
            plan = self._require_plan(plan_id)
            if plan.status.is_terminal:
                raise PlanningError(f"cannot remove tasks from terminal plan {plan_id}")
            removed = self._remove_from_list(plan.tasks, task_id)
            if not removed:
                return False
            # Prune dangling dependency references.
            for task in plan.flatten_tasks():
                task.dependencies = [dep for dep in task.dependencies if dep != task_id]
            plan.touch()
            return True

    @staticmethod
    def _remove_from_list(tasks: list[Task], task_id: str) -> bool:
        """Remove *task_id* from *tasks* or any nested subtask list."""

        for index, task in enumerate(tasks):
            if task.id == task_id:
                tasks.pop(index)
                return True
            if Planner._remove_from_list(task.subtasks, task_id):
                return True
        return False

    def update_task_status(
        self,
        plan_id: str,
        task_id: str,
        status: TaskStatus,
        *,
        actual_tokens: int | None = None,
    ) -> Task:
        """Transition a task to *status* and propagate side-effects.

        Sets :attr:`Task.completed_at` for terminal states, accumulates
        ``actual_tokens`` when provided, and propagates ``BLOCKED`` /
        re-``PENDING`` transitions to dependents when a predecessor fails or is
        retried. Completing the last runnable task finalises the plan.
        """

        with self._lock:
            plan = self._require_plan(plan_id)
            task = plan.find_task(task_id)
            if task is None:
                raise PlanningError(f"task {task_id!r} not found in plan {plan_id}")
            task.mark(status)
            if actual_tokens is not None:
                task.actual_tokens += max(0, actual_tokens)
            self._propagate_status(plan, task)
            self._maybe_finalize_plan(plan)
            plan.touch()
            return task.model_copy(deep=True)

    def _propagate_status(self, plan: Plan, changed: Task) -> None:
        """Cascade blocking / unblocking based on *changed*'s new status.

        * If *changed* FAILED, every still-runnable dependent is marked BLOCKED.
        * If *changed* moved to COMPLETED, any BLOCKED dependent whose other
          dependencies are now also satisfied is returned to PENDING.
        """

        dependents = [t for t in plan.flatten_tasks() if changed.id in t.dependencies]
        if changed.status is TaskStatus.FAILED:
            for dep in dependents:
                if dep.status in (TaskStatus.PENDING, TaskStatus.BLOCKED):
                    dep.mark(TaskStatus.BLOCKED)
                    dep.metadata["blocked_by"] = changed.id
        elif changed.status is TaskStatus.COMPLETED:
            task_map = {t.id: t for t in plan.flatten_tasks()}
            for dep in dependents:
                if dep.status is not TaskStatus.BLOCKED:
                    continue
                # Re-check all dependencies: ready to retry?
                deps_ok = True
                for d in dep.dependencies:
                    predecessor = task_map.get(d)
                    if predecessor is None or predecessor.status is not TaskStatus.COMPLETED:
                        deps_ok = False
                        break
                if deps_ok:
                    dep.mark(TaskStatus.PENDING)
                    dep.metadata.pop("blocked_by", None)

    def _maybe_finalize_plan(self, plan: Plan) -> None:
        """Mark the plan COMPLETED/FAILED based on its task states."""

        if plan.status.is_terminal:
            return
        tasks = plan.flatten_tasks()
        if not tasks:
            return
        if any(t.status is TaskStatus.FAILED for t in tasks):
            plan.status = PlanStatus.FAILED
        elif all(t.status is TaskStatus.COMPLETED for t in tasks):
            plan.status = PlanStatus.COMPLETED

    def reorder_tasks(self, plan_id: str, task_ids: Sequence[str]) -> Plan:
        """Reorder the plan's top-level tasks to match *task_ids*.

        *task_ids* must contain exactly the ids of the plan's top-level tasks.
        Subtask ordering is preserved. Returns a snapshot of the updated plan.
        """

        with self._lock:
            plan = self._require_plan(plan_id)
            top_ids = [t.id for t in plan.tasks]
            if set(task_ids) != set(top_ids):
                raise PlanningError("task_ids must contain exactly the plan's top-level task ids")
            order = {tid: index for index, tid in enumerate(task_ids)}
            plan.tasks.sort(key=lambda t: order[t.id])
            plan.touch()
            return plan.model_copy(deep=True)

    # -- scheduling -------------------------------------------------------

    def get_ready_tasks(self, plan_id: str) -> list[Task]:
        """Return all PENDING tasks whose dependencies are COMPLETED.

        Ordered by descending priority then ascending creation time. Use this
        for :attr:`ExecutionStrategy.PARALLEL` scheduling.
        """

        with self._lock:
            plan = self._require_plan(plan_id)
            tasks = plan.flatten_tasks()
            task_map = {t.id: t for t in tasks}
            ready = [
                t
                for t in tasks
                if t.status is TaskStatus.PENDING and self._dependencies_satisfied(t, task_map)
            ]
            ready.sort(key=lambda t: (-t.priority.weight, t.created_at))
            return [t.model_copy(deep=True) for t in ready]

    def get_next_task(
        self,
        plan_id: str,
        *,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
    ) -> Task | None:
        """Return the single next task to execute, or ``None`` if idle.

        Scheduling semantics:

        * ``SEQUENTIAL`` — the first ready task in plan (insertion) order.
        * ``ADAPTIVE`` / ``PARALLEL`` — the highest-priority ready task. For
          true batch parallelism use :meth:`get_ready_tasks` instead.
        """

        with self._lock:
            plan = self._require_plan(plan_id)
            tasks = plan.flatten_tasks()
            task_map = {t.id: t for t in tasks}
            ready = [
                t
                for t in tasks
                if t.status is TaskStatus.PENDING and self._dependencies_satisfied(t, task_map)
            ]
            if not ready:
                return None
            if strategy is ExecutionStrategy.SEQUENTIAL:
                chosen = ready[0]
            else:
                chosen = max(ready, key=lambda t: (t.priority.weight, -t.created_at))
            return chosen.model_copy(deep=True)

    def get_execution_order(
        self,
        plan_id: str,
        *,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        include_completed: bool = False,
    ) -> list[list[Task]]:
        """Return dependency levels for the plan's remaining tasks.

        Each level is an independent batch. For ``SEQUENTIAL`` the caller
        iterates levels and tasks in order; for ``PARALLEL`` / ``ADAPTIVE`` it
        runs each level's tasks concurrently.
        """

        with self._lock:
            plan = self._require_plan(plan_id)
            tasks = [t.model_copy(deep=True) for t in plan.flatten_tasks()]
        return execution_order(tasks, include_completed=include_completed)

    # -- progress ---------------------------------------------------------

    def get_plan_progress(self, plan_id: str) -> dict[str, Any]:
        """Return a progress summary for a plan.

        Includes per-status counts, completion percentage, and token budgets.
        """

        with self._lock:
            plan = self._require_plan(plan_id)
            tasks = plan.flatten_tasks()
            total = len(tasks)
            counts = {status.value: 0 for status in TaskStatus}
            estimated_tokens = 0
            actual_tokens = 0
            for task in tasks:
                counts[task.status.value] += 1
                if task.estimated_tokens is not None:
                    estimated_tokens += task.estimated_tokens
                actual_tokens += task.actual_tokens
            completed = counts[TaskStatus.COMPLETED.value]
            percentage = (completed / total * 100.0) if total else 0.0
            return {
                "plan_id": plan.id,
                "plan_name": plan.name,
                "status": plan.status.value,
                "total": total,
                "counts": counts,
                "pending": counts[TaskStatus.PENDING.value],
                "in_progress": counts[TaskStatus.IN_PROGRESS.value],
                "completed": completed,
                "failed": counts[TaskStatus.FAILED.value],
                "blocked": counts[TaskStatus.BLOCKED.value],
                "completion_percentage": round(percentage, 2),
                "estimated_tokens": estimated_tokens,
                "actual_tokens": actual_tokens,
                "created_at": plan.created_at,
                "updated_at": plan.updated_at,
            }

    # -- LLM-assisted planning -------------------------------------------

    def decompose_task(
        self,
        plan_id: str,
        task_id: str,
        *,
        context: str = "",
    ) -> list[Task]:
        """Decompose a task into subtasks and attach them as children.

        Uses the planner's :class:`TaskDecomposer`. Returns the created
        subtasks (deep copies).
        """

        with self._lock:
            plan = self._require_plan(plan_id)
            task = plan.find_task(task_id)
            if task is None:
                raise PlanningError(f"task {task_id!r} not found in plan {plan_id}")
            description = task.description

        subtasks = self._decomposer.decompose(description, context=context)
        with self._lock:
            plan = self._require_plan(plan_id)
            task = plan.find_task(task_id)
            if task is None:
                raise PlanningError(f"task {task_id!r} disappeared during decomposition")
            task.subtasks = subtasks
            self._validate_plan(plan)
            plan.touch()
            return [t.model_copy(deep=True) for t in subtasks]

    async def adecompose_task(
        self,
        plan_id: str,
        task_id: str,
        *,
        context: str = "",
    ) -> list[Task]:
        """Async variant of :meth:`decompose_task`."""

        return await asyncio.to_thread(self.decompose_task, plan_id, task_id, context=context)

    def refine_plan(self, plan_id: str) -> Plan:
        """Ask the LLM to refine a plan's task list in place.

        Returns the plan unchanged when no LLM gateway is configured (a debug
        log is emitted). On LLM failure the original task list is preserved.
        Raises :class:`PlanningError` if the plan does not exist.
        """

        with self._lock:
            plan = self._require_plan(plan_id)
            snapshot = plan.model_copy(deep=True)

        if not self._decomposer.has_llm:
            logger.debug("No LLM gateway; skipping plan refinement for %s", plan_id)
            return snapshot

        try:
            refined = self._llm_refine(snapshot)
        except Exception as exc:  # noqa: BLE001 - preserve plan on failure
            logger.warning("Plan refinement failed (%s); keeping original tasks", exc)
            return snapshot

        if not refined:
            return snapshot

        with self._lock:
            plan = self._require_plan(plan_id)
            plan.tasks = refined
            self._validate_plan(plan)
            plan.touch()
            return plan.model_copy(deep=True)

    async def arefine_plan(self, plan_id: str) -> Plan:
        """Async variant of :meth:`refine_plan`."""

        return await asyncio.to_thread(self.refine_plan, plan_id)

    def _llm_refine(self, plan: Plan) -> list[Task]:
        """Call the model gateway to produce a refined task list."""

        from justagent.adapters.model_gateway import (
            ChatCompletionRequest,
            ChatMessage,
            ModelGateway,
        )

        gateway: ModelGateway = self._decomposer._gateway  # type: ignore[assignment]
        plan_json = json.dumps(
            [
                {
                    "id": t.id,
                    "description": t.description,
                    "priority": t.priority.value,
                    "dependencies": t.dependencies,
                    "estimated_tokens": t.estimated_tokens,
                }
                for t in plan.flatten_tasks()
            ],
            ensure_ascii=False,
            indent=2,
        )
        prompt = PLAN_REFINEMENT_PROMPT.format(
            plan_name=plan.name,
            plan_json=plan_json,
        )
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=_PLANNING_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=self._decomposer._temperature,
            max_tokens=self._decomposer._max_tokens,
            stream=False,
        )
        logger.debug("Requesting LLM plan refinement for %s", plan.name)
        response = gateway.chat(request)
        return self._parse_refinement(response.content)

    def _parse_refinement(self, content: str) -> list[Task]:
        """Parse the refined JSON array into top-level :class:`Task` objects."""

        raw_items = _extract_json_array(content)
        if not raw_items:
            return []
        tasks: list[Task] = []
        for item in raw_items:
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            task_id = str(item.get("id", "")).strip() or uuid.uuid4().hex
            tasks.append(
                Task(
                    id=task_id,
                    description=description,
                    priority=_coerce_priority(item.get("priority")),
                    dependencies=[
                        str(dep)
                        for dep in (item.get("dependencies") or [])
                        if isinstance(dep, (str, int)) and str(dep)
                    ],
                    estimated_tokens=_coerce_int(item.get("estimated_tokens")),
                )
            )
        return tasks

    # -- execution --------------------------------------------------------

    async def execute_plan(
        self,
        plan_id: str,
        executor: TaskExecutor,
        *,
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
    ) -> Plan:
        """Drive a plan to completion using *executor* for each task.

        ``executor`` is an async callable receiving a :class:`Task` and
        returning ``True`` on success / ``False`` (or raising) on failure. The
        planner repeatedly computes the set of *ready* tasks (PENDING with all
        dependencies COMPLETED) and dispatches them according to *strategy*:

        * ``SEQUENTIAL`` — run one ready task at a time; stop at the first
          failure.
        * ``PARALLEL`` — run every ready task concurrently via
          :func:`asyncio.gather`; keep going even if some fail (their
          dependents are naturally blocked via status propagation, so
          independent branches still complete).
        * ``ADAPTIVE`` — run every ready task concurrently, but stop
          scheduling further batches as soon as one task fails (fail-fast).

        The loop terminates when no ready tasks remain (done or stalled).
        Returns a snapshot of the final plan state.
        """

        with self._lock:
            plan = self._require_plan(plan_id)
            if plan.status.is_terminal:
                raise PlanningError(f"cannot execute terminal plan {plan_id} ({plan.status.value})")
            plan.status = PlanStatus.ACTIVE
            plan.touch()

        while True:
            ready = self.get_ready_tasks(plan_id)
            if not ready:
                break

            if strategy is ExecutionStrategy.SEQUENTIAL:
                task = ready[0]
                ok = await self._run_task(plan_id, task, executor)
                if not ok:
                    break
            else:
                outcomes = await asyncio.gather(
                    *(self._run_task(plan_id, task, executor) for task in ready),
                    return_exceptions=True,
                )
                failed = any(
                    (isinstance(o, bool) and not o) or isinstance(o, BaseException)
                    for o in outcomes
                )
                if failed and strategy is ExecutionStrategy.ADAPTIVE:
                    # Fail-fast: stop scheduling further independent branches.
                    break

        # Finalize the plan status based on terminal task states.
        with self._lock:
            plan = self._require_plan(plan_id)
            self._maybe_finalize_plan(plan)
            if not plan.status.is_terminal:
                # Stalled (e.g. blocked tasks remain) — leave as ACTIVE/PAUSED.
                logger.warning(
                    "Plan %s finished execution without reaching a terminal state",
                    plan_id,
                )
            return plan.model_copy(deep=True)

    async def _run_task(
        self,
        plan_id: str,
        task: Task,
        executor: TaskExecutor,
    ) -> bool:
        """Mark a task IN_PROGRESS, run *executor*, then update its status."""

        self.update_task_status(plan_id, task.id, TaskStatus.IN_PROGRESS)
        try:
            ok = await executor(task)
        except Exception as exc:  # noqa: BLE001 - record failure, keep going
            logger.exception("Executor crashed for task %s: %s", task.id, exc)
            ok = False
        status = TaskStatus.COMPLETED if ok else TaskStatus.FAILED
        self.update_task_status(plan_id, task.id, status)
        return ok

    # -- internals --------------------------------------------------------

    def _require_plan(self, plan_id: str) -> Plan:
        """Return the live plan for *plan_id* or raise."""

        plan = self._plans.get(plan_id)
        if plan is None:
            raise PlanningError(f"plan not found: {plan_id}")
        return plan

    def _set_plan_status(self, plan_id: str, status: PlanStatus) -> Plan:
        """Transition a plan to *status* (with light validation)."""

        with self._lock:
            plan = self._require_plan(plan_id)
            plan.status = status
            plan.touch()
            return plan.model_copy(deep=True)

    @staticmethod
    def _dependencies_satisfied(task: Task, task_map: dict[str, Task]) -> bool:
        """Whether every dependency of *task* is COMPLETED."""

        for dep_id in task.dependencies:
            predecessor = task_map.get(dep_id)
            if predecessor is None:
                # Unknown dependency — treat as unsatisfied (stale reference).
                return False
            if predecessor.status is not TaskStatus.COMPLETED:
                return False
        return True

    @staticmethod
    def _validate_plan(plan: Plan) -> None:
        """Validate the plan's full (flattened) dependency graph."""

        tasks = plan.flatten_tasks()
        # Validate the topological order (raises on cycles / dangling refs).
        topological_sort(tasks)


# ---------------------------------------------------------------------------
# JSON / value coercion helpers
# ---------------------------------------------------------------------------


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Robustly extract a JSON array of objects from *text*.

    Handles raw JSON, markdown-fenced blocks, and leading/trailing prose by
    falling back to the substring between the first ``[`` and the last ``]``.
    """

    if not text:
        return []
    cleaned = text.strip()
    # Strip markdown code fences if present.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Fast path: the whole string is valid JSON.
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    # Fallback: carve out the outermost array.
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        chunk = cleaned[start : end + 1]
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


def _coerce_priority(value: Any) -> TaskPriority:
    """Coerce an arbitrary value into a :class:`TaskPriority`."""

    if isinstance(value, TaskPriority):
        return value
    if isinstance(value, str):
        try:
            return TaskPriority(value.strip().lower())
        except ValueError:
            pass
    return TaskPriority.MEDIUM


def _coerce_int(value: Any) -> int | None:
    """Coerce a value to a non-negative int, or ``None`` if impossible."""

    if value is None:
        return None
    if isinstance(value, bool):  # guard: bools are ints in Python
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            return int(value) if int(value) >= 0 else None
        except ValueError:
            return None
    return None


def item_at(items: Sequence[Any], index: int) -> Any:
    """Return ``items[index]`` or ``{}`` when out of range."""

    if 0 <= index < len(items):
        return items[index]
    return {}


__all__ = [
    "ExecutionStrategy",
    "Plan",
    "PlanStatus",
    "PlanValidationError",
    "Planner",
    "PlanningError",
    "Task",
    "TaskDecomposer",
    "TaskExecutor",
    "TaskPriority",
    "TaskStatus",
    "TASK_DECOMPOSITION_PROMPT",
    "PLAN_REFINEMENT_PROMPT",
    "execution_order",
    "topological_sort",
]
