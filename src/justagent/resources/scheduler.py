"""Task scheduler — priority queues, resource matching and preemption.

The scheduler turns a stream of submitted :class:`Task` objects into
actual executions on the resources catalogued by a
:class:`~justagent.resources.registry.ResourceRegistry`. It owns:

* A **priority queue** (heapq-backed) ordered by :class:`TaskPriority`
  (CRITICAL > HIGH > NORMAL > LOW) with FIFO tie-breaking.
* **Resource matching**: each task carries a :class:`ResourceRequirements`
  spec; the scheduler finds the least-loaded eligible resource via the
  registry's ``best_match`` and reserves capacity on it.
* **Preemptive scheduling**: when a higher-priority task cannot find a free
  resource, the scheduler preempts the lowest-priority running task whose
  resource satisfies the newcomer, re-queues the victim and hands the
  resource to the newcomer.
* **Execution tracking**: timeout, retry-with-backoff, cancellation and a
  pluggable runner (default: local :mod:`subprocess`).

The scheduler is thread-safe. Execution is driven explicitly by the caller
through :meth:`TaskScheduler.run_pending` (concurrent) or
:meth:`TaskScheduler.execute` (single blocking call), so the scheduler can
be embedded in an event loop or driven by a daemon thread.
"""

from __future__ import annotations

import contextlib
import heapq
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from justagent.resources.registry import (
    ResourceRecord,
    ResourceRegistry,
    ResourceType,
)

logger = logging.getLogger("justagent.resources")

#: Default per-resource concurrency ceiling.
DEFAULT_MAX_CONCURRENT_PER_RESOURCE = 1

#: Sentinel timeout meaning "no timeout".
NO_TIMEOUT = 0.0


class TaskPriority(str, Enum):  # noqa: UP042
    """Scheduling priority. Higher enum weight runs first / may preempt."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> int:
        """Numeric weight used for heap ordering and preemption decisions.

        Strictly higher weight means strictly higher priority.
        """

        return _PRIORITY_WEIGHTS[self]


_PRIORITY_WEIGHTS: dict[TaskPriority, int] = {
    TaskPriority.LOW: 1,
    TaskPriority.NORMAL: 5,
    TaskPriority.HIGH: 10,
    TaskPriority.CRITICAL: 100,
}


class TaskStatus(str, Enum):  # noqa: UP042
    """Lifecycle state of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"
    TIMED_OUT = "timed_out"


#: Statuses that represent a finished task (terminal).
_TERMINAL_STATES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT}
)


class ResourceRequirements(BaseModel):
    """Minimum resource needs declared by a task.

    Mirrors the capability axes of
    :class:`~justagent.resources.registry.ResourceSpec`. ``preferred_type``
    and ``preferred_resource_id`` express soft affinity: when set, the
    scheduler prefers (but is not restricted to) that type / resource.
    """

    min_cpu: int = 0
    min_memory: float = 0.0
    min_disk: float = 0.0
    min_gpu: int = 0
    min_gpu_memory: float = 0.0
    min_bandwidth: int = 0
    required_tags: list[str] = Field(default_factory=list)
    architecture: str = ""
    preferred_type: ResourceType | None = None
    preferred_resource_id: str = ""

    def match_kwargs(self) -> dict[str, Any]:
        """Return kwargs for :meth:`ResourceRegistry.best_match`."""

        return {
            "min_cpu": self.min_cpu,
            "min_memory": self.min_memory,
            "min_disk": self.min_disk,
            "min_gpu": self.min_gpu,
            "min_gpu_memory": self.min_gpu_memory,
            "min_bandwidth": self.min_bandwidth,
            "required_tags": list(self.required_tags) or None,
            "architecture": self.architecture,
            "type": self.preferred_type,
        }


class RetryPolicy(BaseModel):
    """Retry-on-failure configuration.

    Attributes:
        max_retries: Maximum number of retry attempts after the first try
            (so total executions = ``max_retries + 1``).
        base_delay: Seconds to wait before the first retry.
        backoff_multiplier: Multiplier applied to the delay after each retry.
        max_delay: Upper bound on the per-retry delay.
        retry_on_timeout: Whether a timeout counts as a retryable failure.
    """

    max_retries: int = 0
    base_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    retry_on_timeout: bool = True


class TaskResult(BaseModel):
    """The outcome of executing a task once (or after retries)."""

    task_id: str
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    attempts: int = 1
    started_at: float = 0.0
    finished_at: float = 0.0
    duration: float = 0.0
    resource_id: str = ""
    error: str = ""

    @property
    def timed_out(self) -> bool:
        """True if the final attempt ended because of a timeout."""

        return not self.success and "timed out" in self.error.lower()


class Task(BaseModel):
    """A unit of schedulable work.

    Attributes:
        id: Stable unique identifier (auto-generated UUID4 hex when omitted).
        name: Human-readable name.
        priority: :class:`TaskPriority` used for queue ordering / preemption.
        command: Shell command executed by the default runner.
        args: Optional argv passed to the command (kept for inspection).
        env: Environment variable overrides for the subprocess.
        requirements: :class:`ResourceRequirements` the assigned resource must meet.
        timeout: Maximum execution seconds per attempt (0 = no timeout).
        retry_policy: :class:`RetryPolicy` applied on failure.
        status: Current :class:`TaskStatus`.
        created_at: Submission timestamp.
        started_at: Timestamp of the first execution attempt.
        finished_at: Timestamp when the task reached a terminal state.
        attempts: Number of execution attempts so far.
        assigned_resource_id: Id of the resource the task is pinned to.
        result: Final :class:`TaskResult` once terminal.
        preempted_by: Id of the task that preempted this one (if any).
        preempted_count: How many times this task has been preempted.
        metadata: Free-form caller metadata.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    priority: TaskPriority = TaskPriority.NORMAL
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    requirements: ResourceRequirements = Field(default_factory=ResourceRequirements)
    timeout: float = NO_TIMEOUT
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    attempts: int = 0
    assigned_resource_id: str = ""
    result: TaskResult | None = None
    preempted_by: str = ""
    preempted_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_terminal(self) -> bool:
        """True if the task has reached a final state."""

        return self.status in _TERMINAL_STATES

    def is_running(self) -> bool:
        """True if the task is currently executing."""

        return self.status is TaskStatus.RUNNING


#: Signature of an injectable execution runner.
#:
#: The runner receives the task, the resource it was assigned to, and a
#: cancellation :class:`threading.Event` that is set when the task is
#: cancelled or preempted. It must return a :class:`TaskResult`.
TaskRunner = Callable[[Task, ResourceRecord, threading.Event], TaskResult]


class SchedulerError(Exception):
    """Raised for invalid scheduler operations."""


class TaskScheduler:
    """Priority-aware, preemptive task scheduler over a resource registry.

    Example:

        >>> registry = ResourceRegistry()
        >>> registry.register(ResourceRecord(
        ...     name="worker-1", type=ResourceType.SERVER,
        ...     status=ResourceStatus.ONLINE,
        ...     capabilities=ResourceSpec(cpu_cores=4),
        ... ))  # doctest: +SKIP
        >>> sched = TaskScheduler(registry)
        >>> tid = sched.submit(Task(name="hello", command="echo hi",
        ...     priority=TaskPriority.HIGH,
        ...     requirements=ResourceRequirements(min_cpu=1)))  # doctest: +SKIP
        >>> results = sched.run_pending()  # doctest: +SKIP
    """

    def __init__(
        self,
        registry: ResourceRegistry,
        *,
        max_concurrent_per_resource: int = DEFAULT_MAX_CONCURRENT_PER_RESOURCE,
        runner: TaskRunner | None = None,
    ) -> None:
        if max_concurrent_per_resource < 1:
            raise SchedulerError("max_concurrent_per_resource must be >= 1")
        self._registry = registry
        self._max_concurrent = max_concurrent_per_resource
        self._runner: TaskRunner = runner or default_runner

        self._tasks: dict[str, Task] = {}
        # Priority heap entries: (-weight, seq, task_id). Lower rank first.
        self._heap: list[tuple[int, int, str]] = []
        self._seq = 0
        # Capacity reservations: resource_id -> set of task_ids running on it.
        self._reservations: dict[str, set[str]] = {}
        # Running subprocesses for cancel/preempt (default runner only).
        self._procs: dict[str, subprocess.Popen[Any]] = {}
        # Cancellation events per running task.
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Submission & lifecycle
    # ------------------------------------------------------------------

    def submit(self, task: Task) -> str:
        """Register a task and enqueue it. Returns the task id.

        If the task is CRITICAL and no free resource is available, the
        scheduler may immediately preempt a lower-priority running task.
        """

        if not task.name:
            raise SchedulerError("task name must not be empty")
        if task.command and not isinstance(task.command, str):
            raise SchedulerError("task.command must be a string")
        if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            task = task.model_copy(update={"status": TaskStatus.PENDING})

        with self._lock:
            if task.id in self._tasks:
                raise SchedulerError(f"task id {task.id!r} already exists")
            task = task.model_copy(update={"status": TaskStatus.PENDING})
            self._tasks[task.id] = task
            self._push(task)
            logger.info(
                "Submitted task %s (%s, priority=%s)",
                task.name,
                task.id,
                task.priority.value,
            )
        # Attempt preemption for urgent tasks outside the main lock to keep
        # the critical section short; preemption re-acquires the lock.
        if task.priority is TaskPriority.CRITICAL:
            self._maybe_preempt_for(task.id)
        return task.id

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending or running task. Returns True if it was active.

        Pending tasks are marked CANCELLED and removed from the queue.
        Running tasks have their cancellation event set (and, for the
        default runner, their subprocess terminated); their status becomes
        CANCELLED once the runner observes the signal.
        """

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.is_terminal():
                return False
            if task.status is TaskStatus.RUNNING:
                event = self._cancel_events.get(task_id)
                if event is not None:
                    event.set()
                proc = self._procs.get(task_id)
                if proc is not None and proc.poll() is None:
                    self._terminate(proc)
                self._release_resource(task_id)
                self._set_terminal(task, TaskStatus.CANCELLED, error="cancelled by user")
                logger.info("Cancelled running task %s", task.name)
            else:
                self._set_terminal(task, TaskStatus.CANCELLED, error="cancelled while pending")
                logger.info("Cancelled pending task %s", task.name)
            return True

    def get(self, task_id: str) -> Task | None:
        """Return a copy of the task, or None."""

        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, status: TaskStatus | None = None) -> list[Task]:
        """Return tasks (copies), optionally filtered by status, newest-first."""

        with self._lock:
            tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status is status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign(self) -> list[Task]:
        """Match pending tasks to free resources; returns newly-assigned tasks.

        Pops tasks from the priority queue; for each, finds the least-loaded
        schedulable resource with spare capacity that satisfies the task's
        requirements. Tasks that cannot be assigned are re-pushed onto the
        queue (preserving priority). Already-assigned tasks are skipped.
        """

        assigned: list[Task] = []
        # Drain the queue, then re-push what we cannot place.
        drained: list[tuple[int, int, str]] = []
        with self._lock:
            while self._heap:
                drained.append(heapq.heappop(self._heap))
            for _rank, _seq, task_id in drained:
                task = self._tasks.get(task_id)
                if task is None or task.status is not TaskStatus.PENDING:
                    continue
                resource = self._find_free_resource(task)
                if resource is None:
                    # Cannot place now; keep it queued.
                    heapq.heappush(self._heap, (_rank, _seq, task_id))
                    continue
                updated = task.model_copy(update={"assigned_resource_id": resource.id})
                self._tasks[task_id] = updated
                self._reservations.setdefault(resource.id, set()).add(task_id)
                self._bump_load(resource.id, delta_tasks=1)
                assigned.append(updated)
                logger.info(
                    "Assigned task %s to resource %s",
                    updated.name,
                    resource.name,
                )
        return assigned

    def _find_free_resource(self, task: Task) -> ResourceRecord | None:
        """Return a schedulable resource with spare capacity for ``task``."""

        kwargs = task.requirements.match_kwargs()
        # Try the preferred resource first (affinity).
        pref_id = task.requirements.preferred_resource_id
        if pref_id:
            pref = self._registry.get(pref_id)
            if (
                pref is not None
                and pref.is_schedulable()
                and self._has_capacity(pref.id)
                and pref.capabilities.satisfies(
                    min_cpu=kwargs["min_cpu"],
                    min_memory=kwargs["min_memory"],
                    min_disk=kwargs["min_disk"],
                    min_gpu=kwargs["min_gpu"],
                    min_gpu_memory=kwargs["min_gpu_memory"],
                    min_bandwidth=kwargs["min_bandwidth"],
                    required_tags=kwargs["required_tags"],
                    architecture=kwargs["architecture"],
                )
            ):
                return pref
        # Scan candidates from the registry, picking the least-loaded one
        # that still has capacity.
        candidates = self._registry.discover(schedulable_only=True)
        eligible = [
            r
            for r in candidates
            if self._has_capacity(r.id)
            and r.capabilities.satisfies(
                min_cpu=kwargs["min_cpu"],
                min_memory=kwargs["min_memory"],
                min_disk=kwargs["min_disk"],
                min_gpu=kwargs["min_gpu"],
                min_gpu_memory=kwargs["min_gpu_memory"],
                min_bandwidth=kwargs["min_bandwidth"],
                required_tags=kwargs["required_tags"],
                architecture=kwargs["architecture"],
            )
            and (kwargs["type"] is None or r.type is kwargs["type"])
        ]
        if not eligible:
            return None
        eligible.sort(key=lambda r: (r.load.score(), r.name))
        return eligible[0]

    def _has_capacity(self, resource_id: str) -> bool:
        """True if ``resource_id`` can accept one more concurrent task."""

        in_use = len(self._reservations.get(resource_id, set()))
        return in_use < self._max_concurrent

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, task_id: str) -> TaskResult:
        """Execute an assigned task to completion (blocking, with retries).

        The task must already be assigned to a resource (via
        :meth:`assign` or by setting ``assigned_resource_id``). Handles
        timeout and retry per the task's policy, then updates the task's
        terminal status and result.
        """

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise SchedulerError(f"unknown task {task_id!r}")
            if task.is_terminal():
                if task.result is not None:
                    return task.result
                raise SchedulerError(f"task {task.name!r} is already terminal")
            if not task.assigned_resource_id:
                raise SchedulerError(
                    f"task {task.name!r} has no assigned resource; call assign() first"
                )
            resource = self._registry.get(task.assigned_resource_id)
            if resource is None:
                raise SchedulerError(
                    f"assigned resource {task.assigned_resource_id!r} no longer exists"
                )
            cancel_event = threading.Event()
            self._cancel_events[task_id] = cancel_event
            self._set_status(task, TaskStatus.RUNNING)
            started = time.time()
            if task.started_at == 0.0:
                task = self._tasks[task_id].model_copy(update={"started_at": started})
                self._tasks[task_id] = task

        # Run outside the lock so the runner is not serialised.
        result = self._run_with_retries(task_id, resource, cancel_event)

        with self._lock:
            final_task = self._tasks.get(task_id)
            if final_task is None:
                return result
            self._cancel_events.pop(task_id, None)
            self._procs.pop(task_id, None)
            self._release_resource(task_id)
            status = (
                TaskStatus.COMPLETED
                if result.success
                else (TaskStatus.TIMED_OUT if result.timed_out else TaskStatus.FAILED)
            )
            self._set_terminal(final_task, status, result=result)
            logger.info(
                "Task %s finished: status=%s attempts=%d duration=%.2fs",
                final_task.name,
                status.value,
                result.attempts,
                result.duration,
            )
            return result

    def run_pending(self, max_workers: int | None = None) -> list[TaskResult]:
        """Assign and concurrently execute all currently-placeable tasks.

        Spawns a worker per assignable task (bounded by ``max_workers``).
        Returns the :class:`TaskResult` of every task that reached a
        terminal state during this call, in completion order.
        """

        assigned = self.assign()
        if not assigned:
            return []
        workers = max_workers or max(1, len(assigned))
        results: list[TaskResult] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: list[Future[TaskResult]] = [
                pool.submit(self.execute, task.id) for task in assigned
            ]
            for future in futures:
                try:
                    results.append(future.result())
                except SchedulerError as exc:
                    logger.error("run_pending task failed: %s", exc)
        return results

    def _run_with_retries(
        self,
        task_id: str,
        resource: ResourceRecord,
        cancel_event: threading.Event,
    ) -> TaskResult:
        """Run the runner, retrying per the task's :class:`RetryPolicy`."""

        policy = self._tasks[task_id].retry_policy
        attempts = 0
        while True:
            attempts += 1
            task = self._tasks[task_id]
            if cancel_event.is_set():
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    exit_code=-1,
                    attempts=attempts,
                    started_at=task.started_at or time.time(),
                    finished_at=time.time(),
                    resource_id=resource.id,
                    error="cancelled",
                )
            result = self._runner(task, resource, cancel_event)
            result = result.model_copy(
                update={
                    "attempts": attempts,
                    "resource_id": resource.id,
                }
            )
            if result.success:
                return result
            retryable = self._is_retryable(result, policy)
            if not retryable or attempts > policy.max_retries:
                return result
            delay = min(
                policy.base_delay * (policy.backoff_multiplier ** (attempts - 1)),
                policy.max_delay,
            )
            logger.info(
                "Task %s attempt %d failed (%s); retrying in %.1fs",
                task.name,
                attempts,
                result.error or f"exit {result.exit_code}",
                delay,
            )
            # Sleep cooperatively so cancellation can interrupt the wait.
            if cancel_event.wait(timeout=delay):
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    exit_code=-1,
                    attempts=attempts,
                    started_at=task.started_at or time.time(),
                    finished_at=time.time(),
                    resource_id=resource.id,
                    error="cancelled during retry backoff",
                )

    @staticmethod
    def _is_retryable(result: TaskResult, policy: RetryPolicy) -> bool:
        """Decide whether a failed result warrants another attempt."""

        if result.success:
            return False
        return not (result.timed_out and not policy.retry_on_timeout)

    # ------------------------------------------------------------------
    # Preemption
    # ------------------------------------------------------------------

    def _maybe_preempt_for(self, urgent_task_id: str) -> None:
        """If ``urgent_task_id`` cannot be placed, preempt a lower task for it.

        Only CRITICAL tasks trigger preemption. The victim is the running
        task with the lowest priority whose resource satisfies the urgent
        task's requirements. The victim is re-queued (status PREEMPTED then
        PENDING) and its resource reservation is released so the next
        :meth:`assign` cycle can place the urgent task.
        """

        with self._lock:
            urgent = self._tasks.get(urgent_task_id)
            if urgent is None or urgent.priority is not TaskPriority.CRITICAL:
                return
            if urgent.status is not TaskStatus.PENDING:
                return
            # Already placeable? Nothing to preempt.
            if self._find_free_resource(urgent) is not None:
                return
            victim = self._pick_preemption_victim(urgent)
            if victim is None:
                return
            self._preempt(victim, by=urgent.id)

    def _pick_preemption_victim(self, urgent: Task) -> Task | None:
        """Choose the lowest-priority running task whose resource fits ``urgent``."""

        running = [t for t in self._tasks.values() if t.is_running() and t.id != urgent.id]
        if not running:
            return None
        urgent_weight = urgent.priority.weight
        candidates: list[Task] = []
        for t in running:
            if t.priority.weight >= urgent_weight:
                continue
            resource = self._registry.get(t.assigned_resource_id)
            if resource is None:
                continue
            kwargs = urgent.requirements.match_kwargs()
            if resource.capabilities.satisfies(
                min_cpu=kwargs["min_cpu"],
                min_memory=kwargs["min_memory"],
                min_disk=kwargs["min_disk"],
                min_gpu=kwargs["min_gpu"],
                min_gpu_memory=kwargs["min_gpu_memory"],
                min_bandwidth=kwargs["min_bandwidth"],
                required_tags=kwargs["required_tags"],
                architecture=kwargs["architecture"],
            ):
                candidates.append(t)
        if not candidates:
            return None
        # Lowest priority first, then the one that started earliest (oldest).
        candidates.sort(key=lambda t: (t.priority.weight, t.started_at))
        return candidates[0]

    def preempt(self, task_id: str, reason: str = "") -> bool:
        """Externally-driven preemption of a running task.

        Returns True if a running task was preempted, False otherwise.
        """

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or not task.is_running():
                return False
            self._preempt(task, reason=reason)
            return True

    def _preempt(self, task: Task, *, by: str = "", reason: str = "") -> None:
        """Re-queue a running task: signal cancellation, release, re-enqueue."""

        event = self._cancel_events.get(task.id)
        if event is not None:
            event.set()
        proc = self._procs.get(task.id)
        if proc is not None and proc.poll() is None:
            self._terminate(proc)
        self._cancel_events.pop(task.id, None)
        self._procs.pop(task.id, None)
        self._release_resource(task.id)
        updated = task.model_copy(
            update={
                "status": TaskStatus.PENDING,
                "preempted_count": task.preempted_count + 1,
                "preempted_by": by,
                "assigned_resource_id": "",
            }
        )
        self._tasks[task.id] = updated
        self._push(updated)
        why = reason or (f"preempted by {by}" if by else "preempted")
        logger.warning("Preempted task %s (%s)", task.name, why)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of scheduler state for dashboards."""

        with self._lock:
            by_status: dict[str, int] = {}
            for task in self._tasks.values():
                by_status[task.status.value] = by_status.get(task.status.value, 0) + 1
            running = [t for t in self._tasks.values() if t.is_running()]
            return {
                "total": len(self._tasks),
                "by_status": by_status,
                "pending": len(self._heap),
                "running": len(running),
                "resources_in_use": sum(1 for s in self._reservations.values() if s),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push(self, task: Task) -> None:
        """Push a task onto the priority queue (caller holds the lock)."""

        rank = -task.priority.weight
        self._seq += 1
        heapq.heappush(self._heap, (rank, self._seq, task.id))

    def _set_status(self, task: Task, status: TaskStatus) -> None:
        updated = task.model_copy(update={"status": status})
        self._tasks[task.id] = updated

    def _set_terminal(
        self,
        task: Task,
        status: TaskStatus,
        *,
        result: TaskResult | None = None,
        error: str = "",
    ) -> None:
        now = time.time()
        update: dict[str, Any] = {
            "status": status,
            "finished_at": now,
        }
        if result is not None:
            update["result"] = result
            update["attempts"] = result.attempts
            update["started_at"] = result.started_at or task.started_at
        elif error:
            update["result"] = TaskResult(
                task_id=task.id,
                success=False,
                exit_code=-1,
                attempts=task.attempts,
                started_at=task.started_at or now,
                finished_at=now,
                resource_id=task.assigned_resource_id,
                error=error,
            )
        self._tasks[task.id] = task.model_copy(update=update)

    def _release_resource(self, task_id: str) -> None:
        """Release the capacity reservation held by ``task_id``."""

        task = self._tasks.get(task_id)
        if task is None or not task.assigned_resource_id:
            return
        resource_id = task.assigned_resource_id
        bucket = self._reservations.get(resource_id)
        if bucket is not None:
            bucket.discard(task_id)
            if not bucket:
                self._reservations.pop(resource_id, None)
        self._bump_load(resource_id, delta_tasks=-1)
        updated = task.model_copy(update={"assigned_resource_id": ""})
        self._tasks[task_id] = updated

    def _bump_load(self, resource_id: str, *, delta_tasks: int) -> None:
        """Adjust the running-task count on a resource's load snapshot."""

        record = self._registry.get(resource_id)
        if record is None:
            return
        load = record.load.model_copy()
        load.running_tasks = max(0, load.running_tasks + delta_tasks)
        load.updated_at = time.time()
        self._registry.update_load(resource_id, load)

    @staticmethod
    def _terminate(proc: subprocess.Popen[Any]) -> None:
        """Terminate a subprocess gracefully, then forcefully if needed."""

        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        except Exception as exc:  # noqa: BLE001 - best-effort cleanup
            logger.debug("subprocess termination error: %s", exc)

    def register_proc(self, task_id: str, proc: subprocess.Popen[Any]) -> None:
        """Register a running subprocess so cancel/preempt can terminate it.

        Called by :func:`default_runner`; safe to call from any thread.
        """

        with self._lock:
            self._procs[task_id] = proc


# ---------------------------------------------------------------------------
# Default runner: local subprocess execution
# ---------------------------------------------------------------------------


def default_runner(
    task: Task, resource: ResourceRecord, cancel_event: threading.Event
) -> TaskResult:
    """Default :data:`TaskRunner`: execute ``task.command`` via :mod:`subprocess`.

    The command runs locally; for remote resources the caller can pass a
    custom runner (e.g. one that prepends ``ssh <address>``). The runner
    honours the task ``timeout`` per attempt and the cancellation event,
    terminating the subprocess when either fires.
    """

    started = time.time()
    if not task.command:
        return TaskResult(
            task_id=task.id,
            success=False,
            exit_code=-1,
            started_at=started,
            finished_at=time.time(),
            resource_id=resource.id,
            error="empty command",
        )

    env = dict(os.environ)
    env.update(task.env)
    try:
        proc = subprocess.Popen(  # noqa: S602 - shell execution is intentional
            task.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            # Run in a new process group so we can kill the whole tree.
            start_new_session=True,
        )
    except OSError as exc:
        return TaskResult(
            task_id=task.id,
            success=False,
            exit_code=-1,
            started_at=started,
            finished_at=time.time(),
            resource_id=resource.id,
            error=f"failed to start: {exc}",
        )

    # Register so the scheduler can terminate us on cancel/preempt.
    # We reach into the scheduler via a thread-local mapping kept on the
    # runner module; the active scheduler registers itself when execute()
    # starts. This keeps default_runner decoupled from a specific instance.
    _active_procs[task.id] = proc
    try:
        return _wait_for_proc(task, resource, proc, cancel_event, started)
    finally:
        _active_procs.pop(task.id, None)


#: Maps task_id -> running Popen for the default runner. The scheduler's
#: ``register_proc`` is the preferred registration path; this dict lets
#: ``default_runner`` work standalone (e.g. in tests) too.
_active_procs: dict[str, subprocess.Popen[Any]] = {}


def _wait_for_proc(
    task: Task,
    resource: ResourceRecord,
    proc: subprocess.Popen[Any],
    cancel_event: threading.Event,
    started: float,
) -> TaskResult:
    """Block until ``proc`` exits, times out, or is cancelled."""

    timeout = task.timeout if task.timeout and task.timeout > 0 else None
    # Poll so we can react to cancellation promptly.
    deadline = (started + timeout) if timeout is not None else None
    while True:
        if cancel_event.is_set():
            _kill_group(proc)
            return _result_from_proc(task, resource, proc, started, error="cancelled")
        if deadline is not None and time.time() >= deadline:
            _kill_group(proc)
            return _result_from_proc(
                task,
                resource,
                proc,
                started,
                error=f"timed out after {timeout:.0f}s",
            )
        try:
            proc.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            continue
    return _result_from_proc(task, resource, proc, started)


def _kill_group(proc: subprocess.Popen[Any]) -> None:
    """Kill the whole process group of ``proc`` (start_new_session=True)."""

    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    if callable(killpg) and callable(getpgid):
        try:
            killpg(getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(Exception):
                proc.kill()
        return
    # POSIX-only APIs unavailable (e.g. Windows stubs).
    with contextlib.suppress(Exception):
        proc.kill()


def _result_from_proc(
    task: Task,
    resource: ResourceRecord,
    proc: subprocess.Popen[Any],
    started: float,
    *,
    error: str = "",
) -> TaskResult:
    """Build a :class:`TaskResult` from a finished (or killed) subprocess."""

    finished = time.time()
    try:
        stdout, stderr = proc.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        stdout, stderr = "", ""
    stdout = stdout or ""
    stderr = stderr or ""
    exit_code = proc.returncode if proc.returncode is not None else -1
    success = (not error) and exit_code == 0
    return TaskResult(
        task_id=task.id,
        success=success,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        started_at=started,
        finished_at=finished,
        duration=finished - started,
        resource_id=resource.id,
        error=error,
    )


__all__ = [
    "DEFAULT_MAX_CONCURRENT_PER_RESOURCE",
    "NO_TIMEOUT",
    "ResourceRequirements",
    "RetryPolicy",
    "SchedulerError",
    "Task",
    "TaskPriority",
    "TaskResult",
    "TaskRunner",
    "TaskScheduler",
    "TaskStatus",
    "default_runner",
]
