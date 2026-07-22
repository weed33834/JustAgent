"""Scheduled tasks — cron-like recurring command execution.

Persist scheduled tasks to ``~/.myagent/schedules.json`` and run them
on a schedule. Supports interval-based schedules (``"30m"``, ``"2h"``,
``"1d"``), daily-at-time schedules (``"daily 09:00"``), and basic
5-field cron expressions (``"*/5 * * * *"``, ``"0 9 * * 1-5"``).

Used by the ``myagent schedule`` command group. The ``daemon`` mode
runs a loop that checks for due tasks every minute and executes them.

Reference: crontab(5) format.
"""

from __future__ import annotations

import calendar
import json
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from myagent.exceptions import MyAgentError
from myagent.utils.atomic_write import atomic_write_text

DEFAULT_SCHEDULE_STORE_PATH = Path.home() / ".myagent" / "schedules.json"

#: Maximum subprocess execution time per task, in seconds.
_TASK_TIMEOUT_SECONDS = 1800

#: Upper bound for the cron next-run scan: 366 days in minutes.
_CRON_SCAN_LIMIT_MINUTES = 366 * 24 * 60


class SchedulerError(MyAgentError):
    """Raised when a schedule expression is invalid or a task cannot run."""


class ScheduleType(str, Enum):  # noqa: UP042 - match existing codebase style
    """The kind of schedule expression backing a :class:`ScheduledTask`."""

    INTERVAL = "interval"
    DAILY = "daily"
    CRON = "cron"


@dataclass(frozen=True)
class ScheduledTask:
    """A persisted scheduled task.

    ``next_run`` is a Unix timestamp (seconds since epoch). ``0`` means
    "never scheduled" or "not yet computed".
    """

    id: str
    name: str
    schedule: str
    schedule_type: ScheduleType
    command: str
    project: str = ""
    enabled: bool = True
    created_at: float = 0.0
    last_run: float = 0.0
    next_run: float = 0.0
    last_exit_code: int = -1
    last_output: str = ""


@dataclass(frozen=True)
class TaskRunResult:
    """The outcome of executing a :class:`ScheduledTask` once."""

    task: ScheduledTask
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    started_at: float
    finished_at: float


# ---------------------------------------------------------------------------
# ScheduleStore
# ---------------------------------------------------------------------------


class ScheduleStore:
    """Manage a JSON store of :class:`ScheduledTask` records.

    Mirrors the resilience pattern of
    :class:`myagent.core.project_store.ProjectStore`: missing or corrupt
    files degrade gracefully to an empty store instead of crashing callers.
    """

    def __init__(self, store_path: Path | None = None) -> None:
        self.store_path: Path = store_path or DEFAULT_SCHEDULE_STORE_PATH

    def add(self, task: ScheduledTask) -> None:
        """Add or replace a task by name."""
        data = self._load()
        data[task.name] = task
        self._save(data)

    def remove(self, name: str) -> bool:
        """Remove a task by name; return True if it was present."""
        data = self._load()
        if name not in data:
            return False
        del data[name]
        self._save(data)
        return True

    def get(self, name: str) -> ScheduledTask | None:
        """Return a task by name, or None if not present."""
        return self._load().get(name)

    def list_all(self) -> list[ScheduledTask]:
        """Return all tasks sorted by name."""
        return sorted(self._load().values(), key=lambda t: t.name)

    def update(self, task: ScheduledTask) -> None:
        """Update an existing task in place (e.g. after running it)."""
        data = self._load()
        data[task.name] = task
        self._save(data)

    def _load(self) -> dict[str, ScheduledTask]:
        """Load the store from disk; missing or corrupt files yield an empty dict."""
        if not self.store_path.exists():
            return {}
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        result: dict[str, ScheduledTask] = {}
        for key, item in raw.items():
            if not isinstance(item, dict):
                continue
            try:
                result[key] = _task_from_dict(item, key)
            except (KeyError, TypeError, ValueError):
                continue
        return result

    def _save(self, data: dict[str, ScheduledTask]) -> None:
        """Persist the store to disk as indented JSON."""
        payload = {name: _task_to_dict(task) for name, task in data.items()}
        atomic_write_text(self.store_path, json.dumps(payload, indent=2))


def _task_to_dict(task: ScheduledTask) -> dict[str, Any]:
    """Serialise a :class:`ScheduledTask` to a JSON-safe dict."""
    return {
        "id": task.id,
        "name": task.name,
        "schedule": task.schedule,
        "schedule_type": task.schedule_type.value,
        "command": task.command,
        "project": task.project,
        "enabled": task.enabled,
        "created_at": task.created_at,
        "last_run": task.last_run,
        "next_run": task.next_run,
        "last_exit_code": task.last_exit_code,
        "last_output": task.last_output,
    }


def _task_from_dict(item: dict[str, Any], fallback_key: str) -> ScheduledTask:
    """Deserialise a dict back into a :class:`ScheduledTask`."""
    return ScheduledTask(
        id=str(item.get("id", fallback_key)),
        name=str(item.get("name", fallback_key)),
        schedule=str(item["schedule"]),
        schedule_type=ScheduleType(str(item["schedule_type"])),
        command=str(item["command"]),
        project=str(item.get("project") or ""),
        enabled=bool(item.get("enabled", True)),
        created_at=float(item.get("created_at", 0.0)),
        last_run=float(item.get("last_run", 0.0)),
        next_run=float(item.get("next_run", 0.0)),
        last_exit_code=int(item.get("last_exit_code", -1)),
        last_output=str(item.get("last_output") or ""),
    )


# ---------------------------------------------------------------------------
# ScheduleParser
# ---------------------------------------------------------------------------


_INTERVAL_RE = re.compile(r"^(\d+)([mhd])$")
_DAILY_RE = re.compile(r"^daily\s+(\d{1,2}):(\d{2})$")


class ScheduleParser:
    """Parse schedule expressions into typed parameters + next-run timestamps."""

    def parse(self, expression: str) -> tuple[ScheduleType, dict[str, Any]]:
        """Parse ``expression`` and return ``(type, params)``.

        Raises :class:`SchedulerError` for malformed expressions.
        """
        text = expression.strip()
        if not text:
            raise SchedulerError("empty schedule expression")

        match = _INTERVAL_RE.match(text)
        if match:
            count = int(match.group(1))
            unit = match.group(2)
            if count <= 0:
                raise SchedulerError(f"interval must be positive: {expression!r}")
            if unit == "m":
                return ScheduleType.INTERVAL, {"minutes": count}
            if unit == "h":
                return ScheduleType.INTERVAL, {"hours": count}
            return ScheduleType.INTERVAL, {"days": count}

        match = _DAILY_RE.match(text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            if hour > 23 or minute > 59:
                raise SchedulerError(f"invalid daily time: {expression!r}")
            return ScheduleType.DAILY, {"hour": hour, "minute": minute}

        # Fall back to cron parsing.
        params = _parse_cron_expression(text)
        return ScheduleType.CRON, params

    def next_run(self, expression: str, after: float | None = None) -> float:
        """Compute the next Unix timestamp at which ``expression`` should fire.

        ``after`` defaults to :func:`time.time`. Always returns a value
        strictly greater than ``after``.
        """
        schedule_type, params = self.parse(expression)
        base = time.time() if after is None else after
        if schedule_type is ScheduleType.INTERVAL:
            seconds = _interval_to_seconds(params)
            return base + seconds
        if schedule_type is ScheduleType.DAILY:
            return _next_daily(base, params["hour"], params["minute"])
        return _next_cron(base, params)


def _interval_to_seconds(params: dict[str, Any]) -> float:
    """Convert parsed interval params to a number of seconds."""
    if "minutes" in params:
        return float(params["minutes"]) * 60.0
    if "hours" in params:
        return float(params["hours"]) * 3600.0
    if "days" in params:
        return float(params["days"]) * 86400.0
    raise SchedulerError(f"unknown interval params: {params!r}")


def _next_daily(base: float, hour: int, minute: int) -> float:
    """Compute the next daily-at-HH:MM timestamp strictly after ``base``."""
    now = datetime.fromtimestamp(base)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return target.timestamp()


def _parse_cron_expression(text: str) -> dict[str, str]:
    """Split a 5-field cron expression into its named fields.

    Raises :class:`SchedulerError` if the field count is wrong or any field
    contains out-of-range literal values.
    """
    fields = text.split()
    if len(fields) != 5:
        raise SchedulerError(
            f"cron expression must have 5 fields, got {len(fields)}: {text!r}"
        )
    minute, hour, day, month, weekday = fields
    # Range-check any plain integer literals so "60 * * * *" is rejected.
    _validate_cron_field(minute, 0, 59, "minute")
    _validate_cron_field(hour, 0, 23, "hour")
    _validate_cron_field(day, 1, 31, "day")
    _validate_cron_field(month, 1, 12, "month")
    _validate_cron_field(weekday, 0, 7, "weekday")
    return {
        "minute": minute,
        "hour": hour,
        "day": day,
        "month": month,
        "weekday": weekday,
    }


def _validate_cron_field(field: str, min_val: int, max_val: int, name: str) -> None:
    """Reject obviously invalid literals like ``60`` in a minute field.

    Range/step syntax is checked loosely here; full matching happens in
    :func:`_cron_field_matches`.
    """
    for part in field.split(","):
        token = part.strip()
        if "/" in token:
            base_token, step_str = token.split("/", 1)
            if not step_str.isdigit() or int(step_str) <= 0:
                raise SchedulerError(f"invalid cron step in {name} field: {field!r}")
            token = base_token
        if token in ("", "*"):
            continue
        if "-" in token:
            lo_str, hi_str = token.split("-", 1)
            if not (lo_str.isdigit() and hi_str.isdigit()):
                raise SchedulerError(f"invalid cron range in {name} field: {field!r}")
            lo, hi = int(lo_str), int(hi_str)
            if (
                lo < min_val or hi > max_val or lo > hi
            ) and not (name == "weekday" and hi == 7):
                raise SchedulerError(
                    f"cron {name} out of range [{min_val},{max_val}]: {field!r}"
                )
            continue
        if token.isdigit():
            value = int(token)
            # weekday 7 is a synonym for 0 (Sunday).
            effective_max = max_val if not (name == "weekday" and value == 7) else 7
            if value < min_val or value > effective_max:
                raise SchedulerError(
                    f"cron {name} out of range [{min_val},{max_val}]: {field!r}"
                )
            continue
        raise SchedulerError(f"invalid cron token in {name} field: {field!r}")


def _cron_field_matches(field: str, value: int, min_val: int, max_val: int) -> bool:
    """Check if a cron field matches ``value``.

    Supports ``*``, ``N``, ``N-M``, ``*/S``, ``N-M/S``, and ``N,M,K``.
    """
    for raw_part in field.split(","):
        part = raw_part.strip()
        if not part:
            continue
        step = 1
        range_token = part
        if "/" in part:
            range_token, step_str = part.split("/", 1)
            if not step_str.isdigit():
                continue
            step = int(step_str)
            if step <= 0:
                continue
        if range_token in ("", "*"):
            lo, hi = min_val, max_val
        elif "-" in range_token:
            lo_str, hi_str = range_token.split("-", 1)
            if not (lo_str.isdigit() and hi_str.isdigit()):
                continue
            lo, hi = int(lo_str), int(hi_str)
            # weekday 7 is shorthand for Sunday (0); allow either.
            if hi == 7 and min_val == 0 and max_val == 6:
                hi = 6
        else:
            if not range_token.isdigit():
                continue
            n = int(range_token)
            if n == 7 and min_val == 0 and max_val == 6:
                n = 0
            if step > 1:
                # ``N/S`` is non-standard but tolerable: treat as N..max/S.
                lo, hi = n, max_val
            else:
                if value == n:
                    return True
                continue
        if lo <= value <= hi and (value - lo) % step == 0:
            return True
    return False


def _next_cron(base: float, params: dict[str, str]) -> float:
    """Scan forward minute-by-minute to find the next cron match.

    Raises :class:`SchedulerError` if no match is found within
    :data:`_CRON_SCAN_LIMIT_MINUTES` (covers impossible expressions like
    ``0 0 30 2 *``).
    """
    start = datetime.fromtimestamp(base)
    # Truncate to the start of the next minute.
    candidate = start.replace(second=0, microsecond=0) + timedelta(minutes=1)
    weekday_field = params["weekday"]
    for _ in range(_CRON_SCAN_LIMIT_MINUTES):
        if (
            _cron_field_matches(params["minute"], candidate.minute, 0, 59)
            and _cron_field_matches(params["hour"], candidate.hour, 0, 23)
            and _cron_field_matches(params["day"], candidate.day, 1, 31)
            and _cron_field_matches(params["month"], candidate.month, 1, 12)
            and _cron_field_matches(
                weekday_field,
                _python_to_cron_weekday(candidate.weekday()),
                0,
                6,
            )
            and calendar.monthrange(candidate.year, candidate.month)[1] >= candidate.day
        ):
            return candidate.timestamp()
        candidate = candidate + timedelta(minutes=1)
    raise SchedulerError(
        f"no cron match within {_CRON_SCAN_LIMIT_MINUTES} minutes for {params!r}"
    )


def _python_to_cron_weekday(python_weekday: int) -> int:
    """Convert Python ``date.weekday()`` (Mon=0..Sun=6) to cron (Sun=0..Sat=6)."""
    return (python_weekday + 1) % 7


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """High-level façade over :class:`ScheduleStore` and :class:`ScheduleParser`.

    Owns task lifecycle (add/remove/enable/disable), runs due tasks via
    :mod:`subprocess`, and exposes a blocking ``daemon`` loop.
    """

    def __init__(
        self,
        store: ScheduleStore | None = None,
        project_store: Any | None = None,
    ) -> None:
        self.store = store or ScheduleStore()
        self.project_store = project_store
        self.parser = ScheduleParser()

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def add_task(
        self,
        name: str,
        schedule: str,
        command: str,
        project: str = "",
        enabled: bool = True,
    ) -> ScheduledTask:
        """Create and persist a new task; returns the created task."""
        if not name:
            raise SchedulerError("task name must not be empty")
        schedule_type, _params = self.parser.parse(schedule)
        created_at = time.time()
        next_run = self.parser.next_run(schedule, after=created_at)
        task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name,
            schedule=schedule,
            schedule_type=schedule_type,
            command=command,
            project=project,
            enabled=enabled,
            created_at=created_at,
            next_run=next_run,
        )
        self.store.add(task)
        return task

    def remove_task(self, name: str) -> bool:
        """Remove a task by name; return True if it was present."""
        return self.store.remove(name)

    def list_tasks(self) -> list[ScheduledTask]:
        """Return all tasks sorted by name."""
        return self.store.list_all()

    def get_task(self, name: str) -> ScheduledTask | None:
        """Return a task by name, or None if not present."""
        return self.store.get(name)

    def enable_task(self, name: str) -> bool:
        """Enable a task; return True if the task exists."""
        return self._set_enabled(name, True)

    def disable_task(self, name: str) -> bool:
        """Disable a task; return True if the task exists."""
        return self._set_enabled(name, False)

    def _set_enabled(self, name: str, enabled: bool) -> bool:
        task = self.store.get(name)
        if task is None:
            return False
        updated = ScheduledTask(
            id=task.id,
            name=task.name,
            schedule=task.schedule,
            schedule_type=task.schedule_type,
            command=task.command,
            project=task.project,
            enabled=enabled,
            created_at=task.created_at,
            last_run=task.last_run,
            next_run=task.next_run,
            last_exit_code=task.last_exit_code,
            last_output=task.last_output,
        )
        self.store.update(updated)
        return True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_due(self, now: float | None = None) -> list[TaskRunResult]:
        """Run every enabled task whose ``next_run`` is due.

        Updates ``last_run``, ``next_run``, ``last_exit_code`` and
        ``last_output`` for each task that ran.
        """
        current = time.time() if now is None else now
        results: list[TaskRunResult] = []
        for task in self.store.list_all():
            if not task.enabled:
                continue
            if task.next_run <= 0:
                continue
            if task.next_run > current:
                continue
            result = self._execute_task(task)
            results.append(result)
            updated = ScheduledTask(
                id=task.id,
                name=task.name,
                schedule=task.schedule,
                schedule_type=task.schedule_type,
                command=task.command,
                project=task.project,
                enabled=task.enabled,
                created_at=task.created_at,
                last_run=result.started_at,
                next_run=self.parser.next_run(task.schedule, after=current),
                last_exit_code=result.exit_code,
                last_output=_compose_output(result.stdout, result.stderr),
            )
            self.store.update(updated)
        return results

    def run_task(self, name: str) -> TaskRunResult:
        """Manually run a task immediately, regardless of schedule."""
        task = self.store.get(name)
        if task is None:
            raise SchedulerError(f"no scheduled task named {name!r}")
        result = self._execute_task(task)
        current = time.time()
        updated = ScheduledTask(
            id=task.id,
            name=task.name,
            schedule=task.schedule,
            schedule_type=task.schedule_type,
            command=task.command,
            project=task.project,
            enabled=task.enabled,
            created_at=task.created_at,
            last_run=result.started_at,
            next_run=self.parser.next_run(task.schedule, after=current),
            last_exit_code=result.exit_code,
            last_output=_compose_output(result.stdout, result.stderr),
        )
        self.store.update(updated)
        return result

    def run_daemon(self, check_interval: float = 60.0) -> None:
        """Block forever, calling :meth:`run_due` every ``check_interval`` seconds.

        Stops cleanly on :class:`KeyboardInterrupt`.
        """
        try:
            while True:
                self.run_due()
                time.sleep(check_interval)
        except KeyboardInterrupt:
            return

    def _execute_task(self, task: ScheduledTask) -> TaskRunResult:
        """Resolve the project cwd (if any) and shell-execute ``task.command``."""
        cwd = self._resolve_cwd(task)
        started_at = time.time()
        try:
            completed = subprocess.run(
                task.command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=_TASK_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            finished_at = time.time()
            stdout = _stream_to_str(exc.stdout)
            stderr = _stream_to_str(exc.stderr) or f"command timed out after {exc.timeout}s"
            return TaskRunResult(
                task=task,
                success=False,
                exit_code=-1,
                stdout=stdout,
                stderr=stderr,
                started_at=started_at,
                finished_at=finished_at,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            finished_at = time.time()
            return TaskRunResult(
                task=task,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                started_at=started_at,
                finished_at=finished_at,
            )
        finished_at = time.time()
        return TaskRunResult(
            task=task,
            success=completed.returncode == 0,
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            started_at=started_at,
            finished_at=finished_at,
        )

    def _resolve_cwd(self, task: ScheduledTask) -> str | None:
        """Return the working directory for ``task``, or None to inherit.

        If the task references a project that is no longer registered, falls
        back to None (the current directory) rather than raising.
        """
        if not task.project or self.project_store is None:
            return None
        try:
            project = self.project_store.get(task.project)
        except Exception:  # noqa: BLE001 - project store is pluggable
            return None
        if project is None:
            return None
        path = getattr(project, "path", None)
        if not path:
            return None
        return str(path)


def _stream_to_str(value: str | bytes | None) -> str:
    """Normalise a subprocess output stream to a string."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _compose_output(stdout: str, stderr: str) -> str:
    """Combine stdout/stderr into a single human-readable output blob."""
    parts: list[str] = []
    if stdout:
        parts.append(stdout.rstrip())
    if stderr:
        parts.append(stderr.rstrip())
    return "\n".join(parts)


__all__ = [
    "DEFAULT_SCHEDULE_STORE_PATH",
    "ScheduleParser",
    "ScheduleStore",
    "ScheduleType",
    "ScheduledTask",
    "Scheduler",
    "SchedulerError",
    "TaskRunResult",
]
