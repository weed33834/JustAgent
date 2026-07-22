"""Tests for :mod:`myagent.core.scheduler`."""

from __future__ import annotations

import json
import time
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from myagent.core.project_store import ProjectStore
from myagent.core.scheduler import (
    ScheduledTask,
    ScheduleParser,
    Scheduler,
    SchedulerError,
    ScheduleStore,
    ScheduleType,
    TaskRunResult,
    _cron_field_matches,
)
from myagent.models.project import ManagedProject

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> CompletedProcess[str]:
    """Build a fake :class:`subprocess.CompletedProcess`."""
    return CompletedProcess(
        args="fake", returncode=returncode, stdout=stdout, stderr=stderr
    )


def _make_task(
    name: str = "task",
    schedule: str = "30m",
    command: str = "echo hi",
    project: str = "",
    enabled: bool = True,
    next_run: float = 0.0,
    last_run: float = 0.0,
) -> ScheduledTask:
    """Build a :class:`ScheduledTask` with sensible test defaults."""
    schedule_type, _ = ScheduleParser().parse(schedule)
    return ScheduledTask(
        id=name,
        name=name,
        schedule=schedule,
        schedule_type=schedule_type,
        command=command,
        project=project,
        enabled=enabled,
        created_at=1000.0,
        last_run=last_run,
        next_run=next_run,
    )


# ---------------------------------------------------------------------------
# TestScheduleType
# ---------------------------------------------------------------------------


class TestScheduleType:
    def test_values(self) -> None:
        assert ScheduleType.INTERVAL.value == "interval"
        assert ScheduleType.DAILY.value == "daily"
        assert ScheduleType.CRON.value == "cron"

    def test_is_str(self) -> None:
        assert isinstance(ScheduleType.INTERVAL, str)

    def test_from_value(self) -> None:
        assert ScheduleType("daily") is ScheduleType.DAILY


# ---------------------------------------------------------------------------
# TestScheduledTask
# ---------------------------------------------------------------------------


class TestScheduledTask:
    def test_construction(self) -> None:
        task = ScheduledTask(
            id="abc",
            name="nightly",
            schedule="daily 09:00",
            schedule_type=ScheduleType.DAILY,
            command="echo hi",
        )
        assert task.name == "nightly"
        assert task.project == ""
        assert task.enabled is True
        assert task.created_at == 0.0
        assert task.last_run == 0.0
        assert task.next_run == 0.0
        assert task.last_exit_code == -1
        assert task.last_output == ""

    def test_frozen(self) -> None:
        task = _make_task()
        with pytest.raises(FrozenInstanceError):
            task.name = "other"  # type: ignore[misc]

    def test_defaults_independent(self) -> None:
        task = _make_task()
        # last_output should default to "" per task instance
        assert task.last_output == ""


# ---------------------------------------------------------------------------
# TestTaskRunResult
# ---------------------------------------------------------------------------


class TestTaskRunResult:
    def test_construction(self) -> None:
        task = _make_task()
        result = TaskRunResult(
            task=task,
            success=True,
            exit_code=0,
            stdout="ok",
            stderr="",
            started_at=1.0,
            finished_at=2.0,
        )
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "ok"
        assert result.started_at == 1.0

    def test_frozen(self) -> None:
        task = _make_task()
        result = TaskRunResult(
            task=task,
            success=False,
            exit_code=1,
            stdout="",
            stderr="err",
            started_at=0.0,
            finished_at=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            result.success = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestScheduleStore
# ---------------------------------------------------------------------------


class TestScheduleStore:
    def test_add_and_get(self, tmp_path: Path) -> None:
        store = ScheduleStore(store_path=tmp_path / "schedules.json")
        store.add(_make_task(name="alpha", next_run=1000.0))
        fetched = store.get("alpha")
        assert fetched is not None
        assert fetched.name == "alpha"
        assert fetched.next_run == 1000.0

    def test_add_overwrites(self, tmp_path: Path) -> None:
        store = ScheduleStore(store_path=tmp_path / "schedules.json")
        store.add(_make_task(name="alpha", command="echo one"))
        store.add(_make_task(name="alpha", command="echo two"))
        fetched = store.get("alpha")
        assert fetched is not None
        assert fetched.command == "echo two"

    def test_remove_found(self, tmp_path: Path) -> None:
        store = ScheduleStore(store_path=tmp_path / "schedules.json")
        store.add(_make_task(name="alpha"))
        assert store.remove("alpha") is True
        assert store.get("alpha") is None

    def test_remove_missing_returns_false(self, tmp_path: Path) -> None:
        store = ScheduleStore(store_path=tmp_path / "schedules.json")
        assert store.remove("ghost") is False

    def test_list_all_sorted_by_name(self, tmp_path: Path) -> None:
        store = ScheduleStore(store_path=tmp_path / "schedules.json")
        store.add(_make_task(name="zeta"))
        store.add(_make_task(name="alpha"))
        store.add(_make_task(name="mid"))
        assert [t.name for t in store.list_all()] == ["alpha", "mid", "zeta"]

    def test_update_existing(self, tmp_path: Path) -> None:
        store = ScheduleStore(store_path=tmp_path / "schedules.json")
        store.add(_make_task(name="alpha", last_run=0.0))
        original = store.get("alpha")
        assert original is not None
        updated = ScheduledTask(
            id=original.id,
            name=original.name,
            schedule=original.schedule,
            schedule_type=original.schedule_type,
            command=original.command,
            project=original.project,
            enabled=original.enabled,
            created_at=original.created_at,
            last_run=999.0,
            next_run=1000.0,
            last_exit_code=0,
            last_output="done",
        )
        store.update(updated)
        fetched = store.get("alpha")
        assert fetched is not None
        assert fetched.last_run == 999.0
        assert fetched.last_output == "done"

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "schedules.json"
        ScheduleStore(store_path=path).add(_make_task(name="alpha", next_run=123.0))
        fresh = ScheduleStore(store_path=path)
        fetched = fresh.get("alpha")
        assert fetched is not None
        assert fetched.next_run == 123.0

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        store = ScheduleStore(store_path=tmp_path / "missing.json")
        assert store.list_all() == []

    def test_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "schedules.json"
        path.write_text("{not valid json", encoding="utf-8")
        store = ScheduleStore(store_path=path)
        assert store.list_all() == []

    def test_skips_invalid_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "schedules.json"
        path.write_text(
            json.dumps(
                {
                    "good": {
                        "id": "good",
                        "name": "good",
                        "schedule": "30m",
                        "schedule_type": "interval",
                        "command": "echo ok",
                    },
                    "bad": {"name": "bad"},
                }
            ),
            encoding="utf-8",
        )
        store = ScheduleStore(store_path=path)
        assert [t.name for t in store.list_all()] == ["good"]

    def test_atomic_write_uses_tmp_then_rename(self, tmp_path: Path) -> None:
        path = tmp_path / "schedules.json"
        store = ScheduleStore(store_path=path)
        with patch(
            "myagent.core.scheduler.atomic_write_text"
        ) as mock_write:
            store.add(_make_task(name="alpha"))
        mock_write.assert_called_once()
        # Ensure the target path passed is the configured store path.
        args, _ = mock_write.call_args
        assert args[0] == path


# ---------------------------------------------------------------------------
# TestScheduleParserInterval
# ---------------------------------------------------------------------------


class TestScheduleParserInterval:
    def test_parse_minutes(self) -> None:
        t, params = ScheduleParser().parse("30m")
        assert t is ScheduleType.INTERVAL
        assert params == {"minutes": 30}

    def test_parse_hours(self) -> None:
        t, params = ScheduleParser().parse("2h")
        assert t is ScheduleType.INTERVAL
        assert params == {"hours": 2}

    def test_parse_days(self) -> None:
        t, params = ScheduleParser().parse("1d")
        assert t is ScheduleType.INTERVAL
        assert params == {"days": 1}

    def test_parse_90_minutes(self) -> None:
        t, params = ScheduleParser().parse("90m")
        assert t is ScheduleType.INTERVAL
        assert params == {"minutes": 90}

    def test_invalid_no_unit(self) -> None:
        with pytest.raises(SchedulerError):
            ScheduleParser().parse("30")

    def test_invalid_unit(self) -> None:
        with pytest.raises(SchedulerError):
            ScheduleParser().parse("30x")

    def test_invalid_zero_interval(self) -> None:
        with pytest.raises(SchedulerError):
            ScheduleParser().parse("0m")

    @freeze_time("2025-06-15 12:00:00", tz_offset=0)
    def test_next_run_interval(self) -> None:
        ts = ScheduleParser().next_run("30m")
        # 30 minutes after the frozen now.
        expected = datetime(2025, 6, 15, 12, 30, 0, tzinfo=UTC).timestamp()
        assert ts == pytest.approx(expected, abs=1.0)


# ---------------------------------------------------------------------------
# TestScheduleParserDaily
# ---------------------------------------------------------------------------


class TestScheduleParserDaily:
    def test_parse_daily_morning(self) -> None:
        t, params = ScheduleParser().parse("daily 09:00")
        assert t is ScheduleType.DAILY
        assert params == {"hour": 9, "minute": 0}

    def test_parse_daily_afternoon(self) -> None:
        t, params = ScheduleParser().parse("daily 14:30")
        assert t is ScheduleType.DAILY
        assert params == {"hour": 14, "minute": 30}

    def test_invalid_hour(self) -> None:
        with pytest.raises(SchedulerError):
            ScheduleParser().parse("daily 25:00")

    def test_invalid_missing_time(self) -> None:
        with pytest.raises(SchedulerError):
            ScheduleParser().parse("daily")

    @freeze_time("2025-06-15 08:00:00", tz_offset=0)
    def test_next_run_before_target_today(self) -> None:
        # 09:00 hasn't happened yet → today 09:00.
        ts = ScheduleParser().next_run("daily 09:00")
        expected = datetime(2025, 6, 15, 9, 0, 0, tzinfo=UTC).timestamp()
        assert ts == pytest.approx(expected, abs=1.0)

    @freeze_time("2025-06-15 10:00:00", tz_offset=0)
    def test_next_run_after_target_today(self) -> None:
        # 09:00 already passed → tomorrow 09:00.
        ts = ScheduleParser().next_run("daily 09:00")
        expected = datetime(2025, 6, 16, 9, 0, 0, tzinfo=UTC).timestamp()
        assert ts == pytest.approx(expected, abs=1.0)


# ---------------------------------------------------------------------------
# TestScheduleParserCron
# ---------------------------------------------------------------------------


class TestScheduleParserCron:
    def test_parse_every_5_minutes(self) -> None:
        t, params = ScheduleParser().parse("*/5 * * * *")
        assert t is ScheduleType.CRON
        assert params["minute"] == "*/5"
        assert params["hour"] == "*"

    def test_parse_daily_at_9(self) -> None:
        t, params = ScheduleParser().parse("0 9 * * *")
        assert t is ScheduleType.CRON
        assert params["minute"] == "0"
        assert params["hour"] == "9"

    def test_parse_weekdays(self) -> None:
        t, params = ScheduleParser().parse("0 9 * * 1-5")
        assert t is ScheduleType.CRON
        assert params["weekday"] == "1-5"

    def test_parse_every_15(self) -> None:
        t, params = ScheduleParser().parse("*/15 * * * *")
        assert t is ScheduleType.CRON
        assert params["minute"] == "*/15"

    def test_invalid_too_few_fields(self) -> None:
        with pytest.raises(SchedulerError):
            ScheduleParser().parse("* * *")

    def test_invalid_minute_out_of_range(self) -> None:
        with pytest.raises(SchedulerError):
            ScheduleParser().parse("60 * * * *")

    # Field matching
    def test_match_star(self) -> None:
        assert _cron_field_matches("*", 5, 0, 59) is True

    def test_match_literal(self) -> None:
        assert _cron_field_matches("5", 5, 0, 59) is True
        assert _cron_field_matches("5", 6, 0, 59) is False

    def test_match_range(self) -> None:
        assert _cron_field_matches("1-5", 3, 0, 59) is True
        assert _cron_field_matches("1-5", 6, 0, 59) is False

    def test_match_step(self) -> None:
        assert _cron_field_matches("*/15", 0, 0, 59) is True
        assert _cron_field_matches("*/15", 15, 0, 59) is True
        assert _cron_field_matches("*/15", 30, 0, 59) is True
        assert _cron_field_matches("*/15", 7, 0, 59) is False

    def test_match_range_step(self) -> None:
        assert _cron_field_matches("1-30/2", 1, 0, 59) is True
        assert _cron_field_matches("1-30/2", 3, 0, 59) is True
        assert _cron_field_matches("1-30/2", 31, 0, 59) is False

    def test_match_comma_list(self) -> None:
        assert _cron_field_matches("1,3,5", 1, 0, 59) is True
        assert _cron_field_matches("1,3,5", 5, 0, 59) is True
        assert _cron_field_matches("1,3,5", 2, 0, 59) is False

    @freeze_time("2025-06-15 12:00:00", tz_offset=0)
    def test_next_run_every_5_minutes(self) -> None:
        ts = ScheduleParser().next_run("*/5 * * * *")
        expected = datetime(2025, 6, 15, 12, 5, 0, tzinfo=UTC).timestamp()
        assert ts == pytest.approx(expected, abs=1.0)

    @freeze_time("2025-06-15 12:00:00", tz_offset=0)
    def test_next_run_hourly(self) -> None:
        ts = ScheduleParser().next_run("0 * * * *")
        expected = datetime(2025, 6, 15, 13, 0, 0, tzinfo=UTC).timestamp()
        assert ts == pytest.approx(expected, abs=1.0)

    def test_impossible_cron_raises(self) -> None:
        # Feb 30 never exists → scan should exhaust the limit.
        with pytest.raises(SchedulerError):
            ScheduleParser().next_run("0 0 30 2 *")


# ---------------------------------------------------------------------------
# TestScheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_add_task(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        task = scheduler.add_task(name="alpha", schedule="30m", command="echo hi")
        assert task.name == "alpha"
        assert task.schedule_type is ScheduleType.INTERVAL
        assert task.next_run > 0
        assert scheduler.get_task("alpha") is not None

    def test_add_task_invalid_schedule(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        with pytest.raises(SchedulerError):
            scheduler.add_task(name="bad", schedule="not a schedule", command="x")

    def test_remove_task(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="alpha", schedule="30m", command="x")
        assert scheduler.remove_task("alpha") is True
        assert scheduler.remove_task("alpha") is False

    def test_list_tasks(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="beta", schedule="30m", command="x")
        scheduler.add_task(name="alpha", schedule="30m", command="x")
        assert [t.name for t in scheduler.list_tasks()] == ["alpha", "beta"]

    def test_get_task_missing(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        assert scheduler.get_task("ghost") is None

    def test_enable_disable(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="alpha", schedule="30m", command="x", enabled=True)
        assert scheduler.disable_task("alpha") is True
        assert scheduler.get_task("alpha").enabled is False  # type: ignore[union-attr]
        assert scheduler.enable_task("alpha") is True
        assert scheduler.get_task("alpha").enabled is True  # type: ignore[union-attr]

    def test_enable_missing_returns_false(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        assert scheduler.enable_task("ghost") is False

    @freeze_time("2025-06-15 12:00:00", tz_offset=0)
    def test_run_due_runs_due_task(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        # next_run in the past → due now.
        scheduler.add_task(name="alpha", schedule="30m", command="echo hi")
        # Force next_run into the past.
        task = scheduler.get_task("alpha")
        assert task is not None
        scheduler.store.update(
            ScheduledTask(
                id=task.id,
                name=task.name,
                schedule=task.schedule,
                schedule_type=task.schedule_type,
                command=task.command,
                project=task.project,
                enabled=task.enabled,
                created_at=task.created_at,
                last_run=task.last_run,
                next_run=time.time() - 60,
                last_exit_code=task.last_exit_code,
                last_output=task.last_output,
            )
        )
        with patch(
            "myagent.core.scheduler.subprocess.run",
            return_value=_completed(0, "hi\n", ""),
        ):
            results = scheduler.run_due()
        assert len(results) == 1
        assert results[0].success is True
        # next_run should be recomputed into the future.
        updated = scheduler.get_task("alpha")
        assert updated is not None
        assert updated.next_run > time.time()
        assert updated.last_exit_code == 0

    def test_run_task_manual(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="alpha", schedule="30m", command="echo hi")
        with patch(
            "myagent.core.scheduler.subprocess.run",
            return_value=_completed(0, "hi\n", ""),
        ) as mock_run:
            result = scheduler.run_task("alpha")
        mock_run.assert_called_once()
        assert result.success is True
        assert "hi" in result.stdout

    def test_run_task_missing_raises(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        with pytest.raises(SchedulerError):
            scheduler.run_task("ghost")

    @freeze_time("2025-06-15 12:00:00", tz_offset=0)
    def test_next_run_updated_after_run(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="alpha", schedule="30m", command="echo hi")
        original = scheduler.get_task("alpha")
        assert original is not None
        original_next = original.next_run
        with patch(
            "myagent.core.scheduler.subprocess.run",
            return_value=_completed(0, "", ""),
        ):
            scheduler.run_task("alpha")
        updated = scheduler.get_task("alpha")
        assert updated is not None
        # 30m later than the original next_run (which itself was ~30m after creation).
        assert updated.next_run >= original_next


# ---------------------------------------------------------------------------
# TestSchedulerExecution
# ---------------------------------------------------------------------------


class TestSchedulerExecution:
    def test_command_executed_with_cwd_none_when_no_project(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="alpha", schedule="30m", command="echo hi")
        with patch(
            "myagent.core.scheduler.subprocess.run",
            return_value=_completed(0, "hi", ""),
        ) as mock_run:
            scheduler.run_task("alpha")
        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] is None
        assert kwargs["shell"] is True

    def test_command_executed_in_project_cwd(self, tmp_path: Path) -> None:
        project_store = ProjectStore(store_path=tmp_path / "p.json")
        project_dir = tmp_path / "myproj"
        project_dir.mkdir()
        project_store.add(
            ManagedProject(name="myproj", path=str(project_dir), added_at=1.0)
        )
        scheduler = Scheduler(
            store=ScheduleStore(tmp_path / "s.json"),
            project_store=project_store,
        )
        scheduler.add_task(
            name="alpha", schedule="30m", command="pwd", project="myproj"
        )
        with patch(
            "myagent.core.scheduler.subprocess.run",
            return_value=_completed(0, str(project_dir), ""),
        ) as mock_run:
            scheduler.run_task("alpha")
        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] == str(project_dir)

    def test_output_captured_and_exit_code_recorded(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="alpha", schedule="30m", command="echo hi")
        with patch(
            "myagent.core.scheduler.subprocess.run",
            return_value=_completed(2, "out", "err"),
        ):
            result = scheduler.run_task("alpha")
        assert result.exit_code == 2
        assert result.success is False
        assert result.stdout == "out"
        assert result.stderr == "err"
        updated = scheduler.get_task("alpha")
        assert updated is not None
        assert updated.last_exit_code == 2
        assert "out" in updated.last_output
        assert "err" in updated.last_output

    def test_task_updated_in_store_after_run(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="alpha", schedule="30m", command="echo hi")
        with patch(
            "myagent.core.scheduler.subprocess.run",
            return_value=_completed(0, "ok", ""),
        ):
            scheduler.run_task("alpha")
        updated = scheduler.get_task("alpha")
        assert updated is not None
        assert updated.last_run > 0
        assert updated.last_exit_code == 0

    def test_subprocess_failure_returns_failure_result(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="alpha", schedule="30m", command="bad-cmd")
        with patch(
            "myagent.core.scheduler.subprocess.run",
            side_effect=FileNotFoundError("no such binary"),
        ):
            result = scheduler.run_task("alpha")
        assert result.success is False
        assert result.exit_code == -1
        assert "no such binary" in result.stderr


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @freeze_time("2025-06-15 12:00:00", tz_offset=0)
    def test_disabled_task_not_run_by_run_due(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="alpha", schedule="30m", command="echo hi", enabled=False)
        # Force next_run into the past so a normal run would fire.
        task = scheduler.get_task("alpha")
        assert task is not None
        scheduler.store.update(
            ScheduledTask(
                id=task.id,
                name=task.name,
                schedule=task.schedule,
                schedule_type=task.schedule_type,
                command=task.command,
                project=task.project,
                enabled=False,
                created_at=task.created_at,
                last_run=task.last_run,
                next_run=time.time() - 60,
                last_exit_code=task.last_exit_code,
                last_output=task.last_output,
            )
        )
        with patch(
            "myagent.core.scheduler.subprocess.run",
            return_value=_completed(0, "", ""),
        ) as mock_run:
            results = scheduler.run_due()
        assert results == []
        mock_run.assert_not_called()

    def test_task_with_project_resolves_cwd(self, tmp_path: Path) -> None:
        project_store = ProjectStore(store_path=tmp_path / "p.json")
        project_dir = tmp_path / "p"
        project_dir.mkdir()
        project_store.add(
            ManagedProject(name="p", path=str(project_dir), added_at=1.0)
        )
        scheduler = Scheduler(
            store=ScheduleStore(tmp_path / "s.json"),
            project_store=project_store,
        )
        task = scheduler.add_task(
            name="alpha", schedule="30m", command="pwd", project="p"
        )
        cwd = scheduler._resolve_cwd(task)
        assert cwd == str(project_dir)

    def test_task_with_unknown_project_returns_none_cwd(self, tmp_path: Path) -> None:
        project_store = ProjectStore(store_path=tmp_path / "p.json")
        scheduler = Scheduler(
            store=ScheduleStore(tmp_path / "s.json"),
            project_store=project_store,
        )
        task = scheduler.add_task(
            name="alpha", schedule="30m", command="pwd", project="ghost"
        )
        assert scheduler._resolve_cwd(task) is None

    @freeze_time("2025-06-15 12:00:00", tz_offset=0)
    def test_task_with_empty_command_still_runs(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        scheduler.add_task(name="alpha", schedule="30m", command="")
        with patch(
            "myagent.core.scheduler.subprocess.run",
            return_value=_completed(0, "", ""),
        ) as mock_run:
            result = scheduler.run_task("alpha")
        mock_run.assert_called_once()
        assert result.success is True
        _, kwargs = mock_run.call_args
        assert kwargs["shell"] is True

    def test_daemon_loop_stops_on_keyboard_interrupt(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        # Stop after a few iterations by having time.sleep raise KeyboardInterrupt.
        sleep_calls = {"count": 0}

        def fake_sleep(_seconds: float) -> None:
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 3:
                raise KeyboardInterrupt

        with (
            patch(
                "myagent.core.scheduler.subprocess.run",
                return_value=_completed(0, "", ""),
            ),
            patch("myagent.core.scheduler.time.sleep", side_effect=fake_sleep),
        ):
            scheduler.run_daemon(check_interval=0.01)
        # Daemon ran at least 3 cycles before being interrupted.
        assert sleep_calls["count"] >= 3

    def test_daemon_calls_run_due_each_cycle(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        call_count = {"n": 0}

        def fake_run_due(now: float | None = None) -> list[TaskRunResult]:
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise KeyboardInterrupt
            return []

        scheduler.run_due = fake_run_due  # type: ignore[method-assign]
        # Should exit cleanly without raising.
        scheduler.run_daemon(check_interval=0.01)
        assert call_count["n"] >= 2

    def test_parse_empty_expression_raises(self) -> None:
        with pytest.raises(SchedulerError):
            ScheduleParser().parse("   ")

    def test_add_task_empty_name_raises(self, tmp_path: Path) -> None:
        scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
        with pytest.raises(SchedulerError):
            scheduler.add_task(name="", schedule="30m", command="x")


# ---------------------------------------------------------------------------
# Daemon mock helpers
# ---------------------------------------------------------------------------


def test_run_daemon_with_mock_subprocess(tmp_path: Path) -> None:
    """Daemon should not crash when subprocess succeeds; stops on interrupt."""
    scheduler = Scheduler(store=ScheduleStore(tmp_path / "s.json"))
    sleep_calls = {"n": 0}

    def fake_sleep(_s: float) -> None:
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise KeyboardInterrupt

    with (
        patch(
            "myagent.core.scheduler.subprocess.run",
            return_value=_completed(0, "", ""),
        ),
        patch("myagent.core.scheduler.time.sleep", side_effect=fake_sleep),
    ):
        scheduler.run_daemon(check_interval=0.0)
    assert sleep_calls["n"] >= 2


def test_module_exports() -> None:
    """Sanity check: public symbols are importable from the module."""
    from myagent.core import scheduler as module

    for name in (
        "Scheduler",
        "SchedulerError",
        "ScheduleStore",
        "ScheduleParser",
        "ScheduledTask",
        "TaskRunResult",
        "ScheduleType",
    ):
        assert hasattr(module, name)
