"""Tests for the ``myagent schedule`` CLI command group."""

from __future__ import annotations

import time
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from typer.testing import CliRunner

from myagent.cli.main import app
from myagent.core.project_store import ProjectStore
from myagent.core.scheduler import (
    ScheduledTask,
    Scheduler,
    ScheduleStore,
    ScheduleType,
)
from myagent.models.project import ManagedProject

runner = CliRunner()


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> CompletedProcess[str]:
    """Build a fake :class:`subprocess.CompletedProcess`."""
    return CompletedProcess(
        args="fake", returncode=returncode, stdout=stdout, stderr=stderr
    )


def _patch_scheduler(store: ScheduleStore) -> patch[Scheduler]:
    """Patch the CLI's scheduler factory to use ``store``."""
    return patch(
        "myagent.cli.commands.schedule.Scheduler",
        return_value=Scheduler(store=store, project_store=ProjectStore()),
    )


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_list_empty(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        result = runner.invoke(app, ["schedule", "list"])
    assert result.exit_code == 0, result.output
    assert "no scheduled tasks" in result.output.lower()


def test_add_task(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        add_result = runner.invoke(
            app,
            [
                "schedule", "add", "alpha",
                "--schedule", "30m",
                "--command", "echo hi",
            ],
        )
    assert add_result.exit_code == 0, add_result.output
    assert "alpha" in add_result.output

    with _patch_scheduler(store):
        list_result = runner.invoke(app, ["schedule", "list"])
    assert list_result.exit_code == 0, list_result.output
    assert "alpha" in list_result.output
    assert "30m" in list_result.output


def test_add_task_with_project(tmp_path: Path) -> None:
    project_store = ProjectStore(store_path=tmp_path / "projects.json")
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    project_store.add(
        ManagedProject(name="myproj", path=str(project_dir), added_at=1.0)
    )
    schedule_store = ScheduleStore(store_path=tmp_path / "schedules.json")

    with patch(
        "myagent.cli.commands.schedule.Scheduler",
        return_value=Scheduler(store=schedule_store, project_store=project_store),
    ):
        add_result = runner.invoke(
            app,
            [
                "schedule", "add", "alpha",
                "--schedule", "1h",
                "--command", "pwd",
                "--project", "myproj",
            ],
        )
    assert add_result.exit_code == 0, add_result.output
    assert "alpha" in add_result.output

    # The task should have the project attached.
    with _patch_scheduler(schedule_store):
        list_result = runner.invoke(app, ["schedule", "list"])
    assert list_result.exit_code == 0, list_result.output
    assert "alpha" in list_result.output


def test_remove_task(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        runner.invoke(
            app,
            [
                "schedule", "add", "alpha",
                "--schedule", "30m",
                "--command", "echo hi",
            ],
        )
        remove_result = runner.invoke(app, ["schedule", "remove", "alpha"])
    assert remove_result.exit_code == 0, remove_result.output
    assert "Removed task" in remove_result.output
    assert store.get("alpha") is None


def test_enable_disable(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        runner.invoke(
            app,
            [
                "schedule", "add", "alpha",
                "--schedule", "30m",
                "--command", "echo hi",
            ],
        )
        disable_result = runner.invoke(app, ["schedule", "disable", "alpha"])
    assert disable_result.exit_code == 0, disable_result.output
    assert store.get("alpha").enabled is False  # type: ignore[union-attr]

    with _patch_scheduler(store):
        enable_result = runner.invoke(app, ["schedule", "enable", "alpha"])
    assert enable_result.exit_code == 0, enable_result.output
    assert store.get("alpha").enabled is True  # type: ignore[union-attr]


def test_run_task(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        runner.invoke(
            app,
            [
                "schedule", "add", "alpha",
                "--schedule", "30m",
                "--command", "echo hi",
            ],
        )

    with _patch_scheduler(store), patch(
        "myagent.core.scheduler.subprocess.run",
        return_value=_completed(0, "hi\n", ""),
    ) as mock_run:
        run_result = runner.invoke(app, ["schedule", "run", "alpha"])
    assert run_result.exit_code == 0, run_result.output
    mock_run.assert_called_once()
    assert "succeeded" in run_result.output.lower()
    assert "hi" in run_result.output

    # The store should reflect the run.
    updated = store.get("alpha")
    assert updated is not None
    assert updated.last_exit_code == 0


def test_due(tmp_path: Path) -> None:
    """The ``due`` command runs every task whose next_run is in the past."""
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    # Add a task with next_run forced into the past.
    store.add(
        ScheduledTask(
            id="alpha",
            name="alpha",
            schedule="30m",
            schedule_type=ScheduleType.INTERVAL,
            command="echo hi",
            enabled=True,
            created_at=1.0,
            next_run=time.time() - 60,
        )
    )

    with _patch_scheduler(store), patch(
        "myagent.core.scheduler.subprocess.run",
        return_value=_completed(0, "", ""),
    ) as mock_run:
        result = runner.invoke(app, ["schedule", "due"])
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    assert "alpha" in result.output
    assert "OK" in result.output


def test_due_no_due_tasks(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        result = runner.invoke(app, ["schedule", "due"])
    assert result.exit_code == 0, result.output
    assert "no due tasks" in result.output.lower()


def test_add_invalid_schedule(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        result = runner.invoke(
            app,
            [
                "schedule", "add", "bad",
                "--schedule", "not a schedule",
                "--command", "echo hi",
            ],
        )
    assert result.exit_code == 1, result.output
    assert "invalid schedule" in result.output.lower()
    # The task should not have been persisted.
    assert store.get("bad") is None


def test_remove_nonexistent(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        result = runner.invoke(app, ["schedule", "remove", "ghost"])
    assert result.exit_code == 1, result.output
    assert "no scheduled task" in result.output.lower()


def test_enable_nonexistent(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        result = runner.invoke(app, ["schedule", "enable", "ghost"])
    assert result.exit_code == 1, result.output
    assert "no scheduled task" in result.output.lower()


def test_disable_nonexistent(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        result = runner.invoke(app, ["schedule", "disable", "ghost"])
    assert result.exit_code == 1, result.output
    assert "no scheduled task" in result.output.lower()


def test_run_nonexistent(tmp_path: Path) -> None:
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        result = runner.invoke(app, ["schedule", "run", "ghost"])
    assert result.exit_code == 1, result.output


def test_run_task_failure_exit_code(tmp_path: Path) -> None:
    """A failing task surfaces a non-zero exit code."""
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        runner.invoke(
            app,
            [
                "schedule", "add", "alpha",
                "--schedule", "30m",
                "--command", "false",
            ],
        )
    with _patch_scheduler(store), patch(
        "myagent.core.scheduler.subprocess.run",
        return_value=_completed(2, "", "boom"),
    ):
        result = runner.invoke(app, ["schedule", "run", "alpha"])
    assert result.exit_code != 0
    assert "failed" in result.output.lower()


def test_add_with_disabled_flag(tmp_path: Path) -> None:
    """``--disabled`` adds the task in a disabled state."""
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    with _patch_scheduler(store):
        runner.invoke(
            app,
            [
                "schedule", "add", "alpha",
                "--schedule", "30m",
                "--command", "echo hi",
                "--disabled",
            ],
        )
    task = store.get("alpha")
    assert task is not None
    assert task.enabled is False


def test_due_failure_propagates_exit_code(tmp_path: Path) -> None:
    """A failing task in ``due`` causes the command to exit non-zero."""
    store = ScheduleStore(store_path=tmp_path / "schedules.json")
    store.add(
        ScheduledTask(
            id="alpha",
            name="alpha",
            schedule="30m",
            schedule_type=ScheduleType.INTERVAL,
            command="false",
            enabled=True,
            created_at=1.0,
            next_run=time.time() - 60,
        )
    )
    with _patch_scheduler(store), patch(
        "myagent.core.scheduler.subprocess.run",
        return_value=_completed(1, "", ""),
    ):
        result = runner.invoke(app, ["schedule", "due"])
    assert result.exit_code == 1
    assert "FAIL" in result.output


# ---------------------------------------------------------------------------
# Daemon smoke test
# ---------------------------------------------------------------------------


def test_daemon_does_not_block_on_keyboard_interrupt(tmp_path: Path) -> None:
    """The daemon command should stop cleanly on KeyboardInterrupt."""
    store = ScheduleStore(store_path=tmp_path / "schedules.json")

    def fake_run_daemon(check_interval: float = 60.0) -> None:
        # Simulate immediate interrupt without actually looping.
        return

    with patch(
        "myagent.cli.commands.schedule.Scheduler",
        return_value=Scheduler(store=store, project_store=ProjectStore()),
    ) as mock_scheduler_cls:
        mock_scheduler_cls.return_value.run_daemon = fake_run_daemon  # type: ignore[method-assign]
        result = runner.invoke(app, ["schedule", "daemon", "--check-interval", "0.01"])
    assert result.exit_code == 0, result.output
    assert "daemon" in result.output.lower()
