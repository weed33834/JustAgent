"""Tests for :mod:`justagent.core.batch_ops`."""

from __future__ import annotations

import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from justagent.core.batch_ops import (
    BatchOperation,
    BatchResult,
    BatchRunner,
    BatchSummary,
)
from justagent.core.project_store import ProjectStore
from justagent.models.project import ManagedProject


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> CompletedProcess[str]:
    """Build a fake :class:`subprocess.CompletedProcess`."""
    return CompletedProcess(args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr)


def _add_project(store: ProjectStore, tmp_path: Path, name: str) -> Path:
    """Create a project directory and register it in ``store``."""
    directory = tmp_path / name
    directory.mkdir()
    store.add(ManagedProject(name=name, path=str(directory), added_at=1.0))
    return directory


# ---------------------------------------------------------------------------
# TestBatchOperation
# ---------------------------------------------------------------------------


class TestBatchOperation:
    def test_values(self) -> None:
        assert BatchOperation.STATUS.value == "status"
        assert BatchOperation.CLEAN.value == "clean"
        assert BatchOperation.VERIFY.value == "verify"
        assert BatchOperation.COMMIT.value == "commit"
        assert BatchOperation.SHIP.value == "ship"
        assert BatchOperation.RUN.value == "run"

    def test_is_str(self) -> None:
        assert isinstance(BatchOperation.SHIP, str)

    def test_from_value(self) -> None:
        assert BatchOperation("clean") is BatchOperation.CLEAN


# ---------------------------------------------------------------------------
# TestBatchResult
# ---------------------------------------------------------------------------


class TestBatchResult:
    def test_construction(self) -> None:
        result = BatchResult(
            project_name="p",
            project_path="/p",
            operation=BatchOperation.STATUS,
            success=True,
        )
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == ""
        assert result.elapsed_seconds == 0.0

    def test_frozen(self) -> None:
        result = BatchResult(
            project_name="p",
            project_path="/p",
            operation=BatchOperation.STATUS,
            success=True,
        )
        with pytest.raises(FrozenInstanceError):
            result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestBatchSummary
# ---------------------------------------------------------------------------


class TestBatchSummary:
    def test_construction(self) -> None:
        results = [
            BatchResult("a", "/a", BatchOperation.STATUS, True),
            BatchResult("b", "/b", BatchOperation.STATUS, False, exit_code=1),
        ]
        summary = BatchSummary(
            operation=BatchOperation.STATUS,
            results=results,
            total=2,
            succeeded=1,
            failed=1,
            elapsed_seconds=0.5,
        )
        assert summary.total == 2
        assert summary.succeeded == 1
        assert summary.failed == 1
        assert summary.operation == BatchOperation.STATUS

    def test_totals_computed_by_runner(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        _add_project(store, tmp_path, "p0")
        _add_project(store, tmp_path, "p1")
        runner = BatchRunner(store)
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            side_effect=[_completed(0, "", ""), _completed(1, "", "err")],
        ):
            summary = runner.run_status(None)
        assert summary.total == 2
        assert summary.succeeded == 1
        assert summary.failed == 1


# ---------------------------------------------------------------------------
# TestBatchRunnerStatus
# ---------------------------------------------------------------------------


class TestBatchRunnerStatus:
    def test_single_project_clean(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        proj_dir = _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            return_value=_completed(0, "", ""),
        ) as mock_run:
            summary = runner.run_status(["proj"])
        assert summary.total == 1
        assert summary.succeeded == 1
        assert summary.results[0].success is True
        assert summary.results[0].stdout == ""
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["git", "status", "--porcelain"]
        assert kwargs["cwd"] == str(proj_dir)

    def test_single_project_dirty(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            return_value=_completed(0, " M file.txt\n", ""),
        ):
            summary = runner.run_status(["proj"])
        assert summary.results[0].success is True
        assert "file.txt" in summary.results[0].stdout

    def test_multiple_projects(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        for name in ("p0", "p1", "p2"):
            _add_project(store, tmp_path, name)
        runner = BatchRunner(store)
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            return_value=_completed(0, "", ""),
        ):
            summary = runner.run_status(None)
        assert summary.total == 3
        assert summary.succeeded == 3
        assert [r.project_name for r in summary.results] == ["p0", "p1", "p2"]

    def test_missing_project_recorded_as_error(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        runner = BatchRunner(store)
        summary = runner.run_status(["ghost"])
        assert summary.total == 1
        assert summary.failed == 1
        assert "not found" in summary.results[0].error
        assert summary.results[0].success is False

    def test_parallel_execution(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        for name in ("p0", "p1", "p2"):
            _add_project(store, tmp_path, name)
        runner = BatchRunner(store, parallel=True, max_workers=2)
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            return_value=_completed(0, "", ""),
        ) as mock_run:
            summary = runner.run_status(None)
        assert summary.total == 3
        assert summary.succeeded == 3
        assert mock_run.call_count == 3
        assert [r.project_name for r in summary.results] == ["p0", "p1", "p2"]


# ---------------------------------------------------------------------------
# TestBatchRunnerCommand
# ---------------------------------------------------------------------------


class TestBatchRunnerCommand:
    def test_run_echo_command(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        summary = runner.run_command(None, ["echo", "hello"])
        assert summary.total == 1
        assert summary.succeeded == 1
        assert "hello" in summary.results[0].stdout

    def test_run_failing_command(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        summary = runner.run_command(None, ["sh", "-c", "exit 3"])
        assert summary.results[0].success is False
        assert summary.results[0].exit_code == 3
        assert summary.failed == 1

    def test_command_output_captured(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        summary = runner.run_command(None, ["sh", "-c", "echo captured"])
        assert "captured" in summary.results[0].stdout
        assert summary.results[0].stderr == ""


# ---------------------------------------------------------------------------
# TestBatchRunnerPipeline
# ---------------------------------------------------------------------------


class TestBatchRunnerPipeline:
    def test_single_stage(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        proj_dir = _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            return_value=_completed(0, "ok", ""),
        ) as mock_run:
            summary = runner.run_pipeline(["proj"], [BatchOperation.CLEAN])
        assert summary.total == 1
        assert summary.succeeded == 1
        assert summary.operation == BatchOperation.CLEAN
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["justagent", "clean"]
        assert kwargs["cwd"] == str(proj_dir)

    def test_multiple_stages(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        stages = [BatchOperation.CLEAN, BatchOperation.VERIFY, BatchOperation.SHIP]
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            return_value=_completed(0, "", ""),
        ) as mock_run:
            summary = runner.run_pipeline(["proj"], stages)
        assert summary.total == 3
        assert summary.succeeded == 3
        assert summary.operation == BatchOperation.SHIP
        assert mock_run.call_count == 3

    def test_stage_failure_aborts_remaining_stages(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        stages = [BatchOperation.CLEAN, BatchOperation.VERIFY, BatchOperation.SHIP]
        # First stage (clean) fails; verify and ship should be skipped.
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            return_value=_completed(1, "", "fail"),
        ) as mock_run:
            summary = runner.run_pipeline(["proj"], stages)
        assert summary.total == 1
        assert summary.failed == 1
        assert summary.results[0].operation == BatchOperation.CLEAN
        assert mock_run.call_count == 1

    def test_pipeline_shells_to_myagent(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            return_value=_completed(0, "", ""),
        ) as mock_run:
            runner.run_pipeline(["proj"], [BatchOperation.SHIP])
        args, _ = mock_run.call_args
        assert args[0] == ["justagent", "ship"]


# ---------------------------------------------------------------------------
# TestFormatSummary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_format_all_success(self) -> None:
        runner = BatchRunner(ProjectStore())
        results = [
            BatchResult("p0", "/p0", BatchOperation.STATUS, True, elapsed_seconds=0.1),
            BatchResult("p1", "/p1", BatchOperation.STATUS, True, elapsed_seconds=0.2),
        ]
        summary = BatchSummary(
            operation=BatchOperation.STATUS,
            results=results,
            total=2,
            succeeded=2,
            failed=0,
            elapsed_seconds=0.3,
        )
        output = runner.format_summary(summary)
        assert "OK" in output
        assert "FAIL" not in output
        assert "p0" in output
        assert "Succeeded: 2" in output

    def test_format_mixed(self) -> None:
        runner = BatchRunner(ProjectStore())
        results = [
            BatchResult("p0", "/p0", BatchOperation.STATUS, True),
            BatchResult("p1", "/p1", BatchOperation.STATUS, False, exit_code=1),
        ]
        summary = BatchSummary(
            operation=BatchOperation.STATUS,
            results=results,
            total=2,
            succeeded=1,
            failed=1,
            elapsed_seconds=0.0,
        )
        output = runner.format_summary(summary)
        assert "OK" in output
        assert "FAIL" in output
        assert "Failed: 1" in output

    def test_format_empty(self) -> None:
        runner = BatchRunner(ProjectStore())
        summary = BatchSummary(
            operation=BatchOperation.STATUS,
            results=[],
            total=0,
            succeeded=0,
            failed=0,
            elapsed_seconds=0.0,
        )
        output = runner.format_summary(summary)
        assert "Total: 0" in output
        assert "Succeeded: 0" in output


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_project_list(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        runner = BatchRunner(store)
        summary = runner.run_status(None)
        assert summary.total == 0
        assert summary.results == []

    def test_nonexistent_project_names(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        runner = BatchRunner(store)
        summary = runner.run_command(["ghost"], ["echo", "hi"])
        assert summary.total == 1
        assert summary.failed == 1
        assert "not found" in summary.results[0].error

    def test_timeout_handling(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=300),
        ):
            summary = runner.run_status(["proj"])
        assert summary.total == 1
        assert summary.failed == 1
        assert "timed out" in summary.results[0].error
        assert summary.results[0].exit_code == -1

    def test_oserror_handling(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        _add_project(store, tmp_path, "proj")
        runner = BatchRunner(store)
        with patch(
            "justagent.core.batch_ops.subprocess.run",
            side_effect=FileNotFoundError("no such binary"),
        ):
            summary = runner.run_status(["proj"])
        assert summary.failed == 1
        assert "no such binary" in summary.results[0].error

    def test_max_workers_floored_to_one(self, tmp_path: Path) -> None:
        store = ProjectStore(store_path=tmp_path / "projects.json")
        runner = BatchRunner(store, parallel=True, max_workers=0)
        assert runner.max_workers == 1
