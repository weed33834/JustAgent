"""Tests for the ``autoship ui`` command (textual dashboard wrapper).

The textual TUI itself is never started — ``_build_app`` is exercised to
construct the dashboard, and the ``action_run_*`` handlers are driven with
``subprocess.run`` and the textual DOM (``query_one``/``mount``) mocked out so
no real app loop or subprocess runs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from autoship.cli.commands import ui as ui_module
from autoship.cli.commands.ui import (
    _build_app,
    _check_textual_available,
    _project_summary,
    _tail_audit,
)
from autoship.cli.main import app
from autoship.core.audit_logger import AuditLogger
from autoship.core.i18n import get_i18n
from autoship.models.config import AppConfig

runner = CliRunner()


def _write_config(tmp_path: Path, log_dir: Path) -> Path:
    config_path = tmp_path / ".autoship.toml"
    config_path.write_text(
        f'schema_version = 1\nproject_root = "{tmp_path}"\n\n[audit]\nlog_dir = "{log_dir}"\n',
        encoding="utf-8",
    )
    return config_path


def _make_logger(tmp_path: Path) -> AuditLogger:
    config = AppConfig(project_root=tmp_path, audit_log_dir=tmp_path / "logs")
    return AuditLogger(config)


def _make_dashboard(tmp_path: Path) -> object:
    config = AppConfig(project_root=tmp_path, audit_log_dir=tmp_path / "logs")
    logger = AuditLogger(config)
    return _build_app(config, logger, get_i18n("en"))


# ---------------------------------------------------------------------------
# _check_textual_available
# ---------------------------------------------------------------------------


def test_check_textual_available_true_when_installed() -> None:
    # textual is a dev/test dependency, so find_spec returns a real spec.
    assert _check_textual_available() is True


def test_check_textual_available_false_when_find_spec_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert _check_textual_available() is False


def test_check_textual_available_false_when_find_spec_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert _check_textual_available() is False


# ---------------------------------------------------------------------------
# _tail_audit
# ---------------------------------------------------------------------------


def test_tail_audit_empty_when_no_logs(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    assert _tail_audit(logger) == []


def test_tail_audit_returns_last_records_most_recent_first(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "audit.2024-01-01.jsonl").write_text(
        json.dumps({"ts": "2024-01-01T00:00:00", "event": "a"})
        + "\n"
        + json.dumps({"ts": "2024-01-01T00:00:01", "event": "b"})
        + "\n",
        encoding="utf-8",
    )
    logger = _make_logger(tmp_path)
    lines = _tail_audit(logger, lines=10)
    assert [line.split(maxsplit=1) for line in lines] == [
        ["2024-01-01T00:00:01", "b"],
        ["2024-01-01T00:00:00", "a"],
    ]


def test_tail_audit_respects_lines_limit(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "audit.2024-01-01.jsonl").write_text(
        "".join(
            json.dumps({"ts": f"2024-01-01T00:00:{i:02d}", "event": f"e{i}"}) + "\n"
            for i in range(5)
        ),
        encoding="utf-8",
    )
    logger = _make_logger(tmp_path)
    assert len(_tail_audit(logger, lines=2)) == 2


def test_tail_audit_skips_corrupt_lines(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "audit.2024-01-01.jsonl").write_text(
        "not json\n" + json.dumps({"ts": "2024-01-01T00:00:00", "event": "good"}) + "\n" + "{bad\n",
        encoding="utf-8",
    )
    logger = _make_logger(tmp_path)
    lines = _tail_audit(logger)
    assert len(lines) == 1
    assert "good" in lines[0]


def test_tail_audit_skips_export_files(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "audit.export.abc.jsonl").write_text(
        json.dumps({"ts": "2024-01-01T00:00:00", "event": "export"}) + "\n",
        encoding="utf-8",
    )
    (log_dir / "audit.2024-01-01.jsonl").write_text(
        json.dumps({"ts": "2024-01-01T00:00:00", "event": "real"}) + "\n",
        encoding="utf-8",
    )
    logger = _make_logger(tmp_path)
    lines = _tail_audit(logger)
    assert len(lines) == 1
    assert "real" in lines[0]


def test_tail_audit_skips_unreadable_files(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    # A directory named like a log file triggers OSError on read_text regardless
    # of the running uid, exercising the ``except OSError`` skip branch.
    (log_dir / "audit.2024-01-01.jsonl").mkdir()
    (log_dir / "audit.2024-01-02.jsonl").write_text(
        json.dumps({"ts": "2024-01-02T00:00:00", "event": "ok"}) + "\n",
        encoding="utf-8",
    )
    logger = _make_logger(tmp_path)
    lines = _tail_audit(logger)
    assert len(lines) == 1
    assert "ok" in lines[0]


def test_tail_audit_uses_defaults_for_missing_fields(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "audit.2024-01-01.jsonl").write_text(
        json.dumps({"event": "e"}) + "\n" + json.dumps({"ts": "x"}) + "\n",
        encoding="utf-8",
    )
    logger = _make_logger(tmp_path)
    lines = _tail_audit(logger)
    # Records are read most-recent-first; the second line ({"ts": "x"}) has no
    # event, the first line ({"event": "e"}) has no ts.
    assert lines == ["x  ?", "?  e"]


# ---------------------------------------------------------------------------
# _project_summary
# ---------------------------------------------------------------------------


def test_project_summary_generic_when_no_markers(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    config = AppConfig(project_root=tmp_path, audit_log_dir=tmp_path / "logs")
    text = "\n".join(_project_summary(config, logger))
    assert f"Project root  : {tmp_path}" in text
    assert "Detected type : generic" in text
    # No rule matches a marker-less dir → primary falls back to detected.
    assert "Primary lang  : generic" in text
    assert "Rules applied : 0" in text
    assert logger.trace_id[:8] in text
    assert str(logger.log_dir) in text


def test_project_summary_python_project_uses_python_rule(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    logger = _make_logger(tmp_path)
    config = AppConfig(project_root=tmp_path, audit_log_dir=tmp_path / "logs")
    text = "\n".join(_project_summary(config, logger))
    assert "Detected type : python" in text
    # A python LanguageRule now exists, so primary_language returns
    # "python" via the rule (rather than falling back to the detected type).
    assert "Primary lang  : python" in text
    assert "Rules applied : 1" in text


def test_project_summary_go_project_has_matching_rule(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    logger = _make_logger(tmp_path)
    config = AppConfig(project_root=tmp_path, audit_log_dir=tmp_path / "logs")
    text = "\n".join(_project_summary(config, logger))
    assert "Detected type : go" in text
    assert "Primary lang  : go" in text
    assert "Rules applied : 1" in text


# ---------------------------------------------------------------------------
# _build_app
# ---------------------------------------------------------------------------


def test_build_app_returns_dashboard_with_bindings(tmp_path: Path) -> None:
    dashboard = _make_dashboard(tmp_path)
    assert dashboard is not None
    assert dashboard.__class__.__name__ == "AutoShipDashboard"
    # Quit binding + one per action (clean/verify/commit/upload/doctor).
    assert len(dashboard.BINDINGS) == 6  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# AutoShipDashboard action_run_* — subprocess.run + DOM mocked.
# ---------------------------------------------------------------------------


_ACTION_COMMANDS = [
    ("action_run_clean", [sys.executable, "-m", "autoship", "clean", "--yes"]),
    ("action_run_verify", [sys.executable, "-m", "autoship", "verify", "pytest"]),
    ("action_run_commit", [sys.executable, "-m", "autoship", "commit"]),
    ("action_run_upload", [sys.executable, "-m", "autoship", "upload", "--dry-run"]),
    ("action_run_doctor", [sys.executable, "-m", "autoship", "doctor"]),
]


def _wire_dashboard(dashboard: object, *, mount: object) -> object:
    """Replace ``query_one`` with a mock returning a right-pane stub."""
    right = MagicMock()
    right.mount = mount
    right.scroll_end = MagicMock()
    dashboard.query_one = MagicMock(return_value=right)  # type: ignore[method-assign]
    return right


@pytest.mark.parametrize("action,expected_cmd", _ACTION_COMMANDS)
def test_action_run_invokes_correct_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    expected_cmd: list[str],
) -> None:
    dashboard = _make_dashboard(tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="warn")

    monkeypatch.setattr(subprocess, "run", fake_run)
    right = _wire_dashboard(dashboard, mount=AsyncMock())

    asyncio.run(getattr(dashboard, action)())

    assert calls == [expected_cmd]
    assert right.mount.await_count >= 1
    right.scroll_end.assert_called_once_with(animate=False)


def test_run_command_no_output_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = _make_dashboard(tmp_path)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    right = _wire_dashboard(dashboard, mount=AsyncMock())

    asyncio.run(dashboard._run_command([sys.executable, "-m", "autoship", "doctor"]))  # type: ignore[attr-defined]

    # The "(no output, exit=...)" fallback is mounted into the right pane.
    assert right.mount.await_count >= 1
    right.scroll_end.assert_called_once_with(animate=False)


def test_run_command_handles_subprocess_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dashboard = _make_dashboard(tmp_path)

    def boom(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("kaboom")

    monkeypatch.setattr(subprocess, "run", boom)
    # _flash calls mount synchronously (no await), so mount is a plain Mock.
    right = _wire_dashboard(dashboard, mount=MagicMock())

    asyncio.run(dashboard._run_command([sys.executable, "-m", "autoship", "doctor"]))  # type: ignore[attr-defined]

    # The error path surfaces a red error line in the right pane.
    right.mount.assert_called_once()
    # subprocess.run was still invoked (the error came from inside it).
    right.scroll_end.assert_not_called()


# ---------------------------------------------------------------------------
# ui command entry point (CliRunner)
# ---------------------------------------------------------------------------


def test_ui_command_exits_two_when_textual_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, tmp_path / "logs")
    monkeypatch.setattr(ui_module, "_check_textual_available", lambda: False)
    result = runner.invoke(app, ["--config", str(config_path), "ui"])
    assert result.exit_code == 2
    assert "textual" in result.output.lower()


def test_ui_command_runs_dashboard_when_textual_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, tmp_path / "logs")
    fake_dashboard = MagicMock()
    monkeypatch.setattr(ui_module, "_build_app", lambda *a, **k: fake_dashboard)
    result = runner.invoke(app, ["--config", str(config_path), "ui"])
    assert result.exit_code == 0
    fake_dashboard.run.assert_called_once()
