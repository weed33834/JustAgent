"""Tests for the ``autoship hooks`` command."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from autoship.cli.main import app

runner = CliRunner()


def _write_config(tmp_path: Path, body: str) -> Path:
    config_path = tmp_path / ".autoship.toml"
    config_path.write_text(
        f'schema_version = 1\nproject_root = "{tmp_path}"\n{body}', encoding="utf-8"
    )
    return config_path


# ---------------------------------------------------------------------------
# hooks list
# ---------------------------------------------------------------------------


def test_hooks_list_disabled(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "")
    result = runner.invoke(app, ["--config", str(config_path), "hooks", "list"])
    assert result.exit_code == 0
    assert "disabled" in result.output.lower()


def test_hooks_list_empty(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "[hooks]\nenabled = true\n")
    result = runner.invoke(app, ["--config", str(config_path), "hooks", "list"])
    assert result.exit_code == 0
    assert "no on-save hooks" in result.output.lower() or "no on-save" in result.output.lower()


def test_hooks_list_shows_configured(tmp_path: Path) -> None:
    body = (
        "[hooks]\nenabled = true\n"
        '[[hooks.on_save]]\ncommand = "clean"\ninclude = ["**/*.py"]\ndebounce_ms = 100\n'
        '[[hooks.on_save]]\ncommand = "verify"\nverify_command = "pytest"\n'
        'include = ["tests/**"]\ndebounce_ms = 200\n'
    )
    config_path = _write_config(tmp_path, body)
    result = runner.invoke(app, ["--config", str(config_path), "hooks", "list"])
    assert result.exit_code == 0
    assert "clean" in result.output
    assert "verify" in result.output
    assert "pytest" in result.output


# ---------------------------------------------------------------------------
# hooks run
# ---------------------------------------------------------------------------


def test_hooks_run_no_match(tmp_path: Path) -> None:
    body = "[hooks]\nenabled = true\n[[hooks.on_save]]\ncommand = 'clean'\ninclude = ['**/*.py']\n"
    config_path = _write_config(tmp_path, body)
    result = runner.invoke(
        app,
        ["--config", str(config_path), "hooks", "run", "--file", str(tmp_path / "README.md")],
    )
    assert result.exit_code == 0
    assert "no on-save hooks match" in result.output.lower()


def test_hooks_run_disabled(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "")
    result = runner.invoke(
        app,
        ["--config", str(config_path), "hooks", "run", "--file", str(tmp_path / "a.py")],
    )
    assert result.exit_code == 0
    assert "disabled" in result.output.lower()


def test_hooks_run_executes_matching_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = "[hooks]\nenabled = true\n[[hooks.on_save]]\ncommand = 'clean'\ninclude = ['**/*.py']\n"
    config_path = _write_config(tmp_path, body)
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="cleaned", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(
        app,
        ["--config", str(config_path), "hooks", "run", "--file", str(target)],
    )
    assert result.exit_code == 0
    assert "OK" in result.output
    assert len(calls) == 1
    assert "clean" in calls[0]


def test_hooks_run_failure_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = "[hooks]\nenabled = true\n[[hooks.on_save]]\ncommand = 'clean'\ninclude = ['**/*.py']\n"
    config_path = _write_config(tmp_path, body)
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="bad"),
    )
    result = runner.invoke(
        app,
        ["--config", str(config_path), "hooks", "run", "--file", str(target)],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output


# ---------------------------------------------------------------------------
# hooks watch (smoke test only — the watcher loop is long-running)
# ---------------------------------------------------------------------------


def test_hooks_watch_disabled_exits_zero(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "")
    result = runner.invoke(app, ["--config", str(config_path), "hooks", "watch", str(tmp_path)])
    assert result.exit_code == 0
    assert "disabled" in result.output.lower()


# ---------------------------------------------------------------------------
# hooks watch (full flow — uses a fake Observer so no real FS watcher starts)
# ---------------------------------------------------------------------------


class _FakeObserver:
    """Stand-in for ``watchdog.observers.Observer``.

    The first ``join`` raises ``KeyboardInterrupt`` so the watch loop exits
    immediately; the second ``join`` (called from the ``finally`` block)
    returns normally. This lets us exercise the full start/stop lifecycle
    without blocking on a real filesystem watcher.
    """

    def __init__(self) -> None:
        self.join_calls = 0
        self.started = False
        self.stopped = False
        self.scheduled: list[tuple[Any, str, bool]] = []
        self.handler: Any = None

    def schedule(self, handler: Any, path: str, recursive: bool) -> None:
        self.scheduled.append((handler, path, recursive))
        self.handler = handler

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1
        if self.join_calls == 1:
            raise KeyboardInterrupt

    def stop(self) -> None:
        self.stopped = True


def _patch_observer(monkeypatch: pytest.MonkeyPatch) -> _FakeObserver:
    """Patch ``watchdog.observers.Observer`` to return a fake instance."""
    observer = _FakeObserver()
    monkeypatch.setattr("watchdog.observers.Observer", lambda: observer)
    return observer


def test_hooks_watch_missing_path_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path whose parent does not exist prints ``watch_missing`` and exits 1."""
    config_path = _write_config(tmp_path, "[hooks]\nenabled = true\n")
    observer = _patch_observer(monkeypatch)
    # Parent does not exist → target.exists() is False → watch_missing + skip.
    missing = tmp_path / "missing_parent" / "child"
    result = runner.invoke(app, ["--config", str(config_path), "hooks", "watch", str(missing)])
    assert result.exit_code == 1
    assert "does not exist" in result.output.lower()
    assert not observer.started
    assert observer.scheduled == []


def test_hooks_watch_all_paths_missing_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no path resolves, ``watched`` is empty → exit 1, start never called."""
    config_path = _write_config(tmp_path, "[hooks]\nenabled = true\n")
    observer = _patch_observer(monkeypatch)
    missing_a = tmp_path / "no_a" / "child"
    missing_b = tmp_path / "no_b" / "child"
    result = runner.invoke(
        app,
        ["--config", str(config_path), "hooks", "watch", str(missing_a), str(missing_b)],
    )
    assert result.exit_code == 1
    assert not observer.started
    assert observer.scheduled == []


def test_hooks_watch_keyboard_interrupt_stops_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C stops the observer: start, schedule, stop and join all run."""
    body = "[hooks]\nenabled = true\n[[hooks.on_save]]\ncommand = 'clean'\ninclude = ['**/*.py']\n"
    config_path = _write_config(tmp_path, body)
    observer = _patch_observer(monkeypatch)
    result = runner.invoke(app, ["--config", str(config_path), "hooks", "watch", str(tmp_path)])
    assert result.exit_code == 0
    assert observer.started
    assert observer.stopped
    assert len(observer.scheduled) == 1
    assert observer.scheduled[0][1] == str(tmp_path)
    assert observer.join_calls == 2
    assert "stopped" in result.output.lower()


def _watch_and_get_handler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Any:
    """Invoke ``hooks watch`` on ``tmp_path`` and return the captured handler."""
    config_path = _write_config(tmp_path, body)
    observer = _patch_observer(monkeypatch)
    result = runner.invoke(app, ["--config", str(config_path), "hooks", "watch", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert observer.handler is not None
    return observer.handler


def _fake_subprocess_ok(calls: list[list[str]]) -> Any:
    """Return a ``subprocess.run`` replacement that records cmds and succeeds."""

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_run


def test_hooks_watch_handler_on_modified_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``on_modified`` routes through ``_dispatch`` to ``run_for_path``."""
    body = "[hooks]\nenabled = true\n[[hooks.on_save]]\ncommand = 'clean'\ninclude = ['**/*.py']\n"
    handler = _watch_and_get_handler(tmp_path, monkeypatch, body)

    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")

    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_ok(calls))

    from watchdog.events import FileSystemEvent

    event = FileSystemEvent(str(target))
    event.is_directory = False
    handler.on_modified(event)

    assert len(calls) == 1
    assert "clean" in calls[0]


def test_hooks_watch_handler_on_created_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``on_created`` routes through ``_dispatch`` to ``run_for_path``."""
    body = "[hooks]\nenabled = true\n[[hooks.on_save]]\ncommand = 'clean'\ninclude = ['**/*.py']\n"
    handler = _watch_and_get_handler(tmp_path, monkeypatch, body)

    target = tmp_path / "created.py"
    target.write_text("y = 2\n", encoding="utf-8")

    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_ok(calls))

    from watchdog.events import FileSystemEvent

    event = FileSystemEvent(str(target))
    event.is_directory = False
    handler.on_created(event)

    assert len(calls) == 1
    assert "clean" in calls[0]


def test_hooks_watch_handler_skips_excluded_glob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files matching the exclude glob are skipped (no subprocess)."""
    body = (
        "[hooks]\nenabled = true\n"
        "[[hooks.on_save]]\ncommand = 'clean'\n"
        "include = ['**/*.py']\nexclude = ['**/test_*']\n"
    )
    handler = _watch_and_get_handler(tmp_path, monkeypatch, body)

    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_ok(calls))

    from watchdog.events import FileSystemEvent

    target = tmp_path / "test_skip.py"
    event = FileSystemEvent(str(target))
    event.is_directory = False
    handler.on_modified(event)

    assert calls == []


def test_hooks_watch_handler_skips_excluded_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files inside an ``_EXCLUDED_DIRS`` entry (e.g. ``.git``) are skipped."""
    body = "[hooks]\nenabled = true\n[[hooks.on_save]]\ncommand = 'clean'\ninclude = ['**/*.py']\n"
    handler = _watch_and_get_handler(tmp_path, monkeypatch, body)

    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_ok(calls))

    from watchdog.events import FileSystemEvent

    target = tmp_path / ".git" / "config.py"
    event = FileSystemEvent(str(target))
    event.is_directory = False
    handler.on_modified(event)

    assert calls == []
