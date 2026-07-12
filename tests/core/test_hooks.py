"""Tests for the on-save hook runner."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoship.core.hooks import (
    OnSaveHookRunner,
    _glob_to_regex,
    _matches_any,
)
from autoship.models.config import AppConfig, HookConfig, HooksConfig

# ---------------------------------------------------------------------------
# Glob translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,path,expected",
    [
        ("*.py", "bar.py", True),
        ("*.py", "src/bar.py", False),
        ("**/*.py", "bar.py", True),
        ("**/*.py", "src/foo/bar.py", True),
        ("**/*.py", "src/foo/bar.txt", False),
        ("src/**", "src/foo/bar.py", True),
        ("src/**", "tests/bar.py", False),
        ("**/*", "anything/here.txt", True),
        ("tests/**", "tests/core/test_x.py", True),
        ("*.py", "bar.pyc", False),
        ("?at.py", "cat.py", True),
        ("?at.py", "scat.py", False),
        ("[abc].py", "a.py", True),
        ("[!abc].py", "d.py", True),
        ("[!abc].py", "a.py", False),
    ],
)
def test_glob_to_regex_matches(pattern: str, path: str, expected: bool) -> None:
    assert bool(_glob_to_regex(pattern).match(path)) is expected


def test_matches_any_empty_patterns_is_false() -> None:
    assert _matches_any("foo.py", []) is False


# ---------------------------------------------------------------------------
# HookConfig validation
# ---------------------------------------------------------------------------


def test_hook_config_verify_requires_verify_command() -> None:
    with pytest.raises(ValueError, match="verify_command is required"):
        HookConfig(command="verify")


def test_hook_config_verify_accepts_verify_command() -> None:
    hook = HookConfig(command="verify", verify_command="pytest")
    assert hook.verify_command == "pytest"


def test_hook_config_clean_does_not_require_verify_command() -> None:
    hook = HookConfig(command="clean")
    assert hook.verify_command is None


def test_hook_config_debounce_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        HookConfig(command="clean", debounce_ms=-1)


# ---------------------------------------------------------------------------
# matching_hooks
# ---------------------------------------------------------------------------


def _config_with_hooks(
    project_root: Path, hooks: list[HookConfig], *, enabled: bool = True
) -> AppConfig:
    return AppConfig(
        project_root=project_root,
        hooks=HooksConfig(enabled=enabled, on_save=hooks),
    )


def test_matching_hooks_empty_when_disabled(project_root: Path) -> None:
    config = _config_with_hooks(project_root, [HookConfig(command="clean")], enabled=False)
    runner = OnSaveHookRunner(config)
    assert runner.matching_hooks(project_root / "a.py") == []


def test_matching_hooks_include_exclude(project_root: Path) -> None:
    (project_root / "src").mkdir()
    (project_root / "tests").mkdir()
    config = _config_with_hooks(
        project_root,
        [
            HookConfig(command="clean", include=["src/**/*.py"], exclude=["src/**/_*.py"]),
            HookConfig(command="clean", include=["tests/**/*.py"]),
        ],
    )
    runner = OnSaveHookRunner(config)
    assert [i for i, _ in runner.matching_hooks(project_root / "src" / "foo.py")] == [0]
    assert [i for i, _ in runner.matching_hooks(project_root / "src" / "_skip.py")] == []
    assert [i for i, _ in runner.matching_hooks(project_root / "tests" / "test_x.py")] == [1]
    assert runner.matching_hooks(project_root / "README.md") == []


def test_matching_hooks_default_include_matches_everything(project_root: Path) -> None:
    config = _config_with_hooks(project_root, [HookConfig(command="clean")])
    runner = OnSaveHookRunner(config)
    matches = runner.matching_hooks(project_root / "deep" / "nested" / "file.txt")
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------


def test_build_command_clean(project_root: Path) -> None:
    config = _config_with_hooks(project_root, [HookConfig(command="clean", args=["--check"])])
    runner = OnSaveHookRunner(config, python_executable="/usr/bin/python3")
    cmd = runner.build_command(config.hooks.on_save[0], project_root / "a.py")
    assert cmd[0] == "/usr/bin/python3"
    assert cmd[1:4] == ["-m", "autoship", "clean"]
    assert "--yes" in cmd
    assert str(project_root / "a.py") in cmd
    assert "--check" in cmd


def test_build_command_verify(project_root: Path) -> None:
    config = _config_with_hooks(
        project_root, [HookConfig(command="verify", verify_command="pytest")]
    )
    runner = OnSaveHookRunner(config, python_executable="python")
    cmd = runner.build_command(config.hooks.on_save[0], project_root / "a.py")
    assert cmd[1:4] == ["-m", "autoship", "verify"]
    assert "pytest" in cmd


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------


def _fake_clock() -> tuple[MagicMock, list[float]]:
    now = [0.0]

    def _t() -> float:
        return now[0]

    return _t, now


def test_debounce_suppresses_rapid_runs(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, now = _fake_clock()
    config = _config_with_hooks(project_root, [HookConfig(command="clean", debounce_ms=500)])
    runner = OnSaveHookRunner(config, clock=clock, python_executable="python")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    # First run sets the timestamp (real run_hook records _last_run).
    assert runner.run_for_path(project_root / "a.py")  # runs
    now[0] = 0.2  # 200ms < 500ms window
    assert runner.run_for_path(project_root / "a.py") == []  # debounced
    now[0] = 0.6  # 600ms > 500ms window
    assert runner.run_for_path(project_root / "a.py")  # runs again


def test_reset_clears_debounce(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock, now = _fake_clock()
    config = _config_with_hooks(project_root, [HookConfig(command="clean", debounce_ms=500)])
    runner = OnSaveHookRunner(config, clock=clock, python_executable="python")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    runner.run_for_path(project_root / "a.py")
    now[0] = 0.1
    runner.reset()
    assert runner.run_for_path(project_root / "a.py")  # runs after reset


# ---------------------------------------------------------------------------
# run_hook (subprocess)
# ---------------------------------------------------------------------------


def _make_runner(project_root: Path, hooks: list[HookConfig]) -> OnSaveHookRunner:
    config = _config_with_hooks(project_root, hooks)
    return OnSaveHookRunner(config, audit_logger=None, python_executable="python")


def test_run_hook_success(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _make_runner(project_root, [HookConfig(command="clean")])
    hook = runner.config.hooks.on_save[0]

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.run_hook(hook, project_root / "a.py")
    assert result.ok
    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    assert not result.timed_out


def test_run_hook_failure_recorded(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _make_runner(project_root, [HookConfig(command="clean")])
    hook = runner.config.hooks.on_save[0]

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.run_hook(hook, project_root / "a.py")
    assert not result.ok
    assert result.exit_code == 1
    assert result.stderr == "boom"


def test_run_hook_timeout(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _make_runner(project_root, [HookConfig(command="clean")])
    hook = runner.config.hooks.on_save[0]

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1.0, output=b"partial", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.run_hook(hook, project_root / "a.py")
    assert result.timed_out
    assert result.exit_code == 124
    assert result.stdout == "partial"


def test_run_hook_oserror(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _make_runner(project_root, [HookConfig(command="clean")])
    hook = runner.config.hooks.on_save[0]

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("no such executable")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.run_hook(hook, project_root / "a.py")
    assert result.exit_code == 127
    assert "no such executable" in result.stderr


def test_run_hook_records_audit(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit = MagicMock()
    config = _config_with_hooks(project_root, [HookConfig(command="clean")])
    runner = OnSaveHookRunner(config, audit_logger=audit, python_executable="python")
    hook = config.hooks.on_save[0]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    runner.run_hook(hook, project_root / "a.py")
    events = [call.args[0] for call in audit.record.call_args_list]
    assert "hook.run.start" in events
    assert "hook.run.done" in events


def test_run_hook_records_error_audit_on_failure(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = MagicMock()
    config = _config_with_hooks(project_root, [HookConfig(command="clean")])
    runner = OnSaveHookRunner(config, audit_logger=audit, python_executable="python")
    hook = config.hooks.on_save[0]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 2, stdout="", stderr=""),
    )
    runner.run_hook(hook, project_root / "a.py")
    events = [call.args[0] for call in audit.record.call_args_list]
    assert "hook.run.error" in events


def test_run_for_path_runs_only_matching(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_with_hooks(
        project_root,
        [
            HookConfig(command="clean", include=["src/**/*.py"]),
            HookConfig(command="clean", include=["tests/**/*.py"]),
        ],
    )
    runner = OnSaveHookRunner(config, audit_logger=None, python_executable="python")
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    (project_root / "tests").mkdir()
    results = runner.run_for_path(project_root / "tests" / "test_x.py")
    assert len(results) == 1
    assert len(seen) == 1
