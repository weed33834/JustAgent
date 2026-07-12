"""Tests for the verify command."""

from __future__ import annotations

import logging
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoship.cli.commands import verify
from autoship.core.context import CommandContext
from autoship.core.fix import FixSuggestion
from autoship.exceptions import VerifyError
from autoship.hookspec import hookimpl
from autoship.models.config import AppConfig
from autoship.plugin_manager import manager as plugin_manager


class _FixPlugin:
    suggestion: FixSuggestion | None = None

    @hookimpl
    def on_error(self, context, error):
        return self.suggestion


@pytest.fixture
def fix_plugin():
    """Register a test plugin that returns a configurable FixSuggestion."""
    plugin = _FixPlugin()
    plugin_manager.pm.register(plugin)
    try:
        yield plugin
    finally:
        plugin_manager.pm.unregister(plugin)


def _make_context(
    app_config: AppConfig, *, fix: bool = False, dry_run: bool = False, yes: bool = False
) -> MagicMock:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": dry_run,
        "yes": yes,
        "verbose": False,
    }
    return ctx


def test_verify_success(project_root, app_config: AppConfig) -> None:
    app_config.verify.allowed_commands = ["echo"]
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "OK"
        mock_run.return_value.stderr = ""
        verify.verify(ctx, command="echo ok")


def test_verify_failure(project_root, app_config: AppConfig) -> None:
    app_config.verify.allowed_commands = ["false"]
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "FAILED"
        with pytest.raises(VerifyError, match="exit code 1") as exc:
            verify.verify(ctx, command="false")
    assert "false" in str(exc.value)


def test_verify_os_error(project_root, app_config: AppConfig) -> None:
    app_config.verify.allowed_commands = ["false"]
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with (
        patch("subprocess.run", side_effect=OSError("permission denied")),
        pytest.raises(VerifyError, match="permission denied"),
    ):
        verify.verify(ctx, command="false")


def test_verify_dry_run(project_root, app_config: AppConfig) -> None:
    app_config.verify.allowed_commands = ["echo"]
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": True,
        "yes": True,
        "verbose": False,
    }
    with pytest.raises(verify.typer.Exit) as exc:
        verify.verify(ctx, command="echo ok")
    assert exc.value.exit_code == 0


def test_verify_on_error_suggestion_no_patch(fix_plugin, app_config: AppConfig) -> None:
    app_config.verify.allowed_commands = ["false"]
    fix_plugin.suggestion = FixSuggestion(description="Try increasing the timeout.")
    ctx = _make_context(app_config, fix=True, yes=True)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "FAILED"
        with pytest.raises(VerifyError):
            verify.verify(ctx, command="false")

    audit = ctx.obj["audit_logger"]
    audit.record.assert_any_call(
        "verify.fix.suggestion", {"description": "Try increasing the timeout."}
    )


def test_verify_on_error_patch_auto_apply(fix_plugin, app_config: AppConfig) -> None:
    app_config.verify.allowed_commands = ["false"]
    fix_plugin.suggestion = FixSuggestion(
        description="Add a newline.",
        patch="diff content",
    )
    ctx = _make_context(app_config, fix=True, yes=True)

    with (
        patch("subprocess.run") as mock_run,
        patch.object(verify, "_apply_patch", return_value=(True, None)),
    ):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "FAILED"
        with pytest.raises(VerifyError):
            verify.verify(ctx, command="false")

    audit = ctx.obj["audit_logger"]
    audit.record.assert_any_call(
        "verify.fix.applied",
        {"description": "Add a newline.", "patch": "diff content"},
    )


def test_verify_on_error_patch_declined(fix_plugin, app_config: AppConfig) -> None:
    app_config.verify.allowed_commands = ["false"]
    fix_plugin.suggestion = FixSuggestion(description="Add a newline.", patch="diff content")
    ctx = _make_context(app_config, fix=True, yes=False)

    with (
        patch("subprocess.run") as mock_run,
        patch("autoship.cli.commands.verify.typer.confirm", return_value=False),
    ):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "FAILED"
        with pytest.raises(VerifyError):
            verify.verify(ctx, command="false")

    audit = ctx.obj["audit_logger"]
    audit.record.assert_any_call("verify.fix.declined", {"description": "Add a newline."})


def test_verify_rejects_shell_metacharacters(app_config: AppConfig, i18n) -> None:
    with pytest.raises(VerifyError, match="not allowed"):
        verify._validate_verify_command("pytest; rm -rf /", app_config.verify, i18n)


def test_verify_accepts_command_with_parentheses(app_config: AppConfig, i18n) -> None:
    cmd_parts = verify._validate_verify_command(
        "python -c \"print('ok')\"", app_config.verify, i18n
    )
    assert cmd_parts == ["python", "-c", "print('ok')"]


def test_verify_rejects_disallowed_executable(app_config: AppConfig, i18n) -> None:
    with pytest.raises(VerifyError, match="not allowed"):
        verify._validate_verify_command("curl https://example.com", app_config.verify, i18n)


def test_verify_accepts_allowed_command(app_config: AppConfig, i18n) -> None:
    cmd_parts = verify._validate_verify_command("pytest -x", app_config.verify, i18n)
    assert cmd_parts == ["pytest", "-x"]


def test_present_suggestion_dry_run(app_config: AppConfig, i18n) -> None:
    ctx = CommandContext(
        command="verify",
        project_root=app_config.project_root,
        config=app_config,
        dry_run=True,
        yes=True,
    )
    audit = MagicMock()
    suggestion = FixSuggestion(description="Add a newline.", patch="diff content")

    verify._present_suggestion(ctx, suggestion, 1, audit, i18n)

    audit.record.assert_any_call(
        "verify.fix.dry_run",
        {"description": "Add a newline.", "patch": "diff content"},
    )


def test_apply_patch_git_success(app_config: AppConfig) -> None:
    patch_text = "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"

    def _run(cmd, **kwargs):
        result = MagicMock()
        if (cmd[:2] == ["git", "apply"] and "--check" in cmd) or cmd == ["git", "apply"]:
            result.returncode = 0
        else:
            result.returncode = 1
        result.stderr = ""
        return result

    with (
        patch(
            "shutil.which",
            side_effect=lambda name: "/usr/bin/git" if name == "git" else None,
        ),
        patch("subprocess.run", side_effect=_run),
    ):
        applied, reason = verify._apply_patch(app_config.project_root, patch_text)

    assert applied is True
    assert reason is None


def test_apply_patch_no_tools(app_config: AppConfig) -> None:
    with patch("shutil.which", return_value=None):
        applied, reason = verify._apply_patch(app_config.project_root, "diff")

    assert applied is False
    assert "Neither git nor patch" in reason


def test_write_error_log_redacts_secrets(monkeypatch, tmp_path: Path) -> None:
    """Secrets embedded in stdout/stderr must be masked before persistence."""
    error_dir = tmp_path / "autoship"
    error_file = error_dir / "last_error.txt"
    monkeypatch.setattr(verify, "ERROR_LOG_DIR", error_dir)
    monkeypatch.setattr(verify, "ERROR_LOG_PATH", error_file)

    token = "ghp_" + "a" * 36
    stdout = "cloning into repo"
    stderr = f"fatal: Authentication failed for https://user:{token}@github.com"

    verify._write_error_log(stdout, stderr)

    content = error_file.read_text(encoding="utf-8")
    assert token not in content
    assert "***" in content
    assert stdout in content


def test_write_error_log_sets_permissions(monkeypatch, tmp_path: Path) -> None:
    """The error log directory and file must be readable only by the owner."""
    error_dir = tmp_path / "autoship"
    error_file = error_dir / "last_error.txt"
    monkeypatch.setattr(verify, "ERROR_LOG_DIR", error_dir)
    monkeypatch.setattr(verify, "ERROR_LOG_PATH", error_file)

    verify._write_error_log("out", "err")

    assert error_dir.exists()
    assert error_file.exists()
    assert stat.S_IMODE(error_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(error_file.stat().st_mode) == 0o600


def test_write_error_log_warns_when_permissions_are_too_broad(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """A warning is emitted if an existing log path has overly broad permissions."""
    error_dir = tmp_path / "autoship"
    error_file = error_dir / "last_error.txt"
    monkeypatch.setattr(verify, "ERROR_LOG_DIR", error_dir)
    monkeypatch.setattr(verify, "ERROR_LOG_PATH", error_file)

    error_dir.mkdir(parents=True)
    error_dir.chmod(0o755)
    error_file.write_text("old content", encoding="utf-8")
    error_file.chmod(0o644)

    with caplog.at_level(logging.WARNING, logger="autoship"):
        verify._write_error_log("out", "err")

    assert error_file.exists()
    assert stat.S_IMODE(error_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(error_file.stat().st_mode) == 0o600
    assert "too broad" in caplog.text
