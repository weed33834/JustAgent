"""The ``justagent verify`` command."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import structlog
import typer

from justagent.core.audit_logger import AuditLogger, redact_text
from justagent.core.context import CommandContext
from justagent.core.i18n import I18n, get_i18n, get_i18n_from_ctx
from justagent.exceptions import VerifyError
from justagent.models.config import ToolsConfig, VerifyConfig
from justagent.plugin_manager import manager as plugin_manager
from justagent.plugins.defaults import FixSuggestion
from justagent.utils.hashing import ToolVerifier
from justagent.utils.patch import apply_patch, patch_paths_are_safe
from justagent.utils.permissions import ensure_dir_permissions, ensure_file_permissions
from justagent.utils.shell_safety import contains_shell_metacharacters

ERROR_LOG_DIR = Path.home() / ".local" / "state" / "justagent"
ERROR_LOG_PATH = ERROR_LOG_DIR / "last_error.txt"

logger = structlog.get_logger("justagent")


def _write_error_log(stdout: str, stderr: str) -> None:
    """Persist redacted verify output with restrictive permissions.

    stdout/stderr are redacted using the same logic as the audit logger before
    being written. The containing directory is created with ``0o700`` and the
    log file with ``0o600``. Existing paths with overly broad permissions are
    tightened and a warning is emitted.
    """
    try:
        ensure_dir_permissions(ERROR_LOG_DIR, 0o700)
        content = f"STDOUT:\n{redact_text(stdout)}\n\nSTDERR:\n{redact_text(stderr)}"
        ERROR_LOG_PATH.write_text(content, encoding="utf-8")
        ensure_file_permissions(ERROR_LOG_PATH, 0o600)
    except OSError:
        pass


def validate_verify_command(
    command: str, verify_config: VerifyConfig, i18n: I18n | None = None
) -> list[str]:
    """校验 ``command`` 是否在配置的白名单内。

    成功返回拆分后的命令列表；若含 shell 元字符或可执行文件不在白名单
    则抛 ``VerifyError``。

    公共接口：``justagent lsp`` 导入此函数在拉起 verify 前校验，
    确保 LSP 无法绕过白名单策略。

    ``i18n`` is accepted for call-site compatibility with tests that pass an
    :class:`~justagent.core.i18n.I18n` instance; when ``None`` the env-detected
    default (:func:`get_i18n`) is used so the localised message is always
    available.
    """
    i18n = i18n or get_i18n()
    try:
        cmd_parts = shlex.split(command)
    except ValueError as exc:
        raise VerifyError(
            i18n._("verify.command_disallowed", command=command),
            details={"command": command, "reason": str(exc)},
        ) from exc

    if not cmd_parts:
        raise VerifyError(
            i18n._("verify.command_disallowed", command=command),
            details={"command": command, "reason": "empty_command"},
        )

    if contains_shell_metacharacters(command):
        raise VerifyError(
            i18n._("verify.command_disallowed", command=command),
            details={"command": command, "reason": "shell_metacharacters"},
        )

    executable_name = Path(cmd_parts[0]).name
    allowed = verify_config.allowed_commands
    if executable_name not in allowed:
        raise VerifyError(
            i18n._("verify.command_disallowed", command=command),
            details={"command": command, "executable": executable_name, "allowed": allowed},
        )

    return cmd_parts


# Backward-compat alias: tests and downstream code reference the private
# ``verify._validate_verify_command``. The function is now part of the
# public surface so ``justagent lsp`` can import it without a
# ``pyright: ignore[reportPrivateUsage]`` escape hatch; this alias keeps
# the old call sites working.
_validate_verify_command = validate_verify_command

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.command(name="verify")(verify)


@app.command(name="verify")
def verify(
    ctx: typer.Context,
    command: str = typer.Argument(..., help="Command to run for verification, e.g. `pytest`"),
    fix: bool = typer.Option(False, "--fix", help="Ask the model to suggest fixes on failure"),
    timeout: int | None = typer.Option(
        None,
        "--timeout",
        help="Kill the verification command after this many seconds (default: no timeout).",
    ),
) -> None:
    """Run a verification command and capture errors for AI-assisted fixing."""
    # RBAC gate: verify executes arbitrary allowlisted commands.

    config = ctx.obj["config"]
    i18n = get_i18n_from_ctx(ctx)

    audit: AuditLogger = ctx.obj["audit_logger"]
    dry_run: bool = ctx.obj.get("dry_run", False)
    yes: bool = ctx.obj.get("yes", False)
    verbose: bool = ctx.obj.get("verbose", False)

    # Normalise ``timeout``: when ``verify`` is called directly (e.g. from
    # tests) the default is Typer's ``OptionInfo`` sentinel rather than
    # ``None``. Treat OptionInfo and non-positive ints as "no timeout".
    timeout_seconds: int | None = timeout if isinstance(timeout, int) and timeout > 0 else None

    context = CommandContext(
        command="verify",
        project_root=config.project_root,
        config=config,
        dry_run=dry_run,
        yes=yes,
        trace_id=audit.trace_id,
        extras={"verify_command": command, "fix": fix, "timeout": timeout_seconds},
    )

    audit.record("verify.start", {"command": command, "fix": fix, "timeout": timeout_seconds})
    plugin_manager.call("pre_verify", context=context, fail_fast=False)

    if dry_run:
        typer.echo(i18n._("verify.dry_run", command=command))
        audit.record("verify.dry_run", {"command": command})
        plugin_manager.call("post_verify", context=context, fail_fast=False)
        raise typer.Exit(code=0)

    cmd_parts = validate_verify_command(command, config.verify, i18n)
    executable = shutil.which(cmd_parts[0])
    if executable is None:
        error = VerifyError(
            i18n._("verify.command_not_found", cmd=cmd_parts[0]),
            details={"command": command},
        )
        audit.record("verify.error", {"command": command, "error": str(error)})
        raise error

    try:
        result = subprocess.run(
            cmd_parts,
            cwd=config.project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        audit.record(
            "verify.error",
            {
                "command": command,
                "error": f"timed out after {timeout_seconds}s",
                "timeout": timeout_seconds,
            },
        )
        error = VerifyError(
            i18n._("verify.timed_out", command=command, timeout=timeout_seconds),
            details={"command": command, "timeout": timeout_seconds},
        )
        _handle_error(context, error, audit)
        raise error from exc
    except (FileNotFoundError, OSError) as exc:
        audit.record("verify.error", {"command": command, "error": str(exc)})
        _handle_error(context, exc, audit)
        raise VerifyError(i18n._("verify.run_failed", exc=exc)) from exc

    stdout_redacted = redact_text(result.stdout)
    stderr_redacted = redact_text(result.stderr)

    if verbose:
        typer.echo(stdout_redacted)
    if result.stderr:
        typer.secho(stderr_redacted, fg=typer.colors.YELLOW, err=True)

    if result.returncode != 0:
        _write_error_log(result.stdout, result.stderr)
        audit.record(
            "verify.failure",
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": stdout_redacted,
                "stderr": stderr_redacted,
            },
        )
        error = VerifyError(
            i18n._("verify.failed", code=result.returncode, command=command),
            details={
                "command": command,
                "stdout": stdout_redacted,
                "stderr": stderr_redacted,
            },
        )
        _handle_error(context, error, audit)
        raise error

    audit.record("verify.done", {"command": command})
    plugin_manager.call("post_verify", context=context, fail_fast=False)
    typer.echo(i18n._("verify.verified", command=command))


def _handle_error(
    context: CommandContext, error: Exception, audit: AuditLogger
) -> None:
    """调用 ``on_error`` hooks 并可选地应用修复补丁。"""
    hook_results = plugin_manager.call("on_error", context=context, error=error, fail_fast=False)

    if not context.extras.get("fix"):
        return

    suggestions: list[FixSuggestion] = [
        suggestion for suggestion in hook_results if isinstance(suggestion, FixSuggestion)
    ]

    for index, suggestion in enumerate(suggestions, start=1):
        _present_suggestion(context, suggestion, index, audit)


def _present_suggestion(
    context: CommandContext,
    suggestion: FixSuggestion,
    index: int,
    audit: AuditLogger,
) -> None:
    """展示修复建议，并在用户确认后应用其补丁。"""
    i18n = get_i18n()
    typer.secho("\n" + i18n._("verify.suggested_fix", index=index), fg=typer.colors.CYAN)
    typer.echo(suggestion.description)

    if not suggestion.patch:
        audit.record("verify.fix.suggestion", {"description": suggestion.description})
        return

    typer.secho("\n" + i18n._("verify.proposed_patch"), fg=typer.colors.CYAN)
    typer.echo(suggestion.patch)

    if context.dry_run:
        audit.record(
            "verify.fix.dry_run",
            {"description": suggestion.description, "patch": suggestion.patch},
        )
        typer.echo(i18n._("verify.patch_dry_run"))
        return

    if not context.yes and not typer.confirm(i18n._("verify.apply_patch")):
        audit.record(
            "verify.fix.declined",
            {"description": suggestion.description},
        )
        typer.echo(i18n._("verify.patch_not_applied"))
        return

    applied, reason = _apply_patch(context.project_root, suggestion.patch, context.config.tools)
    if applied:
        audit.record(
            "verify.fix.applied",
            {"description": suggestion.description, "patch": suggestion.patch},
        )
        typer.echo(i18n._("verify.patch_applied"))
    else:
        audit.record(
            "verify.fix.failed",
            {"description": suggestion.description, "patch": suggestion.patch, "reason": reason},
        )
        typer.secho(i18n._("verify.patch_failed", reason=reason), fg=typer.colors.YELLOW, err=True)


def _apply_patch(
    project_root: Path,
    patch: str,
    tools: ToolsConfig | None = None,
) -> tuple[bool, str | None]:
    """Apply a unified diff patch to the project.

    Delegates the git/patch apply mechanics to
    :func:`justagent.utils.patch.apply_patch` so the logic is shared with the
    ``fix`` command. Returns a tuple of ``(success, reason)`` so callers can
    explain failures.

    When ``tools`` is provided the pinned binary paths / hashes from
    ``config.tools`` are honoured via :class:`ToolVerifier`.

    Patches are first run through :func:`patch_paths_are_safe` — the same
    traversal / test-file guard the ``fix`` command uses — so a malicious or
    buggy plugin cannot use ``verify --fix`` to write outside the project
    root or to mutate tests.
    """
    if not patch_paths_are_safe(project_root, patch):
        return False, "patch contains unsafe paths (traversal or test file)"
    verifier = ToolVerifier(tools) if tools else ToolVerifier()
    return apply_patch(project_root, patch, verifier)
