"""The ``autoship commit`` command."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import typer

from autoship.adapters.git_adapter import GitAdapter
from autoship.core.audit_logger import AuditLogger
from autoship.core.context import CommandContext
from autoship.core.model_router import ModelRouter
from autoship.exceptions import GitError, ModelGatewayError
from autoship.plugin_manager import manager as plugin_manager
from autoship.utils.hashing import ToolVerifier
from autoship.utils.shell_safety import contains_shell_metacharacters

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.command(name="commit")(commit)


@app.command()
def commit(
    ctx: typer.Context,
    message: str | None = typer.Option(None, "--message", "-m", help="Use given commit message"),
    edit: bool = typer.Option(True, "--edit/--no-edit", help="Open editor to refine message"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmations"),
) -> None:
    """Generate a commit message and commit staged/unstaged changes."""
    # RBAC gate: commit writes to git history.

    config = ctx.obj["config"]

    audit: AuditLogger = ctx.obj["audit_logger"]
    dry_run: bool = ctx.obj.get("dry_run", False)
    yes = yes or ctx.obj.get("yes", False)
    verbose: bool = ctx.obj.get("verbose", False)

    git = GitAdapter(config.project_root, tool_verifier=ToolVerifier(config.tools))

    if not git.is_git_repo():
        typer.echo("commit.not_git_repo")
        raise typer.Exit(code=1)

    if not git.has_changes():
        typer.echo("commit.nothing")
        return

    context = CommandContext(
        command="commit",
        project_root=config.project_root,
        config=config,
        dry_run=dry_run,
        yes=yes,
        trace_id=audit.trace_id,
    )

    audit.record("commit.start")

    diff = git.diff()
    stats = git.stats()

    final_message = message
    if final_message is None:
        with ModelRouter(config) as router:
            try:
                final_message = router.generate_commit_message(diff=diff, stats=stats)
            except ModelGatewayError:
                if verbose:
                    typer.echo("commit.model_failed", err=True)
                final_message = "Update files"

    if edit and not yes:
        final_message = _open_editor(None, final_message, config.commit.allowed_editors)

    # Make the final message available to pre-commit hooks (e.g. commit-policy
    # plugins) and allow them to abort the commit.
    context.extras["message"] = final_message
    try:
        plugin_manager.call("pre_commit", context=context, fail_fast=True)
    except Exception as exc:
        audit.record("commit.aborted", {"message": final_message, "reason": str(exc)})
        raise GitError(
            "commit.pre_commit_failed",
            details={"message": final_message},
        ) from exc

    if dry_run:
        typer.echo("commit.dry_run")
        audit.record("commit.dry_run", {"message": final_message})
        return

    git.commit(final_message)

    audit.record("commit.done", {"message": final_message})
    plugin_manager.call("post_commit", context=context, fail_fast=False)
    typer.echo("commit.done")


def _validate_editor(editor: str, allowed_editors: list[str], i18n) -> str:
    """Validate ``editor`` against the configured allowlist.

    Returns the executable path/token on success, or raises ``GitError`` if the
    value contains shell metacharacters, path traversal, or an unknown editor.
    Only the first token of ``editor`` is returned; extra arguments are ignored
    to prevent command injection.
    """
    try:
        cmd_parts = shlex.split(editor)
    except ValueError as exc:
        raise GitError(
            "commit.editor_disallowed",
            details={"editor": editor, "reason": str(exc)},
        ) from exc

    if not cmd_parts:
        raise GitError(
            "commit.editor_disallowed",
            details={"editor": editor, "reason": "empty_command"},
        )

    if contains_shell_metacharacters(editor):
        raise GitError(
            "commit.editor_disallowed",
            details={"editor": editor, "reason": "shell_metacharacters"},
        )

    executable = cmd_parts[0]
    if ".." in Path(executable).parts:
        raise GitError(
            "commit.editor_disallowed",
            details={"editor": editor, "reason": "path_traversal"},
        )

    executable_name = Path(executable).name
    if executable_name not in allowed_editors:
        raise GitError(
            "commit.editor_unknown",
            details={"editor": editor, "executable": executable_name, "allowed": allowed_editors},
        )

    return executable


def _open_editor(i18n, initial: str, allowed_editors: list[str]) -> str:
    """Open the user's preferred editor to review/modify a commit message."""
    editor = os.environ.get("EDITOR", "vim")
    executable = _validate_editor(editor, allowed_editors, i18n)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
        f.write(initial)
        f.flush()
        path = Path(f.name)
    try:
        subprocess.run([executable, str(path)], check=True)
        return path.read_text(encoding="utf-8").strip()
    except subprocess.CalledProcessError as exc:
        raise GitError("commit.editor_error") from exc
    finally:
        path.unlink(missing_ok=True)
