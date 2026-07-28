"""The ``justagent commit`` command."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import typer

from justagent.adapters.git_adapter import GitAdapter
from justagent.core.audit_logger import AuditLogger
from justagent.core.context import CommandContext
from justagent.core.i18n import I18n, get_i18n_from_ctx
from justagent.core.model_router import ModelRouter
from justagent.exceptions import GitError, ModelGatewayError
from justagent.plugin_manager import manager as plugin_manager
from justagent.utils.hashing import ToolVerifier
from justagent.utils.shell_safety import contains_shell_metacharacters

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
    i18n = get_i18n_from_ctx(ctx)

    audit: AuditLogger = ctx.obj["audit_logger"]
    dry_run: bool = ctx.obj.get("dry_run", False)
    yes = yes or ctx.obj.get("yes", False)
    verbose: bool = ctx.obj.get("verbose", False)

    git = GitAdapter(config.project_root, tool_verifier=ToolVerifier(config.tools))

    if not git.is_git_repo():
        typer.echo(i18n._("commit.not_git_repo"))
        raise typer.Exit(code=1)

    if not git.has_changes():
        typer.echo(i18n._("commit.nothing"))
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
            except ModelGatewayError as exc:
                if verbose:
                    typer.echo(i18n._("commit.model_failed", exc=exc), err=True)
                final_message = "Update files"

    if edit and not yes:
        final_message = _open_editor(i18n, final_message, config.commit.allowed_editors)

    # Make the final message available to pre-commit hooks (e.g. commit-policy
    # plugins) and allow them to abort the commit.
    context.extras["message"] = final_message
    try:
        plugin_manager.call("pre_commit", context=context, fail_fast=True)
    except Exception as exc:
        audit.record("commit.aborted", {"message": final_message, "reason": str(exc)})
        raise GitError(
            i18n._("commit.pre_commit_failed", exc=exc),
            details={"message": final_message},
        ) from exc

    if dry_run:
        typer.echo(i18n._("commit.dry_run", message=final_message))
        audit.record("commit.dry_run", {"message": final_message})
        return

    git.commit(final_message)

    audit.record("commit.done", {"message": final_message})
    plugin_manager.call("post_commit", context=context, fail_fast=False)
    typer.echo(i18n._("commit.done", message=final_message))


def _validate_editor(editor: str, allowed_editors: list[str], i18n: I18n) -> str:
    """校验 ``editor`` 是否在白名单内。

    成功返回可执行路径；若含 shell 元字符、路径穿越或未知编辑器则抛 ``GitError``。
    仅取 ``editor`` 的第一个 token，忽略其余参数以防止命令注入。
    """
    try:
        cmd_parts = shlex.split(editor)
    except ValueError as exc:
        raise GitError(
            i18n._("commit.editor_disallowed", editor=editor),
            details={"editor": editor, "reason": str(exc)},
        ) from exc

    if not cmd_parts:
        raise GitError(
            i18n._("commit.editor_disallowed", editor=editor),
            details={"editor": editor, "reason": "empty_command"},
        )

    if contains_shell_metacharacters(editor):
        raise GitError(
            i18n._("commit.editor_disallowed", editor=editor),
            details={"editor": editor, "reason": "shell_metacharacters"},
        )

    executable = cmd_parts[0]
    if ".." in Path(executable).parts:
        raise GitError(
            i18n._("commit.editor_disallowed", editor=editor),
            details={"editor": editor, "reason": "path_traversal"},
        )

    executable_name = Path(executable).name
    if executable_name not in allowed_editors:
        raise GitError(
            i18n._(
                "commit.editor_unknown",
                editor=executable_name,
                allowed=", ".join(allowed_editors),
            ),
            details={"editor": editor, "executable": executable_name, "allowed": allowed_editors},
        )

    return executable


def _open_editor(i18n: I18n, initial: str, allowed_editors: list[str]) -> str:
    """打开用户偏好编辑器以审阅/修改提交信息。"""
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
        raise GitError(i18n._("commit.editor_error", code=exc.returncode)) from exc
    finally:
        path.unlink(missing_ok=True)
