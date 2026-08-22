"""``justagent session`` 命令：管理会话历史。

支持列出、查看、恢复、删除已保存的 agent 会话。会话存储在
``~/.justagent/sessions/`` 下（可通过 ``MYAGENT_SESSIONS_DIR`` 环境变量覆盖）。

参考 Cline 的 task history 与 OpenCode 的 session persistence。
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime

import typer

from justagent.agent.session import (
    SessionError,
    get_session_store,
)
from justagent.core.i18n import I18n, get_i18n_from_ctx

app = typer.Typer(help="Manage session history")


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="session")


def _format_ts(ts: float) -> str:
    """Format a unix timestamp as ``YYYY-MM-DD HH:MM``."""

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _short(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` chars, appending ``…`` if cut."""

    text = text.replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


@app.command("list", help="List all saved sessions.")
def list_sessions(ctx: typer.Context) -> None:
    """List all saved sessions (sorted by update time, descending)."""

    i18n: I18n = get_i18n_from_ctx(ctx)
    store = get_session_store()
    sessions = store.list_sessions()
    if not sessions:
        typer.echo(i18n._("session.empty"))
        return

    typer.echo(f"{'ID':<18} {'Date':<16} {'Mode':<6} {'Model':<18} {'Tokens':>8}  Preview")
    typer.echo("-" * 90)
    for m in sessions:
        typer.echo(
            f"{m.id:<18} {_format_ts(m.updated_at):<16} {m.mode:<6} "
            f"{_short(m.model, 18):<18} {m.total_tokens:>8}  "
            f"{_short(m.prompt_preview, 30)}"
        )


@app.command("show", help="Show details of a specific session.")
def show_session(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Session ID"),
) -> None:
    """Show metadata and message stats for a specific session."""

    i18n: I18n = get_i18n_from_ctx(ctx)
    store = get_session_store()
    try:
        session = store.load(session_id)
    except SessionError as exc:
        typer.secho(i18n._("error.prefix", exc=exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    m = session.metadata
    typer.echo(f"ID:           {m.id}")
    typer.echo(f"Created:      {_format_ts(m.created_at)}")
    typer.echo(f"Updated:      {_format_ts(m.updated_at)}")
    typer.echo(f"Status:       {m.status.value}")
    typer.echo(f"Mode:         {m.mode}")
    typer.echo(f"Model:        {m.model}")
    typer.echo(f"CWD:          {m.cwd}")
    typer.echo(f"Iterations:   {m.iterations}")
    typer.echo(f"Total tokens: {m.total_tokens}")
    typer.echo(f"Messages:     {m.message_count}")
    if m.files_changed:
        typer.echo("Files changed:")
        for path in m.files_changed:
            typer.echo(f"  - {path}")
    typer.echo(f"Preview:      {m.prompt_preview}")


@app.command("resume", help="Resume a specific session (starts interactive mode).")
def resume_session(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Session ID"),
) -> None:
    """Resume a saved session by launching the interactive agent.

    Validates that the session exists, then spawns
    ``justagent agent -i --resume <session_id>`` as a subprocess so the
    restored conversation continues in the interactive REPL. Relevant
    global options (``--config``, ``--lang``, ``--verbose``, ``--yes``)
    are forwarded from the current invocation so the resumed run matches
    the caller's context.
    """

    i18n: I18n = get_i18n_from_ctx(ctx)
    store = get_session_store()
    if not store.exists(session_id):
        typer.secho(
            i18n._("session.not_found", session_id=session_id),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Launch the interactive agent with the session restored. We use a
    # subprocess (rather than invoking the ``agent`` callback in-process)
    # because the interactive REPL needs direct access to the controlling
    # TTY for stdin/stdout, and the ``agent`` command depends on
    # ``ctx.obj`` state (config, audit logger, ...) that is set up by the
    # top-level callback — a fresh process reproduces that setup cleanly.
    cmd: list[str] = [
        sys.executable,
        "-m",
        "justagent",
        "agent",
        "-i",
        "--resume",
        session_id,
    ]

    # Forward the global options that affect the resumed run so it behaves
    # like the caller typed ``justagent agent -i --resume <ID>`` with the
    # same configuration context.
    obj = ctx.obj if isinstance(ctx.obj, dict) else {}
    config_path = obj.get("config_path")
    if config_path:
        cmd.extend(["--config", str(config_path)])
    lang = getattr(obj.get("i18n"), "lang", None)
    if lang:
        cmd.extend(["--lang", str(lang)])
    if obj.get("verbose"):
        cmd.append("--verbose")
    if obj.get("yes"):
        cmd.append("--yes")

    typer.echo(i18n._("session.resuming", session_id=session_id))
    try:
        result = subprocess.run(cmd)
    except OSError as exc:
        typer.secho(
            i18n._("error.prefix", exc=exc),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    raise typer.Exit(code=result.returncode)


@app.command("delete", help="Delete a specific session.")
def delete_session(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Session ID"),
) -> None:
    """Delete a saved session file."""

    i18n: I18n = get_i18n_from_ctx(ctx)
    store = get_session_store()
    if not store.delete(session_id):
        typer.secho(
            i18n._("session.not_found", session_id=session_id),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(i18n._("session.deleted", session_id=session_id))


__all__ = ["app", "register"]
