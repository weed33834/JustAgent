"""``myagent session`` 命令：管理会话历史。

支持列出、查看、恢复、删除已保存的 agent 会话。会话存储在
``~/.myagent/sessions/`` 下（可通过 ``MYAGENT_SESSIONS_DIR`` 环境变量覆盖）。

参考 Cline 的 task history 与 OpenCode 的 session persistence。
"""

from __future__ import annotations

from datetime import datetime

import typer

from myagent.agent.session import (
    SessionError,
    get_session_store,
)
from myagent.core.i18n import I18n, get_i18n_from_ctx

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

    typer.echo(
        f"{'ID':<18} {'Date':<16} {'Mode':<6} {'Model':<18} "
        f"{'Tokens':>8}  Preview"
    )
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
    """Resume a saved session.

    The current implementation prints resume instructions; for full
    inline resume use: ``myagent agent -i --resume <ID>``.
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
    typer.echo(i18n._("session.resume_ready", session_id=session_id))
    typer.echo(f"  myagent agent -i --resume {session_id}")


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
