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

app = typer.Typer(help="管理会话历史")


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


@app.command("list", help="列出所有已保存的会话。")
def list_sessions() -> None:
    """列出所有已保存的会话（按更新时间倒序）。"""

    store = get_session_store()
    sessions = store.list_sessions()
    if not sessions:
        typer.echo("没有保存的会话。")
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


@app.command("show", help="显示指定会话的详细信息。")
def show_session(
    session_id: str = typer.Argument(..., help="会话 ID"),
) -> None:
    """显示指定会话的元数据与消息统计。"""

    store = get_session_store()
    try:
        session = store.load(session_id)
    except SessionError as exc:
        typer.secho(f"错误：{exc}", fg=typer.colors.RED, err=True)
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


@app.command("resume", help="恢复指定会话（启动交互模式）。")
def resume_session(
    session_id: str = typer.Argument(..., help="会话 ID"),
) -> None:
    """恢复一个已保存的会话。

    当前实现打印恢复指令；完整的内联恢复请使用：
    ``myagent agent -i --resume <ID>``。
    """

    store = get_session_store()
    if not store.exists(session_id):
        typer.secho(
            f"错误：找不到会话 {session_id}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    typer.echo(f"会话 {session_id} 已就绪。使用以下命令恢复：")
    typer.echo(f"  myagent agent -i --resume {session_id}")


@app.command("delete", help="删除指定会话。")
def delete_session(
    session_id: str = typer.Argument(..., help="会话 ID"),
) -> None:
    """删除一个已保存的会话文件。"""

    store = get_session_store()
    if not store.delete(session_id):
        typer.secho(
            f"错误：找不到会话 {session_id}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    typer.echo(f"已删除会话 {session_id}。")


__all__ = ["app", "register"]
