"""The ``justagent web`` command — start the local Web chat interface."""

from __future__ import annotations

import typer

from justagent.models.config import AppConfig

app = typer.Typer(help="Start the local Web chat interface")


def register(parent: typer.Typer) -> None:
    parent.command(name="web", help="Start the local Web chat interface")(web)


def web(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Listen port"),
) -> None:
    """Start a browser chat interface for JustAgent.

    The agent (with the judicial tool) and the judicial dashboard are served
    at http://<host>:<port>/. Judicial features work without an LLM; chatting
    requires a configured model backend.
    """
    config: AppConfig = ctx.obj["config"]
    try:
        from justagent.web.app import run
    except ImportError as exc:
        raise typer.BadParameter(
            f"Web support requires fastapi+uvicorn: {exc}"
        ) from exc

    typer.secho(
        f"JustAgent Web 已启动：http://{host}:{port}/  (Ctrl+C 退出)",
        fg=typer.colors.GREEN,
    )
    run(config, host=host, port=port)
