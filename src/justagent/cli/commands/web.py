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
    no_auth: bool = typer.Option(
        False, "--no-auth", help="Disable authentication (local development only)"
    ),
) -> None:
    """Start a browser chat interface for JustAgent.

    Vertical tools contributed by installed vertical packages and their dashboards are served
    at http://<host>:<port>/. Judicial features work without an LLM; chatting
    requires a configured model backend.

    Authentication is enforced by default: log in with the admin account
    (password from JUSTAGENT_WEB_ADMIN_PASSWORD on first start, printed once
    otherwise), or set JUSTAGENT_WEB_TOKEN for shared-token access.
    """
    config: AppConfig = ctx.obj["config"]
    try:
        from justagent.web.app import run
    except ImportError as exc:
        raise typer.BadParameter(f"Web support requires fastapi+uvicorn: {exc}") from exc

    if no_auth:
        typer.secho(
            "警告：鉴权已禁用（--no-auth）。仅限本机开发调试，切勿暴露到网络。",
            fg=typer.colors.RED,
        )
    typer.secho(
        f"JustAgent Web 已启动：http://{host}:{port}/  (Ctrl+C 退出)",
        fg=typer.colors.GREEN,
    )
    run(config, host=host, port=port, no_auth=no_auth)
