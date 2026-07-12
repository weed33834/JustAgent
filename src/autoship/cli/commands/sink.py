"""The ``autoship sink`` command: a thin CLI front-end for the self-hosted
audit/telemetry sink.

The receiving server itself lives in :class:`the sink server (removed)`;
this module exposes ``serve`` (run the sink HTTP server) and ``status``
(query a running sink's ``/status`` endpoint for aggregated counters).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from autoship.models.config import AppConfig

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="sink")


def _config_from_ctx(ctx: typer.Context) -> AppConfig:
    config = ctx.obj.get("config") if ctx.obj else None
    if not isinstance(config, AppConfig):
        typer.secho(
            "AutoShip config not loaded. Run from a project with .autoship.toml.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from None
    return config


def _i18n(ctx: typer.Context):
    return None


def _resolve_store_dir(config: AppConfig) -> Path:
    if config.sink.server_store_dir is not None:
        return Path(config.sink.server_store_dir)
    return Path.home() / ".autoship" / "sink"


@app.command("serve")
def serve(
    ctx: typer.Context,
    bind: str | None = typer.Option(
        None, "--bind", help="Bind address (default: config sink.server_bind)."
    ),
    port: int | None = typer.Option(
        None, "--port", help="Listen port (default: config sink.server_port)."
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Shared bearer token clients must send (default: config sink.server_token).",
    ),
    store_dir: Path | None = typer.Option(
        None,
        "--store-dir",
        help="Directory to store aggregated records (default: ~/.autoship/sink).",
    ),
) -> None:
    """Run the self-hosted sink HTTP server (Ctrl+C to stop)."""
    # RBAC gate: serve exposes an HTTP endpoint receiving audit/telemetry.

    config = _config_from_ctx(ctx)
    _i18n_local = _i18n(ctx)

    effective_bind = bind or config.sink.server_bind
    effective_port = port or config.sink.server_port
    effective_token = token or config.sink.server_token
    effective_store = store_dir or _resolve_store_dir(config)

    # Loopback is safe without a token. Anything else MUST have a token.
    if effective_bind not in {"127.0.0.1", "::1", "localhost"} and not effective_token:
        typer.secho(
            "sink.token_required",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    server = SinkServer(  # noqa: F821
        effective_store,
        bind=effective_bind,
        port=effective_port,
        token=effective_token,
        retention_days=config.sink.retention_days,
    )
    typer.echo("sink.serving")
    if effective_token:
        typer.echo("sink.token_set")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("sink.stopped")
    finally:
        server.shutdown()
        server.server_close()


@app.command("status")
def status(
    ctx: typer.Context,
    url: str | None = typer.Option(None, "--url", help="Sink base URL (default: config sink.url)."),
    token: str | None = typer.Option(
        None, "--token", help="Bearer token (default: config sink.token)."
    ),
) -> None:
    """Query a running sink's /status endpoint and print its counters."""
    config = _config_from_ctx(ctx)
    _i18n_local = _i18n(ctx)

    base = (url or (str(config.sink.url) if config.sink.url else None) or "").rstrip("/")
    if not base:
        typer.echo("sink.no_url", err=True)
        raise typer.Exit(code=2)
    effective_token = token or config.sink.token

    import httpx

    headers: dict[str, str] = {}
    if effective_token:
        headers["Authorization"] = f"Bearer {effective_token}"
    try:
        resp = httpx.get(f"{base}/status", headers=headers, timeout=5.0)
    except httpx.HTTPError:
        typer.echo("sink.unreachable", err=True)
        raise typer.Exit(code=1) from None
    if resp.status_code != 200:
        typer.echo("sink.status_failed", err=True)
        raise typer.Exit(code=1)
    try:
        data: dict[str, Any] = resp.json()
    except json.JSONDecodeError:
        typer.echo("sink.status_invalid", err=True)
        raise typer.Exit(code=1) from None

    typer.echo("sink.status_header")
    typer.echo(f"  status:            {data.get('status', '?')}")
    typer.echo(f"  audit_records:     {data.get('audit_records', 0)}")
    typer.echo(f"  telemetry_records: {data.get('telemetry_records', 0)}")
    typer.echo(f"  latest_ts:         {data.get('latest_ts', '-')}")
    typer.echo(f"  started_at:        {data.get('started_at', '-')}")
    typer.echo(f"  retention_days:    {data.get('retention_days', '-')}")
    typer.echo(f"  store_dir:         {data.get('store_dir', '-')}")
