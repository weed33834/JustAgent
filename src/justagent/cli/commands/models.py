"""The ``justagent models`` command — inspect configured model backends.

Lists every configured model backend (provider, model, tier, endpoint) and,
optionally, runs a live connectivity health check against each. Useful after
adding/changing backends and for verifying a provider is reachable without
starting a full agent run.
"""

from __future__ import annotations

import typer

from justagent.adapters.providers.unified_gateway import UnifiedGateway
from justagent.models.config import AppConfig

app = typer.Typer(help="Inspect configured model backends")


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="models", help="Inspect configured model backends")


@app.command("list", help="List model backends; with --check run a live health check")
def list_backends(
    ctx: typer.Context,
    check: bool = typer.Option(False, "--check", "-c", help="Run a live connectivity health check"),
) -> None:
    """Print a table of configured model backends."""
    config: AppConfig = ctx.obj["config"]
    backends = config.model.backends

    if not backends:
        typer.secho(
            "No model backends configured. Add them in your config or use `justagent config`.",
            fg=typer.colors.YELLOW,
        )
        return

    typer.echo(f"{'PROVIDER':<14}{'MODEL':<22}{'TIER':<6}{'API_KEY':<9}{'HEALTH':<8}ENDPOINT")
    typer.echo("-" * 110)
    ok = 0
    for backend in backends:
        provider = backend.provider.value
        model = backend.model or "-"
        tier = str(backend.tier)
        key = "set" if backend.api_key else "none"
        endpoint = str(backend.base_url)
        health = "-"
        if check:
            try:
                health = "OK" if UnifiedGateway(backend).health() else "FAIL"
            except Exception:  # noqa: BLE001 - report rather than crash the table
                health = "ERROR"
            if health == "OK":
                ok += 1
        typer.echo(f"{provider:<14}{model:<22}{tier:<6}{key:<9}{health:<8}{endpoint}")

    if check:
        typer.echo("-" * 110)
        typer.secho(f"{ok}/{len(backends)} backend(s) healthy", fg=typer.colors.GREEN)
    else:
        typer.echo("-" * 110)
        typer.echo("Run `justagent models list --check` to verify connectivity.")
