"""The ``autoship team`` command: signed team profile management.

Provides ``verify`` (check the detached signature of ``.autoship.team.toml``),
``show`` (print the effective merged config with signature status), ``sign``
(produce a detached Ed25519 signature), and ``keygen`` (generate a fresh
Ed25519 keypair).
"""

from __future__ import annotations

import base64
from pathlib import Path

import typer

from autoship.core.config_center import TEAM_CONFIG_NAME
from autoship.core.team_config import (
    TeamConfigError,
    generate_keypair,
    sign_team_config,
    signature_path_for,
    verify_team_config,
)
from autoship.models.config import AppConfig

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="team")


def _resolve_team_path(config: AppConfig, override: Path | None) -> Path:
    if override is not None:
        return override
    return Path(config.project_root) / TEAM_CONFIG_NAME


@app.command("verify")
def verify(
    ctx: typer.Context,
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the team config (defaults to .autoship.team.toml in the project root).",
    ),
) -> None:
    """Verify the detached signature of the team config file."""
    config: AppConfig = ctx.obj["config"]
    team_path = _resolve_team_path(config, config_path)

    if not team_path.exists():
        typer.echo(f"Team config not found: {team_path}", err=True)
        raise typer.Exit(code=2)

    public_key = config.team.public_key
    if not public_key:
        typer.secho(
            "No public key pinned — signature verification skipped.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=0)

    try:
        verify_team_config(team_path, public_key)
    except TeamConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Signature is valid.")


@app.command("show")
def show(
    ctx: typer.Context,
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the team config (defaults to .autoship.team.toml in the project root).",
    ),
) -> None:
    """Print the effective team config and its signature status."""
    config: AppConfig = ctx.obj["config"]
    team_path = _resolve_team_path(config, config_path)

    if not team_path.exists():
        typer.echo(f"Team config not found: {team_path}")
        raise typer.Exit(code=0)

    public_key = config.team.public_key
    sig_path = signature_path_for(team_path)
    signed = sig_path.exists()

    typer.echo(f"# {team_path}")
    if public_key:
        typer.echo(f"# pinned public_key: {public_key}")

    ok: bool | None = None
    verify_error: TeamConfigError | None = None
    if public_key and signed:
        try:
            verify_team_config(team_path, public_key)
            ok = True
        except TeamConfigError as exc:
            ok = False
            verify_error = exc

    if not public_key:
        typer.secho("No public key pinned — skipped.", fg=typer.colors.YELLOW)
    elif ok is None:
        typer.secho("No signature file found.", fg=typer.colors.RED)
    elif ok:
        typer.secho("Signature is valid.", fg=typer.colors.GREEN)
    else:
        typer.secho("Signature is INVALID.", fg=typer.colors.RED)
        if verify_error is not None:
            raise typer.Exit(code=1) from verify_error

    typer.echo(team_path.read_text(encoding="utf-8"))


@app.command("sign")
def sign(
    ctx: typer.Context,
    config_path: Path = typer.Argument(
        ...,
        help="Path to the team config file to sign.",
    ),
    private_key: str = typer.Option(
        ...,
        "--key",
        "-k",
        help="Base64 URL-safe Ed25519 private key (32 raw bytes).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the signature here (defaults to <config>.sig).",
    ),
    stdout: bool = typer.Option(
        False,
        "--stdout",
        help="Print the base64 signature to stdout instead of writing a file.",
    ),
) -> None:
    """Produce a detached signature for a team config file."""
    try:
        signature_bytes = sign_team_config(config_path, private_key)
    except TeamConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    encoded = base64.urlsafe_b64encode(signature_bytes).rstrip(b"=").decode("ascii")

    if stdout:
        typer.echo(encoded)
        return

    out_path = output or signature_path_for(config_path)
    out_path.write_text(encoded + "\n", encoding="utf-8")
    typer.echo(f"Signature written to {out_path}")


@app.command("keygen")
def keygen(
    ctx: typer.Context,
) -> None:
    """Generate a fresh Ed25519 keypair for signing team configs."""
    public_b64, private_b64 = generate_keypair()
    typer.echo("Generated new Ed25519 keypair:")
    typer.echo("")
    typer.echo("# Paste into .autoship.toml:")
    typer.echo(f'[team]\npublic_key = "{public_b64}"')
    typer.echo("")
    typer.echo("# Private key — store in a password manager or KMS, never commit:")
    typer.echo(f"# {private_b64}")
    typer.echo("")
    typer.echo("Use: autoship team sign <config> --key <private_key>")
