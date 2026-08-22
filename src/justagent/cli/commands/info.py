"""The ``justagent info`` command — environment & project diagnostics.

Prints a compact summary of the runtime environment, the resolved project
config, model backends, installed plugins, and git state. Every section is
defensively wrapped so the command never crashes on a missing optional
component.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

import typer

from justagent.models.config import AppConfig


def register(parent: typer.Typer) -> None:
    parent.command(name="info", help="Show environment and project diagnostics")(info)


def _pkg(name: str) -> str:
    try:
        return _pkg_version(name)
    except Exception:  # noqa: BLE001 - version lookup must never crash
        return "?"


def _git_branch(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        branch = out.stdout.strip()
        return branch if out.returncode == 0 and branch else None
    except Exception:  # noqa: BLE001
        return None


def _git_dirty(root: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def info(ctx: typer.Context) -> None:
    """Print environment, config, backends, plugins and git state."""
    config: AppConfig = ctx.obj["config"]
    config_path = ctx.obj.get("config_path")

    typer.secho("JustAgent", bold=True)
    typer.echo(f"  version       : justagent {_pkg('justagent')}")

    typer.secho("Runtime", bold=True)
    typer.echo(f"  python        : {platform.python_version()} @ {sys.executable}")
    typer.echo(f"  platform      : {platform.platform()}")

    typer.secho("Config", bold=True)
    typer.echo(f"  project root  : {config.project_root}")
    typer.echo(f"  config path   : {config_path if config_path else '(default)'}")

    # Model backends
    backends = config.model.backends
    if backends:
        providers = ", ".join(sorted({b.provider.value for b in backends}))
        typer.echo(f"  model backends: {len(backends)} ({providers})")
        for b in backends[:6]:
            key = "set" if b.api_key else "none"
            typer.echo(
                f"    - {b.provider.value:<12} {b.model or '-':<20} tier={b.tier} key={key} {b.base_url}"
            )
        if len(backends) > 6:
            typer.echo(f"    ... and {len(backends) - 6} more")
    else:
        typer.echo("  model backends: none configured")

    # Plugins
    try:
        from justagent.core.plugin_registry import PluginRegistry

        plugins = PluginRegistry().list()
        typer.echo(f"  plugins       : {len(plugins)}")
    except Exception:  # noqa: BLE001
        typer.echo("  plugins       : ?")

    # Git state
    branch = _git_branch(config.project_root)
    if branch is not None:
        dirty = _git_dirty(config.project_root)
        state = "dirty" if dirty else "clean"
        typer.echo(f"  git           : {branch} ({state}) @ {config.project_root}")
    else:
        typer.echo("  git           : not a git repository")
