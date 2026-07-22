"""The ``myagent init`` command."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from myagent.core.audit_logger import AuditLogger
from myagent.core.context import CommandContext
from myagent.core.hardware_profiler import detect_hardware
from myagent.models.config import AppConfig
from myagent.plugin_manager import manager as plugin_manager
from myagent.utils.project_detector import PROJECT_MARKERS, detect_project_type


def _render_default_config(project_type: str, default_tier: int = 2) -> str:
    """Render a default ``.myagent.toml`` for the given project type."""
    return f'''# MyAgent configuration
schema_version = 1
project_type = "{project_type}"

[model]
default_tier = {default_tier}
fallback = true

# Local model backend example (Ollama).
# Replace with your own endpoint / model as needed.
[[model.backends]]
provider = "ollama"
base_url = "http://127.0.0.1:11434/v1"
model = "qwen2.5:7b"
timeout = 30.0

[clean]
enabled = true
tools = ["autoflake", "black"]

[commit]
enabled = true
max_tokens = 512
conventional_commits = true
'''


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text to ``path`` atomically via write-temp-then-rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.command(name="init")(init)


@app.command()
def init(
    ctx: typer.Context,
    project_type: str | None = typer.Option(None, "--type", help="Override project type"),
    output: Path = typer.Option(Path(".myagent.toml"), "--output", "-o", help="Config file path"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmations"),
) -> None:
    """Initialize an MyAgent configuration file for the current project."""
    config: AppConfig = ctx.obj["config"]
    audit: AuditLogger = ctx.obj["audit_logger"]
    dry_run: bool = ctx.obj.get("dry_run", False)
    yes = yes or ctx.obj.get("yes", False)

    # Normalise ``project_type``: when ``init`` is called directly (e.g. from
    # tests) the default is Typer's ``OptionInfo`` sentinel rather than
    # ``None``. Treat OptionInfo as "no explicit override".
    override_type: str | None = project_type if isinstance(project_type, str) else None

    # Validate an explicit ``--type`` against the known project types so a
    # typo (e.g. ``--type pyton``) is caught at init time rather than
    # silently producing a config that no language rule matches.
    if override_type is not None:
        known = set(PROJECT_MARKERS.keys()) | {"generic"}
        if override_type not in known:
            typer.secho(
                f"Unknown project type '{override_type}'. Known types: {', '.join(sorted(known))}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

    detected = override_type or detect_project_type(config.project_root)
    hardware = detect_hardware()
    context = CommandContext(
        command="init",
        project_root=config.project_root,
        config=config,
        dry_run=dry_run,
        yes=yes,
        trace_id=audit.trace_id,
    )

    audit.record(
        "init.start",
        {
            "detected": detected,
            "output": str(output),
            "recommended_tier": hardware.recommended_tier,
        },
    )
    plugin_manager.call("pre_init", context=context, fail_fast=False)

    if output.exists():
        if not yes:
            if not typer.confirm(f"File {output} already exists. Overwrite?"):
                typer.echo("Initialization aborted.")
                audit.record("init.aborted", {"reason": "overwrite_declined"})
                raise typer.Exit(code=0)
        else:
            typer.secho(
                f"Overwriting existing file {output}",
                fg=typer.colors.YELLOW,
                err=True,
            )

    rendered = _render_default_config(detected, default_tier=hardware.recommended_tier)

    if dry_run:
        typer.echo(f"Would write config to {output}")
        typer.echo(rendered)
        audit.record("init.dry_run", {"output": str(output), "project_type": detected})
    else:
        _atomic_write_text(output, rendered)
        audit.record("init.done", {"output": str(output), "project_type": detected})
        typer.echo(f"Configuration written to {output}")

    plugin_manager.call("post_init", context=context, fail_fast=False)
