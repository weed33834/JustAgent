"""The ``justagent project`` command group: manage multiple local projects.

Projects are tracked in ``~/.justagent/projects.json`` via
:class:`justagent.core.project_store.ProjectStore`. The group exposes
``list``, ``add``, ``remove``, ``run``, ``scan`` and ``batch-*`` subcommands.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import typer

from justagent.core.batch_ops import BatchOperation, BatchRunner
from justagent.core.project_discovery import (
    DiscoveryConfig,
    ProjectDiscovery,
    ProjectDiscoveryError,
)
from justagent.core.project_store import ProjectStore
from justagent.exceptions import MyAgentError
from justagent.models.config import AppConfig
from justagent.models.project import ManagedProject

app = typer.Typer(help="Manage multiple local projects.")


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="project")


def _config_from_ctx(ctx: typer.Context) -> AppConfig:
    config = ctx.obj.get("config") if ctx.obj else None
    if not isinstance(config, AppConfig):
        typer.secho(
            "JustAgent config not loaded. Run from a project with .justagent.toml.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    return config


@app.command("list")
def list_projects(ctx: typer.Context) -> None:
    """List all managed projects."""
    projects = ProjectStore().list_all()
    if not projects:
        typer.echo("No managed projects. Use 'justagent project add PATH' to add one.")
        return
    typer.echo(f"{'NAME':<24} {'PATH':<48} TAGS")
    for project in projects:
        tags = ",".join(project.tags) if project.tags else "-"
        typer.echo(f"{project.name:<24} {project.path:<48} {tags}")


@app.command("add")
def add_project(
    ctx: typer.Context,
    path: Path = typer.Argument(..., help="Path to the project directory."),
    name: str | None = typer.Option(
        None, "--name", help="Project name (default: directory basename)."
    ),
    tag: list[str] | None = typer.Option(None, "--tag", help="Tag to attach (repeatable)."),
) -> None:
    """Add a project to the managed list."""
    resolved = path.resolve()
    project_name = name or resolved.name
    tags = list(tag) if tag else []
    ProjectStore().add(
        ManagedProject(
            name=project_name,
            path=str(resolved),
            added_at=time.time(),
            tags=tags,
        )
    )
    typer.secho(f"Added project '{project_name}' -> {resolved}", fg=typer.colors.GREEN)


@app.command("remove")
def remove_project(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name of the project to remove."),
) -> None:
    """Remove a project from the managed list."""
    if not ProjectStore().remove(name):
        typer.secho(f"No project named '{name}'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"Removed project '{name}'.", fg=typer.colors.GREEN)


@app.command("run")
def run_command(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Managed project name."),
    command: list[str] | None = typer.Argument(
        None, help="Command and args to run in the project directory."
    ),
) -> None:
    """Run a shell command in a project's directory."""
    if not command:
        typer.secho("No command specified.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    project = ProjectStore().get(name)
    if project is None:
        typer.secho(f"No project named '{name}'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    config = _config_from_ctx(ctx)
    cwd = Path(project.path)
    if not cwd.is_absolute():
        cwd = config.project_root / cwd
    if not cwd.exists():
        typer.secho(f"Project path does not exist: {cwd}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"$ {' '.join(command)}  (in {cwd})")
    try:
        result = subprocess.run(command, cwd=str(cwd))
    except (subprocess.SubprocessError, OSError) as exc:
        typer.secho(f"Failed to run command: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@app.command("scan")
def scan(
    ctx: typer.Context,
    root: Path | None = typer.Argument(
        None, help="Directory to scan (default: current directory)."
    ),
    tag: list[str] | None = typer.Option(
        None, "--tag", help="Tag to attach to discovered projects (repeatable)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List discovered projects without registering them."
    ),
    max_depth: int = typer.Option(4, "--max-depth", help="Maximum directory depth to scan."),
) -> None:
    """Scan a directory for projects and optionally register them."""
    target = root.resolve() if root else Path.cwd()
    discovery = ProjectDiscovery(DiscoveryConfig(max_depth=max_depth))
    try:
        if dry_run:
            discovered = discovery.discover(target)
        else:
            store = ProjectStore()
            added = discovery.discover_and_register(
                target,
                store,
                tags=list(tag) if tag else None,
                dry_run=False,
            )
            for project in added:
                typer.secho(
                    f"Registered '{project.name}' -> {project.path}",
                    fg=typer.colors.GREEN,
                )
            typer.echo(f"Registered {len(added)} new project(s).")
            return
    except (ProjectDiscoveryError, MyAgentError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if not discovered:
        typer.echo("No projects found.")
        return
    typer.echo(f"{'NAME':<24} {'TYPE':<8} PATH")
    for found in discovered:
        typer.echo(f"{found.name:<24} {found.project_type.value:<8} {found.path}")
    typer.echo(f"Found {len(discovered)} project(s) (dry run, not registered).")


@app.command("batch-status")
def batch_status(
    ctx: typer.Context,
    names: list[str] | None = typer.Argument(
        None, help="Project names to check (default: all managed projects)."
    ),
) -> None:
    """Run ``git status`` across managed projects."""
    runner = BatchRunner(ProjectStore())
    summary = runner.run_status(names)
    typer.echo(runner.format_summary(summary))


@app.command("batch-run")
def batch_run(
    ctx: typer.Context,
    command: list[str] | None = typer.Argument(
        None, help="Command and args to run in each project directory."
    ),
) -> None:
    """Run a shell command in each managed project's directory."""
    if not command:
        typer.secho("No command specified.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    runner = BatchRunner(ProjectStore())
    summary = runner.run_command(None, command)
    typer.echo(runner.format_summary(summary))


@app.command("batch-ship")
def batch_ship(
    ctx: typer.Context,
    names: list[str] | None = typer.Argument(
        None, help="Project names to ship (default: all managed projects)."
    ),
    stages: str = typer.Option(
        "clean,verify,commit,ship",
        "--stages",
        help="Comma-separated workflow stages (clean, verify, commit, ship).",
    ),
) -> None:
    """Run workflow stages across managed projects."""
    try:
        stage_list = [BatchOperation(token.strip()) for token in stages.split(",") if token.strip()]
    except ValueError as exc:
        typer.secho(f"Invalid stage: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    runner = BatchRunner(ProjectStore())
    summary = runner.run_pipeline(names, stage_list)
    typer.echo(runner.format_summary(summary))
