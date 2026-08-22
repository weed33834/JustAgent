"""The ``justagent schedule`` command group: manage scheduled tasks.

Wraps :class:`justagent.core.scheduler.Scheduler` so users can configure
cron-like recurring tasks without leaving the CLI. The ``daemon``
subcommand is meant to run as a long-lived process; ``due`` is meant to
be invoked by an external cron (one-minute granularity).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import typer

from justagent.core.project_store import ProjectStore
from justagent.core.scheduler import (
    Scheduler,
    SchedulerError,
    ScheduleStore,
)
from justagent.exceptions import MyAgentError

app = typer.Typer(help="Manage scheduled tasks.")


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="schedule")


def _make_scheduler() -> Scheduler:
    """Build a :class:`Scheduler` bound to a :class:`ProjectStore`."""
    return Scheduler(store=ScheduleStore(), project_store=ProjectStore())


def _format_timestamp(ts: float) -> str:
    """Render a Unix timestamp as a human-readable UTC string (or 'never')."""
    if ts <= 0:
        return "never"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _print_task_row(task: Any) -> None:
    """Pretty-print a single scheduled task as one terminal line."""
    status = "enabled" if task.enabled else "disabled"
    typer.echo(
        f"{task.name:<24} {task.schedule:<16} {status:<8} "
        f"next={_format_timestamp(task.next_run)} last={_format_timestamp(task.last_run)}"
    )


@app.command("list")
def list_tasks() -> None:
    """List all scheduled tasks."""
    scheduler = _make_scheduler()
    tasks = scheduler.list_tasks()
    if not tasks:
        typer.echo("No scheduled tasks. Use 'justagent schedule add' to create one.")
        return
    typer.echo(f"{'NAME':<24} {'SCHEDULE':<16} {'STATUS':<8} TIMINGS")
    for task in tasks:
        _print_task_row(task)


@app.command("add")
def add_task(
    name: str = typer.Argument(..., help="Unique task name."),
    schedule: str = typer.Option(
        ...,
        "--schedule",
        "-s",
        help="Schedule expression (e.g. '30m', 'daily 09:00', '*/5 * * * *').",
    ),
    command: str = typer.Option(..., "--command", "-c", help="Shell command to run."),
    project: str = typer.Option("", "--project", "-p", help="Managed project name (optional)."),
    disabled: bool = typer.Option(False, "--disabled", help="Add the task in a disabled state."),
) -> None:
    """Add a new scheduled task."""
    scheduler = _make_scheduler()
    try:
        task = scheduler.add_task(
            name=name,
            schedule=schedule,
            command=command,
            project=project,
            enabled=not disabled,
        )
    except SchedulerError as exc:
        typer.secho(f"Invalid schedule: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(
        f"Added task '{task.name}' (next run: {_format_timestamp(task.next_run)}).",
        fg=typer.colors.GREEN,
    )


@app.command("remove")
def remove_task(
    name: str = typer.Argument(..., help="Name of the task to remove."),
) -> None:
    """Remove a scheduled task."""
    scheduler = _make_scheduler()
    if not scheduler.remove_task(name):
        typer.secho(f"No scheduled task named '{name}'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"Removed task '{name}'.", fg=typer.colors.GREEN)


@app.command("enable")
def enable_task(
    name: str = typer.Argument(..., help="Name of the task to enable."),
) -> None:
    """Enable a scheduled task."""
    scheduler = _make_scheduler()
    if not scheduler.enable_task(name):
        typer.secho(f"No scheduled task named '{name}'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"Enabled task '{name}'.", fg=typer.colors.GREEN)


@app.command("disable")
def disable_task(
    name: str = typer.Argument(..., help="Name of the task to disable."),
) -> None:
    """Disable a scheduled task."""
    scheduler = _make_scheduler()
    if not scheduler.disable_task(name):
        typer.secho(f"No scheduled task named '{name}'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"Disabled task '{name}'.", fg=typer.colors.GREEN)


@app.command("run")
def run_task(
    name: str = typer.Argument(..., help="Name of the task to run."),
) -> None:
    """Manually run a scheduled task immediately."""
    scheduler = _make_scheduler()
    try:
        result = scheduler.run_task(name)
    except (SchedulerError, MyAgentError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if result.success:
        typer.secho(f"Task '{name}' succeeded.", fg=typer.colors.GREEN)
    else:
        typer.secho(
            f"Task '{name}' failed with exit code {result.exit_code}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=result.exit_code if result.exit_code > 0 else 1)
    if result.stdout:
        typer.echo(result.stdout.rstrip())


@app.command("due")
def due() -> None:
    """Run all due tasks once and exit (for external cron invocation)."""
    scheduler = _make_scheduler()
    results = scheduler.run_due()
    if not results:
        typer.echo("No due tasks.")
        return
    for result in results:
        status = "OK" if result.success else "FAIL"
        typer.echo(
            f"{result.task.name:<24} {status:<5} exit={result.exit_code} "
            f"dur={result.finished_at - result.started_at:.2f}s"
        )
    failed = sum(1 for r in results if not r.success)
    if failed:
        raise typer.Exit(code=1)


@app.command("daemon")
def daemon(
    check_interval: float = typer.Option(
        60.0, "--check-interval", help="Seconds between due-checks."
    ),
) -> None:
    """Run the scheduler daemon: poll for due tasks until interrupted."""
    scheduler = _make_scheduler()
    typer.echo(
        f"Scheduler daemon running (check_interval={check_interval}s). Press Ctrl+C to stop."
    )
    try:
        scheduler.run_daemon(check_interval=check_interval)
    except KeyboardInterrupt:
        typer.echo("\nScheduler daemon stopped.")
