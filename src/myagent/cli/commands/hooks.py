"""The ``myagent hooks`` command: a thin CLI front-end for run-on-save hook
management.

The actual execution, glob filtering and debouncing live in
:class:`myagent.core.hooks.OnSaveHookRunner`; this module exposes ``list``,
``run`` (one-shot, for editor integrations) and ``watch`` (filesystem-event
driven, with per-hook debouncing) subcommands that work with any editor
without requiring LSP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from myagent.core.hooks import OnSaveHookRunner
from myagent.core.i18n import I18n, get_i18n_from_ctx
from myagent.models.config import AppConfig

app = typer.Typer()

# Directories that never count as a "save". Mirrors the exclusions used by
# ``clean``'s file collector so watchers do not chase their own artifacts.
_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="hooks")


def _config_from_ctx(ctx: typer.Context) -> AppConfig:
    config = ctx.obj.get("config") if ctx.obj else None
    if not isinstance(config, AppConfig):
        typer.secho(
            "MyAgent config not loaded. Run from a project with .myagent.toml.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from None
    return config


def _audit_from_ctx(ctx: typer.Context) -> Any:
    return ctx.obj.get("audit_logger") if ctx.obj else None


@app.command("list")
def list_hooks(ctx: typer.Context) -> None:
    """列出已配置的保存时钩子。"""
    config = _config_from_ctx(ctx)
    i18n: I18n = get_i18n_from_ctx(ctx)
    if not config.hooks.enabled:
        typer.echo(i18n._("hooks.disabled"))
        raise typer.Exit(code=0)
    if not config.hooks.on_save:
        typer.echo(i18n._("hooks.empty"))
        raise typer.Exit(code=0)
    typer.echo(i18n._("hooks.list_header"))
    for idx, hook in enumerate(config.hooks.on_save):
        include = ", ".join(hook.include) if hook.include else "**/*"
        exclude = ", ".join(hook.exclude) if hook.exclude else "-"
        extra = f"  verify_command={hook.verify_command}" if hook.command == "verify" else ""
        typer.echo(
            f"  [{idx}] {hook.command}  include={include}  exclude={exclude}  "
            f"debounce_ms={hook.debounce_ms}{extra}"
        )


@app.command("run")
def run_hooks(
    ctx: typer.Context,
    file: Path = typer.Option(..., "--file", "-f", help="Path of the saved file."),
) -> None:
    """Run every on-save hook matching ``--file`` once.

    Each invocation builds a fresh runner, so debouncing (which is
    in-memory and per-runner) does not suppress a one-shot ``run`` —
    debounce only applies inside ``watch`` and the LSP server, where the
    same runner is reused across many saves.
    """
    config = _config_from_ctx(ctx)
    i18n: I18n = get_i18n_from_ctx(ctx)
    audit = _audit_from_ctx(ctx)
    if not config.hooks.enabled:
        typer.echo(i18n._("hooks.disabled"))
        raise typer.Exit(code=0)
    runner = OnSaveHookRunner(config, audit_logger=audit)
    matches = runner.matching_hooks(file)
    if not matches:
        typer.echo(i18n._("hooks.no_match"))
        raise typer.Exit(code=0)
    results = runner.run_for_path(file)
    failed = 0
    for res in results:
        status = "OK" if res.ok else f"FAIL(exit={res.exit_code})"
        typer.echo(f"  {res.hook.command}: {status}  ({res.duration_ms:.0f} ms)")
        if res.stdout.strip():
            typer.echo(res.stdout.rstrip())
        if res.stderr.strip():
            typer.echo(res.stderr.rstrip(), err=True)
        if not res.ok:
            failed += 1
    if failed:
        raise typer.Exit(code=1)


@app.command("watch")
def watch(
    ctx: typer.Context,
    paths: list[Path] = typer.Argument(
        default_factory=lambda: [Path(".")], help="Directories to watch (default: cwd)."
    ),
    timeout: float = typer.Option(
        30.0, "--timeout", help="Per-hook subprocess timeout in seconds."
    ),
) -> None:
    """Watch directories and run on-save hooks on file changes."""
    config = _config_from_ctx(ctx)
    i18n: I18n = get_i18n_from_ctx(ctx)
    audit = _audit_from_ctx(ctx)
    if not config.hooks.enabled:
        typer.echo(i18n._("hooks.disabled"))
        raise typer.Exit(code=0)

    try:
        from watchdog.events import FileSystemEvent, FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:  # pragma: no cover - environment-dependent
        typer.echo(i18n._("hooks.watchdog_missing"), err=True)
        raise typer.Exit(code=2) from None

    runner = OnSaveHookRunner(config, audit_logger=audit, timeout=timeout)
    typer.echo(i18n._("hooks.watching"))
    for p in paths:
        typer.echo(f"  → {p}")

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event: FileSystemEvent) -> None:
            self._dispatch(event)

        def on_created(self, event: FileSystemEvent) -> None:
            self._dispatch(event)

        def _dispatch(self, event: FileSystemEvent) -> None:
            if event.is_directory:
                return
            src = event.src_path
            if not isinstance(src, str) or not src:
                return
            path = Path(src)
            if any(part in _EXCLUDED_DIRS for part in path.parts):
                return
            matches = runner.matching_hooks(path)
            if not matches:
                return
            results = runner.run_for_path(path)
            for res in results:
                status = "OK" if res.ok else f"FAIL(exit={res.exit_code})"
                typer.echo(f"[{path}] {res.hook.command}: {status}  ({res.duration_ms:.0f} ms)")
                if not res.ok and res.stderr.strip():
                    typer.echo(res.stderr.rstrip(), err=True)

    observer = Observer()
    handler = _Handler()
    watched: list[Path] = []
    for p in paths:
        target = p if p.is_dir() else p.parent
        if not target.exists():
            typer.echo(i18n._("hooks.watch_missing", path=target), err=True)
            continue
        observer.schedule(handler, str(target), recursive=True)
        watched.append(target)
    if not watched:
        raise typer.Exit(code=1)
    observer.start()
    try:
        while True:
            observer.join(0.5)
    except KeyboardInterrupt:
        typer.echo(i18n._("hooks.watch_stop"))
    finally:
        observer.stop()
        observer.join()
