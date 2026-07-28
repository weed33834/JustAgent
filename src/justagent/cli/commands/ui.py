"""The ``justagent ui`` command — a textual TUI dashboard."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import typer

from justagent.core.audit_logger import AuditLogger
from justagent.models.config import AppConfig

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.command(name="ui")(ui)


def _check_textual_available() -> bool:
    try:
        return importlib.util.find_spec("textual") is not None
    except (ImportError, ValueError):
        return False


def _tail_audit(audit_logger: AuditLogger, lines: int = 20) -> list[str]:
    import json as _json

    records: list[dict[str, Any]] = []
    for log_file in sorted(audit_logger.log_dir.glob("audit.*.jsonl"), reverse=True):
        if log_file.name.startswith("audit.export."):
            continue
        try:
            text = log_file.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_json.loads(line))
            except _json.JSONDecodeError:
                continue
            if len(records) >= lines:
                break
        if len(records) >= lines:
            break
    return [f"{r.get('ts', '?')[:19]}  {r.get('event', '?')}" for r in records[:lines]]


def _project_summary(config: AppConfig, audit_logger: AuditLogger) -> list[str]:
    from justagent.core.language_rules import primary_language, rules_for_project
    from justagent.utils.project_detector import detect_project_type

    project_root = Path(config.project_root)
    detected = detect_project_type(project_root)
    primary = primary_language(project_root) or detected
    applicable = rules_for_project(project_root)
    plugin_count = sum(1 for _ in applicable)

    return [
        f"Project root  : {project_root}",
        f"Detected type : {detected}",
        f"Primary lang  : {primary}",
        f"Rules applied : {plugin_count}",
        f"Trace id      : {audit_logger.trace_id[:8]}",
        f"Audit dir     : {audit_logger.log_dir}",
    ]


def _build_app(config: AppConfig, audit_logger: AuditLogger) -> Any:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Footer, Static

    project_lines = _project_summary(config, audit_logger)
    audit_lines = _tail_audit(audit_logger, lines=30)

    verify_command = "pytest"
    for hook in config.hooks.on_save:
        if hook.command == "verify" and hook.verify_command:
            verify_command = hook.verify_command
            break

    actions: list[tuple[str, str, list[str]]] = [
        ("1", "clean", [sys.executable, "-m", "justagent", "clean", "--yes"]),
        ("2", "verify", [sys.executable, "-m", "justagent", "verify", verify_command]),
        ("3", "commit", [sys.executable, "-m", "justagent", "commit"]),
        ("4", "upload --dry-run", [sys.executable, "-m", "justagent", "upload", "--dry-run"]),
        ("5", "doctor", [sys.executable, "-m", "justagent", "doctor"]),
    ]

    class MyAgentDashboard(App[None]):
        CSS = """
        Screen { layout: vertical; }
        #header { height: 3; background: $primary; color: $text; padding: 0 1; }
        #body { height: 1fr; }
        #left { width: 40%; border-right: solid $accent; padding: 0 1; }
        #right { width: 60%; padding: 0 1; }
        .panel-title { text-style: bold; margin-bottom: 1; }
        .log-line { height: auto; }
        """

        BINDINGS = cast(
            "list[Any]",
            [Binding("q", "quit", "Quit")]
            + [Binding(key, f"run_{name.split()[0]}", name) for key, name, _ in actions],
        )

        def compose(self) -> ComposeResult:
            yield Static(f"  JustAgent dashboard — {config.project_root}", id="header")
            with Horizontal(id="body"):
                with Vertical(id="left"):
                    yield Static("Project", classes="panel-title")
                    yield Static("\n".join(project_lines))
                    yield Static("Actions", classes="panel-title", id="actions-title")
                    for key, name, _ in actions:
                        yield Static(f"  [{key}] {name}")
                    yield Static("  [q] quit")
                with VerticalScroll(id="right"):
                    yield Static("Audit log (recent)", classes="panel-title")
                    if audit_lines:
                        for line in audit_lines:
                            yield Static(line, classes="log-line")
                    else:
                        yield Static("(no audit records yet)")
            yield Footer()

        async def _run_command(self, cmd: list[str]) -> None:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(config.project_root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                self._flash(f"[red]error:[/red] {exc}")
                return
            output_lines: list[str] = []
            if result.stdout:
                output_lines.append(result.stdout.rstrip())
            if result.stderr:
                output_lines.append(f"[red]{result.stderr.rstrip()}[/red]")
            if not output_lines:
                output_lines.append(f"(no output, exit={result.returncode})")
            right = self.query_one("#right", VerticalScroll)
            await right.mount(Static(f"\n[dim]> {' '.join(cmd)}[/dim]", classes="log-line"))
            for line in "\n".join(output_lines).splitlines():
                await right.mount(Static(line, classes="log-line"))
            right.scroll_end(animate=False)

        def _flash(self, message: str) -> None:
            right = self.query_one("#right", VerticalScroll)
            right.mount(Static(message, classes="log-line"))

        async def action_run_clean(self) -> None:
            await self._run_command(actions[0][2])

        async def action_run_verify(self) -> None:
            await self._run_command(actions[1][2])

        async def action_run_commit(self) -> None:
            await self._run_command(actions[2][2])

        async def action_run_upload(self) -> None:
            await self._run_command(actions[3][2])

        async def action_run_doctor(self) -> None:
            await self._run_command(actions[4][2])

    return MyAgentDashboard()


def ui(ctx: typer.Context) -> None:
    config: AppConfig = ctx.obj["config"]
    audit_logger: AuditLogger = ctx.obj["audit_logger"]

    if not _check_textual_available():
        typer.secho(
            "textual is required for the TUI dashboard (pip install textual)",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=2)

    dashboard = _build_app(config, audit_logger)
    dashboard.run()
