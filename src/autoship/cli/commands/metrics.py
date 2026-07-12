"""The ``autoship metrics`` command for observability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import typer

from autoship.core.audit_logger import AuditLogger
from autoship.core.metrics import get_registry
from autoship.models.config import AppConfig
from autoship.utils.json_io import atomic_write_text

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="metrics", help="Inspect runtime metrics")


def _collect_audit_counters(
    audit: AuditLogger, exclude_trace_id: str | None = None
) -> dict[str, dict[str, Any]]:
    """Aggregate event counts from the audit log into counter-shaped metrics.

    The in-process :class:`MetricsRegistry` is reset on every CLI invocation
    (each ``autoship`` call is a fresh process), so ``registry.snapshot()``
    alone is always empty for a ``metrics show`` command. To give operators
    real numbers we walk the local audit JSONL files and count occurrences of
    each event type, presenting them as counters. This makes ``metrics show``
    reflect actual CLI activity.

    ``exclude_trace_id`` skips records from the current invocation so the
    ``cli.invoked`` event written by this very process does not show up as a
    self-referential metric.
    """
    counters: dict[str, dict[str, Any]] = {}
    try:
        for log_file in sorted(audit.log_dir.glob("audit.*.jsonl")):
            if log_file.name.startswith("audit.export."):
                continue
            try:
                text = log_file.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                event_entry: dict[str, Any] = cast(dict[str, Any], entry)
                if exclude_trace_id is not None:
                    trace_id = event_entry.get("trace_id")
                    if isinstance(trace_id, str) and trace_id == exclude_trace_id:
                        continue
                event = event_entry.get("event")
                if not isinstance(event, str):
                    continue
                bucket = counters.setdefault(
                    f"audit.{event}",
                    {"type": "counter", "value": 0, "description": f"audit event: {event}"},
                )
                bucket["value"] = int(bucket["value"]) + 1
    except OSError:
        pass
    return counters


def _merged_snapshot(ctx: typer.Context) -> dict[str, dict[str, Any]]:
    """Return in-process registry counters merged with audit-log aggregates."""
    registry = get_registry()
    snapshot = registry.snapshot()
    config: AppConfig | None = ctx.obj.get("config") if ctx.obj else None
    audit: AuditLogger | None = ctx.obj.get("audit_logger") if ctx.obj else None
    if config is not None and audit is not None:
        for name, data in _collect_audit_counters(audit, exclude_trace_id=audit.trace_id).items():
            # In-process registry values (if any) take precedence so a fresh
            # counter observed in the current process is not hidden by the
            # audit-log aggregate.
            if name not in snapshot:
                snapshot[name] = data
    return snapshot


@app.command("show")
def show(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output metrics as JSON"),
    reset: bool = typer.Option(False, "--reset", help="Reset metrics after displaying"),
) -> None:
    """Display collected runtime metrics."""

    snapshot = _merged_snapshot(ctx)

    if json_output:
        typer.echo(json.dumps(snapshot, indent=2, ensure_ascii=False))
    else:
        _render_table(None, snapshot)

    if reset:
        get_registry().reset()


@app.command("export")
def export(
    ctx: typer.Context,
    output: Path = typer.Option(
        Path.home() / ".autoship" / "metrics.json",
        "--output",
        "-o",
        help="Path to write the metrics JSON file",
    ),
    reset: bool = typer.Option(False, "--reset", help="Reset metrics after exporting"),
) -> None:
    """Export collected metrics to a JSON file."""

    snapshot = _merged_snapshot(ctx)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(snapshot, indent=2, ensure_ascii=False))
    typer.echo("metrics.exported")
    if reset:
        get_registry().reset()


def _render_table(i18n, snapshot: dict[str, dict[str, object]]) -> None:
    if not snapshot:
        typer.echo("metrics.empty")
        return

    typer.echo("metrics.title")
    typer.echo("-" * 70)
    for name, data in sorted(snapshot.items()):
        metric_type = data.get("type", "unknown")
        description = data.get("description", "")
        if metric_type == "counter":
            value = f"count={data['value']}"
        elif metric_type == "gauge":
            value = f"value={data['value']}"
        elif metric_type == "histogram":
            value = (
                f"count={data['count']} mean={data['mean']}ms "
                f"p50={data['p50']}ms p95={data['p95']}ms p99={data['p99']}ms"
            )
        else:
            value = str(data)
        typer.echo(f"{name:<40} {value}")
        if description:
            typer.echo(f"  {description}")
    typer.echo("-" * 70)
