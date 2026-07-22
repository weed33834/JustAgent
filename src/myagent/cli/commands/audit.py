"""The ``myagent audit`` command."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer

from myagent.core.audit_logger import AuditLogger
from myagent.models.config import AppConfig

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="audit")


def get_audit_logger_from_ctx(ctx: typer.Context) -> AuditLogger:
    """Return the ``AuditLogger`` stored in ``ctx.obj`` or build one from config."""
    obj = getattr(ctx, "obj", None)
    audit_logger = obj.get("audit_logger") if obj else None
    if isinstance(audit_logger, AuditLogger):
        return audit_logger
    config = obj.get("config") if obj else None
    if isinstance(config, AppConfig):
        return AuditLogger(config)
    return AuditLogger(AppConfig())


def _parse_since(value: str) -> datetime:
    """Parse ``--since`` value as ISO datetime or relative days (``1d``, ``7d``)."""
    stripped = value.strip().lower()
    if stripped.endswith("d"):
        try:
            days = int(stripped[:-1])
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid relative time: {value}") from exc
        return datetime.now(UTC) - timedelta(days=days)
    try:
        parsed = datetime.fromisoformat(stripped)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@app.command("export")
def export_logs(
    ctx: typer.Context,
    since: str | None = typer.Option(
        None,
        "--since",
        "-s",
        help="Export records after this time (ISO or 1d/7d/30d)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (jsonl) or output directory (siem-bundle).",
    ),
    fmt: str = typer.Option(
        "jsonl",
        "--format",
        "-f",
        help="Export format: jsonl (default) or siem-bundle (JSONL + MANIFEST + sha256).",
    ),
) -> None:
    """Export audit logs.

    Two formats are supported:

    * ``jsonl`` (default) — a single ``audit.jsonl`` file at ``--output``.
    * ``siem-bundle`` — a directory at ``--output`` containing ``audit.jsonl``,
      ``MANIFEST.json`` (record count, time range, sha256), and
      ``audit.jsonl.sha256``. Use this when shipping audit data to a SIEM
      ingestion pipeline that needs tamper evidence.
    """

    audit_logger = get_audit_logger_from_ctx(ctx)
    try:
        since_dt = _parse_since(since) if since else None
        fmt_norm = fmt.strip().lower()

        # RBAC gate: export lifts audit data out of the host.

        if fmt_norm == "siem-bundle":
            if output is None:
                typer.echo(
                    "siem-bundle 格式需要 --output 指定输出目录。",
                    err=True,
                )
                raise typer.Exit(code=1)
            try:
                _bundle_dir = audit_logger.export_siem_bundle(output, since=since_dt)
            except RuntimeError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=1) from exc
            typer.echo(f"SIEM 包已写入 {_bundle_dir}（audit.jsonl + MANIFEST.json + audit.jsonl.sha256）。")
            return

        if fmt_norm != "jsonl":
            typer.echo(
                f"未知导出格式：{fmt_norm}。可用：jsonl 或 siem-bundle。",
                err=True,
            )
            raise typer.Exit(code=1)

        try:
            exported_path = audit_logger.export(since=since_dt, output=output)
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        _count_lines = 0
        try:
            text = exported_path.read_text()
            _count = len([line for line in text.splitlines() if line.strip()])
        except OSError:
            pass
        typer.echo(f"已导出 {_count_lines} 条记录到 {exported_path}")
    finally:
        # ``get_audit_logger_from_ctx`` may construct a fresh ``AuditLogger``
        # (with its own SIEM/sink HTTP clients) when ``ctx.obj`` has none.
        # ``close()`` is idempotent, so closing a shared logger here is
        # harmless and guarantees the freshly-built one does not leak.
        audit_logger.close()


@app.command("cleanup")
def cleanup_logs(
    ctx: typer.Context,
    retention_days: int | None = typer.Option(
        None,
        "--retention-days",
        help="Retention period in days",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show actions without executing"),
) -> None:
    """Remove audit log files older than the retention period."""

    audit_logger = get_audit_logger_from_ctx(ctx)
    try:
        # RBAC gate: cleanup destroys audit data; export is left open so
        # viewer-tier operators can still pull evidence.

        if dry_run:
            typer.echo("[dry-run] 将删除过期审计日志文件")
            return
        _removed = audit_logger.cleanup(retention_days=retention_days)
        typer.echo(f"已删除 {_removed} 个过期审计日志文件")
    finally:
        # See ``export_logs``: a freshly-built logger must be closed, and
        # ``close()`` is idempotent for a shared one.
        audit_logger.close()
