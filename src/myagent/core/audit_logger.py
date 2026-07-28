"""Structured audit logging — append-only JSONL with optional redaction."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import structlog

from myagent.models.config import AppConfig
from myagent.utils.permissions import ensure_dir_permissions, ensure_file_permissions
from myagent.utils.redaction import (
    SENSITIVE_KEYS,
    redact_dict,
    redact_scalar,
    redact_text,
)

logger = structlog.get_logger("myagent")

__all__ = ["AuditLogger", "SENSITIVE_KEYS", "redact_text"]

_SAFE_KEYS = frozenset(
    {
        "ts",
        "trace_id",
        "event",
        "payload",
        "event_type",
        "command",
        "commands",
        "returncode",
        "duration_ms",
        "status",
        "action",
        "actions",
        "user",
        "env",
        "environment",
        "plugin_name",
        "plugin",
        "version",
        "path",
        "paths",
        "target",
        "targets",
        "message",
        "messages",
        "error",
        "errors",
        "description",
        "output",
        "result",
        "details",
        "cwd",
        "project_type",
        "detected",
        "fix",
        "count",
        "removed",
        "since",
        "reason",
        "operation",
        "enabled",
        "value",
        "values",
        "name",
        "names",
        "id",
        "ids",
        "type",
        "types",
        "source",
        "sources",
    }
)


class AuditLogger:
    """Append-only JSONL audit logger.

    Logs written to ``~/.myagent/logs/audit.{YYYY-MM-DD}.jsonl`` by default,
    or to ``config.audit.log_dir`` when provided. Each command invocation shares
    a single ``trace_id``.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.trace_id = str(uuid.uuid4())
        log_dir = (
            config.audit_log_dir or config.audit.log_dir or (Path.home() / ".myagent" / "logs")
        )
        self.log_dir = Path(log_dir)
        try:
            ensure_dir_permissions(self.log_dir, 0o700)
        except OSError as exc:
            logger.warning("Cannot create audit log directory %s: %s", self.log_dir, exc)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"audit.{today}.jsonl"
        self._context: dict[str, Any] = {}
        self._lock = threading.Lock()

    def record(self, event: str, payload: dict[str, Any] | None = None) -> None:
        """Append a structured audit record. IO failures are logged, not raised."""
        with self._lock:
            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "trace_id": self.trace_id,
                "event": event,
                "payload": self._redact({**self._context, **(payload or {})}),
            }
            try:
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
                ensure_file_permissions(self.log_file, 0o600)
            except OSError as exc:
                logger.warning("Failed to write audit record for %s: %s", event, exc)

    def export(self, since: datetime | None = None, output: Path | None = None) -> Path:
        """Export audit records to a single JSONL file.

        If *since* is given, only records with ``ts >= since`` are included.
        """
        if output is None:
            output = self.log_dir / f"audit.export.{self.trace_id}.jsonl"
        records: list[dict[str, Any]] = []
        for log_file in sorted(self.log_dir.glob("audit.*.jsonl")):
            if log_file == output or log_file.name.startswith("audit.export."):
                continue
            try:
                text = log_file.read_text()
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry_raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry_raw, dict):
                    continue
                entry = cast(dict[str, Any], entry_raw)
                if since is not None:
                    ts = entry.get("ts")
                    if isinstance(ts, str):
                        try:
                            dt = datetime.fromisoformat(ts)
                        except ValueError:
                            continue
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=UTC)
                        if dt < since:
                            continue
                records.append(entry)
        try:
            output.write_text("".join(json.dumps(r) + "\n" for r in records))
        except FileNotFoundError as exc:
            raise RuntimeError(f"Output directory does not exist: {output.parent}") from exc
        ensure_file_permissions(output, 0o600)
        return output

    def cleanup(self, retention_days: int | None = None) -> int:
        """Remove audit log files older than the retention period.

        Returns the number of files removed. Uses file mtime as a pragmatic
        proxy for record age.
        """
        if retention_days is None:
            retention_days = self.config.audit.retention_days
        cutoff = datetime.now(UTC).timestamp() - retention_days * 86400
        removed = 0
        for log_file in self.log_dir.glob("audit.*.jsonl"):
            if log_file == self.log_file or log_file.name.startswith("audit.export."):
                continue
            try:
                if log_file.stat().st_mtime < cutoff:
                    log_file.unlink()
                    removed += 1
            except OSError:
                continue
        return removed

    def export_siem_bundle(
        self, output: Path, since: datetime | None = None
    ) -> Path:
        """Export audit records as a SIEM ingestion bundle.

        Creates a directory at *output* containing ``audit.jsonl``,
        ``audit.jsonl.sha256`` and ``MANIFEST.json`` (record count, time range
        and sha256). Raises ``RuntimeError`` when the bundle cannot be written.
        Returns the bundle directory path.
        """
        bundle_dir = Path(output)
        try:
            bundle_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create SIEM bundle directory: {bundle_dir}"
            ) from exc

        jsonl_path = bundle_dir / "audit.jsonl"
        self.export(since=since, output=jsonl_path)
        content = jsonl_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

        record_count = 0
        time_min: datetime | None = None
        time_max: datetime | None = None
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            record_count += 1
            try:
                entry_raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = entry_raw.get("ts") if isinstance(entry_raw, dict) else None
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts)
                except ValueError:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                if time_min is None or dt < time_min:
                    time_min = dt
                if time_max is None or dt > time_max:
                    time_max = dt

        sha_path = bundle_dir / "audit.jsonl.sha256"
        sha_path.write_text(f"{digest}  audit.jsonl\n", encoding="utf-8")
        ensure_file_permissions(sha_path, 0o600)

        manifest = {
            "schema": "myagent-audit-siem-bundle/v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "record_count": record_count,
            "time_range": {
                "start": time_min.isoformat() if time_min else None,
                "end": time_max.isoformat() if time_max else None,
            },
            "audit_jsonl_sha256": digest,
            "sha256": digest,
            "files": ["audit.jsonl", "audit.jsonl.sha256", "MANIFEST.json"],
        }
        manifest_path = bundle_dir / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        ensure_file_permissions(manifest_path, 0o600)
        return bundle_dir

    def close(self) -> None:
        """Release resources held by the logger.

        The default file-based logger opens files per-record, so this is a
        no-op kept for interface compatibility with callers that manage the
        logger lifecycle (e.g. the CLI entrypoint). It is idempotent.
        """
        return

    def bind_context(self, **kwargs: Any) -> AuditLogger:
        """Bind extra context to future records. Returns self for chaining."""
        with self._lock:
            self._context.update(kwargs)
        return self

    def _redact(self, value: Any) -> Any:
        """Recursively redact sensitive keys and secret-like values."""
        unknown = self.config.audit.redact_unknown_fields
        if isinstance(value, dict):
            return redact_dict(
                cast(dict[str, Any], value),
                redact_unknown=unknown,
                safe_keys=_SAFE_KEYS,
            )
        if isinstance(value, list):
            return [self._redact(item) for item in cast(list[Any], value)]
        return redact_scalar(value)
