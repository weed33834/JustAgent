"""Shared helpers for CLI command modules.

These accessors were copy-pasted across knowledge/security/skill (and
friends); they live here once so behaviour changes land everywhere.
"""

from __future__ import annotations

import time
from contextlib import suppress
from typing import Any

import typer

from justagent.models.config import AppConfig


def get_config(ctx: typer.Context) -> AppConfig:
    """Return the ``AppConfig`` from ``ctx.obj`` or a default instance."""

    obj = getattr(ctx, "obj", None)
    config = obj.get("config") if obj else None
    return config if isinstance(config, AppConfig) else AppConfig()


def get_verbose(ctx: typer.Context) -> bool:
    """Return the global ``--verbose`` flag."""

    obj = getattr(ctx, "obj", None)
    return bool(obj.get("verbose")) if obj else False


def get_dry_run(ctx: typer.Context) -> bool:
    """Return the global ``--dry-run`` flag."""

    obj = getattr(ctx, "obj", None)
    return bool(obj.get("dry_run")) if obj else False


def short(text: str, width: int) -> str:
    """Truncate *text* to *width* chars, appending ``…`` when cut."""

    text = (text or "").replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def format_ts(ts: float) -> str:
    """Format a Unix timestamp as a local ``YYYY-MM-DD HH:MM`` string."""

    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def audit(ctx: typer.Context, event: str, payload: dict[str, Any] | None = None) -> None:
    """Record an audit event best-effort (never raises)."""

    obj = getattr(ctx, "obj", None)
    audit_logger = obj.get("audit_logger") if obj else None
    if audit_logger is None:
        return
    with suppress(Exception):  # audit must never break a command
        audit_logger.record(event, payload or {})
