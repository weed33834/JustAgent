"""Command execution context shared across the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoship.models.config import AppConfig


@dataclass(frozen=True)
class CommandContext:
    """Immutable context for a single command invocation."""

    command: str
    project_root: Path
    config: AppConfig
    verbose: bool = False
    dry_run: bool = False
    yes: bool = False
    trace_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict[str, Any])
