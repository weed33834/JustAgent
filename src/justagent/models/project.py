"""Managed project model for the ``justagent project`` command."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ManagedProject:
    """A local project tracked by ``justagent project``."""

    name: str
    path: str
    added_at: float
    tags: list[str] = field(default_factory=list)
    description: str = ""
