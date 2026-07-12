"""Shared utilities for AutoShip-CLI."""

from __future__ import annotations

from pathlib import Path


def is_within_project(path: Path, project_root: Path) -> bool:
    """Return True when ``path`` resolves to a location inside ``project_root``.

    Returns False on resolution errors (broken symlinks, permission denied,
    etc.) so callers can treat unresolvable paths as outside the project.
    """
    try:
        resolved = path.resolve()
        root = project_root.resolve()
        return resolved.is_relative_to(root)
    except (OSError, ValueError, RuntimeError):
        return False


__all__ = ["is_within_project"]
