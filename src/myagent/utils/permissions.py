"""Filesystem permission helpers shared across MyAgent modules.

All sensitive files (audit logs, registry caches, plugin stats, etc.) must be
created with restrictive permissions.  These helpers centralise the logic so
that every caller applies the same mode and emits the same warning when
existing permissions are too broad.
"""

from __future__ import annotations

import stat
from pathlib import Path

import structlog

_logger = structlog.get_logger("myagent")


def ensure_dir_permissions(path: Path, mode: int) -> None:
    """Create *path* and enforce *mode*, warning if it was too broad."""
    path.mkdir(parents=True, exist_ok=True)
    if path.exists():
        warn_if_too_broad(path, mode)
        path.chmod(mode)


def ensure_file_permissions(path: Path, mode: int) -> None:
    """Enforce *mode* on *path*, warning if it was too broad."""
    if path.exists():
        warn_if_too_broad(path, mode)
    path.chmod(mode)


def warn_if_too_broad(path: Path, mode: int) -> None:
    """Log a warning when *path* has permission bits beyond *mode*."""
    current = stat.S_IMODE(path.stat().st_mode)
    if current & ~mode:
        _logger.warning(
            "Permissions on %s (%04o) are too broad; tightening to %04o",
            path,
            current,
            mode,
        )
