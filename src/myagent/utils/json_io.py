"""Shared JSON file I/O helpers for MyAgent-CLI's local stores.

Several on-disk JSON stores (plugin registry, plugin stats, registry cache)
share the same load/save skeleton: read with utf-8, swallow
``OSError``/``JSONDecodeError`` and warn, write atomically with strict file
permissions. This module centralises that skeleton so the stores only carry
their own data-shape and validation logic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

from myagent.utils.permissions import (
    ensure_dir_permissions,
    ensure_file_permissions,
    warn_if_too_broad,
)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text to ``path`` atomically via write-temp-then-rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


logger = structlog.get_logger("myagent")


def load_json(path: Path, *, label: str, check_perms: bool = False) -> Any | None:
    """Read JSON from ``path``, returning ``None`` on missing file or parse/IO error.

    ``label`` is included in warning logs so each store keeps its own
    diagnostic message (e.g. ``"plugin registry"``, ``"plugin stats"``).
    When ``check_perms`` is True, a too-permissive file mode is warned about
    before reading; the data is still loaded so the CLI stays functional.
    """
    if not path.exists():
        return None
    if check_perms:
        warn_if_too_broad(path, 0o600)
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load %s: %s", label, exc)
        return None


def save_json(
    path: Path,
    data: Any,
    *,
    label: str,
    dir_mode: int = 0o700,
    file_mode: int = 0o600,
) -> None:
    """Persist ``data`` to ``path`` as indented JSON with strict file permissions.

    Best-effort: any ``OSError`` is logged at warning level and swallowed so
    the CLI keeps working when the store is on a read-only filesystem.
    """
    try:
        ensure_dir_permissions(path.parent, dir_mode)
        atomic_write_text(path, json.dumps(data, indent=2))
        ensure_file_permissions(path, file_mode)
    except OSError as exc:
        logger.warning("Failed to save %s: %s", label, exc)


__all__ = ["load_json", "save_json"]
