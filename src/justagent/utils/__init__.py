"""Shared utilities for JustAgent-CLI."""

from __future__ import annotations

import hashlib
import re
import time
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


def now() -> float:
    """Return the current Unix timestamp.

    Centralised so all modules use the same time source — previously this
    was duplicated as ``_now()`` in 6+ files.
    """
    return time.time()


def utcnow() -> str:
    """Return the current UTC timestamp in ISO-8601 format.

    Previously duplicated in ``resources/storage.py`` and
    ``knowledge/document.py``.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_hex(data: str | bytes) -> str:
    """Compute the SHA-256 hex digest of *data*.

    Centralised for hash-chain audit logic previously duplicated in
    ``security/compliance.py``, ``security/judicial_security.py``, and
    ``communication/audit.py``.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Glob → regex conversion (centralised — previously duplicated in 3 files)
# ---------------------------------------------------------------------------

_REGEX_CACHE: dict[str, re.Pattern[str]] = {}


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a glob pattern (with ``**`` support) to a compiled regex.

    Translation rules:

    * ``**/`` → ``(?:.*/)?`` (matches zero or more leading path components).
    * ``**``  → ``.*`` (matches anything, including ``/``).
    * ``*``   → ``[^/]*`` (matches a single path segment, no ``/``).
    * ``?``   → ``[^/]`` (matches a single character, no ``/``).
    * every other character is regex-escaped.

    Results are cached for repeated patterns.
    """
    if pattern in _REGEX_CACHE:
        return _REGEX_CACHE[pattern]

    parts: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        if pattern[i : i + 3] == "**/":
            parts.append("(?:.*/)?")
            i += 3
        elif pattern[i : i + 2] == "**":
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    compiled = re.compile("".join(parts))
    _REGEX_CACHE[pattern] = compiled
    return compiled


def matches_any(path_str: str, patterns: list[str]) -> bool:
    """Return True if *path_str* matches any of the glob *patterns*.

    Supports ``**`` for recursive path matching.
    """
    return any(glob_to_regex(p).fullmatch(path_str) for p in patterns)


__all__ = [
    "glob_to_regex",
    "is_within_project",
    "matches_any",
    "now",
    "sha256_hex",
    "utcnow",
]
