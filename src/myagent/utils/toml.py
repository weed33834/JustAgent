"""Shared TOML loader shim.

Centralises the ``tomllib`` / ``tomli`` fallback so the rest of the codebase
can import a single canonical handle instead of repeating the same
``try/except ImportError`` block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

try:
    import tomllib  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
except ImportError:  # pragma: no cover
    import tomli as tomllib  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]


def load_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file as a ``dict``.

    Raises ``FileNotFoundError`` when ``path`` is missing and re-raises
    ``tomllib.TOMLDecodeError`` / ``OSError`` from the underlying loader
    so callers can map them onto their preferred domain error.
    """
    with path.open("rb") as handle:
        loader: Any = tomllib
        return cast(dict[str, Any], loader.load(handle))


__all__ = ["load_toml", "tomllib"]
