"""Path-related helpers shared across JustAgent-CLI."""

from __future__ import annotations

from pathlib import Path


class PathResolutionError(ValueError):
    """Raised when a resolved path escapes its allowed working directory."""


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


def resolve_path(
    cwd: str | Path,
    input_path: str,
    restrict_to_cwd: bool = True,
) -> Path:
    """Resolve ``input_path`` against ``cwd``, optionally enforcing containment.

    Absolute paths bypass the containment check (mirroring the original
    per-module behaviour). When ``restrict_to_cwd`` is True and ``input_path``
    is relative, the resolved path must remain inside ``cwd``; otherwise a
    :class:`PathResolutionError` is raised.
    """

    base = Path(cwd)
    is_absolute = Path(input_path).is_absolute()
    resolved = (Path(input_path) if is_absolute else base / input_path).resolve()
    if not restrict_to_cwd or is_absolute:
        return resolved
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise PathResolutionError(f"Path must stay within cwd: {input_path}") from exc
    return resolved


__all__ = ["PathResolutionError", "is_within_project", "resolve_path"]
