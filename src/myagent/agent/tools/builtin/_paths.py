"""Shared path-safety helper for file-based tools."""

from __future__ import annotations

from pathlib import Path

from myagent.agent.tools.base import ToolError


class PathSafetyError(ToolError):
    """Raised when a path escapes the project cwd."""


def resolve_under_cwd(
    cwd: str | Path,
    input_path: str,
    *,
    restrict: bool = True,
) -> Path:
    """Resolve ``input_path`` against ``cwd``, optionally enforcing containment.

    Mirrors the logic in ``myagent.agent.patch._resolve_path`` but raises
    :class:`PathSafetyError` (a :class:`ToolError` subclass) so the
    runtime can catch it uniformly.

    Absolute paths bypass the containment check (callers can still
    reject them by inspecting the resolved path).
    """

    cwd_path = Path(cwd).resolve()
    is_absolute = Path(input_path).is_absolute()
    resolved = Path(input_path) if is_absolute else (cwd_path / input_path)
    try:
        resolved = resolved.resolve()
    except (OSError, RuntimeError) as exc:
        raise PathSafetyError(
            f"Cannot resolve path {input_path!r}: {exc}"
        ) from exc
    if not restrict or is_absolute:
        return resolved
    try:
        resolved.relative_to(cwd_path)
    except ValueError as exc:
        raise PathSafetyError(
            f"Path must stay within cwd: {input_path} "
            f"(resolved to {resolved})"
        ) from exc
    return resolved


__all__ = ["PathSafetyError", "resolve_under_cwd"]
