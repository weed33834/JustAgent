"""Tests for ``justagent.agent.tools.builtin._paths`` (path safety helper)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from justagent.agent.tools.base import ToolError
from justagent.agent.tools.builtin._paths import (
    PathSafetyError,
    resolve_under_cwd,
)


def test_resolve_relative_path(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file.txt").write_text("hi")

    resolved = resolve_under_cwd(tmp_path, "sub/file.txt")
    assert resolved == (tmp_path / "sub" / "file.txt").resolve()


def test_resolve_absolute_path_bypasses_containment(tmp_path: Path) -> None:
    """Absolute paths skip the cwd containment check (caller's responsibility)."""

    outside = tmp_path.parent / "elsewhere.txt"
    outside.write_text("hi")
    try:
        resolved = resolve_under_cwd(tmp_path, str(outside))
        assert resolved == outside.resolve()
    finally:
        outside.unlink()


def test_resolve_rejects_dotdot_escape(tmp_path: Path) -> None:
    with pytest.raises(PathSafetyError, match="Path must stay within cwd"):
        resolve_under_cwd(tmp_path, "../escape.txt")


def test_resolve_restrict_false_allows_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("hi")
    try:
        resolved = resolve_under_cwd(tmp_path, "../outside.txt", restrict=False)
        assert resolved == outside.resolve()
    finally:
        outside.unlink()


def test_path_safety_error_is_tool_error() -> None:
    assert issubclass(PathSafetyError, ToolError)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks behave differently on Windows")
def test_resolve_handles_symlink_inside_cwd(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "file.txt").write_text("hi")
    (tmp_path / "link").symlink_to(tmp_path / "real")

    resolved = resolve_under_cwd(tmp_path, "link/file.txt")
    assert resolved == (tmp_path / "real" / "file.txt").resolve()


def test_resolve_with_string_cwd(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("hi")
    resolved = resolve_under_cwd(str(tmp_path), "file.txt")
    assert resolved == (tmp_path / "file.txt").resolve()


def test_resolve_with_pathlib_cwd(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("hi")
    resolved = resolve_under_cwd(tmp_path, "file.txt")
    assert resolved == (tmp_path / "file.txt").resolve()


@pytest.mark.skipif(os.name == "nt", reason="path semantics differ on Windows")
def test_resolve_dangling_symlink_raises(tmp_path: Path) -> None:
    (tmp_path / "dangling").symlink_to(tmp_path / "nope.txt")
    # resolve() on a dangling link returns the link path itself; the
    # containment check still passes, so this is actually OK.
    resolved = resolve_under_cwd(tmp_path, "dangling")
    assert resolved == (tmp_path / "dangling").resolve()
