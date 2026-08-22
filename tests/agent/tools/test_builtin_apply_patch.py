"""Tests for the ``apply_patch`` built-in tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from justagent.agent.tools.base import InvalidArgumentsError, ToolContext
from justagent.agent.tools.builtin.apply_patch import (
    ApplyPatchInput,
    make_apply_patch_tool,
)


def _make_ctx(cwd: str | Path) -> ToolContext:
    return ToolContext(
        tool_call_id="call-1",
        iteration=1,
        cwd=str(cwd),
    )


def _patch(body: str) -> str:
    # Strip trailing newlines so we don't introduce empty lines (which the
    # parser treats as stop markers).
    body = body.rstrip("\n")
    if body:
        return f"*** Begin Patch\n{body}\n*** End Patch\n"
    return "*** Begin Patch\n*** End Patch\n"


@pytest.mark.asyncio
async def test_apply_patch_add_file(tmp_path: Path) -> None:
    tool = make_apply_patch_tool()
    patch = _patch("*** Add File: new.txt\n+hello\n+world\n")
    result = await tool.invoke({"patch": patch}, _make_ctx(tmp_path))
    assert not result.is_error
    assert (tmp_path / "new.txt").read_text() == "hello\nworld"


@pytest.mark.asyncio
async def test_apply_patch_update_file(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("def hello():\n    return 1\n")
    tool = make_apply_patch_tool()
    patch = _patch("*** Update File: f.py\n@@\n def hello():\n-    return 1\n+    return 2\n")
    result = await tool.invoke({"patch": patch}, _make_ctx(tmp_path))
    assert not result.is_error
    assert (tmp_path / "f.py").read_text() == "def hello():\n    return 2\n"


@pytest.mark.asyncio
async def test_apply_patch_delete_file(tmp_path: Path) -> None:
    (tmp_path / "to_delete.txt").write_text("x\n")
    tool = make_apply_patch_tool()
    patch = _patch("*** Delete File: to_delete.txt\n")
    result = await tool.invoke({"patch": patch}, _make_ctx(tmp_path))
    assert not result.is_error
    assert not (tmp_path / "to_delete.txt").exists()


@pytest.mark.asyncio
async def test_apply_patch_missing_file_for_update(tmp_path: Path) -> None:
    tool = make_apply_patch_tool()
    patch = _patch("*** Update File: nope.py\n@@\n-x\n+y\n")
    result = await tool.invoke({"patch": patch}, _make_ctx(tmp_path))
    assert result.is_error


@pytest.mark.asyncio
async def test_apply_patch_invalid_syntax(tmp_path: Path) -> None:
    tool = make_apply_patch_tool()
    patch = _patch("garbage line that isn't a valid action")
    result = await tool.invoke({"patch": patch}, _make_ctx(tmp_path))
    assert result.is_error


@pytest.mark.asyncio
async def test_apply_patch_empty_patch(tmp_path: Path) -> None:
    tool = make_apply_patch_tool()
    patch = _patch("")
    result = await tool.invoke({"patch": patch}, _make_ctx(tmp_path))
    assert not result.is_error
    assert "no files were changed" in result.output.lower()


@pytest.mark.asyncio
async def test_apply_patch_multiple_actions(tmp_path: Path) -> None:
    (tmp_path / "old.py").write_text("a\nb\n")
    tool = make_apply_patch_tool()
    patch = _patch("*** Add File: new.py\n+created\n*** Update File: old.py\n@@\n a\n-b\n+B\n")
    result = await tool.invoke({"patch": patch}, _make_ctx(tmp_path))
    assert not result.is_error
    assert (tmp_path / "new.py").read_text() == "created"
    assert (tmp_path / "old.py").read_text() == "a\nB\n"
    assert "new.py" in result.metadata["touched"]
    assert "old.py" in result.metadata["touched"]


@pytest.mark.asyncio
async def test_apply_patch_input_validation(tmp_path: Path) -> None:
    tool = make_apply_patch_tool()
    with pytest.raises(InvalidArgumentsError):
        await tool.invoke({}, _make_ctx(tmp_path))


def test_apply_patch_input_model() -> None:
    inp = ApplyPatchInput(patch="x")
    assert inp.patch == "x"


def test_make_apply_patch_tool_metadata() -> None:
    tool = make_apply_patch_tool()
    assert tool.id == "apply_patch"
    assert tool.timeout_ms == 60_000
    assert tool.completes_run is False
