"""Tests for the ``write_to_file`` built-in tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from myagent.agent.tools.base import InvalidArgumentsError, ToolContext
from myagent.agent.tools.builtin.write import (
    WriteToFileInput,
    make_write_to_file_tool,
)


def _make_ctx(cwd: str | Path) -> ToolContext:
    return ToolContext(
        tool_call_id="call-1",
        iteration=1,
        cwd=str(cwd),
    )


@pytest.mark.asyncio
async def test_write_creates_new_file(tmp_path: Path) -> None:
    tool = make_write_to_file_tool()
    result = await tool.invoke(
        {"path": "new.txt", "content": "hello\nworld\n"},
        _make_ctx(tmp_path),
    )
    assert not result.is_error
    assert (tmp_path / "new.txt").read_text() == "hello\nworld\n"
    assert result.metadata["line_count"] == 2


@pytest.mark.asyncio
async def test_write_overwrites_existing(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("old content\n")
    tool = make_write_to_file_tool()
    result = await tool.invoke(
        {"path": "f.txt", "content": "new content\n"},
        _make_ctx(tmp_path),
    )
    assert not result.is_error
    assert (tmp_path / "f.txt").read_text() == "new content\n"


@pytest.mark.asyncio
async def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    tool = make_write_to_file_tool()
    result = await tool.invoke(
        {"path": "deep/nested/path/f.txt", "content": "x"},
        _make_ctx(tmp_path),
    )
    assert not result.is_error
    assert (tmp_path / "deep/nested/path/f.txt").read_text() == "x"


@pytest.mark.asyncio
async def test_write_no_create_dirs_fails(tmp_path: Path) -> None:
    tool = make_write_to_file_tool()
    result = await tool.invoke(
        {
            "path": "missing_dir/f.txt",
            "content": "x",
            "create_dirs": False,
        },
        _make_ctx(tmp_path),
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_write_rejects_path_escape(tmp_path: Path) -> None:
    tool = make_write_to_file_tool()
    result = await tool.invoke(
        {"path": "../escape.txt", "content": "x"},
        _make_ctx(tmp_path),
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_write_atomic_does_not_corrupt_on_failure(tmp_path: Path) -> None:
    """Even if the parent dir creation fails, no partial file is left."""
    # Make the target parent a file, so mkdir fails.
    (tmp_path / "block").write_text("not a dir")
    tool = make_write_to_file_tool()
    result = await tool.invoke(
        {"path": "block/sub.txt", "content": "x"},
        _make_ctx(tmp_path),
    )
    assert result.is_error
    # The blocking file is untouched.
    assert (tmp_path / "block").read_text() == "not a dir"
    # No spurious file was created.
    assert not (tmp_path / "block" / "sub.txt").exists()


@pytest.mark.asyncio
async def test_write_input_validation_missing_content(tmp_path: Path) -> None:
    tool = make_write_to_file_tool()
    with pytest.raises(InvalidArgumentsError):
        await tool.invoke({"path": "x.txt"}, _make_ctx(tmp_path))


@pytest.mark.asyncio
async def test_write_input_validation_missing_path(tmp_path: Path) -> None:
    tool = make_write_to_file_tool()
    with pytest.raises(InvalidArgumentsError):
        await tool.invoke({"content": "x"}, _make_ctx(tmp_path))


@pytest.mark.asyncio
async def test_write_to_file_json_schema() -> None:
    tool = make_write_to_file_tool()
    schema = tool.json_schema()
    assert schema["type"] == "object"
    assert "path" in schema["properties"]
    assert "content" in schema["properties"]
    assert "create_dirs" in schema["properties"]


def test_write_input_defaults() -> None:
    inp = WriteToFileInput(path="x", content="y")
    assert inp.create_dirs is True


def test_make_write_to_file_tool_metadata() -> None:
    tool = make_write_to_file_tool()
    assert tool.id == "write_to_file"
    assert tool.timeout_ms == 30_000
    assert tool.completes_run is False
