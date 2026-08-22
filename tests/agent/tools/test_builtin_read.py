"""Tests for the ``read_file`` built-in tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from justagent.agent.tools.base import ToolContext
from justagent.agent.tools.builtin.read import (
    ReadFileInput,
    make_read_file_tool,
)


def _make_ctx(cwd: str | Path) -> ToolContext:
    return ToolContext(
        tool_call_id="call-1",
        iteration=1,
        cwd=str(cwd),
    )


@pytest.mark.asyncio
async def test_read_file_basic(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello\nworld\n")
    tool = make_read_file_tool()
    result = await tool.invoke({"path": "hello.txt"}, _make_ctx(tmp_path))
    assert not result.is_error
    # Line numbers in cat -n format.
    assert "1\thello" in result.output
    assert "2\tworld" in result.output
    assert result.metadata["line_count"] == 2


@pytest.mark.asyncio
async def test_read_file_missing(tmp_path: Path) -> None:
    tool = make_read_file_tool()
    result = await tool.invoke({"path": "nope.txt"}, _make_ctx(tmp_path))
    assert result.is_error
    assert "does not exist" in result.error.lower()


@pytest.mark.asyncio
async def test_read_file_with_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "lines.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)))
    tool = make_read_file_tool()
    result = await tool.invoke(
        {"path": "lines.txt", "offset": 3, "limit": 4},
        _make_ctx(tmp_path),
    )
    assert not result.is_error
    assert "Showing lines 3-6 of 10" in result.output
    assert "3\tline3" in result.output
    assert "6\tline6" in result.output
    assert "7\t" not in result.output
    assert result.metadata["offset"] == 3
    assert result.metadata["limit"] == 4


@pytest.mark.asyncio
async def test_read_file_offset_beyond_end(tmp_path: Path) -> None:
    (tmp_path / "small.txt").write_text("only one line\n")
    tool = make_read_file_tool()
    result = await tool.invoke(
        {"path": "small.txt", "offset": 999},
        _make_ctx(tmp_path),
    )
    assert not result.is_error
    # Should return empty content but with a sensible range header.
    assert "Showing lines" in result.output


@pytest.mark.asyncio
async def test_read_directory_listing(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    (tmp_path / "alpha.txt").write_text("a")
    (tmp_path / "beta.py").write_text("b")
    tool = make_read_file_tool()
    result = await tool.invoke({"path": "."}, _make_ctx(tmp_path))
    assert not result.is_error
    assert "subdir/" in result.output
    assert "alpha.txt" in result.output
    assert "beta.py" in result.output
    # Directories suffixed with /.
    assert result.metadata["is_directory"] is True


@pytest.mark.asyncio
async def test_read_file_path_escape_rejected(tmp_path: Path) -> None:
    tool = make_read_file_tool()
    result = await tool.invoke({"path": "../escape.txt"}, _make_ctx(tmp_path))
    assert result.is_error
    assert (
        "must stay within cwd" in result.error.lower() or "cannot resolve" in result.error.lower()
    )


@pytest.mark.asyncio
async def test_read_file_input_validation() -> None:
    """offset must be >= 1."""
    tool = make_read_file_tool()
    # Pass an invalid offset; pydantic raises ValidationError which Tool.invoke
    # converts to InvalidArgumentsError.
    from justagent.agent.tools.base import InvalidArgumentsError

    with pytest.raises(InvalidArgumentsError):
        await tool.invoke(
            {"path": "x.txt", "offset": 0},
            _make_ctx("/tmp"),
        )


@pytest.mark.asyncio
async def test_read_file_json_schema() -> None:
    tool = make_read_file_tool()
    schema = tool.json_schema()
    assert schema["type"] == "object"
    assert "path" in schema["properties"]
    assert "offset" in schema["properties"]
    assert "limit" in schema["properties"]
    assert "title" not in schema


@pytest.mark.asyncio
async def test_read_file_preserves_trailing_newline(tmp_path: Path) -> None:
    (tmp_path / "n.txt").write_text("a\nb\nc\n")
    tool = make_read_file_tool()
    result = await tool.invoke({"path": "n.txt"}, _make_ctx(tmp_path))
    assert not result.is_error
    # Output should end with a newline (preserved from file).
    assert result.output.endswith("\n")


@pytest.mark.asyncio
async def test_read_file_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("")
    tool = make_read_file_tool()
    result = await tool.invoke({"path": "empty.txt"}, _make_ctx(tmp_path))
    assert not result.is_error
    assert result.metadata["line_count"] == 0


@pytest.mark.asyncio
async def test_read_file_with_subpath(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "nested.txt").write_text("nested content\n")
    tool = make_read_file_tool()
    result = await tool.invoke({"path": "dir/nested.txt"}, _make_ctx(tmp_path))
    assert not result.is_error
    assert "nested content" in result.output


def test_read_file_input_model_defaults() -> None:
    inp = ReadFileInput(path="x.txt")
    assert inp.path == "x.txt"
    assert inp.offset is None
    assert inp.limit is None


def test_make_read_file_tool_metadata() -> None:
    tool = make_read_file_tool()
    assert tool.id == "read_file"
    assert tool.timeout_ms == 30_000
    assert tool.completes_run is False
    assert "Read the contents" in tool.description
