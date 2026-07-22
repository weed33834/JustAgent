"""Tests for the ``replace_in_file`` built-in tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoship.agent.tools.base import InvalidArgumentsError, ToolContext
from autoship.agent.tools.builtin.edit import (
    ReplaceInFileInput,
    make_replace_in_file_tool,
)


def _make_ctx(cwd: str | Path) -> ToolContext:
    return ToolContext(
        tool_call_id="call-1",
        iteration=1,
        cwd=str(cwd),
    )


def _block(fname: str, search: str, replace: str) -> str:
    return "\n".join(
        [
            fname,
            "```",
            "<<<<<<< SEARCH",
            search,
            "=======",
            replace,
            ">>>>>>> REPLACE",
            "```",
        ]
    )


@pytest.mark.asyncio
async def test_replace_simple(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("def hello():\n    return 1\n")
    tool = make_replace_in_file_tool()
    diff = _block(
        "f.py",
        "    return 1",
        "    return 2",
    )
    result = await tool.invoke({"path": "f.py", "diff": diff}, _make_ctx(tmp_path))
    assert not result.is_error
    assert (tmp_path / "f.py").read_text() == "def hello():\n    return 2\n"
    assert "f.py" in result.metadata["touched"]


@pytest.mark.asyncio
async def test_replace_appends_with_empty_search(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("line1\n")
    tool = make_replace_in_file_tool()
    diff = _block("f.py", "", "line2\n")
    result = await tool.invoke({"path": "f.py", "diff": diff}, _make_ctx(tmp_path))
    assert not result.is_error
    assert (tmp_path / "f.py").read_text() == "line1\nline2\n"


@pytest.mark.asyncio
async def test_replace_no_match_returns_failure(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("line1\n")
    tool = make_replace_in_file_tool()
    diff = _block("f.py", "doesn't exist anywhere", "anything")
    result = await tool.invoke({"path": "f.py", "diff": diff}, _make_ctx(tmp_path))
    assert result.is_error
    assert "failed" in result.error.lower()


@pytest.mark.asyncio
async def test_replace_missing_file(tmp_path: Path) -> None:
    tool = make_replace_in_file_tool()
    diff = _block("missing.py", "x", "y")
    result = await tool.invoke({"path": "missing.py", "diff": diff}, _make_ctx(tmp_path))
    assert result.is_error


@pytest.mark.asyncio
async def test_replace_multiple_blocks_same_file(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("a\nb\nc\nd\n")
    tool = make_replace_in_file_tool()
    diff = "\n".join(
        [
            "f.py",
            "```",
            "<<<<<<< SEARCH",
            "a",
            "=======",
            "A",
            ">>>>>>> REPLACE",
            "<<<<<<< SEARCH",
            "d",
            "=======",
            "D",
            ">>>>>>> REPLACE",
            "```",
        ]
    )
    result = await tool.invoke({"path": "f.py", "diff": diff}, _make_ctx(tmp_path))
    assert not result.is_error
    assert (tmp_path / "f.py").read_text() == "A\nb\nc\nD\n"


@pytest.mark.asyncio
async def test_replace_input_validation(tmp_path: Path) -> None:
    tool = make_replace_in_file_tool()
    with pytest.raises(InvalidArgumentsError):
        await tool.invoke({"path": "f.py"}, _make_ctx(tmp_path))


@pytest.mark.asyncio
async def test_replace_json_schema() -> None:
    tool = make_replace_in_file_tool()
    schema = tool.json_schema()
    assert schema["type"] == "object"
    assert "path" in schema["properties"]
    assert "diff" in schema["properties"]


def test_replace_input_model() -> None:
    inp = ReplaceInFileInput(path="x", diff="y")
    assert inp.path == "x"
    assert inp.diff == "y"


def test_make_replace_in_file_tool_metadata() -> None:
    tool = make_replace_in_file_tool()
    assert tool.id == "replace_in_file"
    assert tool.timeout_ms == 30_000
    assert tool.completes_run is False
