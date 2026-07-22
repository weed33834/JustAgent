"""Tests for the ``search`` built-in tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from myagent.agent.tools.base import InvalidArgumentsError, ToolContext
from myagent.agent.tools.builtin.search import (
    MAX_MATCHES_PER_FILE,
    MAX_TOTAL_MATCHES,
    SearchInput,
    make_search_tool,
)


def _make_ctx(cwd: str | Path) -> ToolContext:
    return ToolContext(
        tool_call_id="call-1",
        iteration=1,
        cwd=str(cwd),
    )


def _seed_project(root: Path) -> None:
    (root / "a.py").write_text("import os\n\ndef foo():\n    return 'bar'\n")
    (root / "b.py").write_text("import sys\n\nx = 1\n")
    (root / "sub").mkdir()
    (root / "sub" / "c.py").write_text("# nested\nvalue = 42\n")
    (root / "sub" / "deep").mkdir()
    (root / "sub" / "deep" / "d.py").write_text("nested_match = True\n")
    (root / "data.txt").write_text("not python\nbut has match\n")
    # Use .png which is in the ignored suffix list.
    (root / "binary.png").write_bytes(b"\x00\x01\x02match\x00")


@pytest.mark.asyncio
async def test_search_basic_content_mode(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    tool = make_search_tool()
    result = await tool.invoke(
        {"pattern": "match", "output_mode": "content"}, _make_ctx(tmp_path)
    )
    assert not result.is_error
    assert "d.py" in result.output
    assert "data.txt" in result.output
    assert result.metadata["match_count"] >= 2


@pytest.mark.asyncio
async def test_search_files_with_matches_mode(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    tool = make_search_tool()
    result = await tool.invoke(
        {"pattern": "match", "output_mode": "files_with_matches"},
        _make_ctx(tmp_path),
    )
    assert not result.is_error
    files = result.output.split("\n")
    assert any("d.py" in f for f in files)
    assert any("data.txt" in f for f in files)


@pytest.mark.asyncio
async def test_search_count_mode(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    tool = make_search_tool()
    result = await tool.invoke(
        {"pattern": "import", "output_mode": "count"}, _make_ctx(tmp_path)
    )
    assert not result.is_error
    assert "a.py" in result.output
    assert "b.py" in result.output
    assert result.metadata["total_matches"] == 2


@pytest.mark.asyncio
async def test_search_with_glob_filter(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    tool = make_search_tool()
    result = await tool.invoke(
        {"pattern": "match", "glob": "*.py", "output_mode": "files_with_matches"},
        _make_ctx(tmp_path),
    )
    assert not result.is_error
    files = result.output.split("\n")
    assert any("d.py" in f for f in files)
    # data.txt filtered out.
    assert not any("data.txt" in f for f in files)


@pytest.mark.asyncio
async def test_search_invalid_regex(tmp_path: Path) -> None:
    tool = make_search_tool()
    result = await tool.invoke(
        {"pattern": "[invalid"}, _make_ctx(tmp_path)
    )
    assert result.is_error
    assert "regex" in result.error.lower()


@pytest.mark.asyncio
async def test_search_invalid_output_mode(tmp_path: Path) -> None:
    tool = make_search_tool()
    result = await tool.invoke(
        {"pattern": "x", "output_mode": "bogus"}, _make_ctx(tmp_path)
    )
    assert result.is_error
    assert "output_mode" in result.error.lower()


@pytest.mark.asyncio
async def test_search_missing_path(tmp_path: Path) -> None:
    tool = make_search_tool()
    result = await tool.invoke(
        {"pattern": "x", "path": "nonexistent"}, _make_ctx(tmp_path)
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_search_skips_ignored_dirs(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("match_in_git\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("match_in_nm\n")
    tool = make_search_tool()
    result = await tool.invoke(
        {"pattern": "match_in_", "output_mode": "files_with_matches"},
        _make_ctx(tmp_path),
    )
    assert not result.is_error
    assert ".git" not in result.output
    assert "node_modules" not in result.output


@pytest.mark.asyncio
async def test_search_skips_binary_files(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    tool = make_search_tool()
    # The binary.png file has the word "match" but should be skipped by suffix.
    result = await tool.invoke(
        {"pattern": "match", "glob": "*.png", "output_mode": "files_with_matches"},
        _make_ctx(tmp_path),
    )
    # Either no matches, or empty output.
    assert "binary.png" not in result.output


@pytest.mark.asyncio
async def test_search_input_validation(tmp_path: Path) -> None:
    tool = make_search_tool()
    with pytest.raises(InvalidArgumentsError):
        await tool.invoke({}, _make_ctx(tmp_path))


@pytest.mark.asyncio
async def test_search_no_matches(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    tool = make_search_tool()
    result = await tool.invoke(
        {"pattern": "this_string_doesnt_exist_anywhere_xyz"},
        _make_ctx(tmp_path),
    )
    assert not result.is_error
    assert result.output == ""


@pytest.mark.asyncio
async def test_search_json_schema() -> None:
    tool = make_search_tool()
    schema = tool.json_schema()
    assert schema["type"] == "object"
    assert "pattern" in schema["properties"]
    assert "path" in schema["properties"]
    assert "glob" in schema["properties"]
    assert "output_mode" in schema["properties"]


def test_search_input_model() -> None:
    inp = SearchInput(pattern="x")
    assert inp.pattern == "x"
    assert inp.path is None
    assert inp.glob is None
    assert inp.output_mode == "content"


def test_make_search_tool_metadata() -> None:
    tool = make_search_tool()
    assert tool.id == "search"
    assert tool.timeout_ms == 60_000


def test_search_constants() -> None:
    assert MAX_MATCHES_PER_FILE == 100
    assert MAX_TOTAL_MATCHES == 500
