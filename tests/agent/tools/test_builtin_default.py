"""Tests for the ``make_default_tools`` factory and package exports."""

from __future__ import annotations

from justagent.agent.tools.base import Tool
from justagent.agent.tools.builtin import (
    ApplyPatchInput,
    AskQuestionInput,
    ReadFileInput,
    ReplaceInFileInput,
    RunCommandInput,
    SearchInput,
    WebFetchInput,
    WriteToFileInput,
    make_apply_patch_tool,
    make_ask_question_tool,
    make_default_tools,
    make_read_file_tool,
    make_replace_in_file_tool,
    make_run_command_tool,
    make_search_tool,
    make_web_fetch_tool,
    make_write_to_file_tool,
)


def test_make_default_tools_returns_eight_tools() -> None:
    tools = make_default_tools()
    assert len(tools) == 8


def test_make_default_tools_canonical_ids_and_order() -> None:
    tools = make_default_tools()
    ids = [t.id for t in tools]
    assert ids == [
        "read_file",
        "write_to_file",
        "replace_in_file",
        "apply_patch",
        "search",
        "run_command",
        "web_fetch",
        "ask_question",
    ]


def test_make_default_tools_unique_ids() -> None:
    tools = make_default_tools()
    ids = [t.id for t in tools]
    assert len(ids) == len(set(ids))


def test_all_tools_are_tool_instances() -> None:
    tools = make_default_tools()
    for tool in tools:
        assert isinstance(tool, Tool)


def test_all_tools_have_nonempty_description() -> None:
    tools = make_default_tools()
    for tool in tools:
        assert tool.description.strip(), f"Tool {tool.id} has empty description"


def test_all_tools_have_parameters_model() -> None:
    tools = make_default_tools()
    for tool in tools:
        assert tool.parameters is not None
        # Sanity: each parameters type has at least one field.
        assert len(tool.parameters.model_fields) > 0


def test_all_tools_have_valid_json_schema() -> None:
    tools = make_default_tools()
    for tool in tools:
        schema = tool.json_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        # Title should be stripped (it's just the class name).
        assert "title" not in schema


def test_individual_factories_match_default_tools() -> None:
    """Each make_xxx_tool() returns a tool with the expected id."""

    expected = {
        "read_file": make_read_file_tool,
        "write_to_file": make_write_to_file_tool,
        "replace_in_file": make_replace_in_file_tool,
        "apply_patch": make_apply_patch_tool,
        "search": make_search_tool,
        "run_command": make_run_command_tool,
        "web_fetch": make_web_fetch_tool,
        "ask_question": make_ask_question_tool,
    }
    for tool_id, factory in expected.items():
        tool = factory()
        assert tool.id == tool_id


def test_input_models_round_trip() -> None:
    """Each Input model can be instantiated with its required fields."""

    ReadFileInput(path="x.txt")
    WriteToFileInput(path="x", content="y")
    ReplaceInFileInput(path="x", diff="y")
    ApplyPatchInput(patch="x")
    SearchInput(pattern="x")
    RunCommandInput(command="ls")
    WebFetchInput(url="https://example.com")
    AskQuestionInput(question="Why?")


def test_completes_run_flags() -> None:
    """No built-in tool completes a run on its own (the runtime decides)."""

    tools = make_default_tools()
    for tool in tools:
        assert tool.completes_run is False
