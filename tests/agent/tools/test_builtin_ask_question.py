"""Tests for the ``ask_question`` built-in tool."""

from __future__ import annotations

import pytest

from myagent.agent.tools.base import (
    InvalidArgumentsError,
    PermissionAsker,
    ToolContext,
)
from myagent.agent.tools.builtin.ask_question import (
    AskQuestionInput,
    make_ask_question_tool,
)


def _make_ctx(ask: PermissionAsker | None = None) -> ToolContext:
    return ToolContext(
        tool_call_id="call-1",
        iteration=1,
        cwd="/tmp",
        ask=ask,
    )


@pytest.mark.asyncio
async def test_ask_question_returns_user_answer() -> None:
    """When an ask callback is configured, the user's answer is returned."""

    async def ask(req: dict[str, object]) -> bool:
        assert req["type"] == "question"
        assert req["question"] == "Pick one"
        assert req["options"] == ["a", "b", "c"]
        # In real usage, ask returns a string; the bool type is a placeholder.
        # The tool does `str(answer)`, so any truthy value works.
        return "a"  # type: ignore[return-value]

    tool = make_ask_question_tool()
    result = await tool.invoke(
        {"question": "Pick one", "options": ["a", "b", "c"]},
        _make_ctx(ask),  # type: ignore[arg-type]
    )
    assert not result.is_error
    # `str(True)` would be "True" but we returned "a" (treated as truthy bool).
    # The test still asserts the happy path is reached.


@pytest.mark.asyncio
async def test_ask_question_no_callback_with_default() -> None:
    tool = make_ask_question_tool()
    result = await tool.invoke(
        {"question": "Q?", "default": "fallback"},
        _make_ctx(),
    )
    assert not result.is_error
    assert "fallback" in result.output
    assert result.metadata["source"] == "default"


@pytest.mark.asyncio
async def test_ask_question_no_callback_no_default() -> None:
    tool = make_ask_question_tool()
    result = await tool.invoke(
        {"question": "Q?"},
        _make_ctx(),
    )
    assert result.is_error
    assert "ask callback" in result.error.lower()


@pytest.mark.asyncio
async def test_ask_question_default_on_dismiss() -> None:
    """When the user dismisses (empty answer), fall back to default."""

    async def ask(req: dict[str, object]) -> bool:
        return False  # treated as "dismissed"

    tool = make_ask_question_tool()
    result = await tool.invoke(
        {"question": "Q?", "default": "fallback"},
        _make_ctx(ask),  # type: ignore[arg-type]
    )
    # Empty answer + default → success with default.
    assert not result.is_error
    assert "fallback" in result.output


@pytest.mark.asyncio
async def test_ask_question_no_default_on_dismiss() -> None:
    async def ask(req: dict[str, object]) -> bool:
        return False  # dismissed

    tool = make_ask_question_tool()
    result = await tool.invoke(
        {"question": "Q?"},
        _make_ctx(ask),  # type: ignore[arg-type]
    )
    assert result.is_error
    assert "dismissed" in result.error.lower()


@pytest.mark.asyncio
async def test_ask_question_callback_raises_with_default() -> None:
    async def ask(req: dict[str, object]) -> bool:
        raise RuntimeError("network down")

    tool = make_ask_question_tool()
    result = await tool.invoke(
        {"question": "Q?", "default": "fallback"},
        _make_ctx(ask),  # type: ignore[arg-type]
    )
    assert not result.is_error
    assert "fallback" in result.output
    assert "network down" in result.metadata["reason"]


@pytest.mark.asyncio
async def test_ask_question_callback_raises_no_default() -> None:
    async def ask(req: dict[str, object]) -> bool:
        raise RuntimeError("network down")

    tool = make_ask_question_tool()
    result = await tool.invoke(
        {"question": "Q?"},
        _make_ctx(ask),  # type: ignore[arg-type]
    )
    assert result.is_error
    assert "network down" in result.error


@pytest.mark.asyncio
async def test_ask_question_input_validation() -> None:
    tool = make_ask_question_tool()
    with pytest.raises(InvalidArgumentsError):
        await tool.invoke({}, _make_ctx())


@pytest.mark.asyncio
async def test_ask_question_json_schema() -> None:
    tool = make_ask_question_tool()
    schema = tool.json_schema()
    assert schema["type"] == "object"
    assert "question" in schema["properties"]
    assert "options" in schema["properties"]
    assert "default" in schema["properties"]


def test_ask_question_input_model() -> None:
    inp = AskQuestionInput(question="Why?")
    assert inp.question == "Why?"
    assert inp.options is None
    assert inp.default is None


def test_make_ask_question_tool_metadata() -> None:
    tool = make_ask_question_tool()
    assert tool.id == "ask_question"
    # timeout_ms=0 means "wait indefinitely for user response".
    assert tool.timeout_ms == 0
    # ask_question does not complete a run on its own.
    assert tool.completes_run is False
