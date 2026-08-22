"""Tests for the Tool base class, ToolContext, ToolResult."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from justagent.agent.tools.base import (
    Attachment,
    InvalidArgumentsError,
    PermissionDeniedError,
    Tool,
    ToolAbortedError,
    ToolContext,
    ToolError,
    ToolResult,
    ToolTimeoutError,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class EchoInput(BaseModel):
    message: str
    uppercase: bool = False


async def _echo_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    msg = args.message  # type: ignore[attr-defined]
    if args.uppercase:  # type: ignore[attr-defined]
        msg = msg.upper()
    await ctx.progress("echoing", length=len(msg))
    return ToolResult(output=msg, metadata={"tool_call_id": ctx.tool_call_id})


def _make_echo_tool(
    *,
    timeout_ms: int = 5_000,
    completes_run: bool = False,
) -> Tool:
    return Tool(
        id="echo",
        description="Echo a message back",
        parameters=EchoInput,
        execute=_echo_execute,
        timeout_ms=timeout_ms,
        completes_run=completes_run,
    )


def _make_context(
    *,
    tool_call_id: str = "call-1",
    iteration: int = 1,
    cwd: str = "/tmp",
    emit=None,
    ask=None,
) -> ToolContext:
    return ToolContext(
        tool_call_id=tool_call_id,
        iteration=iteration,
        cwd=cwd,
        emit=emit,
        ask=ask,
    )


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_factory(self) -> None:
        result = ToolResult.success("hello", line_count=1)
        assert result.output == "hello"
        assert result.error is None
        assert result.metadata == {"line_count": 1}
        assert not result.is_error

    def test_failure_factory(self) -> None:
        result = ToolResult.failure("oops", retry=False)
        assert result.output == ""
        assert result.error == "oops"
        assert result.metadata == {"retry": False}
        assert result.is_error

    def test_default_values(self) -> None:
        result = ToolResult()
        assert result.output == ""
        assert result.error is None
        assert result.metadata == {}
        assert result.attachments == []
        assert not result.is_error

    def test_attachments(self) -> None:
        att = Attachment(path="/tmp/out.txt", content="data")
        result = ToolResult(output="ok", attachments=[att])
        assert len(result.attachments) == 1
        assert result.attachments[0].path == "/tmp/out.txt"
        assert result.attachments[0].content == "data"


# ---------------------------------------------------------------------------
# ToolContext
# ---------------------------------------------------------------------------


class TestToolContext:
    def test_progress_no_emit_is_noop(self) -> None:
        ctx = _make_context()
        # Should not raise even though emit is None.
        asyncio.run(ctx.progress("hello", foo=1))

    def test_progress_calls_emit(self) -> None:
        events: list[dict] = []

        async def emit(event: dict) -> None:
            events.append(event)

        ctx = _make_context(emit=emit)
        asyncio.run(ctx.progress("hello", foo=1))
        assert events == [{"title": "hello", "metadata": {"foo": 1}}]

    def test_request_permission_defaults_to_allow(self) -> None:
        ctx = _make_context()  # no ask callback
        result = asyncio.run(ctx.request_permission({"action": "write"}))
        assert result is True

    def test_request_permission_calls_ask(self) -> None:
        async def ask(req: dict) -> bool:
            return req.get("allow", False)

        ctx = _make_context(ask=ask)
        assert asyncio.run(ctx.request_permission({"allow": True})) is True
        assert asyncio.run(ctx.request_permission({"allow": False})) is False

    def test_check_aborted_raises_when_set(self) -> None:
        ctx = _make_context()
        ctx.check_aborted()  # not set, no raise
        ctx.abort.set()
        with pytest.raises(ToolAbortedError):
            ctx.check_aborted()

    def test_check_aborted_mentions_tool_call_id(self) -> None:
        ctx = _make_context(tool_call_id="call-42")
        ctx.abort.set()
        with pytest.raises(ToolAbortedError, match="call-42"):
            ctx.check_aborted()


# ---------------------------------------------------------------------------
# Tool — json_schema
# ---------------------------------------------------------------------------


class TestToolJsonSchema:
    def test_schema_has_type_object(self) -> None:
        tool = _make_echo_tool()
        schema = tool.json_schema()
        assert schema["type"] == "object"

    def test_schema_strips_auto_title(self) -> None:
        tool = _make_echo_tool()
        schema = tool.json_schema()
        assert "title" not in schema

    def test_schema_includes_required_fields(self) -> None:
        tool = _make_echo_tool()
        schema = tool.json_schema()
        # ``message`` is required (no default); ``uppercase`` is optional.
        assert "message" in schema["required"]
        assert "uppercase" not in schema.get("required", [])

    def test_schema_includes_properties(self) -> None:
        tool = _make_echo_tool()
        schema = tool.json_schema()
        assert "message" in schema["properties"]
        assert "uppercase" in schema["properties"]
        assert schema["properties"]["message"]["type"] == "string"
        assert schema["properties"]["uppercase"]["type"] == "boolean"


# ---------------------------------------------------------------------------
# Tool — invoke
# ---------------------------------------------------------------------------


class TestToolInvoke:
    def test_invoke_validates_and_executes(self) -> None:
        tool = _make_echo_tool()
        ctx = _make_context()
        result = asyncio.run(tool.invoke({"message": "hello", "uppercase": True}, ctx))
        assert result.output == "HELLO"
        assert result.metadata["tool_call_id"] == "call-1"

    def test_invoke_uses_default_for_optional_field(self) -> None:
        tool = _make_echo_tool()
        ctx = _make_context()
        result = asyncio.run(tool.invoke({"message": "hello"}, ctx))
        assert result.output == "hello"  # uppercase defaulted to False

    def test_invoke_invalid_args_raises_invalid_arguments_error(self) -> None:
        tool = _make_echo_tool()
        ctx = _make_context()
        # Missing required "message".
        with pytest.raises(InvalidArgumentsError) as exc_info:
            asyncio.run(tool.invoke({}, ctx))
        assert exc_info.value.tool == "echo"
        assert "message" in exc_info.value.detail

    def test_invoke_invalid_type_raises_invalid_arguments_error(self) -> None:
        tool = _make_echo_tool()
        ctx = _make_context()
        # Wrong type for message.
        with pytest.raises(InvalidArgumentsError):
            asyncio.run(tool.invoke({"message": 123}, ctx))

    def test_invoke_extra_fields_rejected_by_default(self) -> None:
        # Pydantic's default is to ignore extra fields, but it can be
        # configured to reject. Our EchoInput uses the default, so
        # extra fields are silently dropped.
        tool = _make_echo_tool()
        ctx = _make_context()
        result = asyncio.run(tool.invoke({"message": "hi", "extra_field": "ignored"}, ctx))
        assert result.output == "hi"

    def test_invoke_timeout_raises_tool_timeout_error(self) -> None:
        async def slow_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
            await asyncio.sleep(1.0)
            return ToolResult(output="never")

        tool = Tool(
            id="slow",
            description="Slow tool",
            parameters=EchoInput,
            execute=slow_execute,
            timeout_ms=50,  # 50ms
        )
        ctx = _make_context()
        with pytest.raises(ToolTimeoutError, match="50ms"):
            asyncio.run(tool.invoke({"message": "hi"}, ctx))

    def test_invoke_zero_timeout_disables_check(self) -> None:
        async def execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
            await asyncio.sleep(0.01)
            return ToolResult(output="ok")

        tool = Tool(
            id="quick",
            description="Quick tool",
            parameters=EchoInput,
            execute=execute,
            timeout_ms=0,  # disabled
        )
        ctx = _make_context()
        result = asyncio.run(tool.invoke({"message": "hi"}, ctx))
        assert result.output == "ok"

    def test_invoke_propagates_tool_error_subclasses(self) -> None:
        async def failing_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
            raise PermissionDeniedError("user said no")

        tool = Tool(
            id="failing",
            description="Always fails",
            parameters=EchoInput,
            execute=failing_execute,
        )
        ctx = _make_context()
        with pytest.raises(PermissionDeniedError, match="user said no"):
            asyncio.run(tool.invoke({"message": "hi"}, ctx))

    def test_invoke_propagates_generic_tool_error(self) -> None:
        async def failing_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
            raise ToolError("boom")

        tool = Tool(
            id="failing",
            description="Always fails",
            parameters=EchoInput,
            execute=failing_execute,
        )
        ctx = _make_context()
        with pytest.raises(ToolError, match="boom"):
            asyncio.run(tool.invoke({"message": "hi"}, ctx))


# ---------------------------------------------------------------------------
# Tool — metadata
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_id_and_description(self) -> None:
        tool = _make_echo_tool()
        assert tool.id == "echo"
        assert tool.description == "Echo a message back"

    def test_completes_run_default_false(self) -> None:
        tool = _make_echo_tool()
        assert tool.completes_run is False

    def test_completes_run_can_be_overridden(self) -> None:
        tool = _make_echo_tool(completes_run=True)
        assert tool.completes_run is True

    def test_default_timeout(self) -> None:
        tool = _make_echo_tool()
        assert tool.timeout_ms == 5_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
