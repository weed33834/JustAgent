"""Tests for the ToolRegistry."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from autoship.agent.tools.base import Tool, ToolContext, ToolResult
from autoship.agent.tools.registry import (
    ToolRegistry,
    make_error_tool,
    make_invalid_tool,
)


class _Input(BaseModel):
    pass


async def _noop(args: BaseModel, ctx: ToolContext) -> ToolResult:
    return ToolResult(output="ok")


def _make_tool(tool_id: str) -> Tool:
    return Tool(
        id=tool_id,
        description=f"Tool {tool_id}",
        parameters=_Input,
        execute=_noop,
    )


class TestToolRegistryBasics:
    def test_empty_registry(self) -> None:
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.ids() == []
        assert registry.all() == []
        assert "anything" not in registry

    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = _make_tool("foo")
        registry.register(tool)
        assert len(registry) == 1
        assert "foo" in registry
        assert registry.get("foo") is tool

    def test_get_missing_returns_none(self) -> None:
        registry = ToolRegistry()
        assert registry.get("missing") is None

    def test_require_missing_raises_keyerror_with_hint(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_tool("alpha"))
        registry.register(_make_tool("beta"))
        with pytest.raises(KeyError) as exc_info:
            registry.require("missing")
        # Error message should list available tools.
        message = str(exc_info.value)
        assert "missing" in message
        assert "alpha" in message
        assert "beta" in message

    def test_require_missing_with_empty_registry(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError, match=r"\(empty\)"):
            registry.require("missing")

    def test_register_replaces_existing(self) -> None:
        registry = ToolRegistry()
        tool1 = _make_tool("foo")
        tool2 = _make_tool("foo")
        registry.register(tool1)
        registry.register(tool2)
        assert len(registry) == 1
        assert registry.get("foo") is tool2

    def test_unregister_existing(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_tool("foo"))
        registry.unregister("foo")
        assert len(registry) == 0
        assert registry.get("foo") is None

    def test_unregister_missing_is_noop(self) -> None:
        registry = ToolRegistry()
        # Should not raise.
        registry.unregister("missing")
        assert len(registry) == 0

    def test_clear(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))
        registry.clear()
        assert len(registry) == 0

    def test_all_preserves_insertion_order(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_tool("charlie"))
        registry.register(_make_tool("alpha"))
        registry.register(_make_tool("bravo"))
        assert registry.ids() == ["charlie", "alpha", "bravo"]
        assert [t.id for t in registry.all()] == ["charlie", "alpha", "bravo"]


class TestInvalidTool:
    def test_invalid_tool_id(self) -> None:
        tool = make_invalid_tool()
        assert tool.id == "invalid"

    def test_invalid_tool_returns_error_result(self) -> None:
        tool = make_invalid_tool()
        ctx = ToolContext(tool_call_id="bad-call", iteration=1, cwd="/tmp")
        result = asyncio.run(tool.invoke({"any_arg": "any_value"}, ctx))
        assert result.is_error
        assert "not a recognized tool" in (result.error or "")

    def test_invalid_tool_accepts_any_args(self) -> None:
        tool = make_invalid_tool()
        ctx = ToolContext(tool_call_id="x", iteration=1, cwd="/tmp")
        # Should not raise on weird args.
        result = asyncio.run(tool.invoke({}, ctx))
        assert result.is_error


class TestErrorTool:
    def test_error_tool_returns_fixed_message(self) -> None:
        tool = make_error_tool("Tool 'foo' is not registered")
        ctx = ToolContext(tool_call_id="x", iteration=1, cwd="/tmp")
        result = asyncio.run(tool.invoke({}, ctx))
        assert result.error == "Tool 'foo' is not registered"

    def test_error_tool_id(self) -> None:
        tool = make_error_tool("oops")
        assert tool.id == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
