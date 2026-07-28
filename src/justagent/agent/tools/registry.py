"""Tool registry.

Mirrors Cline's static ``ToolCatalogEntry`` + runtime-resolved tool
list pattern, plus OpenCode's ``ToolRegistry.Service`` lookup surface.

Unlike both upstream projects (which pass a plain ``Tool[]`` array
through the runtime config), justagent exposes a mutable
:class:`ToolRegistry` so plugins, MCP servers, and user config can all
contribute tools at startup. The runtime then freezes the registry
when a run begins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from justagent.agent.tools.base import Tool, ToolContext, ToolResult

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ToolRegistry:
    """Mutable registry of :class:`Tool` instances keyed by ``id``.

    Example:

    >>> registry = ToolRegistry()
    >>> registry.register(my_tool)
    >>> registry.get("my_tool") is my_tool
    True
    >>> registry.ids()
    ['my_tool']
    """

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        """Register a tool. Replaces any existing tool with the same id."""

        self._tools[tool.id] = tool

    def unregister(self, tool_id: str) -> None:
        """Remove a tool by id. No-op if not registered."""

        self._tools.pop(tool_id, None)

    def get(self, tool_id: str) -> Tool | None:
        """Return the tool with ``tool_id``, or ``None`` if not found."""

        return self._tools.get(tool_id)

    def require(self, tool_id: str) -> Tool:
        """Return the tool with ``tool_id``, or raise :class:`KeyError`.

        The error message is formatted for surfacing back to the LLM
        (it lists the available tool ids as a "did you mean" hint).
        """

        tool = self._tools.get(tool_id)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys())) or "(empty)"
            raise KeyError(
                f"Tool {tool_id!r} is not registered. Available: {available}"
            )
        return tool

    def all(self) -> list[Tool]:
        """Return all registered tools in insertion order."""

        return list(self._tools.values())

    def ids(self) -> list[str]:
        """Return all registered tool ids in insertion order."""

        return list(self._tools.keys())

    def clear(self) -> None:
        """Remove all registered tools."""

        self._tools.clear()

    def __contains__(self, tool_id: object) -> bool:
        return tool_id in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# ---------------------------------------------------------------------------
# Sentinel "invalid" tool
# ---------------------------------------------------------------------------


class _InvalidToolInput(BaseModel):
    """Schema for the sentinel invalid tool — accepts any args."""

    model_config = {"extra": "allow"}


async def _invalid_execute(
    args: BaseModel, context: ToolContext
) -> ToolResult:
    """Return an error result for unknown tool calls.

    Mirrors OpenCode's ``invalid`` tool. The runtime installs this
    sentinel under every unknown tool id the LLM emits so the model
    gets a structured "this tool doesn't exist" message instead of a
    hard crash.
    """

    return ToolResult.failure(
        error=(
            f"Tool {context.tool_call_id!r} is not a recognized tool. "
            "Please use one of the tools described in the system prompt."
        )
    )


def make_invalid_tool() -> Tool:
    """Construct the sentinel "invalid" tool.

    The runtime installs this as a fallback so that an unknown
    ``toolName`` from the model still resolves to a callable tool that
    returns an error message rather than crashing the run.
    """

    return Tool(
        id="invalid",
        description="Do not use. This tool is a sentinel for unknown tool calls.",
        parameters=_InvalidToolInput,
        execute=_invalid_execute,
        timeout_ms=0,
    )


def make_error_tool(error_message: str) -> Tool:
    """Construct a one-shot tool that returns a fixed error.

    Used by the runtime to substitute for a missing tool when the LLM
    references an unknown id. Unlike :func:`make_invalid_tool`, the
    error message is fixed at construction time so it can include the
    offending tool name.
    """

    class _AnyInput(BaseModel):
        model_config = {"extra": "allow"}

    async def _execute(_args: BaseModel, _ctx: ToolContext) -> ToolResult:
        return ToolResult.failure(error=error_message)

    return Tool(
        id="error",
        description="Do not use.",
        parameters=_AnyInput,
        execute=_execute,
        timeout_ms=0,
    )


__all__ = [
    "ToolRegistry",
    "make_error_tool",
    "make_invalid_tool",
]
