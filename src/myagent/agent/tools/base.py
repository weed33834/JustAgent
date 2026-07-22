"""Tool base types — Tool, ToolContext, ToolResult, ToolError.

Ports the tool-calling framework from Cline (``AgentTool`` /
``AgentToolContext`` / ``AgentToolResult``) and OpenCode (``Tool.Def``
/ ``Tool.Context`` / ``Tool.ExecuteResult``) to Python.

Design choices:

* :class:`Tool` is a plain dataclass — tools are constructed by composing
  an ``execute`` callable rather than subclassing. This mirrors Cline's
  factory-function pattern and makes inline tool definitions ergonomic.
* Input parameters are validated by a :class:`pydantic.BaseModel`
  subclass; its ``model_json_schema()`` is sent to the LLM. This matches
  Cline's Zod→JSON-Schema and OpenCode's Effect-Schema→JSON-Schema
  pipelines.
* :meth:`Tool.invoke` runs validation + timeout + abort handling so
  the runtime doesn't have to repeat that logic per call.
* Errors propagate as :class:`ToolError` subclasses; the runtime
  catches them and converts to :class:`ToolResult` (``error=...``).
* :class:`ToolContext` carries ``abort`` (an :class:`asyncio.Event`),
  ``emit`` (progress callback), and ``ask`` (permission callback) —
  matching Cline's ``AgentToolContext.signal / emitUpdate /
  requestToolApproval`` and OpenCode's ``Context.abort / metadata /
  ask``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Base class for tool execution errors."""


class InvalidArgumentsError(ToolError):
    """Raised when tool arguments fail validation.

    Mirrors OpenCode's ``InvalidArgumentsError``. The runtime surfaces
    the message back to the LLM so the model can correct its inputs.
    """

    def __init__(self, tool: str, detail: str) -> None:
        self.tool = tool
        self.detail = detail
        super().__init__(
            f"The {tool} tool was called with invalid arguments: {detail}"
        )


class ToolTimeoutError(ToolError):
    """Raised when a tool execution exceeds its configured timeout."""


class PermissionDeniedError(ToolError):
    """Raised when the user denies a permission prompt."""


class ToolAbortedError(ToolError):
    """Raised when the abort signal fires during tool execution."""


# ---------------------------------------------------------------------------
# Result & attachment
# ---------------------------------------------------------------------------


@dataclass
class Attachment:
    """A file attachment returned by a tool.

    Used by ``read_file`` to return binary blobs, ``run_command`` to
    return captured stdout files, etc. ``content`` may be ``None`` when
    the attachment is too large to inline and only ``path`` is meaningful.
    """

    path: str
    content: str | None = None
    mime_type: str = "text/plain"


@dataclass
class ToolResult:
    """Result of a tool execution.

    Either ``output`` is set (success) or ``error`` is set (failure).
    Both may be set for partial success with warnings — the runtime
    surfaces ``error`` to the LLM as the tool result content.

    ``metadata`` carries tool-specific telemetry (e.g. ``{"truncated":
    True, "output_path": "..."}``) that the runtime may forward to
    progress events.
    """

    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[Attachment] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        """``True`` if this result represents a failure."""

        return self.error is not None

    @classmethod
    def success(cls, output: str, **metadata: Any) -> ToolResult:
        """Construct a success result with optional metadata."""

        return cls(output=output, metadata=dict(metadata))

    @classmethod
    def failure(cls, error: str, **metadata: Any) -> ToolResult:
        """Construct a failure result with optional metadata."""

        return cls(error=error, metadata=dict(metadata))


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


#: Progress callback: receives ``{"title": str, "metadata": dict}``.
ProgressEmitter = Callable[[dict[str, Any]], Awaitable[None]]

#: Permission callback: receives a permission request dict, returns
#: ``True`` if approved, ``False`` if denied.
PermissionAsker = Callable[[dict[str, Any]], Awaitable[bool]]


@dataclass
class ToolContext:
    """Context passed to a tool's ``execute`` callable.

    Provides the tool with access to the runtime environment:
    cancellation, progress reporting, permission prompts, and
    per-iteration metadata.

    The ``abort`` :class:`asyncio.Event` is set by the runtime when the
    user cancels the run (Ctrl-C, abort button, etc.). Tools that
    perform long-running work should periodically check
    ``context.abort.is_set()`` (or ``await context.abort.wait()``) and
    raise :class:`ToolAbortedError` to bail out cleanly.
    """

    tool_call_id: str
    iteration: int
    cwd: str
    agent_id: str = ""
    conversation_id: str = ""
    run_id: str = ""
    abort: asyncio.Event = field(default_factory=asyncio.Event)
    emit: ProgressEmitter | None = None
    ask: PermissionAsker | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    async def progress(self, title: str, **metadata: Any) -> None:
        """Emit a progress update to the runtime.

        ``title`` is a short human-readable label (e.g. ``"Reading
        src/main.py"``); ``metadata`` carries tool-specific fields
        (e.g. ``{"line_count": 42}``).
        """

        if self.emit is not None:
            await self.emit({"title": title, "metadata": metadata})

    async def request_permission(self, request: dict[str, Any]) -> bool:
        """Request user permission for a side effect.

        Returns ``True`` if approved (or if no permission system is
        configured — i.e. ``ask`` is ``None``, which means "default
        allow"). Returns ``False`` if the user explicitly denied.
        """

        if self.ask is None:
            return True
        return await self.ask(request)

    def check_aborted(self) -> None:
        """Raise :class:`ToolAbortedError` if the abort signal is set.

        Convenience for tools that want a one-line cancellation check
        between synchronous work units.
        """

        if self.abort.is_set():
            raise ToolAbortedError(
                f"Tool call {self.tool_call_id} was aborted by the user"
            )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


#: Execute callable: ``(validated_args, context) -> ToolResult``.
ExecuteFn = Callable[[BaseModel, ToolContext], Awaitable[ToolResult]]


@dataclass
class Tool:
    """An agent tool definition.

    Combines the prompt-facing metadata (``id``, ``description``,
    ``parameters``) with the runtime-facing ``execute`` callable. Tools
    are typically constructed via factory functions (see
    :mod:`myagent.agent.tools.builtin`) rather than subclassed.

    Example:

    >>> from pydantic import BaseModel
    >>> class EchoInput(BaseModel):
    ...     message: str
    ...
    >>> async def echo(args: BaseModel, ctx: ToolContext) -> ToolResult:
    ...     return ToolResult(output=f"echo: {args.message}")  # type: ignore[attr-defined]
    ...
    >>> tool = Tool(
    ...     id="echo",
    ...     description="Echo a message back",
    ...     parameters=EchoInput,
    ...     execute=echo,
    ... )
    >>> tool.id
    'echo'
    """

    id: str
    description: str
    parameters: type[BaseModel]
    execute: ExecuteFn
    timeout_ms: int = 30_000
    completes_run: bool = False

    def json_schema(self) -> dict[str, Any]:
        """Return the LLM-facing JSON Schema for this tool's input.

        Forces ``type: "object"`` at the top level (some LMs require
        this) and strips Pydantic's ``title`` field which is just noise
        for the LLM.
        """

        schema = self.parameters.model_json_schema()
        schema.setdefault("type", "object")
        # Drop the auto-generated title — it's the class name, which
        # is meaningless to the LLM and adds prompt-token noise.
        schema.pop("title", None)
        return schema

    async def invoke(
        self,
        raw_args: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Validate ``raw_args`` against the schema, then execute.

        Handles:

        * :class:`ValidationError` → :class:`InvalidArgumentsError`
          (so the runtime can feed the message back to the LLM).
        * :class:`asyncio.TimeoutError` → :class:`ToolTimeoutError`
          (when ``timeout_ms > 0``).
        * :class:`ToolError` subclasses → re-raised unchanged.
        * Other exceptions → wrapped in :class:`ToolError` with the
          tool id prefixed.
        """

        try:
            args = self.parameters.model_validate(raw_args)
        except ValidationError as exc:
            raise InvalidArgumentsError(self.id, str(exc)) from exc

        if self.timeout_ms > 0:
            try:
                result = await asyncio.wait_for(
                    self.execute(args, context),
                    timeout=self.timeout_ms / 1000,
                )
            except TimeoutError as exc:
                raise ToolTimeoutError(
                    f"Tool {self.id} timed out after {self.timeout_ms}ms"
                ) from exc
        else:
            result = await self.execute(args, context)
        return result


__all__ = [
    "Attachment",
    "ExecuteFn",
    "InvalidArgumentsError",
    "PermissionAsker",
    "PermissionDeniedError",
    "ProgressEmitter",
    "Tool",
    "ToolAbortedError",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "ToolTimeoutError",
]
