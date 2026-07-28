"""Agent tool subsystem.

Re-exports the base types, registry, and truncation service. Built-in
tools (read, write, edit, apply_patch, run_command, search, web_fetch,
ask_question) live in :mod:`justagent.agent.tools.builtin` (added in
Wave 1.5).
"""

from __future__ import annotations

from justagent.agent.tools.base import (
    Attachment,
    ExecuteFn,
    InvalidArgumentsError,
    PermissionAsker,
    PermissionDeniedError,
    ProgressEmitter,
    Tool,
    ToolAbortedError,
    ToolContext,
    ToolError,
    ToolResult,
    ToolTimeoutError,
)
from justagent.agent.tools.registry import (
    ToolRegistry,
    make_error_tool,
    make_invalid_tool,
)
from justagent.agent.tools.truncation import (
    DEFAULT_HEAD_LINES,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    DEFAULT_TAIL_LINES,
    TruncationResult,
    TruncationService,
)

__all__ = [
    "Attachment",
    "DEFAULT_HEAD_LINES",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "DEFAULT_TAIL_LINES",
    "ExecuteFn",
    "InvalidArgumentsError",
    "PermissionAsker",
    "PermissionDeniedError",
    "ProgressEmitter",
    "Tool",
    "ToolAbortedError",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolTimeoutError",
    "TruncationResult",
    "TruncationService",
    "make_error_tool",
    "make_invalid_tool",
]
