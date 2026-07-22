"""MyAgent Agent — local-first AI agent kernel.

This package hosts the agent runtime ported/inspired from Cline, OpenCode,
Aider, and Continue.dev. It is intentionally self-contained so it can be
imported without the rest of the MyAgent CLI.

Sub-modules:

* :mod:`myagent.agent.patch` — ``apply_patch`` parser & applier
  (ported from Cline ``apply-patch-parser.ts`` / OpenCode ``patch/index.ts``).
* :mod:`myagent.agent.search_replace` — Aider's SEARCH/REPLACE block
  format with perfect / whitespace-tolerant / ``...``-elision / fuzzy
  fallback matching.
* :mod:`myagent.agent.loop_detection` — repeated-tool-call loop
  detection (ported from Cline ``loop-detection.ts``).
* :mod:`myagent.agent.mistake_tracker` — consecutive-mistake tracker
  with limit callbacks (ported from Cline ``mistake-tracker.ts``).
* :mod:`myagent.agent.tools` — Tool base class + registry +
  truncation service (ported from Cline ``AgentTool`` / OpenCode
  ``Tool.Def`` / OpenCode ``Truncate``).
"""

from __future__ import annotations

from myagent.agent.loop_detection import (
    LoopCheckResult,
    LoopDetectionCall,
    LoopDetectionConfig,
    LoopDetectionState,
    LoopDetectionTracker,
    LoopDetectionVerdict,
    LoopVerdictKind,
    check_repeated_tool_call,
    create_loop_detection_state,
    reset_loop_detection_state,
    tool_call_signature,
)
from myagent.agent.mistake_tracker import (
    AppendRecoveryNotice,
    ConsecutiveMistakeLimitContext,
    ConsecutiveMistakeLimitDecision,
    ContinueDecision,
    ContinueOutcome,
    EmitEvent,
    LeveledLog,
    MistakeOutcome,
    MistakeReason,
    MistakeTracker,
    MistakeTrackerOptions,
    RecordMistakeInput,
    StopDecision,
    StopOutcome,
    build_mistake_limit_stop_message,
    resolve_consecutive_mistake_decision,
)
from myagent.agent.patch import (
    DiffError,
    Patch,
    PatchAction,
    PatchActionType,
    PatchChunk,
    PatchWarning,
    apply_patch_text,
    compute_patch_changes,
    parse_patch,
)
from myagent.agent.search_replace import (
    DEFAULT_FENCE,
    SearchReplaceEdit,
    SearchReplaceError,
    SearchReplaceResult,
    apply_search_replace,
    parse_search_replace,
)
from myagent.agent.tools import (
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
    ToolRegistry,
    ToolResult,
    ToolTimeoutError,
    TruncationResult,
    TruncationService,
    make_error_tool,
    make_invalid_tool,
)

__all__ = [
    "AppendRecoveryNotice",
    "Attachment",
    "ConsecutiveMistakeLimitContext",
    "ConsecutiveMistakeLimitDecision",
    "ContinueDecision",
    "ContinueOutcome",
    "DEFAULT_FENCE",
    "DiffError",
    "EmitEvent",
    "ExecuteFn",
    "InvalidArgumentsError",
    "LeveledLog",
    "LoopCheckResult",
    "LoopDetectionCall",
    "LoopDetectionConfig",
    "LoopDetectionState",
    "LoopDetectionTracker",
    "LoopDetectionVerdict",
    "LoopVerdictKind",
    "MistakeOutcome",
    "MistakeReason",
    "MistakeTracker",
    "MistakeTrackerOptions",
    "Patch",
    "PatchAction",
    "PatchActionType",
    "PatchChunk",
    "PatchWarning",
    "PermissionAsker",
    "PermissionDeniedError",
    "ProgressEmitter",
    "RecordMistakeInput",
    "SearchReplaceEdit",
    "SearchReplaceError",
    "SearchReplaceResult",
    "StopDecision",
    "StopOutcome",
    "Tool",
    "ToolAbortedError",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolTimeoutError",
    "TruncationResult",
    "TruncationService",
    "apply_patch_text",
    "apply_search_replace",
    "build_mistake_limit_stop_message",
    "check_repeated_tool_call",
    "compute_patch_changes",
    "create_loop_detection_state",
    "make_error_tool",
    "make_invalid_tool",
    "parse_patch",
    "parse_search_replace",
    "reset_loop_detection_state",
    "resolve_consecutive_mistake_decision",
    "tool_call_signature",
]
