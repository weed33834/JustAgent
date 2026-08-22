"""Per-session consecutive-mistake tracker.

Ports Cline's :class:`MistakeTracker` to Python. The tracker counts
consecutive mistakes (API errors, invalid tool calls, tool execution
failures) and stops the agent once ``max_consecutive_mistakes`` is
reached. Callers can install an ``on_limit_reached`` callback to
override the default stop decision (e.g. ask the user whether to keep
going) and an ``on_limit_telemetry`` hook for observability.

Reference: ``competitors/cline/sdk/packages/core/src/runtime/safety/mistake-tracker.ts``.

The Python port keeps the same public surface (``record`` / ``reset``
/ ``value``) and the same outcome semantics:

* ``action="continue"`` — agent should keep going (optionally with
  ``guidance`` text the runtime can append as a recovery notice).
* ``action="stop"`` — agent should stop the run with ``message`` and
  optional ``reason``.

Unlike the TypeScript original (which is async because the limit
callback may return a Promise), the Python port is synchronous.
Callers that need an async decision (e.g. prompting the user via a
TUI) should perform that interaction before invoking :meth:`record`
and pass the decision via ``on_limit_reached``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class MistakeReason(str, Enum):  # noqa: UP042 - match existing codebase style
    """Why a mistake is being recorded."""

    API_ERROR = "api_error"
    INVALID_TOOL_CALL = "invalid_tool_call"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"


@dataclass(frozen=True)
class RecordMistakeInput:
    """Input to :meth:`MistakeTracker.record`.

    ``iteration`` is the current agent-loop iteration (1-based). When
    ``force_at_limit`` is true and ``max_consecutive_mistakes`` > 0,
    the counter jumps straight to the limit instead of incrementing by
    one — useful for unrecoverable errors where retrying is pointless.
    """

    iteration: int
    reason: MistakeReason
    details: str | None = None
    force_at_limit: bool = False


@dataclass(frozen=True)
class ConsecutiveMistakeLimitContext:
    """Read-only context passed to limit callbacks."""

    iteration: int
    consecutive_mistakes: int
    max_consecutive_mistakes: int
    reason: MistakeReason
    details: str | None = None


@dataclass(frozen=True)
class ContinueDecision:
    """Decision to keep the run going after a limit hit."""

    action: Literal["continue"] = "continue"
    guidance: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class StopDecision:
    """Decision to stop the run after a limit hit."""

    action: Literal["stop"] = "stop"
    reason: str | None = None


ConsecutiveMistakeLimitDecision = ContinueDecision | StopDecision


@dataclass(frozen=True)
class ContinueOutcome:
    """Outcome directing the runtime to keep running."""

    action: Literal["continue"] = "continue"
    guidance: str | None = None


@dataclass(frozen=True)
class StopOutcome:
    """Outcome directing the runtime to stop the run."""

    message: str
    action: Literal["stop"] = "stop"
    reason: str | None = None


MistakeOutcome = ContinueOutcome | StopOutcome


#: Leveled log callable: ``(level, message, metadata) -> None``.
LeveledLog = Callable[[str, str, "dict[str, Any]"], None]

#: Event-emit callable: the runtime bridges this to its event stream.
#: The event is a plain dict so this module stays decoupled from the
#: eventual agent-event schema.
EmitEvent = Callable[["dict[str, Any]"], None]

#: Recovery-notice callable: ``(message, reason) -> None``. The runtime
#: appends the notice to the conversation so the model sees it on the
#: next iteration.
AppendRecoveryNotice = Callable[[str, MistakeReason], None]


def _noop_log(level: str, message: str, metadata: dict[str, Any]) -> None:
    """Default no-op log sink."""


def _noop_emit(event: dict[str, Any]) -> None:
    """Default no-op event sink."""


def _noop_append_recovery_notice(message: str, reason: MistakeReason) -> None:
    """Default no-op recovery-notice sink."""


def _noop_get_id() -> str:
    """Default no-op ID getter."""

    return ""


@dataclass
class MistakeTrackerOptions:
    """Configuration + dependency-injection surface for :class:`MistakeTracker`.

    The callbacks default to no-ops so a tracker can be constructed
    with just ``max_consecutive_mistakes`` for testing. Production
    callers should provide ``emit``/``log``/``append_recovery_notice``
    for observability and recovery-notice parity with Cline.
    """

    max_consecutive_mistakes: int
    agent_id: str = ""
    get_conversation_id: Callable[[], str] = _noop_get_id
    get_active_run_id: Callable[[], str] = _noop_get_id
    on_limit_reached: (
        Callable[[ConsecutiveMistakeLimitContext], ConsecutiveMistakeLimitDecision] | None
    ) = None
    on_limit_telemetry: Callable[[ConsecutiveMistakeLimitContext], None] | None = None
    emit: EmitEvent = _noop_emit
    log: LeveledLog = _noop_log
    append_recovery_notice: AppendRecoveryNotice = _noop_append_recovery_notice


# ---------------------------------------------------------------------------
# Pure helpers (ported verbatim)
# ---------------------------------------------------------------------------


def build_mistake_limit_stop_message(
    *,
    iteration: int,
    consecutive_mistakes: int,
    max_consecutive_mistakes: int,
    reason: MistakeReason | str,
    details: str | None = None,
    stop_reason: str | None = None,
) -> str:
    """Format the stop message shown to the user when the run is halted.

    Ported verbatim from ``buildMistakeLimitStopMessage``.
    """

    # Normalize enum → value so the message uses ``api_error`` instead
    # of ``MistakeReason.API_ERROR`` (Python 3.11+ ``str(Enum)`` returns
    # the qualified name, not the value).
    reason_str = reason.value if isinstance(reason, MistakeReason) else str(reason)
    parts: list[str] = [
        f"Stopped after {consecutive_mistakes}/{max_consecutive_mistakes} "
        f"consecutive mistakes ({reason_str}) at iteration {iteration}."
    ]
    trimmed_details = details.strip() if details else ""
    if trimmed_details:
        parts.append(f"Error: {trimmed_details}")
    trimmed_stop_reason = stop_reason.strip() if stop_reason else ""
    if trimmed_stop_reason:
        parts.append(f"Decision: {trimmed_stop_reason}")
    parts.append("Session state was preserved. Send a new prompt to resume from the latest state.")
    return " ".join(parts)


def resolve_consecutive_mistake_decision(
    context: ConsecutiveMistakeLimitContext,
    callback: (Callable[[ConsecutiveMistakeLimitContext], ConsecutiveMistakeLimitDecision] | None),
) -> ConsecutiveMistakeLimitDecision:
    """Resolve the limit decision, defaulting to ``stop`` if no callback.

    If the callback raises, the decision is also ``stop`` with the
    error message as the reason — mirroring Cline's try/catch behavior.
    """

    if callback is None:
        return StopDecision(
            reason=(f"maximum consecutive mistakes reached ({context.max_consecutive_mistakes})")
        )
    try:
        return callback(context)
    except Exception as exc:  # noqa: BLE001 — mirror Cline's catch-all
        return StopDecision(
            reason=(
                str(exc)
                if str(exc)
                else (f"maximum consecutive mistakes reached ({context.max_consecutive_mistakes})")
            )
        )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


@dataclass
class MistakeTracker:
    """Per-session consecutive-mistake tracker.

    The agent runtime owns an instance and calls :meth:`record` whenever
    a recoverable mistake occurs (API error, malformed tool call,
    tool-execution failure). The tracker decides whether the agent
    should keep going or stop, and emits the appropriate observability
    events / log entries / recovery notices along the way.

    Example:

    >>> tracker = MistakeTracker(
    ...     options=MistakeTrackerOptions(max_consecutive_mistakes=3)
    ... )
    >>> tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
    ContinueOutcome(action='continue', guidance=None)
    >>> tracker.record(RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR))
    ContinueOutcome(action='continue', guidance=None)
    >>> outcome = tracker.record(
    ...     RecordMistakeInput(iteration=3, reason=MistakeReason.API_ERROR)
    ... )
    >>> outcome.action
    'stop'
    """

    options: MistakeTrackerOptions
    _consecutive_mistakes: int = 0

    def record(self, input_data: RecordMistakeInput) -> MistakeOutcome:
        """Record one mistake and return the outcome.

        Side effects (when configured):

        * Emits a recoverable ``"error"`` event with the iteration.
        * Logs a ``"warn"``-level entry with run/conversation metadata.
        * When the limit is reached: fires ``on_limit_telemetry``
          exactly once, then resolves the decision via
          ``on_limit_reached`` (or defaults to ``stop``). If the
          decision is ``continue`` with guidance, the guidance is
          appended as a recovery notice and the counter resets.
        """

        max_count = self.options.max_consecutive_mistakes
        if input_data.force_at_limit and max_count:
            next_count = max_count
        else:
            next_count = self._consecutive_mistakes + 1
        self._consecutive_mistakes = next_count

        error_message = (
            input_data.details or ""
        ).strip() or f"consecutive mistake ({input_data.reason.value})"
        self.options.emit(
            {
                "type": "error",
                "error": Exception(error_message),
                "recoverable": True,
                "iteration": input_data.iteration,
            }
        )
        self.options.log(
            "warn",
            "Recorded consecutive mistake",
            {
                "agent_id": self.options.agent_id,
                "conversation_id": self.options.get_conversation_id(),
                "run_id": self.options.get_active_run_id(),
                "iteration": input_data.iteration,
                "reason": input_data.reason.value,
                "details": input_data.details,
                "consecutive_mistakes": next_count,
                "max_consecutive_mistakes": self.options.max_consecutive_mistakes,
            },
        )

        if not max_count or next_count < max_count:
            return ContinueOutcome()

        limit_context = ConsecutiveMistakeLimitContext(
            iteration=input_data.iteration,
            consecutive_mistakes=next_count,
            max_consecutive_mistakes=max_count,
            reason=input_data.reason,
            details=input_data.details,
        )
        if self.options.on_limit_telemetry is not None:
            self.options.on_limit_telemetry(limit_context)

        decision = resolve_consecutive_mistake_decision(
            limit_context,
            self.options.on_limit_reached,
        )

        if isinstance(decision, ContinueDecision):
            guidance = decision.guidance.strip() if decision.guidance else None
            if guidance:
                self.options.append_recovery_notice(guidance, input_data.reason)
            self._consecutive_mistakes = 0
            return ContinueOutcome(guidance=guidance)

        return StopOutcome(
            reason=decision.reason,
            message=build_mistake_limit_stop_message(
                iteration=input_data.iteration,
                consecutive_mistakes=next_count,
                max_consecutive_mistakes=max_count,
                reason=input_data.reason,
                details=input_data.details,
                stop_reason=decision.reason,
            ),
        )

    def reset(self) -> None:
        """Clear the consecutive-mistake counter."""

        self._consecutive_mistakes = 0

    @property
    def value(self) -> int:
        """Current consecutive-mistake count (for observability)."""

        return self._consecutive_mistakes


__all__ = [
    "AppendRecoveryNotice",
    "ConsecutiveMistakeLimitContext",
    "ConsecutiveMistakeLimitDecision",
    "ContinueDecision",
    "ContinueOutcome",
    "EmitEvent",
    "LeveledLog",
    "MistakeOutcome",
    "MistakeReason",
    "MistakeTracker",
    "MistakeTrackerOptions",
    "RecordMistakeInput",
    "StopDecision",
    "StopOutcome",
    "build_mistake_limit_stop_message",
    "resolve_consecutive_mistake_decision",
]
