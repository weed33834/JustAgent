"""Repeated tool-call loop detection.

Ports Cline's loop-detection safety primitive to Python. The tracker
detects when an agent makes the same tool call (same name + same input
signature) repeatedly and surfaces a soft warning (so the runtime can
inject a recovery notice) before hard-escalating to a stop (so the
agent doesn't burn tokens in an infinite loop).

Reference: ``competitors/cline/sdk/packages/core/src/runtime/safety/loop-detection.ts``.

The pure helpers (:func:`create_loop_detection_state`,
:func:`reset_loop_detection_state`, :func:`tool_call_signature`,
:func:`check_repeated_tool_call`) are direct ports. The
:class:`LoopDetectionTracker` class wraps a state and exposes the
``inspect()`` / ``reset()`` surface that the agent runtime installs as
a ``before_tool`` hook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Pure helpers (verbatim port)
# ---------------------------------------------------------------------------


@dataclass
class LoopDetectionState:
    """Mutable per-session loop-detection state."""

    last_tool_name: str = ""
    last_tool_signature: str = ""
    consecutive_identical_count: int = 0


def create_loop_detection_state() -> LoopDetectionState:
    """Return a fresh loop-detection state."""

    return LoopDetectionState()


def reset_loop_detection_state(state: LoopDetectionState) -> None:
    """Reset ``state`` in place to its initial values."""

    state.last_tool_name = ""
    state.last_tool_signature = ""
    state.consecutive_identical_count = 0


def _sort_keys(value: Any) -> Any:
    """Recursively sort dict keys so signatures are stable."""

    if isinstance(value, dict):
        return {k: _sort_keys(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, list):
        return [_sort_keys(v) for v in value]
    return value


def tool_call_signature(input_value: Any) -> str:
    """Return a stable string signature for a tool-call input.

    Strings are returned as-is, ``None`` becomes ``"null"``, and
    dicts/lists are JSON-serialized with sorted keys so semantically
    identical inputs (regardless of key order) hash to the same
    signature. Non-serializable values fall back to :func:`str`.
    """

    if input_value is None:
        return "null"
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, (dict, list)):
        try:
            return json.dumps(_sort_keys(input_value), default=str)
        except (TypeError, ValueError):
            return str(input_value)
    return str(input_value)


@dataclass(frozen=True)
class LoopDetectionConfig:
    """Thresholds for soft warning and hard escalation.

    A soft warning fires exactly once when ``consecutive_identical_count``
    reaches ``soft_threshold``. A hard escalation fires whenever the
    count is ``>= hard_threshold``.
    """

    soft_threshold: int = 3
    hard_threshold: int = 5


@dataclass(frozen=True)
class LoopCheckResult:
    """Raw result of :func:`check_repeated_tool_call`."""

    soft_warning: bool
    hard_escalation: bool


def check_repeated_tool_call(
    state: LoopDetectionState,
    tool_name: str,
    signature: str,
    config: LoopDetectionConfig,
) -> LoopCheckResult:
    """Update ``state`` for one tool call and return the verdict."""

    if tool_name == state.last_tool_name and signature == state.last_tool_signature:
        state.consecutive_identical_count += 1
    else:
        state.consecutive_identical_count = 1
    state.last_tool_name = tool_name
    state.last_tool_signature = signature

    return LoopCheckResult(
        soft_warning=state.consecutive_identical_count == config.soft_threshold,
        hard_escalation=state.consecutive_identical_count >= config.hard_threshold,
    )


# ---------------------------------------------------------------------------
# Class wrapper (new — per Cline PLAN.md §3.2.3)
# ---------------------------------------------------------------------------

#: Verdict kind: ``"ok"`` (no repeated call), ``"soft"`` (soft warning),
#: or ``"hard"`` (hard escalation — runtime should stop the run).
LoopVerdictKind = str  # one of "ok" | "soft" | "hard"


@dataclass(frozen=True)
class LoopDetectionVerdict:
    """Verdict returned by :meth:`LoopDetectionTracker.inspect`."""

    kind: LoopVerdictKind
    message: str | None = None


@dataclass(frozen=True)
class LoopDetectionCall:
    """Minimal call shape the tracker needs."""

    name: str
    input: Any = None


@dataclass
class LoopDetectionTracker:
    """Per-session repeated-tool-call detector.

    The agent runtime owns an instance and installs a ``before_tool``
    hook that calls :meth:`inspect` to decide whether to skip the call
    (soft warning) or stop the run (hard escalation).

    Example:

    >>> tracker = LoopDetectionTracker()
    >>> tracker.inspect(LoopDetectionCall(name="read_file", input={"path": "a"}))
    LoopDetectionVerdict(kind='ok', message=None)
    >>> tracker.inspect(LoopDetectionCall(name="read_file", input={"path": "a"}))
    LoopDetectionVerdict(kind='ok', message=None)
    >>> tracker.inspect(LoopDetectionCall(name="read_file", input={"path": "a"}))
    LoopDetectionVerdict(kind='soft', message=...)
    >>> tracker.reset()
    >>> tracker.inspect(LoopDetectionCall(name="read_file", input={"path": "a"}))
    LoopDetectionVerdict(kind='ok', message=None)
    """

    config: LoopDetectionConfig = field(default_factory=LoopDetectionConfig)
    _state: LoopDetectionState = field(default_factory=create_loop_detection_state)

    def inspect(self, call: LoopDetectionCall) -> LoopDetectionVerdict:
        """Inspect one tool call and return the verdict.

        Updates internal state so the next call can be compared against
        this one. Returns:

        * ``kind="ok"`` — no threshold reached.
        * ``kind="soft"`` — soft threshold reached exactly; runtime may
          surface a recovery notice but should not block the call.
        * ``kind="hard"`` — hard threshold reached; runtime should stop
          the run with the provided message.
        """

        signature = tool_call_signature(call.input)
        result = check_repeated_tool_call(
            self._state,
            call.name,
            signature,
            self.config,
        )
        if result.hard_escalation:
            return LoopDetectionVerdict(
                kind="hard",
                message=(
                    f"Detected {self._state.consecutive_identical_count} consecutive "
                    f"identical calls to `{call.name}`; stopping to avoid a loop."
                ),
            )
        if result.soft_warning:
            return LoopDetectionVerdict(
                kind="soft",
                message=(
                    f"Detected {self._state.consecutive_identical_count} consecutive "
                    f"identical calls to `{call.name}`; consider trying a different approach."
                ),
            )
        return LoopDetectionVerdict(kind="ok")

    def reset(self) -> None:
        """Clear internal state."""

        reset_loop_detection_state(self._state)

    @property
    def consecutive_identical_count(self) -> int:
        """Current consecutive-identical-call count (for observability)."""

        return self._state.consecutive_identical_count
