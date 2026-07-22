"""Tests for the consecutive-mistake tracker (ported from Cline)."""

from __future__ import annotations

import pytest

from myagent.agent.mistake_tracker import (
    AppendRecoveryNotice,
    ConsecutiveMistakeLimitContext,
    ConsecutiveMistakeLimitDecision,
    ContinueDecision,
    ContinueOutcome,
    EmitEvent,
    LeveledLog,
    MistakeReason,
    MistakeTracker,
    MistakeTrackerOptions,
    RecordMistakeInput,
    StopDecision,
    StopOutcome,
    build_mistake_limit_stop_message,
    resolve_consecutive_mistake_decision,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestBuildMistakeLimitStopMessage:
    def test_basic_message(self) -> None:
        msg = build_mistake_limit_stop_message(
            iteration=5,
            consecutive_mistakes=3,
            max_consecutive_mistakes=3,
            reason=MistakeReason.API_ERROR,
        )
        assert "Stopped after 3/3 consecutive mistakes (api_error) at iteration 5." in msg
        assert "Session state was preserved." in msg

    def test_message_includes_details_when_present(self) -> None:
        msg = build_mistake_limit_stop_message(
            iteration=2,
            consecutive_mistakes=3,
            max_consecutive_mistakes=3,
            reason=MistakeReason.TOOL_EXECUTION_FAILED,
            details="permission denied",
        )
        assert "Error: permission denied" in msg

    def test_message_omits_details_when_blank(self) -> None:
        msg = build_mistake_limit_stop_message(
            iteration=2,
            consecutive_mistakes=3,
            max_consecutive_mistakes=3,
            reason=MistakeReason.API_ERROR,
            details="   ",
        )
        assert "Error:" not in msg

    def test_message_includes_stop_reason_when_present(self) -> None:
        msg = build_mistake_limit_stop_message(
            iteration=2,
            consecutive_mistakes=3,
            max_consecutive_mistakes=3,
            reason=MistakeReason.INVALID_TOOL_CALL,
            stop_reason="user chose to stop",
        )
        assert "Decision: user chose to stop" in msg

    def test_message_accepts_string_reason(self) -> None:
        msg = build_mistake_limit_stop_message(
            iteration=1,
            consecutive_mistakes=2,
            max_consecutive_mistakes=2,
            reason="custom_reason",
        )
        assert "(custom_reason)" in msg


class TestResolveConsecutiveMistakeDecision:
    def _ctx(self, **kwargs: object) -> ConsecutiveMistakeLimitContext:
        defaults: dict[str, object] = {
            "iteration": 1,
            "consecutive_mistakes": 3,
            "max_consecutive_mistakes": 3,
            "reason": MistakeReason.API_ERROR,
        }
        defaults.update(kwargs)
        return ConsecutiveMistakeLimitContext(**defaults)  # type: ignore[arg-type]

    def test_no_callback_returns_stop(self) -> None:
        ctx = self._ctx()
        decision = resolve_consecutive_mistake_decision(ctx, None)
        assert isinstance(decision, StopDecision)
        assert decision.reason is not None
        assert "3" in decision.reason

    def test_callback_returning_continue_is_respected(self) -> None:
        ctx = self._ctx()

        def cb(_ctx: ConsecutiveMistakeLimitContext) -> ConsecutiveMistakeLimitDecision:
            return ContinueDecision(guidance="try a smaller scope")

        decision = resolve_consecutive_mistake_decision(ctx, cb)
        assert isinstance(decision, ContinueDecision)
        assert decision.guidance == "try a smaller scope"

    def test_callback_returning_stop_is_respected(self) -> None:
        ctx = self._ctx()

        def cb(_ctx: ConsecutiveMistakeLimitContext) -> ConsecutiveMistakeLimitDecision:
            return StopDecision(reason="user aborted")

        decision = resolve_consecutive_mistake_decision(ctx, cb)
        assert isinstance(decision, StopDecision)
        assert decision.reason == "user aborted"

    def test_callback_raising_returns_stop_with_error_message(self) -> None:
        ctx = self._ctx()

        def cb(_ctx: ConsecutiveMistakeLimitContext) -> ConsecutiveMistakeLimitDecision:
            raise RuntimeError("boom")

        decision = resolve_consecutive_mistake_decision(ctx, cb)
        assert isinstance(decision, StopDecision)
        assert decision.reason == "boom"

    def test_callback_raising_with_empty_message_falls_back(self) -> None:
        ctx = self._ctx()

        class _EmptyError(Exception):
            def __str__(self) -> str:
                return ""

        def cb(_ctx: ConsecutiveMistakeLimitContext) -> ConsecutiveMistakeLimitDecision:
            raise _EmptyError()

        decision = resolve_consecutive_mistake_decision(ctx, cb)
        assert isinstance(decision, StopDecision)
        assert decision.reason is not None
        assert "3" in decision.reason


# ---------------------------------------------------------------------------
# MistakeTracker
# ---------------------------------------------------------------------------


def _make_tracker(
    *,
    max_consecutive_mistakes: int = 3,
    on_limit_reached: object | None = None,
    on_limit_telemetry: object | None = None,
    emit: EmitEvent | None = None,
    log: LeveledLog | None = None,
    append_recovery_notice: AppendRecoveryNotice | None = None,
    agent_id: str = "agent-1",
) -> MistakeTracker:
    options = MistakeTrackerOptions(
        max_consecutive_mistakes=max_consecutive_mistakes,
        agent_id=agent_id,
        get_conversation_id=lambda: "conv-1",
        get_active_run_id=lambda: "run-1",
    )
    if on_limit_reached is not None:
        options.on_limit_reached = on_limit_reached  # type: ignore[assignment]
    if on_limit_telemetry is not None:
        options.on_limit_telemetry = on_limit_telemetry  # type: ignore[assignment]
    if emit is not None:
        options.emit = emit
    if log is not None:
        options.log = log
    if append_recovery_notice is not None:
        options.append_recovery_notice = append_recovery_notice
    return MistakeTracker(options=options)


class TestMistakeTrackerRecord:
    def test_first_mistake_returns_continue(self) -> None:
        tracker = _make_tracker()
        outcome = tracker.record(
            RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR)
        )
        assert outcome.action == "continue"
        assert outcome.guidance is None  # type: ignore[attr-defined]
        assert tracker.value == 1

    def test_returns_continue_until_limit_reached(self) -> None:
        tracker = _make_tracker(max_consecutive_mistakes=3)
        outcome1 = tracker.record(
            RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR)
        )
        outcome2 = tracker.record(
            RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR)
        )
        assert outcome1.action == "continue"
        assert outcome2.action == "continue"
        assert tracker.value == 2

    def test_returns_stop_when_limit_reached_without_callback(self) -> None:
        tracker = _make_tracker(max_consecutive_mistakes=3)
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        tracker.record(RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR))
        outcome = tracker.record(
            RecordMistakeInput(iteration=3, reason=MistakeReason.API_ERROR)
        )
        assert isinstance(outcome, StopOutcome)
        assert outcome.action == "stop"
        assert "Stopped after 3/3" in outcome.message

    def test_stop_outcome_carries_default_reason_when_no_callback(self) -> None:
        tracker = _make_tracker(max_consecutive_mistakes=2)
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        outcome = tracker.record(
            RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR)
        )
        assert isinstance(outcome, StopOutcome)
        assert outcome.reason is not None
        assert "maximum consecutive mistakes reached" in outcome.reason

    def test_continue_callback_resets_counter_and_returns_guidance(self) -> None:
        def cb(_ctx: ConsecutiveMistakeLimitContext) -> ConsecutiveMistakeLimitDecision:
            return ContinueDecision(guidance="try a different model")

        tracker = _make_tracker(max_consecutive_mistakes=2, on_limit_reached=cb)
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        outcome = tracker.record(
            RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR)
        )
        assert isinstance(outcome, ContinueOutcome)
        assert outcome.action == "continue"
        assert outcome.guidance == "try a different model"
        # Counter reset after a continue decision.
        assert tracker.value == 0

    def test_stop_callback_returns_stop_with_reason(self) -> None:
        def cb(_ctx: ConsecutiveMistakeLimitContext) -> ConsecutiveMistakeLimitDecision:
            return StopDecision(reason="user aborted")

        tracker = _make_tracker(max_consecutive_mistakes=2, on_limit_reached=cb)
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        outcome = tracker.record(
            RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR)
        )
        assert isinstance(outcome, StopOutcome)
        assert outcome.action == "stop"
        assert outcome.reason == "user aborted"
        assert "user aborted" in outcome.message
        # Counter NOT reset after a stop decision.
        assert tracker.value == 2

    def test_callback_raising_falls_back_to_stop(self) -> None:
        def cb(_ctx: ConsecutiveMistakeLimitContext) -> ConsecutiveMistakeLimitDecision:
            raise RuntimeError("callback blew up")

        tracker = _make_tracker(max_consecutive_mistakes=2, on_limit_reached=cb)
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        outcome = tracker.record(
            RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR)
        )
        assert isinstance(outcome, StopOutcome)
        assert outcome.action == "stop"
        assert outcome.reason == "callback blew up"

    def test_force_at_limit_jumps_straight_to_max(self) -> None:
        tracker = _make_tracker(max_consecutive_mistakes=3)
        outcome = tracker.record(
            RecordMistakeInput(
                iteration=1,
                reason=MistakeReason.API_ERROR,
                force_at_limit=True,
            )
        )
        # Skipped 1 and 2, jumped to 3 → stop.
        assert outcome.action == "stop"
        assert tracker.value == 3

    def test_force_at_limit_with_zero_max_just_increments(self) -> None:
        # Edge case: max=0 disables the limit (no stop), so force_at_limit
        # falls through to the normal +1 increment.
        tracker = _make_tracker(max_consecutive_mistakes=0)
        outcome = tracker.record(
            RecordMistakeInput(
                iteration=1,
                reason=MistakeReason.API_ERROR,
                force_at_limit=True,
            )
        )
        assert outcome.action == "continue"
        assert tracker.value == 1

    def test_zero_max_disables_limit(self) -> None:
        tracker = _make_tracker(max_consecutive_mistakes=0)
        for i in range(10):
            outcome = tracker.record(
                RecordMistakeInput(iteration=i + 1, reason=MistakeReason.API_ERROR)
            )
            assert outcome.action == "continue"
        assert tracker.value == 10


class TestMistakeTrackerReset:
    def test_reset_clears_counter(self) -> None:
        tracker = _make_tracker(max_consecutive_mistakes=5)
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        tracker.record(RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR))
        assert tracker.value == 2
        tracker.reset()
        assert tracker.value == 0
        outcome = tracker.record(
            RecordMistakeInput(iteration=3, reason=MistakeReason.API_ERROR)
        )
        assert outcome.action == "continue"
        assert tracker.value == 1


class TestMistakeTrackerSideEffects:
    def test_emit_called_on_each_record(self) -> None:
        events: list[dict[str, object]] = []

        def emit(event: dict[str, object]) -> None:
            events.append(event)

        tracker = _make_tracker(max_consecutive_mistakes=5, emit=emit)
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        tracker.record(
            RecordMistakeInput(iteration=2, reason=MistakeReason.INVALID_TOOL_CALL)
        )
        assert len(events) == 2
        assert events[0]["type"] == "error"
        assert events[0]["recoverable"] is True
        assert events[0]["iteration"] == 1
        assert events[1]["iteration"] == 2

    def test_log_called_with_metadata(self) -> None:
        log_entries: list[tuple[str, str, dict[str, object]]] = []

        def log(level: str, message: str, metadata: dict[str, object]) -> None:
            log_entries.append((level, message, metadata))

        tracker = _make_tracker(max_consecutive_mistakes=5, log=log)
        tracker.record(
            RecordMistakeInput(
                iteration=3,
                reason=MistakeReason.API_ERROR,
                details="timeout",
            )
        )
        assert len(log_entries) == 1
        level, message, metadata = log_entries[0]
        assert level == "warn"
        assert "consecutive mistake" in message
        assert metadata["agent_id"] == "agent-1"
        assert metadata["conversation_id"] == "conv-1"
        assert metadata["run_id"] == "run-1"
        assert metadata["iteration"] == 3
        assert metadata["reason"] == "api_error"
        assert metadata["details"] == "timeout"
        assert metadata["consecutive_mistakes"] == 1
        assert metadata["max_consecutive_mistakes"] == 5

    def test_on_limit_telemetry_fires_once_when_limit_hit(self) -> None:
        telemetry_calls: list[ConsecutiveMistakeLimitContext] = []

        def on_telemetry(ctx: ConsecutiveMistakeLimitContext) -> None:
            telemetry_calls.append(ctx)

        tracker = _make_tracker(
            max_consecutive_mistakes=3,
            on_limit_telemetry=on_telemetry,
        )
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        tracker.record(RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR))
        # Before the limit, telemetry should not have fired.
        assert telemetry_calls == []
        tracker.record(RecordMistakeInput(iteration=3, reason=MistakeReason.API_ERROR))
        assert len(telemetry_calls) == 1
        ctx = telemetry_calls[0]
        assert ctx.iteration == 3
        assert ctx.consecutive_mistakes == 3
        assert ctx.max_consecutive_mistakes == 3
        assert ctx.reason == MistakeReason.API_ERROR

    def test_append_recovery_notice_called_with_guidance(self) -> None:
        def cb(_ctx: ConsecutiveMistakeLimitContext) -> ConsecutiveMistakeLimitDecision:
            return ContinueDecision(guidance="try a smaller scope")

        notices: list[tuple[str, MistakeReason]] = []

        def append_notice(message: str, reason: MistakeReason) -> None:
            notices.append((message, reason))

        tracker = _make_tracker(
            max_consecutive_mistakes=2,
            on_limit_reached=cb,
            append_recovery_notice=append_notice,
        )
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        tracker.record(
            RecordMistakeInput(
                iteration=2,
                reason=MistakeReason.INVALID_TOOL_CALL,
            )
        )
        assert len(notices) == 1
        message, reason = notices[0]
        assert message == "try a smaller scope"
        assert reason == MistakeReason.INVALID_TOOL_CALL

    def test_append_recovery_notice_not_called_without_guidance(self) -> None:
        def cb(_ctx: ConsecutiveMistakeLimitContext) -> ConsecutiveMistakeLimitDecision:
            return ContinueDecision()

        notices: list[tuple[str, MistakeReason]] = []

        def append_notice(message: str, reason: MistakeReason) -> None:
            notices.append((message, reason))

        tracker = _make_tracker(
            max_consecutive_mistakes=2,
            on_limit_reached=cb,
            append_recovery_notice=append_notice,
        )
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        tracker.record(RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR))
        # ContinueDecision without guidance → no notice appended.
        assert notices == []

    def test_append_recovery_notice_strips_whitespace(self) -> None:
        def cb(_ctx: ConsecutiveMistakeLimitContext) -> ConsecutiveMistakeLimitDecision:
            return ContinueDecision(guidance="  padded guidance  ")

        notices: list[tuple[str, MistakeReason]] = []

        def append_notice(message: str, reason: MistakeReason) -> None:
            notices.append((message, reason))

        tracker = _make_tracker(
            max_consecutive_mistakes=2,
            on_limit_reached=cb,
            append_recovery_notice=append_notice,
        )
        tracker.record(RecordMistakeInput(iteration=1, reason=MistakeReason.API_ERROR))
        tracker.record(RecordMistakeInput(iteration=2, reason=MistakeReason.API_ERROR))
        assert notices == [("padded guidance", MistakeReason.API_ERROR)]


class TestMistakeTrackerDetails:
    def test_details_used_as_error_message_when_present(self) -> None:
        events: list[dict[str, object]] = []

        def emit(event: dict[str, object]) -> None:
            events.append(event)

        tracker = _make_tracker(max_consecutive_mistakes=5, emit=emit)
        tracker.record(
            RecordMistakeInput(
                iteration=1,
                reason=MistakeReason.TOOL_EXECUTION_FAILED,
                details="non-zero exit code 1",
            )
        )
        event = events[0]
        error = event["error"]
        assert isinstance(error, Exception)
        assert "non-zero exit code 1" in str(error)

    def test_details_blank_falls_back_to_reason_message(self) -> None:
        events: list[dict[str, object]] = []

        def emit(event: dict[str, object]) -> None:
            events.append(event)

        tracker = _make_tracker(max_consecutive_mistakes=5, emit=emit)
        tracker.record(
            RecordMistakeInput(
                iteration=1,
                reason=MistakeReason.API_ERROR,
                details="   ",
            )
        )
        event = events[0]
        error = event["error"]
        assert isinstance(error, Exception)
        assert "consecutive mistake (api_error)" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
