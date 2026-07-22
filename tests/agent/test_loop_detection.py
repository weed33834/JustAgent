"""Tests for the loop-detection tracker (ported from Cline)."""

from __future__ import annotations

from autoship.agent.loop_detection import (
    LoopDetectionCall,
    LoopDetectionConfig,
    LoopDetectionState,
    LoopDetectionTracker,
    check_repeated_tool_call,
    create_loop_detection_state,
    reset_loop_detection_state,
    tool_call_signature,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestLoopDetectionState:
    def test_create_initial_state(self) -> None:
        state = create_loop_detection_state()
        assert state.last_tool_name == ""
        assert state.last_tool_signature == ""
        assert state.consecutive_identical_count == 0

    def test_reset_clears_state(self) -> None:
        state = LoopDetectionState(
            last_tool_name="read_file",
            last_tool_signature='{"path":"a"}',
            consecutive_identical_count=3,
        )
        reset_loop_detection_state(state)
        assert state.last_tool_name == ""
        assert state.last_tool_signature == ""
        assert state.consecutive_identical_count == 0


class TestToolCallSignature:
    def test_none(self) -> None:
        assert tool_call_signature(None) == "null"

    def test_string(self) -> None:
        assert tool_call_signature("hello") == "hello"

    def test_int(self) -> None:
        assert tool_call_signature(42) == "42"

    def test_dict_with_sorted_keys(self) -> None:
        sig_a = tool_call_signature({"b": 2, "a": 1})
        sig_b = tool_call_signature({"a": 1, "b": 2})
        assert sig_a == sig_b
        assert '"a"' in sig_a
        assert '"b"' in sig_a

    def test_nested_dict_with_sorted_keys(self) -> None:
        sig_a = tool_call_signature({"outer": {"z": 1, "a": 2}})
        sig_b = tool_call_signature({"outer": {"a": 2, "z": 1}})
        assert sig_a == sig_b

    def test_list(self) -> None:
        sig = tool_call_signature([1, 2, 3])
        assert sig == "[1, 2, 3]"

    def test_dict_with_non_serializable_falls_back(self) -> None:
        # `default=str` converts non-serializable values to str.
        sig = tool_call_signature({"obj": object()})
        assert "object" in sig


class TestCheckRepeatedToolCall:
    def _config(self, soft: int = 3, hard: int = 5) -> LoopDetectionConfig:
        return LoopDetectionConfig(soft_threshold=soft, hard_threshold=hard)

    def test_first_call_count_is_one_no_warnings(self) -> None:
        state = create_loop_detection_state()
        result = check_repeated_tool_call(state, "read_file", "sig1", self._config())
        assert state.consecutive_identical_count == 1
        assert not result.soft_warning
        assert not result.hard_escalation

    def test_second_identical_call_count_is_two(self) -> None:
        state = create_loop_detection_state()
        check_repeated_tool_call(state, "read_file", "sig1", self._config())
        result = check_repeated_tool_call(state, "read_file", "sig1", self._config())
        assert state.consecutive_identical_count == 2
        assert not result.soft_warning
        assert not result.hard_escalation

    def test_third_identical_call_soft_warning(self) -> None:
        state = create_loop_detection_state()
        for _ in range(2):
            check_repeated_tool_call(state, "read_file", "sig1", self._config())
        result = check_repeated_tool_call(state, "read_file", "sig1", self._config())
        assert state.consecutive_identical_count == 3
        assert result.soft_warning
        assert not result.hard_escalation

    def test_fifth_identical_call_hard_escalation(self) -> None:
        state = create_loop_detection_state()
        for _ in range(4):
            check_repeated_tool_call(state, "read_file", "sig1", self._config())
        result = check_repeated_tool_call(state, "read_file", "sig1", self._config())
        assert state.consecutive_identical_count == 5
        assert result.hard_escalation
        # Soft warning only fires *exactly* at threshold (3), not at 5.
        assert not result.soft_warning

    def test_sixth_identical_call_still_hard_escalation(self) -> None:
        state = create_loop_detection_state()
        for _ in range(5):
            check_repeated_tool_call(state, "read_file", "sig1", self._config())
        result = check_repeated_tool_call(state, "read_file", "sig1", self._config())
        assert state.consecutive_identical_count == 6
        assert result.hard_escalation

    def test_different_tool_name_resets_count(self) -> None:
        state = create_loop_detection_state()
        check_repeated_tool_call(state, "read_file", "sig1", self._config())
        check_repeated_tool_call(state, "read_file", "sig1", self._config())
        result = check_repeated_tool_call(state, "write_file", "sig1", self._config())
        assert state.consecutive_identical_count == 1
        assert not result.soft_warning
        assert not result.hard_escalation

    def test_same_name_different_signature_resets_count(self) -> None:
        state = create_loop_detection_state()
        check_repeated_tool_call(state, "read_file", "sig1", self._config())
        check_repeated_tool_call(state, "read_file", "sig1", self._config())
        result = check_repeated_tool_call(state, "read_file", "sig2", self._config())
        assert state.consecutive_identical_count == 1
        assert not result.soft_warning
        assert not result.hard_escalation

    def test_zero_max_means_no_warnings(self) -> None:
        # Edge case: soft=0 would warn on count==0, but count starts at 1.
        # hard=0 means count>=0 always escalates — used to disable hard stop
        # when the runtime wants only soft warnings.
        state = create_loop_detection_state()
        config = LoopDetectionConfig(soft_threshold=3, hard_threshold=0)
        result = check_repeated_tool_call(state, "read_file", "sig1", config)
        assert state.consecutive_identical_count == 1
        assert result.hard_escalation  # 1 >= 0


# ---------------------------------------------------------------------------
# Class wrapper
# ---------------------------------------------------------------------------


class TestLoopDetectionTracker:
    def test_inspect_first_call_ok(self) -> None:
        tracker = LoopDetectionTracker()
        verdict = tracker.inspect(LoopDetectionCall(name="read_file", input={"path": "a"}))
        assert verdict.kind == "ok"
        assert verdict.message is None
        assert tracker.consecutive_identical_count == 1

    def test_inspect_second_identical_call_ok(self) -> None:
        tracker = LoopDetectionTracker()
        call = LoopDetectionCall(name="read_file", input={"path": "a"})
        tracker.inspect(call)
        verdict = tracker.inspect(call)
        assert verdict.kind == "ok"
        assert tracker.consecutive_identical_count == 2

    def test_inspect_soft_warning_at_threshold(self) -> None:
        tracker = LoopDetectionTracker()
        call = LoopDetectionCall(name="read_file", input={"path": "a"})
        tracker.inspect(call)
        tracker.inspect(call)
        verdict = tracker.inspect(call)
        assert verdict.kind == "soft"
        assert verdict.message is not None
        assert "3 consecutive" in verdict.message
        assert "read_file" in verdict.message

    def test_inspect_hard_escalation_at_hard_threshold(self) -> None:
        tracker = LoopDetectionTracker()
        call = LoopDetectionCall(name="read_file", input={"path": "a"})
        for _ in range(4):
            tracker.inspect(call)
        verdict = tracker.inspect(call)
        assert verdict.kind == "hard"
        assert verdict.message is not None
        assert "5 consecutive" in verdict.message
        assert "stopping to avoid a loop" in verdict.message

    def test_inspect_hard_escalation_persists_above_threshold(self) -> None:
        tracker = LoopDetectionTracker()
        call = LoopDetectionCall(name="read_file", input={"path": "a"})
        for _ in range(5):
            tracker.inspect(call)
        verdict = tracker.inspect(call)
        assert verdict.kind == "hard"

    def test_inspect_different_call_resets(self) -> None:
        tracker = LoopDetectionTracker()
        call_a = LoopDetectionCall(name="read_file", input={"path": "a"})
        call_b = LoopDetectionCall(name="read_file", input={"path": "b"})
        tracker.inspect(call_a)
        tracker.inspect(call_a)
        tracker.inspect(call_a)  # soft warning
        verdict = tracker.inspect(call_b)
        assert verdict.kind == "ok"
        assert tracker.consecutive_identical_count == 1

    def test_reset_clears_state(self) -> None:
        tracker = LoopDetectionTracker()
        call = LoopDetectionCall(name="read_file", input={"path": "a"})
        for _ in range(4):
            tracker.inspect(call)
        assert tracker.consecutive_identical_count == 4
        tracker.reset()
        assert tracker.consecutive_identical_count == 0
        verdict = tracker.inspect(call)
        assert verdict.kind == "ok"
        assert tracker.consecutive_identical_count == 1

    def test_custom_config_overrides_defaults(self) -> None:
        config = LoopDetectionConfig(soft_threshold=2, hard_threshold=4)
        tracker = LoopDetectionTracker(config=config)
        call = LoopDetectionCall(name="read_file", input={"path": "a"})
        tracker.inspect(call)
        verdict = tracker.inspect(call)
        assert verdict.kind == "soft"  # threshold=2

    def test_input_signature_normalizes_key_order(self) -> None:
        tracker = LoopDetectionTracker()
        call_a = LoopDetectionCall(name="edit_file", input={"path": "a", "line": 1})
        call_b = LoopDetectionCall(name="edit_file", input={"line": 1, "path": "a"})
        tracker.inspect(call_a)
        tracker.inspect(call_b)
        # Same signature → count is 2.
        assert tracker.consecutive_identical_count == 2

    def test_message_mentions_tool_name(self) -> None:
        tracker = LoopDetectionTracker()
        call = LoopDetectionCall(name="run_command", input={"cmd": "ls"})
        for _ in range(2):
            tracker.inspect(call)
        verdict = tracker.inspect(call)
        assert verdict.kind == "soft"
        assert "run_command" in (verdict.message or "")
