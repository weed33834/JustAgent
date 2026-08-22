"""Tests for ``justagent.agent.compaction`` (context compaction)."""

from __future__ import annotations

import asyncio

from justagent.agent.compaction import (
    CompactionConfig,
    CompactionResult,
    Compactor,
)
from justagent.agent.runtime import LLMClient, LLMRequest, LLMResponse, Message, ToolCall

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeLLMClient(LLMClient):
    """Test double that returns a scripted LLM response."""

    def __init__(self, response: LLMResponse) -> None:
        # Skip parent __init__ — no real API credentials needed.
        self._response = response
        self.calls: list[LLMRequest] = []

    async def complete(
        self,
        request: LLMRequest,
        *,
        abort: asyncio.Event | None = None,
    ) -> LLMResponse:
        self.calls.append(request)
        return self._response


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)  # type: ignore[arg-type]


def _many_messages(count: int) -> list[Message]:
    """Return ``count`` distinct user/assistant messages."""

    messages: list[Message] = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(Message(role=role, content=f"message-{i}"))  # type: ignore[arg-type]
    return messages


# ---------------------------------------------------------------------------
# CompactionConfig defaults
# ---------------------------------------------------------------------------


class TestCompactionConfig:
    def test_defaults(self) -> None:
        config = CompactionConfig()
        assert config.trigger_ratio == 0.9
        assert config.max_context_tokens == 128_000
        assert config.mode == "basic"
        assert config.keep_recent_messages == 6
        assert config.keep_system_prompt is True

    def test_custom_values(self) -> None:
        config = CompactionConfig(
            trigger_ratio=0.8,
            max_context_tokens=8000,
            mode="agentic",
            keep_recent_messages=4,
            keep_system_prompt=False,
        )
        assert config.trigger_ratio == 0.8
        assert config.max_context_tokens == 8000
        assert config.mode == "agentic"
        assert config.keep_recent_messages == 4
        assert config.keep_system_prompt is False


# ---------------------------------------------------------------------------
# should_compact
# ---------------------------------------------------------------------------


class TestShouldCompact:
    def test_below_threshold_returns_false(self) -> None:
        config = CompactionConfig(max_context_tokens=10_000, trigger_ratio=0.9)
        compactor = Compactor(config)
        # threshold = 9000
        assert compactor.should_compact([], current_tokens=8999) is False

    def test_at_threshold_returns_true(self) -> None:
        config = CompactionConfig(max_context_tokens=10_000, trigger_ratio=0.9)
        compactor = Compactor(config)
        assert compactor.should_compact([], current_tokens=9000) is True

    def test_above_threshold_returns_true(self) -> None:
        config = CompactionConfig(max_context_tokens=10_000, trigger_ratio=0.9)
        compactor = Compactor(config)
        assert compactor.should_compact([], current_tokens=9999) is True

    def test_default_ratio(self) -> None:
        config = CompactionConfig()  # 128_000 * 0.9 = 115_200
        compactor = Compactor(config)
        assert compactor.should_compact([], current_tokens=115_199) is False
        assert compactor.should_compact([], current_tokens=115_200) is True


# ---------------------------------------------------------------------------
# Basic compaction
# ---------------------------------------------------------------------------


class TestBasicCompaction:
    def test_removes_old_messages_keeps_recent_and_system(self) -> None:
        config = CompactionConfig(keep_recent_messages=3, mode="basic")
        compactor = Compactor(config)

        messages = [
            _msg("system", "system-prompt"),
            _msg("user", "old-1"),
            _msg("assistant", "old-2"),
            _msg("user", "old-3"),
            _msg("assistant", "old-4"),
            _msg("user", "recent-1"),
            _msg("assistant", "recent-2"),
            _msg("user", "recent-3"),
        ]
        result = compactor.compact(messages)

        # 4 old messages removed, 3 recent kept + system + 1 summary
        assert result.removed_count == 4
        assert len(result.compacted_messages) == 5
        # system prompt preserved
        assert result.compacted_messages[0].role == "system"
        assert result.compacted_messages[0].content == "system-prompt"
        # placeholder summary inserted
        assert result.compacted_messages[1].role == "system"
        assert "4 earlier message(s)" in result.compacted_messages[1].content
        # last 3 recent messages preserved in order
        assert result.compacted_messages[2].content == "recent-1"
        assert result.compacted_messages[3].content == "recent-2"
        assert result.compacted_messages[4].content == "recent-3"

    def test_summary_is_placeholder_in_basic_mode(self) -> None:
        config = CompactionConfig(keep_recent_messages=2, mode="basic")
        compactor = Compactor(config)
        messages = [_msg("user", f"m{i}") for i in range(6)]
        result = compactor.compact(messages)

        assert result.removed_count == 4
        assert "context compacted" in result.summary
        assert "4" in result.summary

    def test_tokens_after_less_than_before(self) -> None:
        config = CompactionConfig(keep_recent_messages=2, mode="basic")
        compactor = Compactor(config)
        messages = [_msg("user", "x" * 100) for _ in range(10)]
        result = compactor.compact(messages)

        assert result.tokens_before > result.tokens_after
        assert result.tokens_before == (100 * 10) // 4

    def test_keep_system_prompt_false(self) -> None:
        config = CompactionConfig(keep_recent_messages=2, mode="basic", keep_system_prompt=False)
        compactor = Compactor(config)
        messages = [
            _msg("system", "system-prompt"),
            _msg("user", "old-1"),
            _msg("user", "old-2"),
            _msg("user", "recent-1"),
            _msg("user", "recent-2"),
        ]
        result = compactor.compact(messages)

        # system prompt NOT preserved; only summary + 2 recent
        assert result.removed_count == 3
        assert len(result.compacted_messages) == 3
        assert result.compacted_messages[0].content != "system-prompt"
        assert result.compacted_messages[1].content == "recent-1"
        assert result.compacted_messages[2].content == "recent-2"

    def test_system_messages_only_at_front_are_preserved(self) -> None:
        """Only leading system messages are treated as the system prompt."""

        config = CompactionConfig(keep_recent_messages=2, mode="basic")
        compactor = Compactor(config)
        messages = [
            _msg("system", "sys-1"),
            _msg("system", "sys-2"),
            _msg("user", "old-1"),
            _msg("user", "old-2"),
            _msg("user", "recent-1"),
            _msg("user", "recent-2"),
        ]
        result = compactor.compact(messages)

        # Both system messages kept, 2 old removed, 2 recent kept, + summary
        assert result.removed_count == 2
        assert result.compacted_messages[0].content == "sys-1"
        assert result.compacted_messages[1].content == "sys-2"
        assert result.compacted_messages[2].role == "system"  # summary
        assert result.compacted_messages[3].content == "recent-1"
        assert result.compacted_messages[4].content == "recent-2"


# ---------------------------------------------------------------------------
# Agentic compaction
# ---------------------------------------------------------------------------


class TestAgenticCompaction:
    def test_uses_llm_to_summarize(self) -> None:
        fake_response = LLMResponse(
            content="Summary of earlier conversation.",
            tool_calls=[],
            finish_reason="stop",
        )
        client = _FakeLLMClient(fake_response)
        config = CompactionConfig(keep_recent_messages=2, mode="agentic")
        compactor = Compactor(config, llm_client=client)

        messages = [
            _msg("system", "sys"),
            _msg("user", "old-1"),
            _msg("assistant", "old-2"),
            _msg("user", "recent-1"),
            _msg("assistant", "recent-2"),
        ]
        result = compactor.compact(messages)

        assert result.removed_count == 2
        assert result.summary == "Summary of earlier conversation."
        # LLM was called exactly once
        assert len(client.calls) == 1
        # The LLM request contained the transcript
        request = client.calls[0]
        assert any("old-1" in m.content for m in request.messages)
        assert any("old-2" in m.content for m in request.messages)

        # compacted = system + summary + 2 recent
        assert len(result.compacted_messages) == 4
        assert result.compacted_messages[0].content == "sys"
        assert result.compacted_messages[1].content == "Summary of earlier conversation."
        assert result.compacted_messages[2].content == "recent-1"
        assert result.compacted_messages[3].content == "recent-2"

    def test_summary_inserted_as_system_message(self) -> None:
        fake_response = LLMResponse(
            content="LLM summary text.",
            tool_calls=[],
            finish_reason="stop",
        )
        client = _FakeLLMClient(fake_response)
        config = CompactionConfig(keep_recent_messages=1, mode="agentic")
        compactor = Compactor(config, llm_client=client)

        messages = [_msg("user", f"msg-{i}") for i in range(5)]
        result = compactor.compact(messages)

        summary_msg = result.compacted_messages[-2]
        assert summary_msg.role == "system"
        assert summary_msg.content == "LLM summary text."

    def test_falls_back_to_placeholder_without_client(self) -> None:
        config = CompactionConfig(keep_recent_messages=2, mode="agentic")
        compactor = Compactor(config, llm_client=None)

        messages = [_msg("user", f"msg-{i}") for i in range(5)]
        result = compactor.compact(messages)

        assert result.removed_count == 3
        assert "context compacted" in result.summary

    def test_empty_llm_response_falls_back_to_placeholder(self) -> None:
        fake_response = LLMResponse(
            content="   ",
            tool_calls=[],
            finish_reason="stop",
        )
        client = _FakeLLMClient(fake_response)
        config = CompactionConfig(keep_recent_messages=2, mode="agentic")
        compactor = Compactor(config, llm_client=client)

        messages = [_msg("user", f"msg-{i}") for i in range(5)]
        result = compactor.compact(messages)

        assert "context compacted" in result.summary

    def test_tool_calls_included_in_transcript(self) -> None:
        fake_response = LLMResponse(
            content="Summary.",
            tool_calls=[],
            finish_reason="stop",
        )
        client = _FakeLLMClient(fake_response)
        config = CompactionConfig(keep_recent_messages=1, mode="agentic")
        compactor = Compactor(config, llm_client=client)

        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="tc1", name="read_file", input={"path": "/a"})],
            ),
            _msg("user", "recent"),
        ]
        result = compactor.compact(messages)

        assert result.removed_count == 1
        assert len(client.calls) == 1
        transcript_msg = client.calls[0].messages[1]
        assert "read_file" in transcript_msg.content


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_messages(self) -> None:
        compactor = Compactor(CompactionConfig())
        assert compactor.estimate_tokens([]) == 0

    def test_rough_accuracy(self) -> None:
        compactor = Compactor(CompactionConfig())
        # 4 chars = 1 token
        msg = _msg("user", "abcd")  # 4 chars -> 1 token
        assert compactor.estimate_tokens([msg]) == 1

    def test_multiple_messages(self) -> None:
        compactor = Compactor(CompactionConfig())
        messages = [
            _msg("user", "abcd"),  # 4 chars
            _msg("assistant", "efgh"),  # 4 chars
        ]
        # 8 chars total -> 2 tokens
        assert compactor.estimate_tokens(messages) == 2

    def test_tool_result_output_counted(self) -> None:
        from justagent.agent.runtime import ToolResultPart

        compactor = Compactor(CompactionConfig())
        msg = Message(
            role="tool",
            tool_result=ToolResultPart(tool_call_id="tc1", name="read", output="x" * 40),
        )
        # 40 chars -> 10 tokens
        assert compactor.estimate_tokens([msg]) == 10


# ---------------------------------------------------------------------------
# keep_recent_messages boundary
# ---------------------------------------------------------------------------


class TestKeepRecentBoundary:
    def test_exactly_keep_recent_messages(self) -> None:
        """When non-system count == keep_recent, nothing is removed."""

        config = CompactionConfig(keep_recent_messages=3, mode="basic")
        compactor = Compactor(config)
        messages = [
            _msg("system", "sys"),
            _msg("user", "m1"),
            _msg("assistant", "m2"),
            _msg("user", "m3"),
        ]
        result = compactor.compact(messages)

        assert result.removed_count == 0
        assert result.summary == ""
        # All messages returned unchanged
        assert len(result.compacted_messages) == 4
        assert result.compacted_messages == messages

    def test_one_more_than_keep_recent(self) -> None:
        """When non-system count == keep_recent + 1, exactly 1 removed."""

        config = CompactionConfig(keep_recent_messages=3, mode="basic")
        compactor = Compactor(config)
        messages = [
            _msg("system", "sys"),
            _msg("user", "m1"),
            _msg("assistant", "m2"),
            _msg("user", "m3"),
            _msg("assistant", "m4"),
        ]
        result = compactor.compact(messages)

        assert result.removed_count == 1
        # system + summary + 3 recent = 5
        assert len(result.compacted_messages) == 5
        assert result.compacted_messages[0].content == "sys"
        assert result.compacted_messages[2].content == "m2"
        assert result.compacted_messages[3].content == "m3"
        assert result.compacted_messages[4].content == "m4"

    def test_keep_recent_zero(self) -> None:
        """keep_recent_messages=0 keeps only system + summary."""

        config = CompactionConfig(keep_recent_messages=0, mode="basic")
        compactor = Compactor(config)
        messages = [
            _msg("system", "sys"),
            _msg("user", "m1"),
            _msg("assistant", "m2"),
        ]
        result = compactor.compact(messages)

        assert result.removed_count == 2
        # system + summary only
        assert len(result.compacted_messages) == 2
        assert result.compacted_messages[0].content == "sys"
        assert result.compacted_messages[1].role == "system"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_messages(self) -> None:
        compactor = Compactor(CompactionConfig(keep_recent_messages=3))
        result = compactor.compact([])

        assert result.removed_count == 0
        assert result.summary == ""
        assert result.compacted_messages == []
        assert result.tokens_before == 0
        assert result.tokens_after == 0

    def test_all_system_messages(self) -> None:
        compactor = Compactor(CompactionConfig(keep_recent_messages=3))
        messages = [
            _msg("system", "sys-1"),
            _msg("system", "sys-2"),
            _msg("system", "sys-3"),
        ]
        result = compactor.compact(messages)

        # All are leading system messages; rest is empty -> nothing removed
        assert result.removed_count == 0
        assert result.compacted_messages == messages

    def test_fewer_messages_than_keep_recent(self) -> None:
        compactor = Compactor(CompactionConfig(keep_recent_messages=10))
        messages = [
            _msg("system", "sys"),
            _msg("user", "u1"),
            _msg("assistant", "a1"),
        ]
        result = compactor.compact(messages)

        assert result.removed_count == 0
        assert result.compacted_messages == messages

    def test_single_message(self) -> None:
        compactor = Compactor(CompactionConfig(keep_recent_messages=3))
        result = compactor.compact([_msg("user", "only")])

        assert result.removed_count == 0
        assert len(result.compacted_messages) == 1

    def test_no_system_prompt_with_system_messages(self) -> None:
        """When keep_system_prompt=False, system messages are removable."""

        config = CompactionConfig(keep_recent_messages=2, keep_system_prompt=False, mode="basic")
        compactor = Compactor(config)
        messages = [
            _msg("system", "sys"),
            _msg("user", "old-1"),
            _msg("user", "old-2"),
            _msg("user", "recent-1"),
            _msg("user", "recent-2"),
        ]
        result = compactor.compact(messages)

        # system message is in the removable block (not treated as system prefix)
        assert result.removed_count == 3
        # No system prompt preserved, just summary + 2 recent
        assert len(result.compacted_messages) == 3
        assert result.compacted_messages[0].role == "system"  # summary
        assert result.compacted_messages[1].content == "recent-1"
        assert result.compacted_messages[2].content == "recent-2"


# ---------------------------------------------------------------------------
# CompactionResult immutability
# ---------------------------------------------------------------------------


class TestCompactionResult:
    def test_frozen_dataclass(self) -> None:
        compactor = Compactor(CompactionConfig(keep_recent_messages=2))
        messages = _many_messages(5)
        result = compactor.compact(messages)

        assert isinstance(result, CompactionResult)
        # Frozen: cannot reassign fields
        try:
            result.removed_count = 999  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised, "CompactionResult should be frozen"
