"""Tests for ``justagent.agent.plan_act`` (Plan / Act / Yolo mode switching).

Covers:

* :class:`AgentMode` properties.
* :func:`filter_tools_for_mode` — edit tools hidden in Plan mode.
* :func:`build_system_prompt` — mode tag always present; Plan/Yolo
  instructions appended only in those modes.
* :func:`format_user_message` — ``<user_input mode="...">`` wrapper.
* :func:`format_mode_switch_notice` — ``<mode_notice>`` block, empty
  when no switch.
* :class:`ModeSwitchTracker` — record/consume/pending, coalescing.
* :class:`ModeConfig` — switch_to + consume_switch_notice.
* :func:`default_plan_file_path` — ``.justagent/plans/plan.md`` layout.
* Runtime integration: mode-aware tool filtering and user-message
  wrapping via :class:`AgentRuntime`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from justagent.agent.plan_act import (
    MODE_TAG_INSTRUCTIONS,
    PLAN_MODE_INSTRUCTIONS,
    YOLO_MODE_INSTRUCTIONS,
    AgentMode,
    ModeConfig,
    ModeSwitch,
    ModeSwitchTracker,
    build_system_prompt,
    default_plan_file_path,
    filter_tools_for_mode,
    format_mode_switch_notice,
    format_user_message,
    is_command_tool,
    is_edit_tool,
)
from justagent.agent.runtime import (
    AgentRuntime,
    AgentRuntimeConfig,
    LLMClient,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from justagent.agent.tools.base import Tool, ToolContext, ToolResult

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _NoArgs(BaseModel):
    pass


async def _noop_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    return ToolResult.success("ok")


def _make_tool(tool_id: str) -> Tool:
    return Tool(
        id=tool_id,
        description=f"tool {tool_id}",
        parameters=_NoArgs,
        execute=_noop_execute,
    )


class _FakeLLMClient(LLMClient):
    """Returns scripted responses and records every request."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._calls: list[LLMRequest] = []

    async def complete(
        self,
        request: LLMRequest,
        *,
        abort: asyncio.Event | None = None,
    ) -> LLMResponse:
        self._calls.append(request)
        if not self._responses:
            raise RuntimeError("No more scripted responses")
        return self._responses.pop(0)


def _response(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="stop",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        model="test",
        latency_ms=0.1,
    )


# ---------------------------------------------------------------------------
# AgentMode
# ---------------------------------------------------------------------------


class TestAgentMode:
    def test_act_allows_edits_and_requires_permission(self) -> None:
        assert AgentMode.ACT.allows_edits is True
        assert AgentMode.ACT.allows_commands is True
        assert AgentMode.ACT.requires_permission is True

    def test_plan_blocks_edits_requires_permission(self) -> None:
        assert AgentMode.PLAN.allows_edits is False
        assert AgentMode.PLAN.allows_commands is True
        assert AgentMode.PLAN.requires_permission is True

    def test_yolo_allows_edits_no_permission(self) -> None:
        assert AgentMode.YOLO.allows_edits is True
        assert AgentMode.YOLO.allows_commands is True
        assert AgentMode.YOLO.requires_permission is False

    def test_str_value_round_trip(self) -> None:
        assert AgentMode("act") is AgentMode.ACT
        assert AgentMode("plan") is AgentMode.PLAN
        assert AgentMode("yolo") is AgentMode.YOLO
        assert AgentMode.ACT.value == "act"


# ---------------------------------------------------------------------------
# filter_tools_for_mode
# ---------------------------------------------------------------------------


class TestFilterToolsForMode:
    def _all_tools(self) -> list[Tool]:
        return [
            _make_tool("read_file"),
            _make_tool("write_to_file"),
            _make_tool("replace_in_file"),
            _make_tool("apply_patch"),
            _make_tool("run_command"),
            _make_tool("ask_question"),
        ]

    def test_act_keeps_all_tools(self) -> None:
        tools = self._all_tools()
        filtered = filter_tools_for_mode(tools, AgentMode.ACT)
        assert [t.id for t in filtered] == [t.id for t in tools]

    def test_yolo_keeps_all_tools(self) -> None:
        tools = self._all_tools()
        filtered = filter_tools_for_mode(tools, AgentMode.YOLO)
        assert [t.id for t in filtered] == [t.id for t in tools]

    def test_plan_removes_edit_tools(self) -> None:
        tools = self._all_tools()
        filtered = filter_tools_for_mode(tools, AgentMode.PLAN)
        ids = [t.id for t in filtered]
        assert "write_to_file" not in ids
        assert "replace_in_file" not in ids
        assert "apply_patch" not in ids

    def test_plan_keeps_read_and_command_tools(self) -> None:
        tools = self._all_tools()
        filtered = filter_tools_for_mode(tools, AgentMode.PLAN)
        ids = [t.id for t in filtered]
        assert "read_file" in ids
        assert "run_command" in ids
        assert "ask_question" in ids

    def test_plan_with_no_edit_tools_keeps_everything(self) -> None:
        tools = [_make_tool("read_file"), _make_tool("search_files")]
        filtered = filter_tools_for_mode(tools, AgentMode.PLAN)
        assert len(filtered) == 2

    def test_filter_returns_new_list(self) -> None:
        tools = self._all_tools()
        filtered = filter_tools_for_mode(tools, AgentMode.ACT)
        assert filtered is not tools


# ---------------------------------------------------------------------------
# is_edit_tool / is_command_tool
# ---------------------------------------------------------------------------


class TestToolClassification:
    def test_is_edit_tool_true_for_known_edit_tools(self) -> None:
        assert is_edit_tool("write_to_file") is True
        assert is_edit_tool("replace_in_file") is True
        assert is_edit_tool("apply_patch") is True

    def test_is_edit_tool_false_for_non_edit(self) -> None:
        assert is_edit_tool("read_file") is False
        assert is_edit_tool("run_command") is False
        assert is_edit_tool("ask_question") is False

    def test_is_command_tool_true_for_run_command(self) -> None:
        assert is_command_tool("run_command") is True

    def test_is_command_tool_false_for_others(self) -> None:
        assert is_command_tool("write_to_file") is False
        assert is_command_tool("read_file") is False


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_act_mode_includes_mode_tag_only(self) -> None:
        prompt = build_system_prompt("Base prompt.", AgentMode.ACT)
        assert "Base prompt." in prompt
        assert MODE_TAG_INSTRUCTIONS in prompt
        assert PLAN_MODE_INSTRUCTIONS not in prompt
        assert YOLO_MODE_INSTRUCTIONS not in prompt

    def test_plan_mode_includes_plan_instructions(self) -> None:
        prompt = build_system_prompt("Base prompt.", AgentMode.PLAN)
        assert "Base prompt." in prompt
        assert MODE_TAG_INSTRUCTIONS in prompt
        assert PLAN_MODE_INSTRUCTIONS in prompt
        assert YOLO_MODE_INSTRUCTIONS not in prompt

    def test_yolo_mode_includes_yolo_instructions(self) -> None:
        prompt = build_system_prompt("Base prompt.", AgentMode.YOLO)
        assert "Base prompt." in prompt
        assert MODE_TAG_INSTRUCTIONS in prompt
        assert YOLO_MODE_INSTRUCTIONS in prompt
        assert PLAN_MODE_INSTRUCTIONS not in prompt

    def test_empty_base_prompt_still_includes_mode_tag(self) -> None:
        prompt = build_system_prompt("", AgentMode.ACT)
        assert prompt == MODE_TAG_INSTRUCTIONS

    def test_extra_rules_appended_between_base_and_mode_tag(self) -> None:
        prompt = build_system_prompt("Base", AgentMode.ACT, extra_rules="Extra rules here")
        base_idx = prompt.index("Base")
        extra_idx = prompt.index("Extra rules here")
        tag_idx = prompt.index(MODE_TAG_INSTRUCTIONS)
        assert base_idx < extra_idx < tag_idx

    def test_whitespace_stripped_from_base_prompt(self) -> None:
        prompt = build_system_prompt("  Base  ", AgentMode.ACT)
        assert prompt.startswith("Base\n\n")


# ---------------------------------------------------------------------------
# format_user_message
# ---------------------------------------------------------------------------


class TestFormatUserMessage:
    def test_act_mode_wrapper(self) -> None:
        result = format_user_message("hello", AgentMode.ACT)
        assert result == '<user_input mode="act">hello</user_input>'

    def test_plan_mode_wrapper(self) -> None:
        result = format_user_message("plan this", AgentMode.PLAN)
        assert result == '<user_input mode="plan">plan this</user_input>'

    def test_yolo_mode_wrapper(self) -> None:
        result = format_user_message("just do it", AgentMode.YOLO)
        assert result == '<user_input mode="yolo">just do it</user_input>'

    def test_multiline_content_preserved(self) -> None:
        result = format_user_message("line1\nline2", AgentMode.ACT)
        assert "line1\nline2" in result


# ---------------------------------------------------------------------------
# format_mode_switch_notice
# ---------------------------------------------------------------------------


class TestFormatModeSwitchNotice:
    def test_act_to_plan(self) -> None:
        notice = format_mode_switch_notice(AgentMode.ACT, AgentMode.PLAN)
        assert "<mode_notice>" in notice
        assert "</mode_notice>" in notice
        assert "act" in notice
        assert "plan" in notice

    def test_plan_to_act(self) -> None:
        notice = format_mode_switch_notice(AgentMode.PLAN, AgentMode.ACT)
        assert "<mode_notice>" in notice
        assert "act" in notice

    def test_act_to_yolo(self) -> None:
        notice = format_mode_switch_notice(AgentMode.ACT, AgentMode.YOLO)
        assert "<mode_notice>" in notice
        assert "yolo" in notice

    def test_same_mode_returns_empty(self) -> None:
        assert format_mode_switch_notice(AgentMode.ACT, AgentMode.ACT) == ""
        assert format_mode_switch_notice(AgentMode.PLAN, AgentMode.PLAN) == ""
        assert format_mode_switch_notice(AgentMode.YOLO, AgentMode.YOLO) == ""


# ---------------------------------------------------------------------------
# ModeSwitchTracker
# ---------------------------------------------------------------------------


class TestModeSwitchTracker:
    def test_initial_pending_is_none(self) -> None:
        tracker = ModeSwitchTracker()
        assert tracker.pending is None

    def test_record_first_switch(self) -> None:
        tracker = ModeSwitchTracker()
        tracker.record(AgentMode.ACT, AgentMode.PLAN)
        assert tracker.pending is not None
        assert tracker.pending.from_mode is AgentMode.ACT
        assert tracker.pending.to_mode is AgentMode.PLAN

    def test_record_same_mode_is_noop(self) -> None:
        tracker = ModeSwitchTracker()
        tracker.record(AgentMode.ACT, AgentMode.ACT)
        assert tracker.pending is None

    def test_consume_returns_and_clears(self) -> None:
        tracker = ModeSwitchTracker()
        tracker.record(AgentMode.ACT, AgentMode.PLAN)
        pending = tracker.consume()
        assert pending is not None
        assert pending.to_mode is AgentMode.PLAN
        assert tracker.pending is None
        assert tracker.consume() is None

    def test_coalesce_round_trip_cancels(self) -> None:
        """plan → act → plan is treated as no switch."""

        tracker = ModeSwitchTracker()
        tracker.record(AgentMode.PLAN, AgentMode.ACT)
        tracker.record(AgentMode.ACT, AgentMode.PLAN)
        assert tracker.pending is None

    def test_coalesce_chained_switches_keep_latest(self) -> None:
        """act → plan → yolo results in a single act → yolo notice."""

        tracker = ModeSwitchTracker()
        tracker.record(AgentMode.ACT, AgentMode.PLAN)
        tracker.record(AgentMode.PLAN, AgentMode.YOLO)
        pending = tracker.pending
        assert pending is not None
        assert pending.from_mode is AgentMode.ACT
        assert pending.to_mode is AgentMode.YOLO

    def test_consume_when_empty_returns_none(self) -> None:
        tracker = ModeSwitchTracker()
        assert tracker.consume() is None


# ---------------------------------------------------------------------------
# ModeConfig
# ---------------------------------------------------------------------------


class TestModeConfig:
    def test_default_mode_is_act(self) -> None:
        config = ModeConfig()
        assert config.mode is AgentMode.ACT

    def test_switch_to_records_and_updates_mode(self) -> None:
        config = ModeConfig()
        config.switch_to(AgentMode.PLAN)
        assert config.mode is AgentMode.PLAN
        assert config.tracker.pending is not None

    def test_switch_to_same_mode_does_not_record(self) -> None:
        config = ModeConfig(mode=AgentMode.ACT)
        config.switch_to(AgentMode.ACT)
        assert config.tracker.pending is None

    def test_consume_switch_notice_returns_string_when_pending(self) -> None:
        config = ModeConfig()
        config.switch_to(AgentMode.PLAN)
        notice = config.consume_switch_notice()
        assert "<mode_notice>" in notice
        assert "plan" in notice

    def test_consume_switch_notice_empty_when_no_pending(self) -> None:
        config = ModeConfig()
        assert config.consume_switch_notice() == ""

    def test_consume_switch_notice_clears_pending(self) -> None:
        config = ModeConfig()
        config.switch_to(AgentMode.PLAN)
        config.consume_switch_notice()
        assert config.consume_switch_notice() == ""


# ---------------------------------------------------------------------------
# default_plan_file_path
# ---------------------------------------------------------------------------


class TestDefaultPlanFilePath:
    def test_returns_myagent_plans_plan_md(self) -> None:
        path = default_plan_file_path("/tmp/project")
        assert path == Path("/tmp/project/.justagent/plans/plan.md")

    def test_accepts_path_object(self) -> None:
        path = default_plan_file_path(Path("/tmp/project"))
        assert path == Path("/tmp/project/.justagent/plans/plan.md")

    def test_does_not_create_directories(self, tmp_path: Path) -> None:
        path = default_plan_file_path(tmp_path)
        assert not path.parent.exists()


# ---------------------------------------------------------------------------
# Runtime integration
# ---------------------------------------------------------------------------


class TestRuntimeModeIntegration:
    @pytest.mark.asyncio
    async def test_runtime_default_mode_is_act(self) -> None:
        client = _FakeLLMClient([_response("hi")])
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(max_iterations=3),
        )
        assert runtime.mode is AgentMode.ACT
        await runtime.run("hello")

    @pytest.mark.asyncio
    async def test_runtime_initial_mode_from_config(self) -> None:
        client = _FakeLLMClient([_response("hi")])
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(max_iterations=3, initial_mode=AgentMode.PLAN),
        )
        assert runtime.mode is AgentMode.PLAN

    @pytest.mark.asyncio
    async def test_runtime_switch_mode_updates_state(self) -> None:
        client = _FakeLLMClient([_response("hi")])
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(max_iterations=3),
        )
        runtime.switch_mode(AgentMode.PLAN)
        assert runtime.mode is AgentMode.PLAN

    @pytest.mark.asyncio
    async def test_runtime_act_mode_keeps_edit_tools(self) -> None:
        client = _FakeLLMClient([_response("hi")])
        tools = [
            _make_tool("read_file"),
            _make_tool("write_to_file"),
            _make_tool("apply_patch"),
        ]
        runtime = AgentRuntime(
            client=client,
            tools=tools,
            config=AgentRuntimeConfig(max_iterations=3),
        )
        await runtime.run("hello")
        # The LLM request should see all tools (act mode keeps edits).
        sent_tool_ids = {t.id for t in client._calls[0].tools}
        assert sent_tool_ids == {"read_file", "write_to_file", "apply_patch"}

    @pytest.mark.asyncio
    async def test_runtime_plan_mode_filters_edit_tools(self) -> None:
        client = _FakeLLMClient([_response("hi")])
        tools = [
            _make_tool("read_file"),
            _make_tool("write_to_file"),
            _make_tool("replace_in_file"),
            _make_tool("apply_patch"),
            _make_tool("run_command"),
        ]
        runtime = AgentRuntime(
            client=client,
            tools=tools,
            config=AgentRuntimeConfig(max_iterations=3, initial_mode=AgentMode.PLAN),
        )
        await runtime.run("explore")
        sent_tool_ids = {t.id for t in client._calls[0].tools}
        assert "read_file" in sent_tool_ids
        assert "run_command" in sent_tool_ids
        assert "write_to_file" not in sent_tool_ids
        assert "replace_in_file" not in sent_tool_ids
        assert "apply_patch" not in sent_tool_ids

    @pytest.mark.asyncio
    async def test_runtime_plan_mode_appends_plan_instructions(self) -> None:
        client = _FakeLLMClient([_response("hi")])
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(
                system_prompt="You are justagent.",
                max_iterations=3,
                initial_mode=AgentMode.PLAN,
            ),
        )
        await runtime.run("explore")
        system_msg = client._calls[0].messages[0]
        assert system_msg.role == "system"
        assert "You are justagent." in system_msg.content
        assert PLAN_MODE_INSTRUCTIONS in system_msg.content

    @pytest.mark.asyncio
    async def test_runtime_act_mode_does_not_append_plan_instructions(self) -> None:
        client = _FakeLLMClient([_response("hi")])
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(
                system_prompt="You are justagent.",
                max_iterations=3,
                initial_mode=AgentMode.ACT,
            ),
        )
        await runtime.run("do something")
        system_msg = client._calls[0].messages[0]
        assert PLAN_MODE_INSTRUCTIONS not in system_msg.content

    @pytest.mark.asyncio
    async def test_runtime_yolo_mode_appends_yolo_instructions(self) -> None:
        client = _FakeLLMClient([_response("hi")])
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(max_iterations=3, initial_mode=AgentMode.YOLO),
        )
        await runtime.run("go")
        system_msg = client._calls[0].messages[0]
        assert YOLO_MODE_INSTRUCTIONS in system_msg.content

    @pytest.mark.asyncio
    async def test_runtime_wraps_user_message_with_mode_tag(self) -> None:
        client = _FakeLLMClient([_response("hi")])
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(max_iterations=3),
        )
        await runtime.run("hello world")
        user_msg = client._calls[0].messages[1]
        assert user_msg.role == "user"
        assert user_msg.content == '<user_input mode="act">hello world</user_input>'

    @pytest.mark.asyncio
    async def test_runtime_switch_mode_prepends_notice_next_run(self) -> None:
        """A mode switch recorded before run() starts is surfaced as a
        ``<mode_notice>`` block at the top of the user message."""

        client = _FakeLLMClient([_response("hi")])
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(max_iterations=3),
        )
        # Simulate the UI toggling to plan mode before run() begins.
        runtime.switch_mode(AgentMode.PLAN)
        await runtime.run("explore")
        user_msg = client._calls[0].messages[1]
        assert user_msg.role == "user"
        assert "<mode_notice>" in user_msg.content
        assert '<user_input mode="plan">explore</user_input>' in user_msg.content
        # Notice appears before the user_input wrapper.
        notice_idx = user_msg.content.index("<mode_notice>")
        input_idx = user_msg.content.index("<user_input")
        assert notice_idx < input_idx

    @pytest.mark.asyncio
    async def test_runtime_yolo_mode_keeps_edit_tools(self) -> None:
        client = _FakeLLMClient([_response("hi")])
        tools = [
            _make_tool("read_file"),
            _make_tool("write_to_file"),
        ]
        runtime = AgentRuntime(
            client=client,
            tools=tools,
            config=AgentRuntimeConfig(max_iterations=3, initial_mode=AgentMode.YOLO),
        )
        await runtime.run("go")
        sent_tool_ids = {t.id for t in client._calls[0].tools}
        assert sent_tool_ids == {"read_file", "write_to_file"}


# ---------------------------------------------------------------------------
# ModeSwitch dataclass
# ---------------------------------------------------------------------------


class TestModeSwitchDataclass:
    def test_fields(self) -> None:
        ms = ModeSwitch(from_mode=AgentMode.ACT, to_mode=AgentMode.PLAN)
        assert ms.from_mode is AgentMode.ACT
        assert ms.to_mode is AgentMode.PLAN

    def test_equality(self) -> None:
        a = ModeSwitch(AgentMode.ACT, AgentMode.PLAN)
        b = ModeSwitch(AgentMode.ACT, AgentMode.PLAN)
        c = ModeSwitch(AgentMode.ACT, AgentMode.YOLO)
        assert a == b
        assert a != c
