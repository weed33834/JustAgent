"""Tests for ``autoship.agent.runtime`` (the agent iterative loop)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from autoship.agent.runtime import (
    AgentRuntime,
    AgentRuntimeConfig,
    LLMClient,
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
    ToolFinishedEvent,
    ToolResultPart,
    ToolStartedEvent,
    TurnStartedEvent,
)
from autoship.agent.tools.base import Tool, ToolContext, ToolResult
from autoship.agent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _StubInput(BaseModel):
    value: str = ""


async def _stub_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    """Echo back the input value as the tool output."""

    assert isinstance(args, _StubInput)
    return ToolResult.success(f"echo: {args.value}")


def _make_echo_tool(tool_id: str = "echo") -> Tool:
    return Tool(
        id=tool_id,
        description="Echo the input value",
        parameters=_StubInput,
        execute=_stub_execute,
    )


class _FakeLLMClient(LLMClient):
    """Test double that returns scripted responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        # Skip parent __init__ — we don't need real API credentials.
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


def _assistant_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="test-model",
        latency_ms=1.0,
    )


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------


class TestMessageModel:
    def test_user_message_to_dict(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.to_dict() == {"role": "user", "content": "hello"}

    def test_system_message_to_dict(self) -> None:
        msg = Message(role="system", content="You are...")
        assert msg.to_dict() == {"role": "system", "content": "You are..."}

    def test_assistant_message_with_tool_calls_to_dict(self) -> None:
        msg = Message(
            role="assistant",
            content="thinking...",
            tool_calls=[
                ToolCall(id="tc1", name="echo", input={"value": "hi"}),
            ],
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "thinking..."
        assert d["tool_calls"][0]["function"]["name"] == "echo"
        assert d["tool_calls"][0]["function"]["arguments"] == '{"value": "hi"}'

    def test_tool_message_to_dict(self) -> None:
        msg = Message(
            role="tool",
            tool_result=ToolResultPart(
                tool_call_id="tc1", name="echo", output="result"
            ),
        )
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "tc1"
        assert d["content"] == "result"

    def test_message_metadata_defaults(self) -> None:
        msg = Message(role="user", content="hi")
        assert msg.tool_calls == []
        assert msg.tool_result is None
        assert msg.metadata == {}


# ---------------------------------------------------------------------------
# AgentRuntime — happy path
# ---------------------------------------------------------------------------


class TestAgentRuntimeHappyPath:
    @pytest.mark.asyncio
    async def test_simple_completion_no_tools(self) -> None:
        """LLM responds with text only → run completes immediately."""

        client = _FakeLLMClient(
            [_assistant_response(content="Hello!", finish_reason="stop")]
        )
        runtime = AgentRuntime(
            client=client,
            tools=[_make_echo_tool()],
            config=AgentRuntimeConfig(max_iterations=5),
        )
        result = await runtime.run("Say hello")
        assert result.status == "completed"
        assert result.final_content == "Hello!"
        assert result.iterations == 1
        # Plan/Act mode tag is always appended to the system prompt, so
        # we get: system(mode tag) + user + assistant = 3 messages.
        assert len(result.messages) == 3
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"
        assert result.messages[2].role == "assistant"

    @pytest.mark.asyncio
    async def test_one_tool_call_then_completion(self) -> None:
        """LLM calls a tool, gets result, then responds with text."""

        client = _FakeLLMClient(
            [
                _assistant_response(
                    content="",
                    tool_calls=[
                        ToolCall(id="tc1", name="echo", input={"value": "test"})
                    ],
                    finish_reason="tool_calls",
                ),
                _assistant_response(content="Got: echo: test", finish_reason="stop"),
            ]
        )
        runtime = AgentRuntime(
            client=client,
            tools=[_make_echo_tool()],
            config=AgentRuntimeConfig(max_iterations=5),
        )
        result = await runtime.run("Echo test")
        assert result.status == "completed"
        assert result.final_content == "Got: echo: test"
        assert result.iterations == 2
        # system(mode tag) + user + assistant(tool_call) + tool(result)
        # + assistant(final) = 5
        assert len(result.messages) == 5

    @pytest.mark.asyncio
    async def test_system_prompt_prepended(self) -> None:
        client = _FakeLLMClient(
            [_assistant_response(content="ok", finish_reason="stop")]
        )
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(system_prompt="You are a bot"),
        )
        result = await runtime.run("hi")
        assert result.status == "completed"
        # First message should be the system prompt, with mode tag appended.
        assert client._calls[0].messages[0].role == "system"
        content = client._calls[0].messages[0].content
        assert content.startswith("You are a bot")
        assert "# Plan / Act Modes" in content

    @pytest.mark.asyncio
    async def test_usage_accumulated(self) -> None:
        client = _FakeLLMClient(
            [
                _assistant_response(
                    content="",
                    tool_calls=[
                        ToolCall(id="tc1", name="echo", input={"value": "x"})
                    ],
                    finish_reason="tool_calls",
                ),
                _assistant_response(content="done", finish_reason="stop"),
            ]
        )
        runtime = AgentRuntime(
            client=client,
            tools=[_make_echo_tool()],
            config=AgentRuntimeConfig(max_iterations=5),
        )
        result = await runtime.run("hi")
        assert result.status == "completed"
        # Two LLM calls, each contributing 15 tokens.
        assert result.total_usage["prompt_tokens"] == 20
        assert result.total_usage["completion_tokens"] == 10
        assert result.total_usage["total_tokens"] == 30


# ---------------------------------------------------------------------------
# AgentRuntime — events
# ---------------------------------------------------------------------------


class TestAgentRuntimeEvents:
    @pytest.mark.asyncio
    async def test_events_emitted_for_simple_run(self) -> None:
        events: list[Any] = []

        async def _emit(event: Any) -> None:
            events.append(event)

        client = _FakeLLMClient(
            [_assistant_response(content="hi", finish_reason="stop")]
        )
        runtime = AgentRuntime(
            client=client,
            tools=[],
            emit=_emit,
            config=AgentRuntimeConfig(max_iterations=3),
        )
        await runtime.run("hello")
        types = [e.type for e in events]
        assert "run-started" in types
        assert "turn-started" in types
        assert "assistant-message" in types

    @pytest.mark.asyncio
    async def test_tool_events_emitted(self) -> None:
        events: list[Any] = []

        async def _emit(event: Any) -> None:
            events.append(event)

        client = _FakeLLMClient(
            [
                _assistant_response(
                    content="",
                    tool_calls=[
                        ToolCall(id="tc1", name="echo", input={"value": "v"})
                    ],
                    finish_reason="tool_calls",
                ),
                _assistant_response(content="done", finish_reason="stop"),
            ]
        )
        runtime = AgentRuntime(
            client=client,
            tools=[_make_echo_tool()],
            emit=_emit,
            config=AgentRuntimeConfig(max_iterations=5),
        )
        await runtime.run("hi")
        types = [e.type for e in events]
        assert "tool-started" in types
        assert "tool-finished" in types
        # Verify the tool-started event carries the right info.
        ts = next(e for e in events if isinstance(e, ToolStartedEvent))
        assert ts.tool_name == "echo"
        assert ts.tool_call_id == "tc1"
        tf = next(e for e in events if isinstance(e, ToolFinishedEvent))
        assert "echo: v" in tf.output
        assert not tf.is_error

    @pytest.mark.asyncio
    async def test_turn_started_event_carries_iteration(self) -> None:
        events: list[Any] = []

        async def _emit(event: Any) -> None:
            events.append(event)

        client = _FakeLLMClient(
            [
                _assistant_response(
                    tool_calls=[
                        ToolCall(id="tc1", name="echo", input={"value": "x"})
                    ],
                    finish_reason="tool_calls",
                ),
                _assistant_response(content="done", finish_reason="stop"),
            ]
        )
        runtime = AgentRuntime(
            client=client,
            tools=[_make_echo_tool()],
            emit=_emit,
        )
        await runtime.run("hi")
        turns = [e for e in events if isinstance(e, TurnStartedEvent)]
        assert len(turns) == 2
        assert turns[0].iteration == 1
        assert turns[1].iteration == 2


# ---------------------------------------------------------------------------
# AgentRuntime — abort
# ---------------------------------------------------------------------------


class TestAgentRuntimeAbort:
    @pytest.mark.asyncio
    async def test_abort_during_run(self) -> None:
        """Abort signal stops the run between iterations."""

        client = _FakeLLMClient(
            [
                _assistant_response(
                    tool_calls=[
                        ToolCall(id="tc1", name="echo", input={"value": "x"})
                    ],
                    finish_reason="tool_calls",
                ),
                _assistant_response(content="done", finish_reason="stop"),
            ]
        )
        runtime = AgentRuntime(
            client=client,
            tools=[_make_echo_tool()],
            config=AgentRuntimeConfig(max_iterations=10),
        )
        # Abort right after the first LLM response.
        original_call = runtime._call_llm

        async def _call_then_abort() -> LLMResponse:
            result = await original_call()
            runtime.abort()
            return result

        runtime._call_llm = _call_then_abort  # type: ignore[assignment]
        result = await runtime.run("hi")
        assert result.status == "aborted"
        assert "abort" in result.error.lower() or "cancel" in result.error.lower()


# ---------------------------------------------------------------------------
# AgentRuntime — unknown tool
# ---------------------------------------------------------------------------


class TestAgentRuntimeUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(self) -> None:
        """LLM calls a tool that doesn't exist → error result fed back."""

        client = _FakeLLMClient(
            [
                _assistant_response(
                    tool_calls=[
                        ToolCall(id="tc1", name="nonexistent", input={})
                    ],
                    finish_reason="tool_calls",
                ),
                _assistant_response(content="sorry", finish_reason="stop"),
            ]
        )
        runtime = AgentRuntime(
            client=client,
            tools=[_make_echo_tool()],
            config=AgentRuntimeConfig(
                max_iterations=5, max_consecutive_mistakes=5
            ),
        )
        result = await runtime.run("hi")
        # After the unknown tool error, the LLM gets another turn.
        assert result.status == "completed", f"failed: {result.error}"
        # The tool result message should be an error.
        tool_msgs = [m for m in result.messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_result is not None
        assert tool_msgs[0].tool_result.is_error


# ---------------------------------------------------------------------------
# AgentRuntime — max iterations
# ---------------------------------------------------------------------------


class TestAgentRuntimeMaxIterations:
    @pytest.mark.asyncio
    async def test_max_iterations_reached(self) -> None:
        """LLM keeps calling tools → run stops at max_iterations."""

        # Build a list of tool-call responses long enough to hit the cap.
        responses = [
            _assistant_response(
                tool_calls=[
                    ToolCall(id=f"tc{i}", name="echo", input={"value": "x"})
                ],
                finish_reason="tool_calls",
            )
            for i in range(10)
        ]
        client = _FakeLLMClient(responses)
        runtime = AgentRuntime(
            client=client,
            tools=[_make_echo_tool()],
            config=AgentRuntimeConfig(max_iterations=3),
        )
        result = await runtime.run("hi")
        assert result.status == "stopped"
        assert "max iterations" in result.error.lower()
        assert result.stop_reason == "max_iterations"
        assert result.iterations == 3


# ---------------------------------------------------------------------------
# AgentRuntime — tool list vs registry
# ---------------------------------------------------------------------------


class TestAgentRuntimeToolSources:
    @pytest.mark.asyncio
    async def test_accepts_tool_registry(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_echo_tool())
        client = _FakeLLMClient(
            [_assistant_response(content="ok", finish_reason="stop")]
        )
        runtime = AgentRuntime(client=client, tools=registry)
        result = await runtime.run("hi")
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_accepts_tool_list(self) -> None:
        client = _FakeLLMClient(
            [_assistant_response(content="ok", finish_reason="stop")]
        )
        runtime = AgentRuntime(client=client, tools=[_make_echo_tool()])
        result = await runtime.run("hi")
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# AgentRuntime — loop detection
# ---------------------------------------------------------------------------


class TestAgentRuntimeLoopDetection:
    @pytest.mark.asyncio
    async def test_hard_loop_triggers_error_result(self) -> None:
        """Five identical tool calls → hard loop → tool returns error."""

        # The LLM makes 5 identical calls; the 5th triggers hard loop.
        responses = [
            _assistant_response(
                tool_calls=[
                    ToolCall(id=f"tc{i}", name="echo", input={"value": "same"})
                ],
                finish_reason="tool_calls",
            )
            for i in range(5)
        ]
        # Add a final response after the loop is broken.
        responses.append(_assistant_response(content="done", finish_reason="stop"))
        client = _FakeLLMClient(responses)
        runtime = AgentRuntime(
            client=client,
            tools=[_make_echo_tool()],
            config=AgentRuntimeConfig(
                max_iterations=10,
                loop_soft_threshold=3,
                loop_hard_threshold=5,
                abort_on_hard_loop=True,
            ),
        )
        result = await runtime.run("hi")
        # The hard-loop tool result should be an error.
        tool_msgs = [m for m in result.messages if m.role == "tool"]
        assert any(tm.tool_result and tm.tool_result.is_error for tm in tool_msgs)
        # The error message should mention "loop".
        loop_errors = [
            tm.tool_result.output
            for tm in tool_msgs
            if tm.tool_result and tm.tool_result.is_error
        ]
        assert any("loop" in e.lower() for e in loop_errors)


# ---------------------------------------------------------------------------
# AgentRuntime — abort propagation to LLM
# ---------------------------------------------------------------------------


class TestAgentRuntimeLLMAbort:
    @pytest.mark.asyncio
    async def test_llm_call_can_be_aborted(self) -> None:
        """When abort fires during the LLM call, the run aborts cleanly."""

        class _SlowClient(LLMClient):
            async def complete(
                self,
                request: LLMRequest,
                *,
                abort: asyncio.Event | None = None,
            ) -> LLMResponse:
                # Simulate a long call.
                if abort is None:
                    await asyncio.sleep(10)
                    return _assistant_response()
                # Wait for either completion or abort.
                done, _ = await asyncio.wait(
                    {
                        asyncio.ensure_future(asyncio.sleep(10)),
                        asyncio.ensure_future(abort.wait()),
                    },
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if abort.is_set():
                    raise asyncio.CancelledError("aborted")
                return _assistant_response()

        client = _SlowClient(model="x")
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(max_iterations=5),
        )

        # Trigger abort shortly after run starts.
        async def _trigger() -> None:
            await asyncio.sleep(0.1)
            runtime.abort()

        trigger = asyncio.create_task(_trigger())
        result = await runtime.run("hi")
        await trigger
        assert result.status == "aborted"


# ---------------------------------------------------------------------------
# AgentRuntime — API error retry (mistake "continue" outcome)
# ---------------------------------------------------------------------------


class TestAgentRuntimeAPIErrorRetry:
    """Regression tests for the LLM-call mistake handling.

    Previously, when :meth:`AgentRuntime._call_llm` raised and the
    :class:`MistakeTracker` returned a "continue" outcome, the runtime
    would *return* the mistake result instead of retrying the LLM call.
    These tests pin the correct behavior: continue outcome → retry,
    stop outcome → end the run.
    """

    @pytest.mark.asyncio
    async def test_api_error_then_retry_then_complete(self) -> None:
        """A transient LLM error should be retried, not end the run."""

        class _FlakyClient(LLMClient):
            def __init__(self) -> None:
                super().__init__(model="flaky")
                self._call_count = 0

            async def complete(
                self,
                request: LLMRequest,
                *,
                abort: asyncio.Event | None = None,
            ) -> LLMResponse:
                self._call_count += 1
                if self._call_count == 1:
                    raise RuntimeError("transient API error")
                return _assistant_response(
                    content="recovered", finish_reason="stop"
                )

        client = _FlakyClient()
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(
                max_iterations=5, max_consecutive_mistakes=3
            ),
        )
        result = await runtime.run("hi")
        assert result.status == "completed", f"failed: {result.error}"
        assert result.final_content == "recovered"
        assert client._call_count == 2

    @pytest.mark.asyncio
    async def test_api_error_exceeds_limit_stops(self) -> None:
        """When the mistake limit is hit, the run stops with an error."""

        class _AlwaysFailingClient(LLMClient):
            def __init__(self) -> None:
                super().__init__(model="failing")

            async def complete(
                self,
                request: LLMRequest,
                *,
                abort: asyncio.Event | None = None,
            ) -> LLMResponse:
                raise RuntimeError("permanent API error")

        client = _AlwaysFailingClient()
        runtime = AgentRuntime(
            client=client,
            tools=[],
            config=AgentRuntimeConfig(
                max_iterations=10, max_consecutive_mistakes=2
            ),
        )
        result = await runtime.run("hi")
        assert result.status == "stopped"
        assert "consecutive mistakes" in result.error.lower()
        assert "permanent API error" in result.error


# ---------------------------------------------------------------------------
# AgentRuntimeConfig defaults
# ---------------------------------------------------------------------------


class TestAgentRuntimeConfig:
    def test_defaults(self) -> None:
        cfg = AgentRuntimeConfig()
        assert cfg.max_iterations == 50
        assert cfg.temperature == 0.7
        assert cfg.max_consecutive_mistakes == 3
        assert cfg.loop_soft_threshold == 3
        assert cfg.loop_hard_threshold == 5
        assert cfg.parallel_tool_execution is False
        assert cfg.abort_on_hard_loop is True

    def test_custom_values(self) -> None:
        cfg = AgentRuntimeConfig(
            max_iterations=10,
            temperature=0.1,
            max_consecutive_mistakes=5,
        )
        assert cfg.max_iterations == 10
        assert cfg.temperature == 0.1
        assert cfg.max_consecutive_mistakes == 5


# ---------------------------------------------------------------------------
# AgentRuntime — messages snapshot
# ---------------------------------------------------------------------------


class TestAgentRuntimeMessages:
    @pytest.mark.asyncio
    async def test_messages_property_returns_copy(self) -> None:
        client = _FakeLLMClient(
            [_assistant_response(content="ok", finish_reason="stop")]
        )
        runtime = AgentRuntime(client=client, tools=[])
        result = await runtime.run("hi")
        snapshot1 = runtime.messages
        snapshot2 = runtime.messages
        assert snapshot1 == snapshot2
        assert snapshot1 is not snapshot2
        # Mutating the snapshot shouldn't affect the runtime.
        snapshot1.append(Message(role="user", content="injected"))
        assert len(runtime.messages) == len(result.messages)

    @pytest.mark.asyncio
    async def test_iteration_property(self) -> None:
        client = _FakeLLMClient(
            [_assistant_response(content="ok", finish_reason="stop")]
        )
        runtime = AgentRuntime(client=client, tools=[])
        assert runtime.iteration == 0
        await runtime.run("hi")
        assert runtime.iteration == 1


# ---------------------------------------------------------------------------
# AgentRuntime — parallel tool execution
# ---------------------------------------------------------------------------


class TestAgentRuntimeParallelTools:
    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self) -> None:
        """When parallel_tool_execution=True, all tool calls run concurrently."""

        execution_order: list[str] = []

        class _SlowInput(BaseModel):
            name: str

        async def _slow_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
            assert isinstance(args, _SlowInput)
            execution_order.append(f"start:{args.name}")
            await asyncio.sleep(0.05)
            execution_order.append(f"end:{args.name}")
            return ToolResult.success(args.name)

        slow_tool = Tool(
            id="slow",
            description="Slow tool",
            parameters=_SlowInput,
            execute=_slow_execute,
        )
        client = _FakeLLMClient(
            [
                _assistant_response(
                    tool_calls=[
                        ToolCall(id="tc1", name="slow", input={"name": "a"}),
                        ToolCall(id="tc2", name="slow", input={"name": "b"}),
                    ],
                    finish_reason="tool_calls",
                ),
                _assistant_response(content="done", finish_reason="stop"),
            ]
        )
        runtime = AgentRuntime(
            client=client,
            tools=[slow_tool],
            config=AgentRuntimeConfig(
                max_iterations=5,
                parallel_tool_execution=True,
            ),
        )
        result = await runtime.run("hi")
        assert result.status == "completed"
        # In parallel mode, both starts happen before either ends.
        starts = [i for i in execution_order if i.startswith("start:")]
        ends = [i for i in execution_order if i.startswith("end:")]
        assert len(starts) == 2
        assert len(ends) == 2
        # Both starts should come before any end.
        assert execution_order.index("start:a") < execution_order.index("end:a")
        assert execution_order.index("start:b") < execution_order.index("end:a") or \
               execution_order.index("start:b") < execution_order.index("end:b")
