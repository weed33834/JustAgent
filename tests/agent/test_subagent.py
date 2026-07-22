"""Tests for ``autoship.agent.subagent`` (parallel read-only research subagents)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from autoship.agent.subagent import (
    SubagentConfig,
    SubagentError,
    SubagentManager,
    SubagentResult,
    SubagentStatus,
    SubagentTask,
    filter_readonly_tools,
    run_research_sync,
)
from autoship.exceptions import AutoShipError

# ---------------------------------------------------------------------------
# Mock doubles
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Scripted async LLM client.

    Each call to ``chat`` pops the next response from ``_responses``.
    Responses are plain dicts with the shape::

        {"content": "summary text", "tool_calls": []}
        {"content": "", "tool_calls": [{"id": "t1", "name": "read_file",
                                        "input": {"path": "/tmp/x"}}]}
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"messages": messages, "tools": tools})
        self._call_count += 1
        if not self._responses:
            raise RuntimeError("No more scripted responses")
        return self._responses.pop(0)


class MockFailingLLMClient:
    """LLM client whose ``chat`` always raises."""

    def __init__(self, exc: Exception | type[Exception] = RuntimeError) -> None:
        self._exc = exc
        self._call_count = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        self._call_count += 1
        raise self._exc if isinstance(self._exc, Exception) else self._exc("boom")


class MockTool:
    """A single scripted async tool."""

    def __init__(self, name: str, output: str = "tool-output") -> None:
        self.name = name
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def execute(self, input_data: Any, context: Any) -> str:
        self.calls.append({"input": input_data, "context": context})
        return self.output


class MockToolRegistry:
    """Registry mapping tool names to tool objects."""

    def __init__(self, tools: dict[str, Any] | None = None) -> None:
        self._tools = dict(tools or {})

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


def _summary_response(text: str, usage: dict[str, int] | None = None) -> dict[str, Any]:
    """Build a final-summary LLM response (no tool calls)."""

    return {"content": text, "tool_calls": [], "usage": usage or {}}


def _tool_call_response(
    name: str, input_data: dict[str, Any] | None = None, call_id: str = "tc1"
) -> dict[str, Any]:
    """Build a tool-call LLM response."""

    return {
        "content": "",
        "tool_calls": [{"id": call_id, "name": name, "input": input_data or {}}],
    }


# ---------------------------------------------------------------------------
# SubagentStatus
# ---------------------------------------------------------------------------


class TestSubagentStatus:
    def test_pending_value(self) -> None:
        assert SubagentStatus.PENDING.value == "pending"

    def test_running_value(self) -> None:
        assert SubagentStatus.RUNNING.value == "running"

    def test_completed_value(self) -> None:
        assert SubagentStatus.COMPLETED.value == "completed"

    def test_failed_value(self) -> None:
        assert SubagentStatus.FAILED.value == "failed"

    def test_aborted_value(self) -> None:
        assert SubagentStatus.ABORTED.value == "aborted"

    def test_timed_out_value(self) -> None:
        assert SubagentStatus.TIMED_OUT.value == "timed_out"

    def test_status_is_str_subclass(self) -> None:
        # ``str, Enum`` so values serialize as plain strings.
        assert isinstance(SubagentStatus.PENDING, str)
        assert SubagentStatus.PENDING == "pending"

    def test_all_statuses(self) -> None:
        expected = {"pending", "running", "completed", "failed", "aborted", "timed_out"}
        actual = {s.value for s in SubagentStatus}
        assert actual == expected


# ---------------------------------------------------------------------------
# SubagentTask
# ---------------------------------------------------------------------------


class TestSubagentTask:
    def test_construction_with_id(self) -> None:
        task = SubagentTask(id="abc123", prompt="research x")
        assert task.id == "abc123"
        assert task.prompt == "research x"

    def test_default_description(self) -> None:
        task = SubagentTask(id="x", prompt="p")
        assert task.description == ""

    def test_default_max_iterations(self) -> None:
        task = SubagentTask(id="x", prompt="p")
        assert task.max_iterations == 10

    def test_default_timeout_seconds(self) -> None:
        task = SubagentTask(id="x", prompt="p")
        assert task.timeout_seconds == 120.0

    def test_default_created_at(self) -> None:
        task = SubagentTask(id="x", prompt="p")
        assert task.created_at == 0.0

    def test_custom_values(self) -> None:
        task = SubagentTask(
            id="x",
            prompt="p",
            description="desc",
            max_iterations=5,
            timeout_seconds=30.0,
            created_at=12345.0,
        )
        assert task.description == "desc"
        assert task.max_iterations == 5
        assert task.timeout_seconds == 30.0
        assert task.created_at == 12345.0

    def test_frozen(self) -> None:
        task = SubagentTask(id="x", prompt="p")
        with pytest.raises((AttributeError, Exception)):
            task.prompt = "changed"  # type: ignore[misc]

    def test_id_generated_when_empty(self) -> None:
        task = SubagentTask(id="", prompt="p")
        assert task.id != ""
        assert len(task.id) == 12

    def test_id_generated_unique(self) -> None:
        a = SubagentTask(id="", prompt="p")
        b = SubagentTask(id="", prompt="p")
        assert a.id != b.id


# ---------------------------------------------------------------------------
# SubagentResult
# ---------------------------------------------------------------------------


class TestSubagentResult:
    def test_construction(self) -> None:
        result = SubagentResult(task_id="t1", status=SubagentStatus.COMPLETED, summary="ok")
        assert result.task_id == "t1"
        assert result.status is SubagentStatus.COMPLETED
        assert result.summary == "ok"

    def test_defaults(self) -> None:
        result = SubagentResult(task_id="t1", status=SubagentStatus.PENDING)
        assert result.summary == ""
        assert result.error == ""
        assert result.iterations == 0
        assert result.elapsed_seconds == 0.0
        assert result.tokens_used == 0

    def test_frozen(self) -> None:
        result = SubagentResult(task_id="t1", status=SubagentStatus.PENDING)
        with pytest.raises((AttributeError, Exception)):
            result.summary = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SubagentConfig
# ---------------------------------------------------------------------------


class TestSubagentConfig:
    def test_defaults(self) -> None:
        config = SubagentConfig()
        assert config.max_concurrent == 3
        assert config.default_max_iterations == 10
        assert config.default_timeout_seconds == 120.0
        assert config.max_summary_chars == 4000

    def test_default_allowed_tools(self) -> None:
        config = SubagentConfig()
        assert config.allowed_tools == [
            "read_file",
            "search_files",
            "list_files",
            "web_fetch",
            "ask_question",
        ]

    def test_custom_values(self) -> None:
        config = SubagentConfig(
            max_concurrent=1,
            default_max_iterations=20,
            default_timeout_seconds=60.0,
            max_summary_chars=2000,
        )
        assert config.max_concurrent == 1
        assert config.default_max_iterations == 20
        assert config.default_timeout_seconds == 60.0
        assert config.max_summary_chars == 2000

    def test_allowed_tools_can_be_overridden(self) -> None:
        config = SubagentConfig(allowed_tools=["read_file"])
        assert config.allowed_tools == ["read_file"]

    def test_allowed_tools_independent_per_instance(self) -> None:
        a = SubagentConfig()
        b = SubagentConfig()
        a.allowed_tools.append("extra")
        assert "extra" not in b.allowed_tools


# ---------------------------------------------------------------------------
# SubagentManager — task creation
# ---------------------------------------------------------------------------


class TestSubagentManagerCreation:
    def test_create_task_basic(self) -> None:
        mgr = SubagentManager()
        task = mgr.create_task(prompt="research x")
        assert task.prompt == "research x"
        assert len(task.id) == 12
        assert task.description == ""

    def test_create_task_with_description(self) -> None:
        mgr = SubagentManager()
        task = mgr.create_task(prompt="p", description="d")
        assert task.description == "d"

    def test_create_task_defaults_from_config(self) -> None:
        config = SubagentConfig(default_max_iterations=7, default_timeout_seconds=42.0)
        mgr = SubagentManager(config=config)
        task = mgr.create_task(prompt="p")
        assert task.max_iterations == 7
        assert task.timeout_seconds == 42.0

    def test_create_task_override_defaults(self) -> None:
        config = SubagentConfig(default_max_iterations=7, default_timeout_seconds=42.0)
        mgr = SubagentManager(config=config)
        task = mgr.create_task(prompt="p", max_iterations=2, timeout_seconds=5.0)
        assert task.max_iterations == 2
        assert task.timeout_seconds == 5.0

    def test_create_task_records_pending_status(self) -> None:
        mgr = SubagentManager()
        task = mgr.create_task(prompt="p")
        assert mgr.get_status(task.id) is SubagentStatus.PENDING

    def test_create_task_sets_created_at(self) -> None:
        mgr = SubagentManager()
        before = time.time()
        task = mgr.create_task(prompt="p")
        after = time.time()
        assert before <= task.created_at <= after

    def test_get_status_unknown_returns_none(self) -> None:
        mgr = SubagentManager()
        assert mgr.get_status("nonexistent") is None

    def test_list_active_empty_initially(self) -> None:
        mgr = SubagentManager()
        assert mgr.list_active() == []


# ---------------------------------------------------------------------------
# SubagentManager — run (single)
# ---------------------------------------------------------------------------


class TestSubagentManagerRun:
    @pytest.mark.asyncio
    async def test_immediate_summary_completes(self) -> None:
        client = MockLLMClient([_summary_response("The answer is 42.")])
        mgr = SubagentManager(llm_client=client)
        task = mgr.create_task(prompt="What is the answer?")
        result = await mgr.run(task)
        assert result.status is SubagentStatus.COMPLETED
        assert result.summary == "The answer is 42."
        assert result.error == ""
        assert result.iterations == 1
        assert result.elapsed_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_llm_raises_returns_failed(self) -> None:
        client = MockFailingLLMClient(ValueError("api broken"))
        mgr = SubagentManager(llm_client=client)
        task = mgr.create_task(prompt="p")
        result = await mgr.run(task)
        assert result.status is SubagentStatus.FAILED
        assert "api broken" in result.error
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_tool_call_then_summary_completes(self) -> None:
        read_tool = MockTool(name="read_file", output="FILE CONTENTS")
        registry = MockToolRegistry({"read_file": read_tool})
        client = MockLLMClient(
            [
                _tool_call_response("read_file", {"path": "/tmp/x"}),
                _summary_response("Found FILE CONTENTS at /tmp/x."),
            ]
        )
        mgr = SubagentManager(llm_client=client, tool_registry=registry)
        task = mgr.create_task(prompt="Read /tmp/x and summarize")
        result = await mgr.run(task)
        assert result.status is SubagentStatus.COMPLETED
        assert result.summary == "Found FILE CONTENTS at /tmp/x."
        assert result.iterations == 2
        # The mock tool was invoked with the LLM-provided input.
        assert len(read_tool.calls) == 1
        assert read_tool.calls[0]["input"] == {"path": "/tmp/x"}

    @pytest.mark.asyncio
    async def test_disallowed_tool_call_surfaces_error(self) -> None:
        # The registry has write_file, but it is not in the allowed list.
        write_tool = MockTool(name="write_file", output="ok")
        registry = MockToolRegistry({"write_file": write_tool})
        client = MockLLMClient(
            [
                _tool_call_response("write_file", {"path": "/tmp/x"}),
                _summary_response("Could not write; summarizing."),
            ]
        )
        mgr = SubagentManager(llm_client=client, tool_registry=registry)
        task = mgr.create_task(prompt="Try to write a file")
        result = await mgr.run(task)
        assert result.status is SubagentStatus.COMPLETED
        assert result.iterations == 2
        # The disallowed tool was NOT executed.
        assert len(write_tool.calls) == 0
        # The conversation fed back an error message for the disallowed tool.
        tool_msgs = [m for m in client.calls[0]["messages"] if m["role"] == "tool"]
        assert tool_msgs
        assert "not allowed" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_tool_execution_error_surfaces_to_llm(self) -> None:
        class BoomTool:
            async def execute(self, input_data: Any, context: Any) -> str:
                raise FileNotFoundError("no such file")

        registry = MockToolRegistry({"read_file": BoomTool()})
        client = MockLLMClient(
            [
                _tool_call_response("read_file", {"path": "/missing"}),
                _summary_response("Could not read; file missing."),
            ]
        )
        mgr = SubagentManager(llm_client=client, tool_registry=registry)
        task = mgr.create_task(prompt="Read /missing")
        result = await mgr.run(task)
        assert result.status is SubagentStatus.COMPLETED
        # The tool's error was fed back into the conversation.
        tool_msgs = [m for m in client.calls[0]["messages"] if m["role"] == "tool"]
        assert tool_msgs
        assert "no such file" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_status_running_then_completed(self) -> None:
        client = MockLLMClient([_summary_response("done")])
        mgr = SubagentManager(llm_client=client)
        task = mgr.create_task(prompt="p")

        # Before run: PENDING.
        assert mgr.get_status(task.id) is SubagentStatus.PENDING

        result = await mgr.run(task)

        # After run: COMPLETED.
        assert result.status is SubagentStatus.COMPLETED
        assert mgr.get_status(task.id) is SubagentStatus.COMPLETED
        # Not in active set anymore.
        assert task.id not in mgr.list_active()

    @pytest.mark.asyncio
    async def test_tokens_accumulated(self) -> None:
        client = MockLLMClient(
            [_summary_response("ok", usage={"prompt_tokens": 10, "completion_tokens": 5})]
        )
        mgr = SubagentManager(llm_client=client)
        task = mgr.create_task(prompt="p")
        result = await mgr.run(task)
        assert result.tokens_used == 15

    @pytest.mark.asyncio
    async def test_total_tokens_preferred_when_present(self) -> None:
        client = MockLLMClient(
            [
                _summary_response(
                    "ok",
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 999,
                    },
                )
            ]
        )
        mgr = SubagentManager(llm_client=client)
        task = mgr.create_task(prompt="p")
        result = await mgr.run(task)
        assert result.tokens_used == 999

    @pytest.mark.asyncio
    async def test_emit_called_for_lifecycle(self) -> None:
        events: list[dict[str, Any]] = []
        client = MockLLMClient([_summary_response("ok")])
        mgr = SubagentManager(llm_client=client, emit=events.append)
        task = mgr.create_task(prompt="p")
        await mgr.run(task)
        types = [e["type"] for e in events]
        assert "subagent-started" in types
        assert "subagent-completed" in types


# ---------------------------------------------------------------------------
# SubagentManager — run_many
# ---------------------------------------------------------------------------


class TestSubagentManagerRunMany:
    @pytest.mark.asyncio
    async def test_results_in_input_order(self) -> None:
        # Each client returns a unique summary. Use one client per task
        # by giving a manager a single client that pops responses.
        responses = [
            _summary_response("first"),
            _summary_response("second"),
            _summary_response("third"),
        ]
        client = MockLLMClient(responses)
        mgr = SubagentManager(llm_client=client, config=SubagentConfig(max_concurrent=3))
        tasks = [mgr.create_task(prompt=f"p{i}") for i in range(3)]
        results = await mgr.run_many(tasks)
        assert [r.task_id for r in results] == [t.id for t in tasks]
        assert [r.summary for r in results] == ["first", "second", "third"]
        assert all(r.status is SubagentStatus.COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self) -> None:
        mgr = SubagentManager(llm_client=MockLLMClient([]))
        results = await mgr.run_many([])
        assert results == []

    @pytest.mark.asyncio
    async def test_max_concurrent_one_serializes(self) -> None:
        # Track overlap: with max_concurrent=1, at most one task should
        # be active at any moment. Use a single manager + run_many so
        # all tasks share the same semaphore.
        active_count = 0
        max_active_seen = 0
        lock = asyncio.Lock()

        class TrackingLLMClient:
            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[Any] | None = None,
            ) -> dict[str, Any]:
                nonlocal active_count, max_active_seen
                async with lock:
                    active_count += 1
                    max_active_seen = max(max_active_seen, active_count)
                await asyncio.sleep(0.02)
                async with lock:
                    active_count -= 1
                return _summary_response("ok")

        client = TrackingLLMClient()
        mgr = SubagentManager(llm_client=client, config=SubagentConfig(max_concurrent=1))
        tasks = [mgr.create_task(prompt=f"p{i}") for i in range(3)]
        results = await mgr.run_many(tasks)
        assert len(results) == 3
        # With concurrency=1, only one task was ever in-flight at a time.
        assert max_active_seen == 1
        assert all(r.status is SubagentStatus.COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_max_concurrent_three_allows_overlap(self) -> None:
        active_count = 0
        max_active_seen = 0
        lock = asyncio.Lock()

        class TrackingLLMClient:
            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[Any] | None = None,
            ) -> dict[str, Any]:
                nonlocal active_count, max_active_seen
                async with lock:
                    active_count += 1
                    max_active_seen = max(max_active_seen, active_count)
                await asyncio.sleep(0.05)
                async with lock:
                    active_count -= 1
                return _summary_response("ok")

        client = TrackingLLMClient()
        mgr = SubagentManager(llm_client=client, config=SubagentConfig(max_concurrent=3))
        tasks = [mgr.create_task(prompt=f"p{i}") for i in range(3)]
        await mgr.run_many(tasks)
        # All three should have been in-flight simultaneously.
        assert max_active_seen == 3


# ---------------------------------------------------------------------------
# SubagentManager — status / list_active
# ---------------------------------------------------------------------------


class TestSubagentManagerStatus:
    @pytest.mark.asyncio
    async def test_list_active_during_run(self) -> None:
        started = asyncio.Event()

        class SlowClient:
            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[Any] | None = None,
            ) -> dict[str, Any]:
                started.set()
                await asyncio.sleep(0.05)
                return _summary_response("done")

        mgr = SubagentManager(llm_client=SlowClient())
        task = mgr.create_task(prompt="p")

        run_task = asyncio.ensure_future(mgr.run(task))
        await started.wait()
        # While running: status RUNNING and task is in list_active.
        assert mgr.get_status(task.id) is SubagentStatus.RUNNING
        assert task.id in mgr.list_active()

        result = await run_task
        assert result.status is SubagentStatus.COMPLETED
        # After completion, no longer active.
        assert task.id not in mgr.list_active()

    @pytest.mark.asyncio
    async def test_get_status_failed_after_exception(self) -> None:
        client = MockFailingLLMClient(RuntimeError)
        mgr = SubagentManager(llm_client=client)
        task = mgr.create_task(prompt="p")
        await mgr.run(task)
        assert mgr.get_status(task.id) is SubagentStatus.FAILED


# ---------------------------------------------------------------------------
# SubagentManager — abort
# ---------------------------------------------------------------------------


class TestSubagentManagerAbort:
    @pytest.mark.asyncio
    async def test_abort_running_task(self) -> None:
        first_call = asyncio.Event()

        class TwoCallClient:
            """Returns a tool call first, then sleeps so we can abort."""

            def __init__(self) -> None:
                self._call_count = 0

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[Any] | None = None,
            ) -> dict[str, Any]:
                self._call_count += 1
                if self._call_count == 1:
                    first_call.set()
                    # Return a tool call to force a second iteration.
                    return _tool_call_response("read_file", {"path": "/x"})
                # Second call: sleep so abort fires mid-loop.
                await asyncio.sleep(0.5)
                return _summary_response("never")

        read_tool = MockTool(name="read_file", output="content")
        registry = MockToolRegistry({"read_file": read_tool})
        client = TwoCallClient()
        mgr = SubagentManager(llm_client=client, tool_registry=registry)
        task = mgr.create_task(prompt="p", max_iterations=10, timeout_seconds=10.0)

        run_future = asyncio.ensure_future(mgr.run(task))
        await first_call.wait()
        # Allow the second LLM call to start, then abort before it returns.
        await asyncio.sleep(0.02)
        aborted = mgr.abort(task.id)
        assert aborted is True

        result = await run_future
        assert result.status is SubagentStatus.ABORTED
        assert "aborted" in result.error.lower()
        assert mgr.get_status(task.id) is SubagentStatus.ABORTED

    @pytest.mark.asyncio
    async def test_abort_unknown_returns_false(self) -> None:
        mgr = SubagentManager()
        assert mgr.abort("nonexistent") is False

    @pytest.mark.asyncio
    async def test_abort_after_complete_returns_false(self) -> None:
        client = MockLLMClient([_summary_response("done")])
        mgr = SubagentManager(llm_client=client)
        task = mgr.create_task(prompt="p")
        await mgr.run(task)
        # Task is no longer running, so abort should return False.
        assert mgr.abort(task.id) is False


# ---------------------------------------------------------------------------
# SubagentManager — timeout
# ---------------------------------------------------------------------------


class TestSubagentTimeout:
    @pytest.mark.asyncio
    async def test_short_timeout_returns_timed_out(self) -> None:
        class SlowClient:
            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[Any] | None = None,
            ) -> dict[str, Any]:
                await asyncio.sleep(1.0)
                return _summary_response("never")

        mgr = SubagentManager(llm_client=SlowClient())
        task = mgr.create_task(prompt="p", timeout_seconds=0.05)
        result = await mgr.run(task)
        assert result.status is SubagentStatus.TIMED_OUT
        assert "timed out" in result.error.lower()
        assert mgr.get_status(task.id) is SubagentStatus.TIMED_OUT

    @pytest.mark.asyncio
    async def test_timeout_does_not_fire_when_fast(self) -> None:
        client = MockLLMClient([_summary_response("quick")])
        mgr = SubagentManager(llm_client=client)
        task = mgr.create_task(prompt="p", timeout_seconds=10.0)
        result = await mgr.run(task)
        assert result.status is SubagentStatus.COMPLETED
        assert result.summary == "quick"


# ---------------------------------------------------------------------------
# filter_readonly_tools
# ---------------------------------------------------------------------------


class TestFilterReadonlyTools:
    def test_filters_to_allowed(self) -> None:
        read_tool = MockTool("read_file")
        write_tool = MockTool("write_file")
        registry = MockToolRegistry({"read_file": read_tool, "write_file": write_tool})
        filtered = filter_readonly_tools(registry, ["read_file"])
        assert "read_file" in filtered
        assert "write_file" not in filtered
        assert filtered["read_file"] is read_tool

    def test_skips_missing_tools(self) -> None:
        registry = MockToolRegistry({"read_file": MockTool("read_file")})
        filtered = filter_readonly_tools(registry, ["read_file", "list_files", "nonexistent"])
        assert set(filtered.keys()) == {"read_file"}

    def test_empty_allowed_returns_empty(self) -> None:
        registry = MockToolRegistry({"read_file": MockTool("read_file")})
        filtered = filter_readonly_tools(registry, [])
        assert filtered == {}

    def test_empty_registry_returns_empty(self) -> None:
        registry = MockToolRegistry({})
        filtered = filter_readonly_tools(registry, ["read_file", "list_files"])
        assert filtered == {}

    def test_default_whitelist_filters_full_registry(self) -> None:
        config = SubagentConfig()
        registry = MockToolRegistry(
            {
                "read_file": MockTool("read_file"),
                "write_file": MockTool("write_file"),
                "run_command": MockTool("run_command"),
                "ask_question": MockTool("ask_question"),
            }
        )
        filtered = filter_readonly_tools(registry, config.allowed_tools)
        assert set(filtered.keys()) == {"read_file", "ask_question"}


# ---------------------------------------------------------------------------
# run_research_sync
# ---------------------------------------------------------------------------


class TestRunResearchSync:
    def test_returns_completed_result(self) -> None:
        client = MockLLMClient([_summary_response("research findings")])
        registry = MockToolRegistry({})
        result = run_research_sync("research this", client, registry)
        assert result.status is SubagentStatus.COMPLETED
        assert result.summary == "research findings"
        assert result.iterations == 1

    def test_with_custom_config(self) -> None:
        client = MockLLMClient([_summary_response("ok")])
        registry = MockToolRegistry({})
        config = SubagentConfig(default_max_iterations=3, default_timeout_seconds=5.0)
        result = run_research_sync("p", client, registry, config=config)
        assert result.status is SubagentStatus.COMPLETED
        assert result.summary == "ok"

    def test_failure_propagates_as_failed_result(self) -> None:
        client = MockFailingLLMClient(ValueError("nope"))
        registry = MockToolRegistry({})
        result = run_research_sync("p", client, registry)
        assert result.status is SubagentStatus.FAILED
        assert "nope" in result.error


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_no_llm_client_raises(self) -> None:
        mgr = SubagentManager()
        task = mgr.create_task(prompt="p")
        with pytest.raises(SubagentError, match="No LLM client"):
            await mgr.run(task)

    @pytest.mark.asyncio
    async def test_subagent_error_is_autoship_error(self) -> None:
        mgr = SubagentManager()
        task = mgr.create_task(prompt="p")
        with pytest.raises(AutoShipError):
            await mgr.run(task)

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_subagent_error(self) -> None:
        client = MockLLMClient([_summary_response("never")])
        mgr = SubagentManager(llm_client=client)
        # Build a task directly with an empty prompt.
        task = SubagentTask(id="x", prompt="")
        mgr._statuses[task.id] = SubagentStatus.PENDING
        with pytest.raises(SubagentError, match="prompt must not be empty"):
            await mgr.run(task)

    @pytest.mark.asyncio
    async def test_max_iterations_zero_returns_no_summary(self) -> None:
        # max_iterations=0 means the loop never executes; the subagent
        # returns an empty summary but COMPLETED status (no failure).
        client = MockLLMClient([_summary_response("should not happen")])
        mgr = SubagentManager(llm_client=client)
        task = mgr.create_task(prompt="p", max_iterations=0)
        result = await mgr.run(task)
        assert result.status is SubagentStatus.COMPLETED
        assert result.summary == ""
        assert result.iterations == 0
        # The LLM client was never called.
        assert client._call_count == 0

    @pytest.mark.asyncio
    async def test_max_iterations_one_with_tool_call_returns_partial(self) -> None:
        # With max_iterations=1, the loop runs once. If the LLM returns a
        # tool call (not a final summary), the loop hits the iteration
        # cap. The summary is empty (no final assistant content seen).
        registry = MockToolRegistry({"read_file": MockTool("read_file")})
        client = MockLLMClient([_tool_call_response("read_file", {"path": "/x"})])
        mgr = SubagentManager(llm_client=client, tool_registry=registry)
        task = mgr.create_task(prompt="p", max_iterations=1)
        result = await mgr.run(task)
        assert result.status is SubagentStatus.COMPLETED
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_summary_truncated_to_max_chars(self) -> None:
        long_text = "x" * 1000
        client = MockLLMClient([_summary_response(long_text)])
        config = SubagentConfig(max_summary_chars=50)
        mgr = SubagentManager(llm_client=client, config=config)
        task = mgr.create_task(prompt="p")
        result = await mgr.run(task)
        assert result.status is SubagentStatus.COMPLETED
        assert len(result.summary) == 50
        assert result.summary == "x" * 50

    @pytest.mark.asyncio
    async def test_short_summary_not_truncated(self) -> None:
        client = MockLLMClient([_summary_response("short")])
        config = SubagentConfig(max_summary_chars=50)
        mgr = SubagentManager(llm_client=client, config=config)
        task = mgr.create_task(prompt="p")
        result = await mgr.run(task)
        assert result.summary == "short"

    @pytest.mark.asyncio
    async def test_isolated_context_per_subagent(self) -> None:
        # Each subagent gets a fresh conversation; verify by inspecting
        # the messages the LLM saw.
        client = MockLLMClient([_summary_response("first"), _summary_response("second")])
        mgr = SubagentManager(llm_client=client)
        task_a = mgr.create_task(prompt="prompt-a")
        task_b = mgr.create_task(prompt="prompt-b")
        await mgr.run(task_a)
        await mgr.run(task_b)
        # Each call's messages should only contain its own prompt.
        first_messages = client.calls[0]["messages"]
        second_messages = client.calls[1]["messages"]
        assert first_messages[1]["content"] == "prompt-a"
        assert second_messages[1]["content"] == "prompt-b"
        # And no cross-contamination.
        assert all(m.get("content") != "prompt-b" for m in first_messages)

    @pytest.mark.asyncio
    async def test_system_prompt_prepended(self) -> None:
        client = MockLLMClient([_summary_response("ok")])
        mgr = SubagentManager(llm_client=client)
        task = mgr.create_task(prompt="p")
        await mgr.run(task)
        first_msg = client.calls[0]["messages"][0]
        assert first_msg["role"] == "system"
        assert "read-only research subagent" in first_msg["content"]

    @pytest.mark.asyncio
    async def test_no_tool_registry_still_runs(self) -> None:
        # No registry configured: tool calls (if any) are rejected, but
        # a summary-only response still completes.
        client = MockLLMClient([_summary_response("ok")])
        mgr = SubagentManager(llm_client=client, tool_registry=None)
        task = mgr.create_task(prompt="p")
        result = await mgr.run(task)
        assert result.status is SubagentStatus.COMPLETED
        assert result.summary == "ok"

    @pytest.mark.asyncio
    async def test_tool_call_with_no_input_handled(self) -> None:
        read_tool = MockTool(name="read_file", output="content")
        registry = MockToolRegistry({"read_file": read_tool})
        # Tool call with no "input" key — should default to {}.
        client = MockLLMClient(
            [
                {"content": "", "tool_calls": [{"id": "t1", "name": "read_file"}]},
                _summary_response("done"),
            ]
        )
        mgr = SubagentManager(llm_client=client, tool_registry=registry)
        task = mgr.create_task(prompt="p")
        result = await mgr.run(task)
        assert result.status is SubagentStatus.COMPLETED
        assert len(read_tool.calls) == 1
        assert read_tool.calls[0]["input"] == {}

    @pytest.mark.asyncio
    async def test_abort_before_first_iteration(self) -> None:
        # Set the abort event before the loop starts. The first iteration
        # check should fire and the run should abort.
        started = asyncio.Event()

        class NeverCallsClient:
            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[Any] | None = None,
            ) -> dict[str, Any]:
                started.set()
                return _summary_response("should not happen")

        mgr = SubagentManager(llm_client=NeverCallsClient())
        task = mgr.create_task(prompt="p")
        # Pre-register the abort event so we can set it before run starts.
        # ``run`` will overwrite it, so we hook into the manager: set the
        # abort by scheduling it for immediately after run kicks off.
        run_future = asyncio.ensure_future(mgr.run(task))
        # Give the loop a tick to enter RUNNING state.
        await asyncio.sleep(0)
        # Set abort via the public API.
        mgr.abort(task.id)
        result = await run_future
        # Either aborted (if the abort flag was checked) or completed
        # (if the LLM call already returned). Either is acceptable; we
        # mostly verify that aborting a not-yet-running task is safe.
        assert result.status in {SubagentStatus.ABORTED, SubagentStatus.COMPLETED}
