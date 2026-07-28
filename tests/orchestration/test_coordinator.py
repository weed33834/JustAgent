"""Tests for :mod:`justagent.orchestration.coordinator` (task coordination)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from justagent.orchestration.coordinator import (
    AgentCapability,
    AgentResult,
    AgentTask,
    CoordinationStrategy,
    CoordinatorConfig,
    CoordinatorError,
    DelegationRecord,
    TaskCoordinator,
)

# ---------------------------------------------------------------------------
# Enums & config
# ---------------------------------------------------------------------------


class TestCoordinationEnums:
    def test_strategy_values(self) -> None:
        assert CoordinationStrategy.ROUND_ROBIN.value == "round_robin"
        assert CoordinationStrategy.LEAST_LOADED.value == "least_loaded"
        assert CoordinationStrategy.CAPABILITY_MATCH.value == "capability_match"
        assert CoordinationStrategy.PRIORITY.value == "priority"
        assert CoordinationStrategy.RANDOM.value == "random"

    def test_strategy_is_str(self) -> None:
        assert isinstance(CoordinationStrategy.ROUND_ROBIN, str)
        assert CoordinationStrategy("round_robin") is CoordinationStrategy.ROUND_ROBIN


class TestCoordinatorConfig:
    def test_defaults(self) -> None:
        config = CoordinatorConfig()
        assert config.strategy is CoordinationStrategy.CAPABILITY_MATCH
        assert config.max_retries == 1
        assert config.timeout == 60.0
        assert config.fallback_enabled is True

    def test_custom(self) -> None:
        config = CoordinatorConfig(
            strategy=CoordinationStrategy.ROUND_ROBIN,
            max_retries=3,
            timeout=5.0,
            fallback_enabled=False,
        )
        assert config.strategy is CoordinationStrategy.ROUND_ROBIN
        assert config.max_retries == 3
        assert config.timeout == 5.0
        assert config.fallback_enabled is False


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class TestAgentTask:
    def test_defaults(self) -> None:
        task = AgentTask(description="do something")
        assert task.description == "do something"
        assert task.required_capabilities == set()
        assert task.priority == 0
        assert task.deadline is None
        assert task.inputs == {}
        assert task.assigned_to is None
        assert task.id

    def test_is_expired_false_without_deadline(self) -> None:
        task = AgentTask(description="x")
        assert task.is_expired is False

    def test_is_expired_true_when_past(self) -> None:
        task = AgentTask(description="x", deadline=time.time() - 10)
        assert task.is_expired is True

    def test_is_expired_false_when_future(self) -> None:
        task = AgentTask(description="x", deadline=time.time() + 100)
        assert task.is_expired is False


class TestAgentResult:
    def test_defaults(self) -> None:
        result = AgentResult(task_id="t1")
        assert result.task_id == "t1"
        assert result.agent_id == ""
        assert result.status == "completed"
        assert result.output == ""
        assert result.data == {}
        assert result.execution_time == 0.0
        assert result.error is None

    def test_succeeded_when_completed_no_error(self) -> None:
        result = AgentResult(task_id="t1", status="completed")
        assert result.succeeded is True

    def test_succeeded_false_when_failed(self) -> None:
        result = AgentResult(task_id="t1", status="failed", error="boom")
        assert result.succeeded is False

    def test_succeeded_false_when_completed_with_error(self) -> None:
        result = AgentResult(task_id="t1", status="completed", error="partial")
        assert result.succeeded is False


class TestDelegationRecord:
    def test_defaults(self) -> None:
        record = DelegationRecord(task_id="t1", from_agent="coordinator", to_agent="a")
        assert record.task_id == "t1"
        assert record.from_agent == "coordinator"
        assert record.to_agent == "a"
        assert record.status == "running"
        assert record.delegated_at is not None

    def test_custom_status(self) -> None:
        record = DelegationRecord(
            task_id="t1", from_agent="mgr", to_agent="worker", status="completed"
        )
        assert record.status == "completed"


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------


class TestAgentRegistration:
    @pytest.mark.asyncio
    async def test_register_agent_returns_descriptor(self) -> None:
        coord = TaskCoordinator()
        desc = await coord.register_agent(
            "coder",
            capabilities={AgentCapability.CODE_GENERATION},
            priority=2,
            name="Coder Agent",
            metadata={"tier": "gold"},
        )
        assert desc.id == "coder"
        assert desc.name == "Coder Agent"
        assert desc.priority == 2
        assert AgentCapability.CODE_GENERATION in desc.capabilities
        assert desc.metadata["tier"] == "gold"

    @pytest.mark.asyncio
    async def test_register_agent_default_name_is_id(self) -> None:
        coord = TaskCoordinator()
        desc = await coord.register_agent("a", capabilities={AgentCapability.REASONING})
        assert desc.name == "a"

    @pytest.mark.asyncio
    async def test_register_agent_empty_id_raises(self) -> None:
        coord = TaskCoordinator()
        with pytest.raises(CoordinatorError, match="empty"):
            await coord.register_agent("")

    @pytest.mark.asyncio
    async def test_register_duplicate_raises(self) -> None:
        coord = TaskCoordinator()
        await coord.register_agent("a", capabilities={AgentCapability.REASONING})
        with pytest.raises(CoordinatorError, match="already registered"):
            await coord.register_agent("a", capabilities={AgentCapability.REASONING})

    @pytest.mark.asyncio
    async def test_deregister_agent_returns_descriptor(self) -> None:
        coord = TaskCoordinator()
        await coord.register_agent("a", capabilities={AgentCapability.REASONING})
        removed = await coord.deregister_agent("a")
        assert removed is not None
        assert removed.id == "a"
        assert await coord.list_agents() == []

    @pytest.mark.asyncio
    async def test_deregister_unknown_returns_none(self) -> None:
        coord = TaskCoordinator()
        assert await coord.deregister_agent("ghost") is None

    @pytest.mark.asyncio
    async def test_list_agents_sorted_by_name(self) -> None:
        coord = TaskCoordinator()
        await coord.register_agent("zeta", capabilities={AgentCapability.REASONING})
        await coord.register_agent("alpha", capabilities={AgentCapability.REASONING})
        agents = await coord.list_agents()
        assert [a.id for a in agents] == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# Agent selection strategies
# ---------------------------------------------------------------------------


def _reasoning_task() -> AgentTask:
    return AgentTask(description="reason", required_capabilities={AgentCapability.REASONING})


class TestSelectionStrategies:
    @pytest.mark.asyncio
    async def test_round_robin_cycles_through_agents(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(strategy=CoordinationStrategy.ROUND_ROBIN)
        )
        await coord.register_agent("a", capabilities={AgentCapability.REASONING})
        await coord.register_agent("b", capabilities={AgentCapability.REASONING})
        await coord.register_agent("c", capabilities={AgentCapability.REASONING})
        task = _reasoning_task()
        # sorted by name: [a, b, c]; rr_index starts at 0.
        assert await coord.select_agent(task) == "a"
        assert await coord.select_agent(task) == "b"
        assert await coord.select_agent(task) == "c"
        assert await coord.select_agent(task) == "a"  # wraps around

    @pytest.mark.asyncio
    async def test_round_robin_skips_ineligible(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(strategy=CoordinationStrategy.ROUND_ROBIN)
        )
        await coord.register_agent("a", capabilities={AgentCapability.REASONING})
        await coord.register_agent("b", capabilities={AgentCapability.CODE_GENERATION})
        task = _reasoning_task()
        # Only "a" is eligible; round-robin always returns "a".
        assert await coord.select_agent(task) == "a"
        assert await coord.select_agent(task) == "a"

    @pytest.mark.asyncio
    async def test_least_loaded_picks_idle_then_loaded(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(strategy=CoordinationStrategy.LEAST_LOADED)
        )
        desc_a = await coord.register_agent("a", capabilities={AgentCapability.REASONING})
        await coord.register_agent("b", capabilities={AgentCapability.REASONING})
        task = _reasoning_task()
        # Both idle; tie broken by name -> "a".
        assert await coord.select_agent(task) == "a"
        # Load "a"; now "b" is least loaded.
        desc_a.active_tasks = 5
        assert await coord.select_agent(task) == "b"

    @pytest.mark.asyncio
    async def test_capability_match_prefers_specialized(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(strategy=CoordinationStrategy.CAPABILITY_MATCH)
        )
        await coord.register_agent("spec", capabilities={AgentCapability.REASONING})
        await coord.register_agent(
            "gen",
            capabilities={AgentCapability.REASONING, AgentCapability.CODE_GENERATION},
        )
        task = _reasoning_task()
        # surplus(spec) = 0, surplus(gen) = 1 -> spec is most specialised.
        assert await coord.select_agent(task) == "spec"

    @pytest.mark.asyncio
    async def test_priority_picks_highest_priority(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(strategy=CoordinationStrategy.PRIORITY)
        )
        await coord.register_agent("low", capabilities={AgentCapability.REASONING}, priority=1)
        await coord.register_agent("high", capabilities={AgentCapability.REASONING}, priority=5)
        task = _reasoning_task()
        assert await coord.select_agent(task) == "high"

    @pytest.mark.asyncio
    async def test_random_returns_eligible_agent(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(strategy=CoordinationStrategy.RANDOM)
        )
        await coord.register_agent("a", capabilities={AgentCapability.REASONING})
        await coord.register_agent("b", capabilities={AgentCapability.REASONING})
        task = _reasoning_task()
        chosen = await coord.select_agent(task)
        assert chosen in {"a", "b"}

    @pytest.mark.asyncio
    async def test_select_agent_override_strategy(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(strategy=CoordinationStrategy.RANDOM)
        )
        await coord.register_agent("a", capabilities={AgentCapability.REASONING})
        await coord.register_agent("b", capabilities={AgentCapability.REASONING})
        task = _reasoning_task()
        # Override the default RANDOM strategy with ROUND_ROBIN.
        assert await coord.select_agent(task, strategy=CoordinationStrategy.ROUND_ROBIN) == "a"

    @pytest.mark.asyncio
    async def test_select_agent_no_capable_raises(self) -> None:
        coord = TaskCoordinator()
        await coord.register_agent("a", capabilities={AgentCapability.CODE_GENERATION})
        task = _reasoning_task()
        with pytest.raises(CoordinatorError, match="no agent capable"):
            await coord.select_agent(task)


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


class TestDelegation:
    @pytest.mark.asyncio
    async def test_delegate_with_handler_succeeds(self) -> None:
        coord = TaskCoordinator()

        async def handler(task: AgentTask) -> AgentResult:
            return AgentResult(task_id=task.id, status="completed", output="done")

        await coord.register_agent(
            "a", capabilities={AgentCapability.REASONING}, handler=handler
        )
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert result.succeeded
        assert result.output == "done"
        assert result.agent_id == "a"
        assert result.task_id == task.id

    @pytest.mark.asyncio
    async def test_delegate_without_handler_simulated(self) -> None:
        coord = TaskCoordinator()
        await coord.register_agent("a", capabilities={AgentCapability.REASONING})
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert result.succeeded
        assert "simulated" in result.output
        assert result.data["simulated"] is True

    @pytest.mark.asyncio
    async def test_delegate_handler_returns_dict_coerced(self) -> None:
        coord = TaskCoordinator()

        async def handler(task: AgentTask) -> dict[str, Any]:
            return {"status": "completed", "output": "from dict", "data": {"k": "v"}}

        await coord.register_agent(
            "a", capabilities={AgentCapability.REASONING}, handler=handler
        )
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert result.succeeded
        assert result.output == "from dict"
        assert result.data == {"k": "v"}
        assert result.agent_id == "a"

    @pytest.mark.asyncio
    async def test_delegate_handler_result_without_agent_id_filled(self) -> None:
        coord = TaskCoordinator()

        async def handler(task: AgentTask) -> AgentResult:
            return AgentResult(task_id=task.id, status="completed", output="ok")

        await coord.register_agent(
            "agent-x", capabilities={AgentCapability.REASONING}, handler=handler
        )
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert result.agent_id == "agent-x"

    @pytest.mark.asyncio
    async def test_delegate_no_capable_returns_failed(self) -> None:
        coord = TaskCoordinator()
        await coord.register_agent("a", capabilities={AgentCapability.CODE_GENERATION})
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert result.status == "failed"
        assert "no capable agent" in result.error

    @pytest.mark.asyncio
    async def test_delegate_expired_task_fails_fast(self) -> None:
        coord = TaskCoordinator()
        await coord.register_agent("a", capabilities={AgentCapability.REASONING})
        task = AgentTask(
            description="late",
            required_capabilities={AgentCapability.REASONING},
            deadline=time.time() - 10,
        )
        result = await coord.delegate(task)
        assert result.status == "failed"
        assert "deadline" in result.error


# ---------------------------------------------------------------------------
# Timeout & retry
# ---------------------------------------------------------------------------


class TestTimeoutAndRetry:
    @pytest.mark.asyncio
    async def test_delegate_timeout(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(
                strategy=CoordinationStrategy.CAPABILITY_MATCH,
                max_retries=0,
                timeout=0.1,
                fallback_enabled=False,
            )
        )

        async def slow_handler(task: AgentTask) -> AgentResult:
            await asyncio.sleep(1.0)
            return AgentResult(task_id=task.id, status="completed")

        await coord.register_agent(
            "a", capabilities={AgentCapability.REASONING}, handler=slow_handler
        )
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert result.status == "timeout"
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_delegate_retries_on_failure(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(max_retries=1, timeout=0, fallback_enabled=False)
        )
        calls = 0

        async def flaky(task: AgentTask) -> AgentResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient")
            return AgentResult(task_id=task.id, status="completed", output="ok")

        await coord.register_agent(
            "a", capabilities={AgentCapability.REASONING}, handler=flaky
        )
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert result.succeeded
        assert calls == 2  # one failure + one retry that succeeds

    @pytest.mark.asyncio
    async def test_delegate_exhausts_retries(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(max_retries=1, timeout=0, fallback_enabled=False)
        )
        calls = 0

        async def always_fail(task: AgentTask) -> AgentResult:
            nonlocal calls
            calls += 1
            raise RuntimeError("always broken")

        await coord.register_agent(
            "a", capabilities={AgentCapability.REASONING}, handler=always_fail
        )
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert result.status == "failed"
        assert "always broken" in (result.error or "")
        assert calls == 2  # max_retries + 1 attempts


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


class TestFallback:
    @pytest.mark.asyncio
    async def test_delegate_fallback_to_another_agent(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(max_retries=0, timeout=0, fallback_enabled=True)
        )
        tried: list[str] = []

        async def failing(task: AgentTask) -> AgentResult:
            tried.append(task.assigned_to or "")
            raise RuntimeError("always fails")

        async def succeeding(task: AgentTask) -> AgentResult:
            tried.append(task.assigned_to or "")
            return AgentResult(task_id=task.id, status="completed", output="ok")

        await coord.register_agent(
            "bad", capabilities={AgentCapability.REASONING}, handler=failing
        )
        await coord.register_agent(
            "good", capabilities={AgentCapability.REASONING}, handler=succeeding
        )
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert result.succeeded
        assert result.agent_id == "good"
        assert "bad" in tried and "good" in tried

    @pytest.mark.asyncio
    async def test_fallback_disabled_does_not_retry_other_agent(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(max_retries=0, timeout=0, fallback_enabled=False)
        )
        tried: list[str] = []

        async def failing(task: AgentTask) -> AgentResult:
            tried.append(task.assigned_to or "")
            raise RuntimeError("fails")

        async def succeeding(task: AgentTask) -> AgentResult:
            tried.append(task.assigned_to or "")
            return AgentResult(task_id=task.id, status="completed")

        await coord.register_agent(
            "bad", capabilities={AgentCapability.REASONING}, handler=failing
        )
        await coord.register_agent(
            "good", capabilities={AgentCapability.REASONING}, handler=succeeding
        )
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert not result.succeeded
        # Only the first-selected agent ("bad") was tried.
        assert tried == ["bad"]


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class TestCancelTask:
    @pytest.mark.asyncio
    async def test_cancel_task_inactive_returns_false(self) -> None:
        coord = TaskCoordinator()
        assert await coord.cancel_task("ghost") is False

    @pytest.mark.asyncio
    async def test_cancel_active_task_returns_true(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(timeout=0.0, max_retries=0, fallback_enabled=False)
        )
        started = asyncio.Event()

        async def blocking(task: AgentTask) -> AgentResult:
            started.set()
            await asyncio.sleep(100)
            return AgentResult(task_id=task.id, status="completed")

        await coord.register_agent(
            "a", capabilities={AgentCapability.REASONING}, handler=blocking
        )
        my_task = _reasoning_task()
        delegate_task = asyncio.create_task(coord.delegate(my_task))

        await started.wait()  # handler is running -> task is active.
        assert my_task.id in await coord.get_active_tasks()

        assert await coord.cancel_task(my_task.id) is True
        # The delegation coroutine surfaces the cancellation.
        with pytest.raises(asyncio.CancelledError):
            await delegate_task
        # The active-task bookkeeping is cleaned up by the finally block.
        assert my_task.id not in await coord.get_active_tasks()


# ---------------------------------------------------------------------------
# Delegation record tracking
# ---------------------------------------------------------------------------


class TestDelegationTracking:
    @pytest.mark.asyncio
    async def test_track_delegation_directly(self) -> None:
        coord = TaskCoordinator()
        record = await coord.track_delegation("t1", "mgr", "worker", "running")
        assert record.task_id == "t1"
        assert record.from_agent == "mgr"
        assert record.to_agent == "worker"
        assert record.status == "running"
        fetched = await coord.get_delegation("t1")
        assert fetched is not None
        assert fetched.to_agent == "worker"

    @pytest.mark.asyncio
    async def test_get_delegation_unknown_returns_none(self) -> None:
        coord = TaskCoordinator()
        assert await coord.get_delegation("ghost") is None

    @pytest.mark.asyncio
    async def test_get_delegation_returns_copy(self) -> None:
        coord = TaskCoordinator()
        await coord.track_delegation("t1", "coordinator", "a", "running")
        first = await coord.get_delegation("t1")
        second = await coord.get_delegation("t1")
        assert first is not None and second is not None
        assert first is not second  # model_copy(deep=True)

    @pytest.mark.asyncio
    async def test_delegation_record_finalised_after_success(self) -> None:
        coord = TaskCoordinator()

        async def handler(task: AgentTask) -> AgentResult:
            return AgentResult(task_id=task.id, status="completed", output="ok")

        await coord.register_agent(
            "a", capabilities={AgentCapability.REASONING}, handler=handler
        )
        task = _reasoning_task()
        await coord.delegate(task)
        record = await coord.get_delegation(task.id)
        assert record is not None
        assert record.to_agent == "a"
        assert record.from_agent == "coordinator"
        assert record.status == "completed"

    @pytest.mark.asyncio
    async def test_delegation_record_finalised_after_failure(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(max_retries=0, timeout=0, fallback_enabled=False)
        )

        async def failing(task: AgentTask) -> AgentResult:
            raise RuntimeError("nope")

        await coord.register_agent(
            "a", capabilities={AgentCapability.REASONING}, handler=failing
        )
        task = _reasoning_task()
        result = await coord.delegate(task)
        assert result.status == "failed"
        record = await coord.get_delegation(task.id)
        assert record is not None
        assert record.status == "failed"

    @pytest.mark.asyncio
    async def test_get_active_tasks_tracks_in_flight(self) -> None:
        coord = TaskCoordinator(
            config=CoordinatorConfig(timeout=0.0, max_retries=0, fallback_enabled=False)
        )
        started = asyncio.Event()

        async def blocking(task: AgentTask) -> AgentResult:
            started.set()
            await asyncio.sleep(100)
            return AgentResult(task_id=task.id, status="completed")

        await coord.register_agent(
            "a", capabilities={AgentCapability.REASONING}, handler=blocking
        )
        my_task = _reasoning_task()
        delegate_task = asyncio.create_task(coord.delegate(my_task))
        await started.wait()
        assert my_task.id in await coord.get_active_tasks()
        await coord.cancel_task(my_task.id)
        with pytest.raises(asyncio.CancelledError):
            await delegate_task
        assert await coord.get_active_tasks() == []


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _result(
    *,
    task_id: str = "t",
    agent: str = "",
    status: str = "completed",
    output: str = "",
    data: dict[str, Any] | None = None,
    error: str | None = None,
    execution_time: float = 0.0,
) -> AgentResult:
    return AgentResult(
        task_id=task_id,
        agent_id=agent,
        status=status,
        output=output,
        data=data or {},
        error=error,
        execution_time=execution_time,
    )


class TestAggregateResults:
    @pytest.mark.asyncio
    async def test_aggregate_empty_returns_failed(self) -> None:
        coord = TaskCoordinator()
        result = await coord.aggregate_results([])
        assert result.status == "failed"
        assert "no results" in (result.error or "")

    @pytest.mark.asyncio
    async def test_aggregate_single_returns_same(self) -> None:
        coord = TaskCoordinator()
        single = _result(output="only", execution_time=1.0)
        result = await coord.aggregate_results([single])
        assert result is single

    @pytest.mark.asyncio
    async def test_aggregate_all_succeeded(self) -> None:
        coord = TaskCoordinator()
        r1 = _result(agent="a", output="first", data={"x": 1}, execution_time=1.0)
        r2 = _result(agent="b", output="second", data={"y": 2}, execution_time=2.0)
        result = await coord.aggregate_results([r1, r2])
        assert result.status == "completed"
        assert "first" in result.output and "second" in result.output
        assert result.data["count"] == 2
        assert result.data["partial_results"] == {"x": 1, "y": 2}
        assert result.execution_time == 3.0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_aggregate_partial_when_some_failed(self) -> None:
        coord = TaskCoordinator()
        r1 = _result(agent="a", status="completed", output="ok")
        r2 = _result(agent="b", status="failed", error="boom")
        result = await coord.aggregate_results([r1, r2])
        assert result.status == "partial"
        assert "boom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_aggregate_all_failed(self) -> None:
        coord = TaskCoordinator()
        r1 = _result(agent="a", status="failed", error="e1")
        r2 = _result(agent="b", status="failed", error="e2")
        result = await coord.aggregate_results([r1, r2])
        assert result.status == "failed"
        assert "e1" in (result.error or "") and "e2" in (result.error or "")

    @pytest.mark.asyncio
    async def test_aggregate_merges_data_later_wins(self) -> None:
        coord = TaskCoordinator()
        r1 = _result(data={"key": "first"})
        r2 = _result(data={"key": "second"})
        result = await coord.aggregate_results([r1, r2])
        assert result.data["partial_results"]["key"] == "second"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestCoordinatorStats:
    @pytest.mark.asyncio
    async def test_stats_reflects_state(self) -> None:
        coord = TaskCoordinator()
        await coord.register_agent(
            "a",
            capabilities={AgentCapability.REASONING, AgentCapability.CODE_GENERATION},
        )
        await coord.register_agent("b", capabilities={AgentCapability.REASONING})
        stats = coord.stats()
        assert stats["agents"] == 2
        assert stats["by_capability"]["reasoning"] == 2
        assert stats["by_capability"]["code_generation"] == 1
        assert stats["active_tasks"] == 0
        assert stats["delegations"] == 0
        assert stats["strategy"] == "capability_match"

    @pytest.mark.asyncio
    async def test_stats_counts_delegations(self) -> None:
        coord = TaskCoordinator()

        async def handler(task: AgentTask) -> AgentResult:
            return AgentResult(task_id=task.id, status="completed")

        await coord.register_agent(
            "a", capabilities={AgentCapability.REASONING}, handler=handler
        )
        await coord.delegate(_reasoning_task())
        stats = coord.stats()
        assert stats["delegations"] == 1
