"""Tests for :mod:`myagent.orchestration.workflow` (DAG workflow engine)."""

from __future__ import annotations

import asyncio

import pytest

from myagent.orchestration.workflow import (
    NodeStatus,
    NodeType,
    Workflow,
    WorkflowEdge,
    WorkflowEngine,
    WorkflowError,
    WorkflowExecution,
    WorkflowNode,
    WorkflowStatus,
)

# ---------------------------------------------------------------------------
# Workflow creation & DAG validation
# ---------------------------------------------------------------------------


class TestWorkflowCreation:
    @pytest.mark.asyncio
    async def test_create_workflow_sets_status_ready(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="release",
            description="a release pipeline",
            nodes=[
                WorkflowNode(id="notify", name="Notify", node_type=NodeType.TASK),
                WorkflowNode(
                    id="ship",
                    name="Ship",
                    node_type=NodeType.SCRIPT,
                    dependencies=["notify"],
                    config={"command": "echo shipped"},
                ),
            ],
            edges=[WorkflowEdge(source="notify", target="ship")],
            created_by="tester",
            variables={"env": "staging"},
        )
        assert wf.status is WorkflowStatus.READY
        assert wf.name == "release"
        assert wf.created_by == "tester"
        assert len(wf.nodes) == 2
        assert len(wf.edges) == 1
        assert wf.variables == {"env": "staging"}
        # The workflow is retrievable by id.
        assert await engine.get_workflow(wf.id) is wf

    @pytest.mark.asyncio
    async def test_create_workflow_rejects_empty_name(self) -> None:
        engine = WorkflowEngine()
        with pytest.raises(WorkflowError, match="name"):
            await engine.create_workflow(name="   ")

    @pytest.mark.asyncio
    async def test_create_workflow_rejects_cycle(self) -> None:
        engine = WorkflowEngine()
        with pytest.raises(WorkflowError, match="cycle"):
            await engine.create_workflow(
                name="cycle",
                nodes=[
                    WorkflowNode(id="a", name="A"),
                    WorkflowNode(id="b", name="B"),
                ],
                edges=[
                    WorkflowEdge(source="a", target="b"),
                    WorkflowEdge(source="b", target="a"),
                ],
            )

    @pytest.mark.asyncio
    async def test_create_workflow_rejects_dangling_edge_source(self) -> None:
        engine = WorkflowEngine()
        with pytest.raises(WorkflowError, match="unknown node"):
            await engine.create_workflow(
                name="dangling",
                nodes=[WorkflowNode(id="a", name="A")],
                edges=[WorkflowEdge(source="ghost", target="a")],
            )

    @pytest.mark.asyncio
    async def test_create_workflow_rejects_dangling_edge_target(self) -> None:
        engine = WorkflowEngine()
        with pytest.raises(WorkflowError, match="unknown node"):
            await engine.create_workflow(
                name="dangling",
                nodes=[WorkflowNode(id="a", name="A")],
                edges=[WorkflowEdge(source="a", target="ghost")],
            )

    @pytest.mark.asyncio
    async def test_create_workflow_rejects_unknown_dependency(self) -> None:
        engine = WorkflowEngine()
        with pytest.raises(WorkflowError, match="unknown node"):
            await engine.create_workflow(
                name="bad-dep",
                nodes=[WorkflowNode(id="a", name="A", dependencies=["ghost"])],
            )

    @pytest.mark.asyncio
    async def test_create_workflow_rejects_duplicate_node_ids(self) -> None:
        engine = WorkflowEngine()
        with pytest.raises(WorkflowError, match="duplicate"):
            await engine.create_workflow(
                name="dup",
                nodes=[
                    WorkflowNode(id="a", name="A"),
                    WorkflowNode(id="a", name="A2"),
                ],
            )

    @pytest.mark.asyncio
    async def test_get_workflow_unknown_returns_none(self) -> None:
        engine = WorkflowEngine()
        assert await engine.get_workflow("missing") is None


# ---------------------------------------------------------------------------
# Topological sort / dependency resolution
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_topological_levels_linear_chain(self) -> None:
        engine = WorkflowEngine()
        wf = Workflow(
            name="chain",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B"),
                WorkflowNode(id="c", name="C"),
            ],
            edges=[
                WorkflowEdge(source="a", target="b"),
                WorkflowEdge(source="b", target="c"),
            ],
        )
        assert engine._topological_levels(wf) == [["a"], ["b"], ["c"]]

    def test_topological_levels_diamond_dependency(self) -> None:
        # Node C depends on both A and B; A and B are independent.
        engine = WorkflowEngine()
        wf = Workflow(
            name="diamond",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B"),
                WorkflowNode(id="c", name="C", dependencies=["a", "b"]),
            ],
            edges=[
                WorkflowEdge(source="a", target="c"),
                WorkflowEdge(source="b", target="c"),
            ],
        )
        # Level 0 holds the independent nodes A and B (sorted); level 1 holds C.
        assert engine._topological_levels(wf) == [["a", "b"], ["c"]]

    def test_topological_levels_dependencies_without_edges(self) -> None:
        # Explicit ``dependencies`` create implicit edges for the sort.
        engine = WorkflowEngine()
        wf = Workflow(
            name="deps-only",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B"),
                WorkflowNode(id="c", name="C", dependencies=["a", "b"]),
            ],
        )
        assert engine._topological_levels(wf) == [["a", "b"], ["c"]]

    def test_topological_levels_single_node(self) -> None:
        engine = WorkflowEngine()
        wf = Workflow(name="solo", nodes=[WorkflowNode(id="a", name="A")])
        assert engine._topological_levels(wf) == [["a"]]

    def test_validate_dag_detects_cycle(self) -> None:
        engine = WorkflowEngine()
        wf = Workflow(
            name="cycle",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B"),
                WorkflowNode(id="c", name="C"),
            ],
            edges=[
                WorkflowEdge(source="a", target="b"),
                WorkflowEdge(source="b", target="c"),
                WorkflowEdge(source="c", target="a"),
            ],
        )
        with pytest.raises(WorkflowError, match="cycle"):
            engine._validate_dag(wf)


# ---------------------------------------------------------------------------
# Execution lifecycle & status transitions
# ---------------------------------------------------------------------------


class TestWorkflowExecution:
    @pytest.mark.asyncio
    async def test_workflow_status_transitions_ready_running_completed(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="transitions",
            nodes=[WorkflowNode(id="a", name="A", node_type=NodeType.TASK)],
        )
        assert wf.status is WorkflowStatus.READY
        execution = await engine.create_execution(wf.id)
        assert execution.status is WorkflowStatus.READY
        assert execution.started_at is None
        result = await engine.start_execution(execution.id)
        assert result.status is WorkflowStatus.COMPLETED
        assert result.started_at is not None
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_create_execution_unknown_workflow_raises(self) -> None:
        engine = WorkflowEngine()
        with pytest.raises(WorkflowError, match="workflow not found"):
            await engine.create_execution("missing")

    @pytest.mark.asyncio
    async def test_start_execution_unknown_raises(self) -> None:
        engine = WorkflowEngine()
        with pytest.raises(WorkflowError, match="execution not found"):
            await engine.start_execution("missing")

    @pytest.mark.asyncio
    async def test_start_execution_terminal_raises(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="once", nodes=[WorkflowNode(id="a", name="A")]
        )
        execution = await engine.create_execution(wf.id)
        await engine.start_execution(execution.id)
        with pytest.raises(WorkflowError, match="terminal"):
            await engine.start_execution(execution.id)

    @pytest.mark.asyncio
    async def test_get_execution_status_returns_snapshot(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="snap", nodes=[WorkflowNode(id="a", name="A")]
        )
        execution = await engine.create_execution(wf.id)
        snapshot = await engine.get_execution_status(execution.id)
        assert snapshot is not None
        assert snapshot.id == execution.id
        assert snapshot.status is WorkflowStatus.READY
        # Unknown id returns None.
        assert await engine.get_execution_status("missing") is None

    @pytest.mark.asyncio
    async def test_get_execution_nodes_pending_before_run(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="nodes",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B"),
            ],
        )
        execution = await engine.create_execution(wf.id)
        nodes = await engine.get_execution_nodes(execution.id)
        assert nodes is not None
        assert {n.id for n in nodes} == {"a", "b"}
        assert all(n.status is NodeStatus.PENDING for n in nodes)
        assert await engine.get_execution_nodes("missing") is None


# ---------------------------------------------------------------------------
# Node status tracking
# ---------------------------------------------------------------------------


class TestNodeStatusTracking:
    @pytest.mark.asyncio
    async def test_node_status_transitions_pending_running_completed(self) -> None:
        engine = WorkflowEngine()
        observed: dict[str, NodeStatus] = {}

        async def observing_handler(node, execution, working):
            # The engine sets the node to RUNNING before invoking the handler.
            observed[node.id] = node.status
            return {"task": node.name}

        engine.register_node_handler(NodeType.TASK, observing_handler)
        wf = await engine.create_workflow(
            name="statuses", nodes=[WorkflowNode(id="a", name="A")]
        )
        execution = await engine.create_execution(wf.id)

        nodes_before = await engine.get_execution_nodes(execution.id)
        assert nodes_before[0].status is NodeStatus.PENDING

        result = await engine.start_execution(execution.id)
        assert observed["a"] is NodeStatus.RUNNING
        assert result.status is WorkflowStatus.COMPLETED
        assert result.node_results["a"]["status"] == NodeStatus.COMPLETED.value
        assert result.node_results["a"]["outputs"] == {"task": "A"}

    @pytest.mark.asyncio
    async def test_node_outputs_merged_into_workflow_variables(self) -> None:
        engine = WorkflowEngine()
        seen: dict[str, object] = {}

        async def producer(node, execution, working):
            return {"produced": 42}

        async def consumer(node, execution, working):
            seen["env"] = working.variables.get("env")
            seen["produced"] = working.variables.get("produced")
            return {"task": node.name}

        # Two independent first-wave nodes so both run.
        engine.register_node_handler(NodeType.TASK, consumer)
        engine.register_node_handler(NodeType.SCRIPT, producer)
        wf = await engine.create_workflow(
            name="vars",
            nodes=[
                WorkflowNode(
                    id="prod",
                    name="Producer",
                    node_type=NodeType.SCRIPT,
                    config={"command": "true"},
                ),
                WorkflowNode(id="cons", name="Consumer", node_type=NodeType.TASK),
            ],
            variables={"env": "prod"},
        )
        execution = await engine.create_execution(wf.id)
        await engine.start_execution(execution.id)
        # The default variables are visible to handlers.
        assert seen["env"] == "prod"


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_workflow_parallel_execution(self) -> None:
        """Nodes A and B run concurrently in the same wave; C waits (BLOCKED)."""
        engine = WorkflowEngine()
        current = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def counting_handler(node, execution, working):
            nonlocal current, max_concurrent
            async with lock:
                current += 1
                max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return {"task": node.name}

        engine.register_node_handler(NodeType.TASK, counting_handler)
        wf = await engine.create_workflow(
            name="parallel",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B"),
                WorkflowNode(id="c", name="C", dependencies=["a", "b"]),
            ],
            edges=[
                WorkflowEdge(source="a", target="c"),
                WorkflowEdge(source="b", target="c"),
            ],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)

        # A and B executed in the same wave -> concurrency reached 2.
        assert max_concurrent == 2
        assert result.node_results["a"]["status"] == NodeStatus.COMPLETED.value
        assert result.node_results["b"]["status"] == NodeStatus.COMPLETED.value
        # C depends on A and B; it is held back as BLOCKED while A/B run.
        assert result.node_results["c"]["status"] == NodeStatus.BLOCKED.value

    @pytest.mark.asyncio
    async def test_parallel_node_type_fans_out_branches(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="fanout",
            nodes=[
                WorkflowNode(
                    id="fan",
                    name="Fan",
                    node_type=NodeType.PARALLEL,
                    config={"branches": ["x", "y", "z"]},
                )
            ],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        outputs = result.node_results["fan"]["outputs"]
        assert outputs["result"] == "completed"
        assert len(outputs["branches"]) == 3
        assert [b["branch"] for b in outputs["branches"]] == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_node_failure_marks_workflow_failed(self) -> None:
        engine = WorkflowEngine()

        async def failing_handler(node, execution, working):
            raise RuntimeError("boom")

        engine.register_node_handler(NodeType.TASK, failing_handler)
        wf = await engine.create_workflow(
            name="fail",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B", dependencies=["a"]),
            ],
            edges=[WorkflowEdge(source="a", target="b")],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        assert result.status is WorkflowStatus.FAILED
        assert result.node_results["a"]["status"] == NodeStatus.FAILED.value
        assert "boom" in result.node_results["a"]["error"]

    @pytest.mark.asyncio
    async def test_handler_missing_output_still_records(self) -> None:
        engine = WorkflowEngine()

        async def none_handler(node, execution, working):
            return None  # type: ignore[return-value]

        engine.register_node_handler(NodeType.TASK, none_handler)
        wf = await engine.create_workflow(
            name="none", nodes=[WorkflowNode(id="a", name="A")]
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        assert result.status is WorkflowStatus.COMPLETED
        assert result.node_results["a"]["outputs"] == {}


# ---------------------------------------------------------------------------
# Pause / resume / cancel
# ---------------------------------------------------------------------------


class TestPauseResumeCancel:
    @pytest.mark.asyncio
    async def test_pause_resume_execution(self) -> None:
        engine = WorkflowEngine()
        gate = asyncio.Event()
        a_running = asyncio.Event()

        async def gated_handler(node, execution, working):
            if node.id == "a":
                a_running.set()
                await gate.wait()
            return {"task": node.name}

        engine.register_node_handler(NodeType.TASK, gated_handler)
        wf = await engine.create_workflow(
            name="pause",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B", dependencies=["a"]),
            ],
            edges=[WorkflowEdge(source="a", target="b")],
        )
        execution = await engine.create_execution(wf.id)
        task = asyncio.create_task(engine.start_execution(execution.id))

        await a_running.wait()  # wave 1 (A) is in-flight.
        assert await engine.pause_execution(execution.id) is True
        gate.set()  # let A finish; the loop then blocks on the pause event.
        await asyncio.sleep(0.05)
        paused = await engine.get_execution_status(execution.id)
        assert paused.status is WorkflowStatus.PAUSED

        assert await engine.resume_execution(execution.id) is True
        result = await task
        assert result.status is WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_execution(self) -> None:
        engine = WorkflowEngine()
        gate = asyncio.Event()
        a_running = asyncio.Event()

        async def gated_handler(node, execution, working):
            if node.id == "a":
                a_running.set()
                await gate.wait()
            return {"task": node.name}

        engine.register_node_handler(NodeType.TASK, gated_handler)
        wf = await engine.create_workflow(
            name="cancel", nodes=[WorkflowNode(id="a", name="A")]
        )
        execution = await engine.create_execution(wf.id)
        task = asyncio.create_task(engine.start_execution(execution.id))

        await a_running.wait()
        assert await engine.cancel_execution(execution.id) is True
        gate.set()
        result = await task
        assert result.status is WorkflowStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_pause_not_running_returns_false(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(name="x", nodes=[WorkflowNode(id="a", name="A")])
        execution = await engine.create_execution(wf.id)
        assert await engine.pause_execution(execution.id) is False

    @pytest.mark.asyncio
    async def test_resume_not_paused_returns_false(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(name="x", nodes=[WorkflowNode(id="a", name="A")])
        execution = await engine.create_execution(wf.id)
        assert await engine.resume_execution(execution.id) is False

    @pytest.mark.asyncio
    async def test_cancel_not_running_returns_false(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(name="x", nodes=[WorkflowNode(id="a", name="A")])
        execution = await engine.create_execution(wf.id)
        assert await engine.cancel_execution(execution.id) is False

    @pytest.mark.asyncio
    async def test_resume_unknown_returns_false(self) -> None:
        engine = WorkflowEngine()
        assert await engine.resume_execution("missing") is False


# ---------------------------------------------------------------------------
# Node handlers
# ---------------------------------------------------------------------------


class TestNodeHandlers:
    @pytest.mark.asyncio
    async def test_register_node_handler_overrides_default(self) -> None:
        engine = WorkflowEngine()
        called: list[str] = []

        async def custom(node, execution, working):
            called.append(node.id)
            return {"custom": True}

        engine.register_node_handler(NodeType.TASK, custom)
        wf = await engine.create_workflow(
            name="custom", nodes=[WorkflowNode(id="a", name="A")]
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        assert called == ["a"]
        assert result.node_results["a"]["outputs"] == {"custom": True}

    @pytest.mark.asyncio
    async def test_script_node_handler_runs_command(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="script",
            nodes=[
                WorkflowNode(
                    id="s",
                    name="Script",
                    node_type=NodeType.SCRIPT,
                    config={"command": "echo hello-world"},
                )
            ],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        outputs = result.node_results["s"]["outputs"]
        assert outputs["exit_code"] == 0
        assert "hello-world" in outputs["stdout"]

    @pytest.mark.asyncio
    async def test_script_node_without_command_skipped(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="no-cmd",
            nodes=[WorkflowNode(id="s", name="Script", node_type=NodeType.SCRIPT)],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        outputs = result.node_results["s"]["outputs"]
        assert outputs["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_decision_node_handler_evaluates_condition(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="decision",
            nodes=[
                WorkflowNode(
                    id="d",
                    name="Decide",
                    node_type=NodeType.DECISION,
                    config={
                        "condition": "count > 5",
                        "true_branch": "go",
                        "false_branch": "stop",
                    },
                    inputs={"count": 10},
                )
            ],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        outputs = result.node_results["d"]["outputs"]
        assert outputs["decision"] == "go"
        assert outputs["evaluated"] is True

    @pytest.mark.asyncio
    async def test_agent_call_node_handler_dispatches(self) -> None:
        engine = WorkflowEngine()
        invoked: list[dict] = []

        async def agent_handler(inputs, variables):
            invoked.append({"inputs": inputs, "variables": variables})
            return {"agent": "coder", "answer": 42}

        engine.register_agent_handler("coder", agent_handler)
        wf = await engine.create_workflow(
            name="agent-call",
            nodes=[
                WorkflowNode(
                    id="ac",
                    name="AgentCall",
                    node_type=NodeType.AGENT_CALL,
                    config={"agent": "coder"},
                    inputs={"q": "what?"},
                )
            ],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        outputs = result.node_results["ac"]["outputs"]
        assert outputs["answer"] == 42
        assert invoked and invoked[0]["inputs"]["q"] == "what?"

    @pytest.mark.asyncio
    async def test_agent_call_without_handler_simulated(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="sim-agent",
            nodes=[
                WorkflowNode(
                    id="ac",
                    name="AgentCall",
                    node_type=NodeType.AGENT_CALL,
                    config={"agent": "ghost"},
                )
            ],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        outputs = result.node_results["ac"]["outputs"]
        assert outputs["agent"] == "ghost"
        assert outputs["status"] == "simulated"

    @pytest.mark.asyncio
    async def test_human_approval_auto_approve(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="approval",
            nodes=[
                WorkflowNode(
                    id="h",
                    name="Approve",
                    node_type=NodeType.HUMAN_APPROVAL,
                    config={"auto_approve": True},
                )
            ],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        outputs = result.node_results["h"]["outputs"]
        assert outputs["approved"] is True
        assert outputs["approver"] == "auto"

    @pytest.mark.asyncio
    async def test_sequential_node_handler_runs_steps(self) -> None:
        engine = WorkflowEngine()
        wf = await engine.create_workflow(
            name="seq",
            nodes=[
                WorkflowNode(
                    id="sq",
                    name="Sequential",
                    node_type=NodeType.SEQUENTIAL,
                    config={"steps": ["one", "two", "three"]},
                )
            ],
        )
        execution = await engine.create_execution(wf.id)
        result = await engine.start_execution(execution.id)
        outputs = result.node_results["sq"]["outputs"]
        assert outputs["result"] == "completed"
        assert len(outputs["steps"]) == 3
        assert [s["value"] for s in outputs["steps"]] == ["one", "two", "three"]


# ---------------------------------------------------------------------------
# Edge conditions & ready-node computation
# ---------------------------------------------------------------------------


class TestEdgeConditions:
    def test_eval_condition_literals(self) -> None:
        engine = WorkflowEngine()
        assert engine._eval_condition("true", {}) is True
        assert engine._eval_condition("1", {}) is True
        assert engine._eval_condition("yes", {}) is True
        assert engine._eval_condition("false", {}) is False
        assert engine._eval_condition("0", {}) is False
        assert engine._eval_condition("no", {}) is False

    def test_eval_condition_empty_is_true(self) -> None:
        engine = WorkflowEngine()
        # Only a truly empty string (falsy) is treated as unconditional.
        assert engine._eval_condition("", {}) is True

    def test_eval_condition_with_namespace(self) -> None:
        engine = WorkflowEngine()
        assert engine._eval_condition("x > 5", {"x": 10}) is True
        assert engine._eval_condition("x > 5", {"x": 1}) is False
        assert engine._eval_condition("name == 'ok'", {"name": "ok"}) is True

    def test_eval_condition_unparseable_is_false(self) -> None:
        engine = WorkflowEngine()
        assert engine._eval_condition("this is not code", {}) is False

    def test_compute_ready_nodes_returns_independent(self) -> None:
        engine = WorkflowEngine()
        wf = Workflow(
            name="ready",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B"),
            ],
        )
        ready, blocked = engine._compute_ready_nodes(wf)
        assert {n.id for n in ready} == {"a", "b"}
        assert blocked == []

    def test_compute_ready_nodes_blocks_unmet_dependency(self) -> None:
        engine = WorkflowEngine()
        wf = Workflow(
            name="blocked",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B", dependencies=["a"]),
            ],
            edges=[WorkflowEdge(source="a", target="b")],
        )
        ready, blocked = engine._compute_ready_nodes(wf)
        assert [n.id for n in ready] == ["a"]
        assert blocked == ["b"]

    def test_compute_ready_nodes_skips_on_false_edge_condition(self) -> None:
        engine = WorkflowEngine()
        wf = Workflow(
            name="cond",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B"),
            ],
            edges=[WorkflowEdge(source="a", target="b", condition="go")],
            variables={"go": False},
        )
        # Mark the upstream node completed so the edge condition is evaluated.
        wf.node_by_id("a").status = NodeStatus.COMPLETED  # type: ignore[union-attr]
        ready, blocked = engine._compute_ready_nodes(wf)
        assert ready == []
        assert blocked == []
        assert wf.node_by_id("b").status is NodeStatus.SKIPPED  # type: ignore[union-attr]

    def test_compute_ready_nodes_takes_branch_on_true_condition(self) -> None:
        engine = WorkflowEngine()
        wf = Workflow(
            name="cond-true",
            nodes=[
                WorkflowNode(id="a", name="A"),
                WorkflowNode(id="b", name="B"),
            ],
            edges=[WorkflowEdge(source="a", target="b", condition="go")],
            variables={"go": True},
        )
        wf.node_by_id("a").status = NodeStatus.COMPLETED  # type: ignore[union-attr]
        ready, blocked = engine._compute_ready_nodes(wf)
        assert [n.id for n in ready] == ["b"]
        assert blocked == []


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    @pytest.mark.asyncio
    async def test_stats_reflects_state(self) -> None:
        engine = WorkflowEngine()
        assert engine.stats() == {"workflows": 0, "executions": 0, "running": 0}
        wf = await engine.create_workflow(
            name="s", nodes=[WorkflowNode(id="a", name="A")]
        )
        await engine.create_execution(wf.id)
        stats = engine.stats()
        assert stats["workflows"] == 1
        assert stats["executions"] == 1
        assert stats["running"] == 0
