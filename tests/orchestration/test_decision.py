"""Tests for :mod:`justagent.orchestration.decision` (intent parsing & execution)."""

from __future__ import annotations

from typing import Any

import pytest

from justagent.orchestration.decision import (
    DecisionAction,
    DecisionError,
    DecisionExecutor,
    DecisionIntent,
    DecisionStatus,
    DecisionType,
    ExecutionResult,
    IntentParser,
)

# ---------------------------------------------------------------------------
# DecisionType / DecisionStatus enums
# ---------------------------------------------------------------------------


class TestDecisionEnums:
    def test_decision_type_values(self) -> None:
        assert DecisionType.NOTIFY.value == "notify"
        assert DecisionType.SCHEDULE.value == "schedule"
        assert DecisionType.ALLOCATE.value == "allocate"
        assert DecisionType.QUERY.value == "query"
        assert DecisionType.APPROVE.value == "approve"
        assert DecisionType.DEPLOY.value == "deploy"
        assert DecisionType.CONFIGURE.value == "configure"
        assert DecisionType.ANALYZE.value == "analyze"

    def test_decision_type_is_str(self) -> None:
        assert isinstance(DecisionType.NOTIFY, str)
        assert DecisionType("notify") is DecisionType.NOTIFY

    def test_decision_status_values(self) -> None:
        assert DecisionStatus.PENDING.value == "pending"
        assert DecisionStatus.VALIDATED.value == "validated"
        assert DecisionStatus.EXECUTING.value == "executing"
        assert DecisionStatus.COMPLETED.value == "completed"
        assert DecisionStatus.FAILED.value == "failed"
        assert DecisionStatus.CANCELLED.value == "cancelled"

    def test_decision_status_is_str(self) -> None:
        assert isinstance(DecisionStatus.COMPLETED, str)


# ---------------------------------------------------------------------------
# DecisionAction / DecisionIntent / ExecutionResult models
# ---------------------------------------------------------------------------


class TestDecisionModels:
    def test_decision_action_defaults(self) -> None:
        action = DecisionAction()
        assert action.action_type is DecisionType.QUERY
        assert action.parameters == {}
        assert action.target == ""
        assert action.priority == 0
        assert action.requires_approval is False
        assert action.id  # auto-generated

    def test_decision_action_construction(self) -> None:
        action = DecisionAction(
            action_type=DecisionType.NOTIFY,
            parameters={"message": "hi"},
            target="engineering",
            priority=2,
            requires_approval=True,
        )
        assert action.action_type is DecisionType.NOTIFY
        assert action.parameters == {"message": "hi"}
        assert action.target == "engineering"
        assert action.priority == 2
        assert action.requires_approval is True

    def test_decision_intent_defaults(self) -> None:
        intent = DecisionIntent(raw_text="notify team")
        assert intent.raw_text == "notify team"
        assert intent.parsed_actions == []
        assert intent.confidence == 0.0
        assert intent.source == "manager"
        assert intent.id

    def test_primary_action_returns_highest_priority(self) -> None:
        low = DecisionAction(action_type=DecisionType.QUERY, priority=0)
        high = DecisionAction(action_type=DecisionType.NOTIFY, priority=2)
        mid = DecisionAction(action_type=DecisionType.SCHEDULE, priority=1)
        intent = DecisionIntent(raw_text="x", parsed_actions=[low, mid, high])
        assert intent.primary_action is high

    def test_primary_action_none_when_empty(self) -> None:
        intent = DecisionIntent(raw_text="x")
        assert intent.primary_action is None

    def test_execution_result_defaults(self) -> None:
        result = ExecutionResult(decision_id="abc")
        assert result.decision_id == "abc"
        assert result.status is DecisionStatus.PENDING
        assert result.results == []
        assert result.errors == []
        assert result.completed_at is None
        assert result.started_at is not None


# ---------------------------------------------------------------------------
# IntentParser — single-clause intents
# ---------------------------------------------------------------------------


class TestIntentParserNotify:
    def test_parse_notify_extracts_audience_and_message(self) -> None:
        parser = IntentParser()
        intent = parser.parse("notify all employees about the new policy")
        assert len(intent.parsed_actions) == 1
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.NOTIFY
        assert action.target == "all employees"
        assert action.parameters["audience"] == "all employees"
        assert action.parameters["message"] == "the new policy"
        # NOTIFY carries priority 1 by default.
        assert action.priority == 1
        assert action.requires_approval is False

    def test_parse_notify_without_about_uses_full_clause_as_message(self) -> None:
        parser = IntentParser()
        intent = parser.parse("tell the engineering team")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.NOTIFY
        assert action.target == "the engineering team"

    def test_parse_alert_classified_as_notify(self) -> None:
        parser = IntentParser()
        intent = parser.parse("alert on-call about the outage")
        assert intent.parsed_actions[0].action_type is DecisionType.NOTIFY


class TestIntentParserSchedule:
    def test_parse_schedule_extracts_attendees(self) -> None:
        parser = IntentParser()
        intent = parser.parse("schedule a meeting for engineering")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.SCHEDULE
        assert action.parameters["attendees"] == "engineering"
        assert action.target == "engineering"
        # SCHEDULE is not in the elevated-priority set.
        assert action.priority == 0
        assert action.requires_approval is False

    def test_parse_schedule_extracts_time_at(self) -> None:
        parser = IntentParser()
        intent = parser.parse("schedule a meeting with leadership at 3pm")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.SCHEDULE
        assert action.parameters["time"] == "3:00 PM"
        assert action.parameters["attendees"] == "leadership"

    def test_parse_schedule_with_in_duration(self) -> None:
        parser = IntentParser()
        intent = parser.parse("schedule a meeting in 30 minutes")
        action = intent.parsed_actions[0]
        assert action.parameters["time"] == "in 30 minutes"

    def test_parse_schedule_tomorrow(self) -> None:
        parser = IntentParser()
        intent = parser.parse("schedule a review meeting tomorrow")
        action = intent.parsed_actions[0]
        assert action.parameters["time"] == "tomorrow"


class TestIntentParserAllocate:
    def test_parse_allocate_extracts_target(self) -> None:
        parser = IntentParser()
        intent = parser.parse("allocate resource Z")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.ALLOCATE
        assert action.target == "resource Z"
        assert action.parameters["count"] == 1
        assert action.parameters["purpose"] == "resource Z"

    def test_parse_allocate_leading_count(self) -> None:
        # ``allocate 5 servers`` — count regex only matches leading integers, and
        # the clause begins with the verb, so count stays at the default 1.
        parser = IntentParser()
        intent = parser.parse("allocate 5 servers")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.ALLOCATE
        assert action.target == "5 servers"

    def test_parse_provision_classified_as_allocate(self) -> None:
        parser = IntentParser()
        intent = parser.parse("provision three database instances")
        assert intent.parsed_actions[0].action_type is DecisionType.ALLOCATE


class TestIntentParserOtherTypes:
    def test_parse_deploy_requires_approval(self) -> None:
        parser = IntentParser()
        intent = parser.parse("deploy to production")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.DEPLOY
        assert action.requires_approval is True
        # "production" matches the urgency regex, so priority is 2 (urgent).
        assert action.priority == 2
        assert action.parameters["environment"] == "production"

    def test_parse_deploy_staging_default(self) -> None:
        parser = IntentParser()
        intent = parser.parse("deploy the api service")
        action = intent.parsed_actions[0]
        assert action.parameters["environment"] == "staging"

    def test_parse_approve_requires_approval(self) -> None:
        parser = IntentParser()
        intent = parser.parse("approve the budget proposal")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.APPROVE
        assert action.requires_approval is True

    def test_parse_configure(self) -> None:
        parser = IntentParser()
        intent = parser.parse("configure the retry limit")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.CONFIGURE
        assert action.parameters["setting"] == "the retry limit"

    def test_parse_analyze(self) -> None:
        parser = IntentParser()
        intent = parser.parse("analyze the quarterly report")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.ANALYZE
        assert action.parameters["subject"] == "the quarterly report"

    def test_parse_unknown_intent_defaults_to_query(self) -> None:
        parser = IntentParser()
        intent = parser.parse("do something weird")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.QUERY
        assert action.parameters["query"] == "do something weird"

    def test_parse_query_strips_verb(self) -> None:
        parser = IntentParser()
        intent = parser.parse("search for active projects")
        action = intent.parsed_actions[0]
        assert action.action_type is DecisionType.QUERY
        assert action.parameters["query"] == "for active projects"


# ---------------------------------------------------------------------------
# IntentParser — compound intents
# ---------------------------------------------------------------------------


class TestIntentParserCompound:
    def test_parse_compound_notify_and_schedule(self) -> None:
        parser = IntentParser()
        intent = parser.parse(
            "notify all employees about the new policy and "
            "schedule a review meeting tomorrow 3pm with engineering"
        )
        assert len(intent.parsed_actions) == 2
        notify, schedule = intent.parsed_actions
        assert notify.action_type is DecisionType.NOTIFY
        assert notify.target == "all employees"
        assert schedule.action_type is DecisionType.SCHEDULE
        assert schedule.parameters["attendees"] == "engineering"

    def test_parse_compound_split_on_then(self) -> None:
        parser = IntentParser()
        intent = parser.parse("allocate servers then deploy the service")
        assert len(intent.parsed_actions) == 2
        assert intent.parsed_actions[0].action_type is DecisionType.ALLOCATE
        assert intent.parsed_actions[1].action_type is DecisionType.DEPLOY

    def test_parse_urgent_elevates_priority_and_approval(self) -> None:
        parser = IntentParser()
        intent = parser.parse("urgent notify team about the outage")
        action = intent.parsed_actions[0]
        assert action.priority == 2
        assert action.requires_approval is True

    def test_parse_source_propagated(self) -> None:
        parser = IntentParser()
        intent = parser.parse("notify team", source="cli")
        assert intent.source == "cli"


# ---------------------------------------------------------------------------
# IntentParser — confidence scoring
# ---------------------------------------------------------------------------


class TestConfidenceScoring:
    def test_confidence_notify_with_target_and_params(self) -> None:
        parser = IntentParser()
        intent = parser.parse("notify all employees about the new policy")
        # 0.5 base + 0.2 matched + 0.1 target + 0.1 params == 0.9
        assert intent.confidence == pytest.approx(0.9)

    def test_confidence_schedule_with_target_and_params(self) -> None:
        parser = IntentParser()
        intent = parser.parse("schedule a meeting for engineering")
        assert intent.confidence == pytest.approx(0.9)

    def test_confidence_urgent_caps_at_0_95(self) -> None:
        parser = IntentParser()
        intent = parser.parse("urgent notify team about the outage")
        assert intent.confidence == pytest.approx(0.95)

    def test_confidence_unknown_intent_lower(self) -> None:
        parser = IntentParser()
        intent = parser.parse("do something weird")
        # 0.5 base + 0.1 target + 0.1 params (no matched bonus) == 0.7
        assert intent.confidence == pytest.approx(0.7)

    def test_confidence_in_range(self) -> None:
        parser = IntentParser()
        for text in [
            "notify team",
            "schedule meeting",
            "allocate resources",
            "deploy to production",
            "query the database",
        ]:
            intent = parser.parse(text)
            assert 0.0 <= intent.confidence <= 0.95


# ---------------------------------------------------------------------------
# IntentParser — error handling
# ---------------------------------------------------------------------------


class TestIntentParserErrors:
    def test_parse_empty_raises(self) -> None:
        parser = IntentParser()
        with pytest.raises(DecisionError, match="empty"):
            parser.parse("")

    def test_parse_whitespace_only_raises(self) -> None:
        parser = IntentParser()
        with pytest.raises(DecisionError, match="empty"):
            parser.parse("   ")


# ---------------------------------------------------------------------------
# DecisionExecutor — validation & permissions
# ---------------------------------------------------------------------------


class TestExecutorValidation:
    def test_validate_decision_rejects_empty_actions(self) -> None:
        executor = DecisionExecutor()
        intent = DecisionIntent(raw_text="x", parsed_actions=[])
        with pytest.raises(DecisionError, match="no parsed actions"):
            executor.validate_decision(intent)

    def test_validate_decision_rejects_low_confidence(self) -> None:
        executor = DecisionExecutor(confidence_threshold=0.95)
        intent = IntentParser().parse("notify team about the update")
        assert intent.confidence < 0.95
        with pytest.raises(DecisionError, match="confidence"):
            executor.validate_decision(intent)

    def test_check_permissions_default_allows(self) -> None:
        executor = DecisionExecutor()
        intent = IntentParser().parse("notify team")
        assert executor.check_permissions(intent) is True

    def test_check_permissions_custom_denies(self) -> None:
        executor = DecisionExecutor()
        executor.register_permission_checker(lambda decision: False)
        intent = IntentParser().parse("notify team")
        assert executor.check_permissions(intent) is False

    def test_check_permissions_checker_error_denies(self) -> None:
        executor = DecisionExecutor()

        def boom(_: DecisionIntent) -> bool:
            raise RuntimeError("checker broken")

        executor.register_permission_checker(boom)
        intent = IntentParser().parse("notify team")
        assert executor.check_permissions(intent) is False


# ---------------------------------------------------------------------------
# DecisionExecutor — execution with mock handlers
# ---------------------------------------------------------------------------


class TestExecutorExecution:
    @pytest.mark.asyncio
    async def test_execute_with_mock_handler_completes(self) -> None:
        executor = DecisionExecutor()
        received: list[DecisionAction] = []

        async def notify_handler(action: DecisionAction) -> dict[str, Any]:
            received.append(action)
            return {"status": "completed", "audience": action.target, "sent": 5}

        executor.register_handler(DecisionType.NOTIFY, notify_handler)
        intent = IntentParser().parse("notify engineering about the deploy")
        result = await executor.execute(intent)

        assert result.status is DecisionStatus.COMPLETED
        assert result.errors == []
        assert len(result.results) == 1
        assert result.results[0]["action_type"] == "notify"
        assert result.results[0]["sent"] == 5
        assert result.completed_at is not None
        assert received[0].action_type is DecisionType.NOTIFY

    @pytest.mark.asyncio
    async def test_execute_default_handler_falls_back_to_simulated(self) -> None:
        executor = DecisionExecutor()
        intent = IntentParser().parse("query the project status")
        result = await executor.execute(intent)
        assert result.status is DecisionStatus.COMPLETED
        assert result.results[0].get("simulated") is True

    @pytest.mark.asyncio
    async def test_execute_permission_denied_returns_failed(self) -> None:
        executor = DecisionExecutor()
        executor.register_permission_checker(lambda decision: False)
        intent = IntentParser().parse("notify team about the update")
        result = await executor.execute(intent)
        assert result.status is DecisionStatus.FAILED
        assert "permission denied" in result.errors
        assert result.results == []

    @pytest.mark.asyncio
    async def test_execute_handler_failure_marks_failed(self) -> None:
        executor = DecisionExecutor()

        async def boom(action: DecisionAction) -> dict[str, Any]:
            raise RuntimeError("handler exploded")

        executor.register_handler(DecisionType.NOTIFY, boom)
        intent = IntentParser().parse("notify team about the update")
        result = await executor.execute(intent)
        assert result.status is DecisionStatus.FAILED
        assert result.errors  # non-empty
        assert result.results[0]["status"] == "failed"
        assert "handler exploded" in result.results[0]["error"]

    @pytest.mark.asyncio
    async def test_execute_low_confidence_raises(self) -> None:
        executor = DecisionExecutor(confidence_threshold=0.95)
        intent = IntentParser().parse("notify team about the update")
        with pytest.raises(DecisionError, match="confidence"):
            await executor.execute(intent)


# ---------------------------------------------------------------------------
# DecisionExecutor — approval gate
# ---------------------------------------------------------------------------


class TestExecutorApprovalGate:
    @pytest.mark.asyncio
    async def test_deploy_denied_without_approval(self) -> None:
        executor = DecisionExecutor()
        intent = IntentParser().parse("deploy to production")
        action = intent.parsed_actions[0]
        assert action.requires_approval is True
        result = await executor.execute(intent)
        # Default approver denies unknown actions.
        assert result.status is DecisionStatus.FAILED
        assert result.results[0]["status"] == "denied"

    @pytest.mark.asyncio
    async def test_deploy_approved_via_pre_approval(self) -> None:
        executor = DecisionExecutor()
        intent = IntentParser().parse("deploy to production")
        action = intent.parsed_actions[0]
        executor.approve_action(action.id)
        result = await executor.execute(intent)
        assert result.status is DecisionStatus.COMPLETED
        assert result.results[0].get("simulated") is True

    @pytest.mark.asyncio
    async def test_deploy_approved_via_custom_approver(self) -> None:
        executor = DecisionExecutor()

        async def always_approve(action: DecisionAction) -> bool:
            return True

        executor.register_approver(always_approve)
        intent = IntentParser().parse("deploy to production")
        result = await executor.execute(intent)
        assert result.status is DecisionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_custom_approver_denies(self) -> None:
        executor = DecisionExecutor()

        async def always_deny(action: DecisionAction) -> bool:
            return False

        executor.register_approver(always_deny)
        intent = IntentParser().parse("deploy to production")
        result = await executor.execute(intent)
        assert result.status is DecisionStatus.FAILED
        assert result.results[0]["status"] == "denied"


# ---------------------------------------------------------------------------
# DecisionExecutor — history & ExecutionResult structure
# ---------------------------------------------------------------------------


class TestExecutorHistory:
    @pytest.mark.asyncio
    async def test_execution_result_structure(self) -> None:
        executor = DecisionExecutor()

        async def handler(action: DecisionAction) -> dict[str, Any]:
            return {"output": "done"}

        executor.register_handler(DecisionType.NOTIFY, handler)
        intent = IntentParser().parse("notify team about the update")
        result = await executor.execute(intent)

        assert isinstance(result, ExecutionResult)
        assert result.decision_id == intent.id
        assert result.status is DecisionStatus.COMPLETED
        assert len(result.results) == 1
        entry = result.results[0]
        assert entry["action_id"]
        assert entry["action_type"] == "notify"
        assert entry["status"] == "completed"
        assert entry["output"] == "done"
        assert result.started_at is not None
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_get_history_records_executions(self) -> None:
        executor = DecisionExecutor()
        intent = IntentParser().parse("notify team about the update")
        await executor.execute(intent)
        history = await executor.get_history()
        assert len(history) == 1
        assert history[0].decision_id == intent.id

    @pytest.mark.asyncio
    async def test_get_history_filtered_by_decision_id(self) -> None:
        executor = DecisionExecutor()
        first = IntentParser().parse("notify team about the update")
        second = IntentParser().parse("query the project status")
        await executor.execute(first)
        await executor.execute(second)
        filtered = await executor.get_history(decision_id=first.id)
        assert len(filtered) == 1
        assert filtered[0].decision_id == first.id

    @pytest.mark.asyncio
    async def test_get_history_empty(self) -> None:
        executor = DecisionExecutor()
        assert await executor.get_history() == []

    @pytest.mark.asyncio
    async def test_no_handler_for_type_returns_failed(self) -> None:
        executor = DecisionExecutor()
        # Remove the registered NOTIFY handler to exercise the dispatch fallback.
        with executor._registry_lock:
            executor._handlers.pop(DecisionType.NOTIFY, None)
        intent = IntentParser().parse("notify team about the update")
        result = await executor.execute(intent)
        assert result.status is DecisionStatus.FAILED
        assert result.results[0]["status"] == "failed"
        assert "no handler for notify" in result.results[0]["error"]
