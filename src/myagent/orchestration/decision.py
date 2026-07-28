"""Decision execution engine — natural-language intent to structured actions.

Converts a manager's free-text instruction (e.g. *"notify all employees about
the new policy and schedule a review meeting tomorrow"*) into a structured
:class:`DecisionIntent` composed of typed :class:`DecisionAction` objects, then
validates, permission-checks and executes each action, dispatching to the
appropriate platform subsystem (communication/notification, resources/scheduler,
knowledge/RAG, ...).

Design:

* :class:`DecisionType` — the verb category of an action.
* :class:`DecisionStatus` — lifecycle of an intent during execution.
* :class:`DecisionAction` — one typed, parameterised, targetable action.
* :class:`DecisionIntent` — a parsed instruction (raw text + actions +
  confidence + provenance).
* :class:`ExecutionResult` — the aggregate outcome of executing an intent.
* :class:`IntentParser` — regex + keyword heuristic NL parser (no LLM
  required; overridable for richer parsing).
* :class:`DecisionExecutor` — async, thread-safe executor with injectable
  per-type handlers, a pluggable permission checker and an approval gate for
  actions flagged ``requires_approval``.

Default action handlers integrate lazily with
:mod:`myagent.communication.notification` and
:mod:`myagent.resources.scheduler` when available, and fall back to simulated
results otherwise, so the engine is fully functional out of the box.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("myagent.orchestration.decision")


class DecisionError(Exception):
    """Raised when a decision cannot be parsed, validated or executed."""


class DecisionType(str, Enum):  # noqa: UP042 - match existing codebase style
    """The verb category of a :class:`DecisionAction`."""

    NOTIFY = "notify"
    SCHEDULE = "schedule"
    ALLOCATE = "allocate"
    QUERY = "query"
    APPROVE = "approve"
    DEPLOY = "deploy"
    CONFIGURE = "configure"
    ANALYZE = "analyze"


class DecisionStatus(str, Enum):  # noqa: UP042
    """Lifecycle status of a :class:`DecisionIntent` during execution."""

    PENDING = "pending"
    VALIDATED = "validated"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class DecisionAction(BaseModel):
    """A single typed, parameterised action extracted from an intent.

    Attributes:
        id: Unique action identifier (auto-generated UUID4 hex when omitted).
        action_type: The :class:`DecisionType` controlling execution dispatch.
        parameters: Type-specific parameters (message body, meeting time,
            resource count, query string, ...).
        target: The subject of the action (audience, resource name, service,
            document id, ...).
        priority: Execution priority (higher runs first); 0 = normal.
        requires_approval: When True the action will not execute until an
            approver (or :meth:`DecisionExecutor.approve_action`) clears it.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    action_type: DecisionType = DecisionType.QUERY
    parameters: dict[str, Any] = Field(default_factory=dict)
    target: str = ""
    priority: int = 0
    requires_approval: bool = False


class DecisionIntent(BaseModel):
    """A parsed natural-language decision request.

    Attributes:
        id: Unique intent identifier.
        raw_text: The original input sentence(s).
        parsed_actions: The structured actions extracted from *raw_text*.
        confidence: Parser confidence in ``[0.0, 1.0]``.
        source: Provenance tag (e.g. ``"manager"``, ``"auto"``, ``"cli"``).
        created_at: UTC creation timestamp.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    raw_text: str
    parsed_actions: list[DecisionAction] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = "manager"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def primary_action(self) -> DecisionAction | None:
        """The highest-priority action, or the first one."""

        if not self.parsed_actions:
            return None
        return max(self.parsed_actions, key=lambda a: a.priority)


class ExecutionResult(BaseModel):
    """The aggregate outcome of executing a :class:`DecisionIntent`.

    Attributes:
        decision_id: The intent that was executed.
        status: Final :class:`DecisionStatus`.
        results: One result dict per action executed.
        errors: Error messages for actions that failed or were denied.
        started_at: UTC timestamp when execution began.
        completed_at: UTC timestamp when execution reached a terminal state.
    """

    decision_id: str
    status: DecisionStatus = DecisionStatus.PENDING
    results: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Natural-language parser
# ---------------------------------------------------------------------------

#: Action keyword patterns mapped to :class:`DecisionType`. Order matters:
#: the first match wins, so more specific verbs are listed first.
_INTENT_PATTERNS: list[tuple[DecisionType, re.Pattern[str]]] = [
    (
        DecisionType.NOTIFY,
        re.compile(
            r"\b(notify|tell|inform|alert|broadcast|"
            r"send\s+(?:a\s+)?message|ping|message)\b",
            re.IGNORECASE,
        ),
    ),
    (
        DecisionType.SCHEDULE,
        re.compile(
            r"\b(schedule|book|arrange|set\s+up\s+(?:a\s+)?meeting|"
            r"calendar|meet\b|appointment)\b",
            re.IGNORECASE,
        ),
    ),
    (
        DecisionType.ALLOCATE,
        re.compile(
            r"\b(allocate|assign|provision|reserve)\b",
            re.IGNORECASE,
        ),
    ),
    (
        DecisionType.APPROVE,
        re.compile(
            r"\b(approve|sign\s+off|authorize|authorise|confirm|"
            r"endorse|ratify)\b",
            re.IGNORECASE,
        ),
    ),
    (
        DecisionType.DEPLOY,
        re.compile(
            r"\b(deploy|release|ship|roll\s+out|publish|"
            r"push\s+to\s+prod|promote)\b",
            re.IGNORECASE,
        ),
    ),
    (
        DecisionType.CONFIGURE,
        re.compile(
            r"\b(configure|config|set\s+up|change\s+setting|"
            r"update\s+config|modify|adjust)\b",
            re.IGNORECASE,
        ),
    ),
    (
        DecisionType.ANALYZE,
        re.compile(
            r"\b(analyz|analys|review|examine|investigate|assess|"
            r"evaluate|inspect|audit)\w*\b",
            re.IGNORECASE,
        ),
    ),
    (
        DecisionType.QUERY,
        re.compile(
            r"\b(query|search|find|lookup|look\s+up|get\s+info|"
            r"fetch|report|show\s+me|list)\b",
            re.IGNORECASE,
        ),
    ),
]

#: Conjunctions used to split a compound instruction into clauses.
_CLAUSE_SPLIT_RE = re.compile(r"\s+(?:and\s+then|then|and|after\s+that|;|,)\s+", re.IGNORECASE)

#: Urgency keywords that elevate priority / require approval.
_URGENT_RE = re.compile(
    r"\b(urgent|critical|emergency|immediate|asap|production|prod)\b",
    re.IGNORECASE,
)

#: Time extraction patterns.
_TIME_AT_RE = re.compile(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_TIME_IN_RE = re.compile(
    r"\bin\s+(\d+)\s+(minutes?|mins?|hours?|hrs?|days?|weeks?)\b",
    re.IGNORECASE,
)
_TIME_TOMORROW_RE = re.compile(r"\btomorrow\b", re.IGNORECASE)
_TIME_TONIGHT_RE = re.compile(r"\btonight\b", re.IGNORECASE)
_TIME_DAY_RE = re.compile(
    r"\b(on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

#: Leading integer / count for allocate-style actions.
_COUNT_RE = re.compile(r"^\s*(\d+)\s+(?:x\s+)?", re.IGNORECASE)

#: Audience / target for notify-style actions.
_NOTIFY_AUDIENCE_RE = re.compile(
    r"(?:notify|tell|inform|alert|message|ping)\s+(.+?)(?:\s+(?:about|that|regarding|on)\s+(.+))?$",
    re.IGNORECASE,
)
_WITH_RE = re.compile(r"\bwith\s+(.+?)(?:\s+(?:at|on|for|tomorrow|tonight)\b|$)", re.IGNORECASE)
_FOR_RE = re.compile(r"\bfor\s+(.+?)(?:\s+(?:at|on|with|tomorrow|tonight)\b|$)", re.IGNORECASE)


class IntentParser:
    """Parses natural-language instructions into :class:`DecisionIntent`.

    Uses a cascade of regex patterns to split a compound instruction into
    clauses, classify each clause's :class:`DecisionType`, and extract
    type-specific parameters (audience, time, count, target, ...). The parser
    is intentionally heuristic — it handles common enterprise phrasing without
    requiring an LLM. Confidence reflects how clearly an intent was identified.

    Example::

        parser = IntentParser()
        intent = parser.parse(
            "notify all employees about the new policy and schedule a "
            "review meeting tomorrow 3pm with engineering"
        )
        assert len(intent.parsed_actions) == 2
        assert intent.parsed_actions[0].action_type is DecisionType.NOTIFY
    """

    def parse(self, text: str, *, source: str = "manager") -> DecisionIntent:
        """Parse *text* into a :class:`DecisionIntent`."""

        if not text or not text.strip():
            raise DecisionError("cannot parse empty decision text")
        raw = text.strip()
        actions = self._extract_actions(raw)
        confidence = self._classify_confidence(raw, actions)
        return DecisionIntent(
            raw_text=raw,
            parsed_actions=actions,
            confidence=confidence,
            source=source,
        )

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def _extract_actions(self, text: str) -> list[DecisionAction]:
        """Split *text* into clauses and build a :class:`DecisionAction` per clause."""

        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]
        if not clauses:
            clauses = [text]
        actions: list[DecisionAction] = []
        for clause in clauses:
            action_type = self._classify_intent(clause)
            params, target = self._extract_parameters(action_type, clause)
            requires_approval = bool(_URGENT_RE.search(clause)) or action_type in (
                DecisionType.DEPLOY,
                DecisionType.APPROVE,
            )
            priority = (
                2
                if _URGENT_RE.search(clause)
                else (1 if action_type in (DecisionType.DEPLOY, DecisionType.NOTIFY) else 0)
            )
            actions.append(
                DecisionAction(
                    action_type=action_type,
                    parameters=params,
                    target=target,
                    priority=priority,
                    requires_approval=requires_approval,
                )
            )
        return actions

    def _classify_intent(self, clause: str) -> DecisionType:
        """Return the :class:`DecisionType` for *clause* (first keyword match wins)."""

        for action_type, pattern in _INTENT_PATTERNS:
            if pattern.search(clause):
                return action_type
        return DecisionType.QUERY

    def _classify_confidence(self, text: str, actions: list[DecisionAction]) -> float:
        """Estimate parser confidence in ``[0.0, 1.0]``."""

        if not actions:
            return 0.2
        confidence = 0.5
        matched = sum(
            1
            for a in actions
            if a.action_type is not DecisionType.QUERY or _INTENT_PATTERNS[-1][1].search(text)
        )
        if matched:
            confidence += 0.2
        if any(a.target for a in actions):
            confidence += 0.1
        if any(a.parameters for a in actions):
            confidence += 0.1
        if _URGENT_RE.search(text):
            confidence += 0.05
        return min(confidence, 0.95)

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    def _extract_parameters(
        self, action_type: DecisionType, clause: str
    ) -> tuple[dict[str, Any], str]:
        """Extract type-specific parameters and target from *clause*."""

        if action_type is DecisionType.NOTIFY:
            return self._extract_notify(clause)
        if action_type is DecisionType.SCHEDULE:
            return self._extract_schedule(clause)
        if action_type is DecisionType.ALLOCATE:
            return self._extract_allocate(clause)
        if action_type is DecisionType.QUERY:
            return self._extract_query(clause)
        if action_type is DecisionType.APPROVE:
            return self._extract_approve(clause)
        if action_type is DecisionType.DEPLOY:
            return self._extract_deploy(clause)
        if action_type is DecisionType.CONFIGURE:
            return self._extract_configure(clause)
        if action_type is DecisionType.ANALYZE:
            return self._extract_analyze(clause)
        return {}, ""

    def _extract_notify(self, clause: str) -> tuple[dict[str, Any], str]:
        match = _NOTIFY_AUDIENCE_RE.search(clause)
        audience = ""
        message = ""
        if match:
            audience = (match.group(1) or "").strip()
            message = (match.group(2) or "").strip()
        return {"message": message or clause, "audience": audience}, audience

    def _extract_schedule(self, clause: str) -> tuple[dict[str, Any], str]:
        time_str = self._extract_time(clause)
        attendees = ""
        with_match = _WITH_RE.search(clause)
        if with_match:
            attendees = with_match.group(1).strip()
        for_match = _FOR_RE.search(clause)
        if not attendees and for_match:
            attendees = for_match.group(1).strip()
        return {"time": time_str, "attendees": attendees}, attendees

    def _extract_allocate(self, clause: str) -> tuple[dict[str, Any], str]:
        count = 1
        count_match = _COUNT_RE.match(clause)
        if count_match:
            count = int(count_match.group(1))
        # Target: text after the allocate verb.
        verb_match = re.search(
            r"\b(?:allocate|assign|provision|reserve)\s+(.+)", clause, re.IGNORECASE
        )
        target = verb_match.group(1).strip() if verb_match else clause
        return {"count": count, "purpose": target}, target

    def _extract_query(self, clause: str) -> tuple[dict[str, Any], str]:
        query = re.sub(
            r"\b(?:query|search|find|lookup|look\s+up|get\s+info|fetch|report|show\s+me|list)\b\s*",
            "",
            clause,
            flags=re.IGNORECASE,
        ).strip()
        return {"query": query or clause}, query or clause

    def _extract_approve(self, clause: str) -> tuple[dict[str, Any], str]:
        target = re.sub(
            r"\b(?:approve|sign\s+off|authorize|authorise|confirm|endorse|ratify)\b\s*",
            "",
            clause,
            flags=re.IGNORECASE,
        ).strip()
        return {"decision": target}, target

    def _extract_deploy(self, clause: str) -> tuple[dict[str, Any], str]:
        target = re.sub(
            r"\b(?:deploy|release|ship|roll\s+out|publish|push\s+to\s+prod|promote)\b\s*",
            "",
            clause,
            flags=re.IGNORECASE,
        ).strip()
        env = (
            "production" if re.search(r"\bprod(?:uction)?\b", clause, re.IGNORECASE) else "staging"
        )
        return {"environment": env}, target

    def _extract_configure(self, clause: str) -> tuple[dict[str, Any], str]:
        target = re.sub(
            r"\b(?:configure|config|set\s+up|change\s+setting|update\s+config|modify|adjust)\b\s*",
            "",
            clause,
            flags=re.IGNORECASE,
        ).strip()
        return {"setting": target}, target

    def _extract_analyze(self, clause: str) -> tuple[dict[str, Any], str]:
        target = re.sub(
            r"\b(?:analyz\w*|analys\w*|review|examine|investigate|assess|evaluate|inspect|audit)\b\s*",
            "",
            clause,
            flags=re.IGNORECASE,
        ).strip()
        return {"subject": target or clause}, target or clause

    def _extract_time(self, clause: str) -> str:
        """Extract a human-readable time expression from *clause*."""

        match = _TIME_AT_RE.search(clause)
        if match:
            hour = match.group(1)
            minute = match.group(2) or "00"
            period = match.group(3).upper()
            return f"{hour}:{minute} {period}"
        match = _TIME_IN_RE.search(clause)
        if match:
            return f"in {match.group(1)} {match.group(2)}"
        if _TIME_TOMORROW_RE.search(clause):
            return "tomorrow"
        if _TIME_TONIGHT_RE.search(clause):
            return "tonight"
        match = _TIME_DAY_RE.search(clause)
        if match:
            return match.group(2).lower()
        return ""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

#: Signature of an async action handler.
ActionHandler = Callable[[DecisionAction], Awaitable[dict[str, Any]]]

#: Signature of a sync permission checker.
PermissionChecker = Callable[[DecisionIntent], bool]

#: Signature of an async approval gate for ``requires_approval`` actions.
Approver = Callable[[DecisionAction], Awaitable[bool]]


class DecisionExecutor:
    """Async, thread-safe executor for :class:`DecisionIntent` objects.

    Each :class:`DecisionType` is backed by an injectable async handler
    (:meth:`register_handler`); sensible defaults integrate lazily with the
    communication and resources subsystems and fall back to simulated
    results. A pluggable permission checker (:meth:`register_permission_checker`)
    gates execution, and actions flagged ``requires_approval`` must be cleared
    by an approver (:meth:`register_approver`) or pre-approved via
    :meth:`approve_action`.

    Example::

        executor = DecisionExecutor()
        parser = IntentParser()
        intent = parser.parse("notify engineering team about the deploy")
        result = await executor.execute(intent)
        assert result.status is DecisionStatus.COMPLETED
    """

    def __init__(self, *, confidence_threshold: float = 0.3) -> None:
        self._handlers: dict[DecisionType, ActionHandler] = {}
        self._permission_checker: PermissionChecker = self._default_permission_checker
        self._approver: Approver = self._default_approver
        self._approved_actions: set[str] = set()
        self._history: list[ExecutionResult] = []
        self._confidence_threshold = confidence_threshold
        self._registry_lock = threading.RLock()
        self._history_lock = asyncio.Lock()
        self._register_default_handlers()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_handler(self, action_type: DecisionType, handler: ActionHandler) -> None:
        """Register or replace the async handler for a :class:`DecisionType`."""

        with self._registry_lock:
            self._handlers[action_type] = handler
        logger.debug("Registered action handler for %s", action_type.value)

    def register_permission_checker(self, checker: PermissionChecker) -> None:
        """Register a synchronous permission checker invoked before execution."""

        with self._registry_lock:
            self._permission_checker = checker

    def register_approver(self, approver: Approver) -> None:
        """Register an async approval gate for ``requires_approval`` actions."""

        with self._registry_lock:
            self._approver = approver

    def approve_action(self, action_id: str) -> bool:
        """Pre-approve a single action so it bypasses the approver gate."""

        with self._registry_lock:
            self._approved_actions.add(action_id)
        logger.info("Pre-approved action %s", action_id)
        return True

    def _register_default_handlers(self) -> None:
        defaults: dict[DecisionType, ActionHandler] = {
            DecisionType.NOTIFY: self._default_notify_handler,
            DecisionType.SCHEDULE: self._default_schedule_handler,
            DecisionType.ALLOCATE: self._default_allocate_handler,
            DecisionType.QUERY: self._default_query_handler,
            DecisionType.APPROVE: self._default_approve_handler,
            DecisionType.DEPLOY: self._default_deploy_handler,
            DecisionType.CONFIGURE: self._default_configure_handler,
            DecisionType.ANALYZE: self._default_analyze_handler,
        }
        self._handlers.update(defaults)

    # ------------------------------------------------------------------
    # Validation & permissions
    # ------------------------------------------------------------------

    def validate_decision(self, decision: DecisionIntent) -> None:
        """Validate a decision before execution. Raises :class:`DecisionError`."""

        if not decision.raw_text.strip():
            raise DecisionError("decision has no raw text")
        if not decision.parsed_actions:
            raise DecisionError("decision has no parsed actions")
        if decision.confidence < self._confidence_threshold:
            raise DecisionError(
                f"decision confidence {decision.confidence:.2f} below threshold "
                f"{self._confidence_threshold:.2f}"
            )
        for action in decision.parsed_actions:
            if not action.action_type:
                raise DecisionError("action has no action_type")
            if action.priority < 0:
                raise DecisionError(f"action {action.id} has negative priority")

    def check_permissions(self, decision: DecisionIntent) -> bool:
        """Return True if the actor is permitted to execute *decision*."""

        with self._registry_lock:
            checker = self._permission_checker
        try:
            allowed = bool(checker(decision))
        except Exception as exc:  # noqa: BLE001 - deny on checker error
            logger.warning("Permission checker raised: %s", exc)
            return False
        if not allowed:
            logger.warning(
                "Permission denied for decision %s (source=%s)",
                decision.id,
                decision.source,
            )
        return allowed

    @staticmethod
    def _default_permission_checker(decision: DecisionIntent) -> bool:
        """Default checker: allow all local-first executions."""

        return True

    async def _default_approver(self, action: DecisionAction) -> bool:
        """Default approval gate: honour pre-approved action ids, else deny."""

        with self._registry_lock:
            return action.id in self._approved_actions

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, decision: DecisionIntent) -> ExecutionResult:
        """Validate, permission-check and execute a decision.

        Returns an :class:`ExecutionResult` with per-action outcomes. Actions
        flagged ``requires_approval`` are gated by the registered approver (or
        pre-approval via :meth:`approve_action`); denied actions are skipped
        and recorded in ``errors``.
        """

        self.validate_decision(decision)
        if not self.check_permissions(decision):
            result = ExecutionResult(
                decision_id=decision.id,
                status=DecisionStatus.FAILED,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                errors=["permission denied"],
            )
            await self._record_history(result)
            return result

        result = ExecutionResult(
            decision_id=decision.id,
            status=DecisionStatus.EXECUTING,
            started_at=datetime.now(UTC),
        )
        logger.info(
            "Executing decision %s (%d action(s), confidence=%.2f)",
            decision.id,
            len(decision.parsed_actions),
            decision.confidence,
        )

        # Run actions concurrently; each is independent by default.
        sorted_actions = sorted(decision.parsed_actions, key=lambda a: a.priority, reverse=True)
        outcomes = await asyncio.gather(
            *[self._execute_action(action, decision) for action in sorted_actions],
            return_exceptions=True,
        )

        errors: list[str] = []
        for action, outcome in zip(sorted_actions, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                errors.append(f"{action.action_type.value}: {outcome}")
                result.results.append(
                    {
                        "action_id": action.id,
                        "action_type": action.action_type.value,
                        "status": "failed",
                        "error": str(outcome),
                    }
                )
            else:
                result.results.append(outcome)
                if outcome.get("status") in ("failed", "denied", "skipped"):
                    errors.append(
                        f"{action.action_type.value}: {outcome.get('error', outcome.get('status'))}"
                    )

        result.errors.extend(errors)
        result.status = DecisionStatus.FAILED if errors else DecisionStatus.COMPLETED
        result.completed_at = datetime.now(UTC)
        await self._record_history(result)
        logger.info(
            "Decision %s finished with status %s",
            decision.id,
            result.status.value,
        )
        return result

    async def _execute_action(
        self, action: DecisionAction, decision: DecisionIntent
    ) -> dict[str, Any]:
        """Dispatch *action* to its handler, honouring the approval gate."""

        if action.requires_approval:
            with self._registry_lock:
                approver = self._approver
            approved = await approver(action)
            if not approved:
                logger.warning(
                    "Action %s (%s) denied by approval gate",
                    action.id,
                    action.action_type.value,
                )
                return {
                    "action_id": action.id,
                    "action_type": action.action_type.value,
                    "status": "denied",
                    "error": "approval required and not granted",
                }

        with self._registry_lock:
            handler = self._handlers.get(action.action_type)
        if handler is None:
            return {
                "action_id": action.id,
                "action_type": action.action_type.value,
                "status": "failed",
                "error": f"no handler for {action.action_type.value}",
            }
        logger.debug(
            "Executing action %s (%s) for decision %s",
            action.id,
            action.action_type.value,
            decision.id,
        )
        try:
            output = await handler(action)
        except Exception as exc:  # noqa: BLE001
            return {
                "action_id": action.id,
                "action_type": action.action_type.value,
                "status": "failed",
                "error": str(exc),
            }
        outcome = {"action_id": action.id, "action_type": action.action_type.value}
        outcome.update(output)
        outcome.setdefault("status", "completed")
        return outcome

    async def _record_history(self, result: ExecutionResult) -> None:
        """Append an execution result to the history (thread-safe)."""

        async with self._history_lock:
            self._history.append(result)

    async def get_history(self, *, decision_id: str | None = None) -> list[ExecutionResult]:
        """Return execution history, optionally filtered by decision id."""

        async with self._history_lock:
            snapshot = list(self._history)
        if decision_id is not None:
            snapshot = [r for r in snapshot if r.decision_id == decision_id]
        return snapshot

    # ------------------------------------------------------------------
    # Default action handlers
    # ------------------------------------------------------------------

    async def _default_notify_handler(self, action: DecisionAction) -> dict[str, Any]:
        """Notify via :mod:`myagent.communication.notification` (lazy import)."""

        message = action.parameters.get("message", "")
        audience = action.target or action.parameters.get("audience", "all")
        try:
            from myagent.communication.notification import (  # lazy import
                NotificationEngine,
                NotificationPriority,
            )

            engine = NotificationEngine()
            records = await engine.notify(
                title=f"Notification for {audience}",
                body=message,
                recipient=audience,
                priority=NotificationPriority.HIGH,
            )
            return {
                "status": "completed",
                "audience": audience,
                "message": message,
                "deliveries": len(records),
            }
        except Exception as exc:  # noqa: BLE001 - fall back to simulated
            logger.debug("Notify handler fell back to simulated: %s", exc)
            return {
                "status": "completed",
                "audience": audience,
                "message": message,
                "simulated": True,
            }

    async def _default_schedule_handler(self, action: DecisionAction) -> dict[str, Any]:
        """Schedule a meeting via :mod:`myagent.communication.meeting` (lazy)."""

        time_str = action.parameters.get("time", "")
        attendees = action.target or action.parameters.get("attendees", "")
        try:
            from myagent.communication.meeting import MeetingService  # lazy import

            service = MeetingService()
            meeting = await service.create_meeting(
                title=f"Meeting with {attendees}" if attendees else "Scheduled meeting",
                organizer="decision-engine",
            )
            return {
                "status": "completed",
                "meeting_id": meeting.id,
                "time": time_str,
                "attendees": attendees,
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Schedule handler fell back to simulated: %s", exc)
            return {
                "status": "completed",
                "time": time_str,
                "attendees": attendees,
                "simulated": True,
            }

    async def _default_allocate_handler(self, action: DecisionAction) -> dict[str, Any]:
        """Allocate a resource via :mod:`myagent.resources.registry` (lazy)."""

        count = int(action.parameters.get("count", 1))
        purpose = action.parameters.get("purpose", action.target)
        try:
            from myagent.resources.registry import (  # lazy import
                ResourceRegistry,
                ResourceStatus,
                ResourceType,
            )

            registry = ResourceRegistry()
            allocated = 0
            for i in range(count):
                from myagent.resources.registry import ResourceRecord

                registry.register(
                    ResourceRecord(
                        name=f"{purpose}-{i + 1}",
                        type=ResourceType.SERVER,
                        status=ResourceStatus.ONLINE,
                    )
                )
                allocated += 1
            return {
                "status": "completed",
                "allocated": allocated,
                "purpose": purpose,
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Allocate handler fell back to simulated: %s", exc)
            return {
                "status": "completed",
                "allocated": count,
                "purpose": purpose,
                "simulated": True,
            }

    async def _default_query_handler(self, action: DecisionAction) -> dict[str, Any]:
        """Run a query (simulated by default; override via register_handler)."""

        query = action.parameters.get("query", action.target)
        await asyncio.sleep(0)
        return {
            "status": "completed",
            "query": query,
            "result": f"query results for {query!r}",
            "simulated": True,
        }

    async def _default_approve_handler(self, action: DecisionAction) -> dict[str, Any]:
        """Record an approval decision (simulated by default)."""

        decision_text = action.parameters.get("decision", action.target)
        await asyncio.sleep(0)
        return {
            "status": "completed",
            "approved": True,
            "decision": decision_text,
            "approver": "decision-engine",
        }

    async def _default_deploy_handler(self, action: DecisionAction) -> dict[str, Any]:
        """Trigger a deployment (simulated by default)."""

        target = action.target or "application"
        env = action.parameters.get("environment", "staging")
        await asyncio.sleep(0)
        return {
            "status": "completed",
            "target": target,
            "environment": env,
            "version": f"release-{uuid.uuid4().hex[:8]}",
            "simulated": True,
        }

    async def _default_configure_handler(self, action: DecisionAction) -> dict[str, Any]:
        """Apply a configuration change (simulated by default)."""

        setting = action.parameters.get("setting", action.target)
        await asyncio.sleep(0)
        return {
            "status": "completed",
            "setting": setting,
            "applied": True,
            "simulated": True,
        }

    async def _default_analyze_handler(self, action: DecisionAction) -> dict[str, Any]:
        """Run an analysis (simulated by default)."""

        subject = action.parameters.get("subject", action.target)
        await asyncio.sleep(0)
        return {
            "status": "completed",
            "subject": subject,
            "findings": [f"analysis of {subject} complete"],
            "simulated": True,
        }


__all__ = [
    "ActionHandler",
    "Approver",
    "DecisionAction",
    "DecisionError",
    "DecisionExecutor",
    "DecisionIntent",
    "DecisionStatus",
    "DecisionType",
    "ExecutionResult",
    "IntentParser",
    "PermissionChecker",
]
