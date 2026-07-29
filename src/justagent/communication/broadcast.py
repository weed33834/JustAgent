"""Message broadcast — natural-language intent parsing, multi-channel delivery, receipt collection.

A manager types one sentence (e.g. *"notify all employees meeting tonight
7pm"*) and this module:

1. **Parses** the natural-language intent into a structured
   :class:`BroadcastIntent` (action, audience, subject, time, channels).
2. **Resolves** the target audience into a concrete list of recipient IDs
   via a pluggable :class:`AudienceResolver`.
3. **Generates** a :class:`BroadcastMessage` with a title and body.
4. **Delivers** the message through the :class:`NotificationEngine` across
   multiple channels concurrently.
5. **Tracks** per-recipient delivery confirmations (receipts).
6. **Schedules** reminders for recipients who have not confirmed within
   a configurable deadline.

Design:

* :class:`AudienceScope` — enum (all, department, role, individual, group).
* :class:`BroadcastStatus` — lifecycle of a broadcast.
* :class:`BroadcastIntent` — parsed NL intent (Pydantic model).
* :class:`BroadcastTarget` — one resolved recipient.
* :class:`BroadcastMessage` — the generated notice.
* :class:`Receipt` — per-recipient confirmation tracking.
* :class:`BroadcastParser` — regex-driven NL parser.
* :class:`AudienceResolver` — pluggable recipient resolver (ABC + default).
* :class:`BroadcastService` — orchestrates the full pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from justagent.communication.notification import (
    DeliveryStatus,
    NotificationChannel,
    NotificationEngine,
    NotificationPriority,
)

logger = logging.getLogger("justagent.communication.broadcast")


class BroadcastError(Exception):
    """Raised when a broadcast cannot be parsed, resolved or delivered."""


class AudienceScope(str, Enum):  # noqa: UP042 - match existing codebase style
    """How the broadcast audience is scoped."""

    ALL = "all"
    DEPARTMENT = "department"
    ROLE = "role"
    INDIVIDUAL = "individual"
    GROUP = "group"


class BroadcastStatus(str, Enum):  # noqa: UP042
    """Lifecycle status of a broadcast."""

    DRAFT = "draft"
    SCHEDULED = "scheduled"
    SENDING = "sending"
    DELIVERED = "delivered"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConfirmationState(str, Enum):  # noqa: UP042
    """Per-recipient confirmation state for a broadcast."""

    PENDING = "pending"
    DELIVERED = "delivered"
    CONFIRMED = "confirmed"
    READ = "read"
    FAILED = "failed"
    REMINDED = "reminded"


class BroadcastAction(str, Enum):  # noqa: UP042
    """The verb parsed from the manager's natural-language instruction."""

    NOTIFY = "notify"
    ANNOUNCE = "announce"
    REMIND = "remind"
    ALERT = "alert"
    BROADCAST = "broadcast"
    INFORM = "inform"
    INVITE = "invite"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BroadcastIntent(BaseModel):
    """Structured representation of a parsed natural-language broadcast request.

    Attributes:
        raw_text: The original input sentence.
        action: The parsed verb.
        audience_scope: How the audience is scoped.
        audience_filter: Department name, role name, or individual ID.
        subject: The main topic/subject of the broadcast.
        message_body: The generated human-readable message body.
        message_title: The generated short title.
        scheduled_at: When the event/notification should fire (UTC), or None.
        channels: Requested delivery channels (empty = use defaults).
        priority: Inferred urgency.
    """

    raw_text: str
    action: BroadcastAction = BroadcastAction.NOTIFY
    audience_scope: AudienceScope = AudienceScope.ALL
    audience_filter: str = ""
    subject: str = ""
    message_body: str = ""
    message_title: str = ""
    scheduled_at: datetime | None = None
    channels: list[NotificationChannel] = Field(default_factory=list)
    priority: NotificationPriority = NotificationPriority.NORMAL

    @property
    def is_scheduled(self) -> bool:
        """True when a future delivery time was parsed."""

        return self.scheduled_at is not None and self.scheduled_at > datetime.now(UTC)


class BroadcastTarget(BaseModel):
    """A single resolved recipient for a broadcast.

    Attributes:
        user_id: Unique user identifier.
        name: Display name (for personalisation).
        email: Email address (for email channel).
        phone: Phone number (for SMS channel).
        department: Department name (for filtering).
        role: Job role / title (for filtering).
    """

    user_id: str
    name: str = ""
    email: str = ""
    phone: str = ""
    department: str = ""
    role: str = ""


class BroadcastMessage(BaseModel):
    """The generated notice that will be delivered to recipients.

    Attributes:
        id: Unique broadcast message identifier.
        intent: The originating parsed intent.
        title: Short headline.
        body: Full message text.
        priority: Urgency ranking.
        channels: Delivery channels.
        created_at: UTC creation timestamp.
        created_by: User ID of the broadcaster (manager).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    intent: BroadcastIntent
    title: str
    body: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: list[NotificationChannel] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str = ""


class Receipt(BaseModel):
    """Per-recipient delivery/confirmation tracking for one broadcast.

    Attributes:
        broadcast_id: The parent broadcast message ID.
        target: The recipient.
        state: Current confirmation state.
        delivered_channels: Which channels succeeded.
        confirmed_at: When the recipient confirmed receipt (UTC), or None.
        last_reminder_at: When the last reminder was sent (UTC), or None.
        reminder_count: Number of reminders sent so far.
        error: Failure reason, if any.
    """

    broadcast_id: str
    target: BroadcastTarget
    state: ConfirmationState = ConfirmationState.PENDING
    delivered_channels: list[NotificationChannel] = Field(default_factory=list)
    confirmed_at: datetime | None = None
    last_reminder_at: datetime | None = None
    reminder_count: int = 0
    error: str = ""

    def mark_delivered(self, channel: NotificationChannel) -> None:
        """Record successful delivery via *channel*."""

        if channel not in self.delivered_channels:
            self.delivered_channels.append(channel)
        if self.state is ConfirmationState.PENDING:
            self.state = ConfirmationState.DELIVERED

    def confirm(self) -> None:
        """Mark this receipt as confirmed by the recipient."""

        self.state = ConfirmationState.CONFIRMED
        self.confirmed_at = datetime.now(UTC)

    def mark_failed(self, error: str) -> None:
        """Mark this receipt as failed."""

        self.state = ConfirmationState.FAILED
        self.error = error

    def record_reminder(self) -> None:
        """Increment the reminder counter and stamp the timestamp."""

        self.reminder_count += 1
        self.last_reminder_at = datetime.now(UTC)
        if self.state is ConfirmationState.DELIVERED:
            self.state = ConfirmationState.REMINDED

    @property
    def is_confirmed(self) -> bool:
        """True when the recipient has confirmed receipt."""

        return self.state in (ConfirmationState.CONFIRMED, ConfirmationState.READ)


# ---------------------------------------------------------------------------
# Natural-language parser
# ---------------------------------------------------------------------------

# Action verbs mapped to BroadcastAction values.
_ACTION_PATTERNS: list[tuple[re.Pattern[str], BroadcastAction]] = [
    (re.compile(r"\b(notify|tell|send)\b", re.IGNORECASE), BroadcastAction.NOTIFY),
    (re.compile(r"\b(announce|announcement)\b", re.IGNORECASE), BroadcastAction.ANNOUNCE),
    (re.compile(r"\b(remind|reminder)\b", re.IGNORECASE), BroadcastAction.REMIND),
    (re.compile(r"\b(alert|warning|warn)\b", re.IGNORECASE), BroadcastAction.ALERT),
    (re.compile(r"\b(broadcast)\b", re.IGNORECASE), BroadcastAction.BROADCAST),
    (re.compile(r"\b(inform)\b", re.IGNORECASE), BroadcastAction.INFORM),
    (re.compile(r"\b(invite|invitation)\b", re.IGNORECASE), BroadcastAction.INVITE),
]

# Audience patterns. Each tuple is (regex, scope, group_index_for_filter).
_ALL_AUDIENCE_RE = re.compile(
    r"\b(all\s+employees|everyone|everybody|all\s+staff|all\s+members|"
    r"whole\s+company|entire\s+company|all\s+team\s+members|all\s+colleagues)\b",
    re.IGNORECASE,
)

_DEPT_AUDIENCE_RE = re.compile(
    r"\b(?:the\s+)?(\w+)\s+(?:department|dept)\b"
    r"|\b(?:department|dept)\s+(\w+)\b"
    r"|\b(?:the\s+)?(\w+)\s+team\b",
    re.IGNORECASE,
)

_ROLE_AUDIENCE_RE = re.compile(
    r"\b(?:all|every|all\s+the)\s+"
    r"(managers?|engineers?|developers?|admins?|directors?|employees?|"
    r"staff|members?|designers?|marketers?|sales(?:people| reps)?|"
    r"hr\s+(?:team|staff)|recruiters?|leads?|vp[s]?|c[- ]?level|executives?)\b",
    re.IGNORECASE,
)

_INDIVIDUAL_AUDIENCE_RE = re.compile(
    r"(?:^|\s)@(\w[\w.\-]*)"
    r"|(?:^|\s)(?:to|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    r"|(?:^|\s)(?:tell|notify|alert|inform|message)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
)

# Time patterns.
_TIME_NOW_RE = re.compile(r"\b(now|immediately|asap|right\s+away|at\s+once)\b", re.IGNORECASE)
_TIME_TONIGHT_RE = re.compile(r"\btonight\b", re.IGNORECASE)
_TIME_TOMORROW_RE = re.compile(r"\btomorrow\b", re.IGNORECASE)
_TIME_AT_RE = re.compile(
    r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
    re.IGNORECASE,
)
_TIME_IN_RE = re.compile(
    r"\bin\s+(\d+)\s+(minutes?|mins?|hours?|hrs?|days?|weeks?)\b",
    re.IGNORECASE,
)
_TIME_DAY_RE = re.compile(
    r"\b(on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

# Channel patterns.
_CHANNEL_PATTERNS: dict[NotificationChannel, re.Pattern[str]] = {
    NotificationChannel.EMAIL: re.compile(r"\b(email|e-mail|mail)\b", re.IGNORECASE),
    NotificationChannel.SMS: re.compile(r"\b(sms|text\s+message|text)\b", re.IGNORECASE),
    NotificationChannel.DESKTOP: re.compile(r"\b(desktop|push)\b", re.IGNORECASE),
    NotificationChannel.IM: re.compile(r"\b(im|slack|teams?|lark|feishu|chat)\b", re.IGNORECASE),
    NotificationChannel.WEBHOOK: re.compile(r"\b(webhook)\b", re.IGNORECASE),
    NotificationChannel.IN_APP: re.compile(r"\b(in[- ]?app)\b", re.IGNORECASE),
    NotificationChannel.MOBILE: re.compile(r"\b(mobile)\b", re.IGNORECASE),
}

#: Days of week → weekday number (Monday=0).
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

#: Subject keywords that elevate priority.
_URGENT_KEYWORDS = re.compile(
    r"\b(urgent|critical|emergency|outage|security|incident|down|failure|"
    r"danger|evacuat|immediate)\b",
    re.IGNORECASE,
)


class BroadcastParser:
    """Parses natural-language broadcast instructions into :class:`BroadcastIntent`.

    Uses a cascade of regex patterns to extract action, audience, subject,
    time and channels. The parser is intentionally heuristic — it handles
    common enterprise phrasing without requiring an LLM. Unrecognised
    portions of the text become the message body.

    Example::

        parser = BroadcastParser()
        intent = parser.parse("notify all employees meeting tonight 7pm")
        assert intent.audience_scope is AudienceScope.ALL
        assert intent.scheduled_at is not None
    """

    def parse(self, text: str) -> BroadcastIntent:
        """Parse *text* into a :class:`BroadcastIntent`."""

        if not text or not text.strip():
            raise BroadcastError("Cannot parse empty broadcast text")
        raw = text.strip()

        action = self._extract_action(raw)
        scope, audience_filter = self._extract_audience(raw)
        scheduled_at = self._extract_time(raw)
        channels = self._extract_channels(raw)
        priority = self._infer_priority(raw, action)
        subject = self._extract_subject(raw, action, scope, audience_filter)
        title = self._build_title(action, subject, scope, audience_filter)
        body = self._build_body(raw, action, subject, scope, audience_filter, scheduled_at)

        return BroadcastIntent(
            raw_text=raw,
            action=action,
            audience_scope=scope,
            audience_filter=audience_filter,
            subject=subject,
            message_body=body,
            message_title=title,
            scheduled_at=scheduled_at,
            channels=channels,
            priority=priority,
        )

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def _extract_action(self, text: str) -> BroadcastAction:
        for pattern, action in _ACTION_PATTERNS:
            if pattern.search(text):
                return action
        return BroadcastAction.NOTIFY

    # ------------------------------------------------------------------
    # Audience extraction
    # ------------------------------------------------------------------

    def _extract_audience(self, text: str) -> tuple[AudienceScope, str]:
        # All employees / everyone.
        if _ALL_AUDIENCE_RE.search(text):
            return AudienceScope.ALL, ""

        # Department / team.
        match = _DEPT_AUDIENCE_RE.search(text)
        if match:
            dept = next((g for g in match.groups() if g), "")
            return AudienceScope.DEPARTMENT, dept.lower()

        # Role-based (all managers, all engineers, etc.).
        match = _ROLE_AUDIENCE_RE.search(text)
        if match:
            role = match.group(1).rstrip("s").lower()
            return AudienceScope.ROLE, role

        # Individual (@username or "tell John").
        match = _INDIVIDUAL_AUDIENCE_RE.search(text)
        if match:
            individual = next((g for g in match.groups() if g), "")
            if individual:
                return AudienceScope.INDIVIDUAL, individual.lstrip("@")

        return AudienceScope.ALL, ""

    # ------------------------------------------------------------------
    # Time extraction
    # ------------------------------------------------------------------

    def _extract_time(self, text: str) -> datetime | None:
        now = datetime.now(UTC)

        # "now" / "immediately"
        if _TIME_NOW_RE.search(text):
            return now

        # Collect day modifier and clock time separately, then combine.
        day_offset = 0

        if _TIME_TOMORROW_RE.search(text):
            day_offset = 1
        elif _TIME_TONIGHT_RE.search(text):
            # "tonight" implies evening of today (or tomorrow if already late).
            pass
        else:
            day_match = _TIME_DAY_RE.search(text)
            if day_match:
                day_name = day_match.group(2).lower()
                target_weekday = _WEEKDAYS[day_name]
                current_weekday = now.weekday()
                day_offset = (target_weekday - current_weekday) % 7
                if day_offset == 0:
                    day_offset = 7  # Next occurrence of same weekday.

        # Try to extract a clock time (HH:MM am/pm).
        clock_match = _TIME_AT_RE.search(text)
        if clock_match:
            hour = int(clock_match.group(1))
            minute = int(clock_match.group(2)) if clock_match.group(2) else 0
            ampm = clock_match.group(3).lower()
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            target += timedelta(days=day_offset)
            # If the time has already passed today and no explicit day, push to tomorrow.
            if target <= now and day_offset == 0:
                target += timedelta(days=1)
            return target

        # "tonight" without explicit time → 19:00.
        if _TIME_TONIGHT_RE.search(text):
            target = now.replace(hour=19, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target

        # "tomorrow" without explicit time → 09:00.
        if _TIME_TOMORROW_RE.search(text):
            return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)

        # Day name without explicit time → 09:00.
        day_match = _TIME_DAY_RE.search(text)
        if day_match:
            target = (now + timedelta(days=day_offset)).replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            return target

        # "in N minutes/hours/days".
        in_match = _TIME_IN_RE.search(text)
        if in_match:
            amount = int(in_match.group(1))
            unit = in_match.group(2).lower()
            if unit.startswith("min"):
                return now + timedelta(minutes=amount)
            if unit.startswith("hour") or unit.startswith("hr"):
                return now + timedelta(hours=amount)
            if unit.startswith("day"):
                return now + timedelta(days=amount)
            if unit.startswith("week"):
                return now + timedelta(weeks=amount)

        return None

    # ------------------------------------------------------------------
    # Channel extraction
    # ------------------------------------------------------------------

    def _extract_channels(self, text: str) -> list[NotificationChannel]:
        found: list[NotificationChannel] = []
        for channel, pattern in _CHANNEL_PATTERNS.items():
            if pattern.search(text):
                found.append(channel)
        return found

    # ------------------------------------------------------------------
    # Priority inference
    # ------------------------------------------------------------------

    def _infer_priority(self, text: str, action: BroadcastAction) -> NotificationPriority:
        if _URGENT_KEYWORDS.search(text):
            return NotificationPriority.URGENT
        if action is BroadcastAction.ALERT:
            return NotificationPriority.HIGH
        if action is BroadcastAction.BROADCAST:
            return NotificationPriority.HIGH
        return NotificationPriority.NORMAL

    # ------------------------------------------------------------------
    # Subject & message generation
    # ------------------------------------------------------------------

    def _extract_subject(
        self,
        text: str,
        action: BroadcastAction,
        scope: AudienceScope,
        audience_filter: str,
    ) -> str:
        """Extract the main subject/topic from the text.

        Strips the action verb, audience references, time references and
        channel keywords, leaving the core message.
        """

        cleaned = text
        # Remove action verbs.
        for pattern, _ in _ACTION_PATTERNS:
            cleaned = pattern.sub("", cleaned)
        # Remove audience phrases.
        cleaned = _ALL_AUDIENCE_RE.sub("", cleaned)
        cleaned = _DEPT_AUDIENCE_RE.sub("", cleaned)
        cleaned = _ROLE_AUDIENCE_RE.sub("", cleaned)
        cleaned = _INDIVIDUAL_AUDIENCE_RE.sub("", cleaned)
        # Remove time phrases.
        cleaned = _TIME_NOW_RE.sub("", cleaned)
        cleaned = _TIME_TONIGHT_RE.sub("", cleaned)
        cleaned = _TIME_TOMORROW_RE.sub("", cleaned)
        cleaned = _TIME_AT_RE.sub("", cleaned)
        cleaned = _TIME_IN_RE.sub("", cleaned)
        cleaned = _TIME_DAY_RE.sub("", cleaned)
        # Remove channel keywords.
        for pattern in _CHANNEL_PATTERNS.values():
            cleaned = pattern.sub("", cleaned)
        # Remove filler words and normalise whitespace.
        cleaned = re.sub(
            r"\b(?:to|about|that|the|of|for|on|at|in)\b", "", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.strip(" :,;-,.!?")  # noqa: B005 - intentional char-set strip
        return cleaned or "general notice"

    def _build_title(
        self,
        action: BroadcastAction,
        subject: str,
        scope: AudienceScope,
        audience_filter: str,
    ) -> str:
        """Generate a concise title for the broadcast."""

        action_word = action.value.capitalize()
        if scope is AudienceScope.ALL:
            return f"{action_word}: {subject}"
        if scope is AudienceScope.DEPARTMENT:
            return f"{action_word} to {audience_filter.title()} dept: {subject}"
        if scope is AudienceScope.ROLE:
            return f"{action_word} to all {audience_filter}s: {subject}"
        if scope is AudienceScope.INDIVIDUAL:
            return f"{action_word} to {audience_filter}: {subject}"
        return f"{action_word}: {subject}"

    def _build_body(
        self,
        raw: str,
        action: BroadcastAction,
        subject: str,
        scope: AudienceScope,
        audience_filter: str,
        scheduled_at: datetime | None,
    ) -> str:
        """Generate the full message body."""

        parts: list[str] = []
        if scope is AudienceScope.ALL:
            parts.append("This message is for all employees.")
        elif scope is AudienceScope.DEPARTMENT:
            parts.append(f"This message is for the {audience_filter} department.")
        elif scope is AudienceScope.ROLE:
            parts.append(f"This message is for all {audience_filter}s.")
        elif scope is AudienceScope.INDIVIDUAL:
            parts.append(f"This message is for {audience_filter}.")
        parts.append(f"Subject: {subject}.")
        if scheduled_at is not None:
            parts.append(f"Scheduled for: {scheduled_at.strftime('%Y-%m-%d %H:%M UTC')}.")
        parts.append(f'Original instruction: "{raw}"')
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Audience resolver
# ---------------------------------------------------------------------------


class AudienceResolver(Protocol):
    """Protocol for resolving a broadcast audience into concrete recipients.

    Implementations may query an HR system, LDAP directory, or a local
    user database. The :class:`StaticAudienceResolver` is a simple
    in-memory implementation suitable for testing.
    """

    def resolve_all(self) -> list[BroadcastTarget]: ...

    def resolve_department(self, department: str) -> list[BroadcastTarget]: ...

    def resolve_role(self, role: str) -> list[BroadcastTarget]: ...

    def resolve_individual(self, user_id: str) -> list[BroadcastTarget]: ...


class StaticAudienceResolver:
    """In-memory audience resolver backed by a list of :class:`BroadcastTarget`.

    Example::

        resolver = StaticAudienceResolver(targets=[
            BroadcastTarget(user_id="alice", department="engineering", role="engineer", email="alice@corp.com"),
            BroadcastTarget(user_id="bob", department="engineering", role="manager", email="bob@corp.com"),
        ])
    """

    def __init__(self, targets: list[BroadcastTarget] | None = None) -> None:
        self._targets: list[BroadcastTarget] = targets or []

    def add_target(self, target: BroadcastTarget) -> None:
        self._targets.append(target)

    def resolve_all(self) -> list[BroadcastTarget]:
        return list(self._targets)

    def resolve_department(self, department: str) -> list[BroadcastTarget]:
        dept = department.lower()
        return [t for t in self._targets if t.department.lower() == dept]

    def resolve_role(self, role: str) -> list[BroadcastTarget]:
        r = role.lower().rstrip("s")
        return [
            t
            for t in self._targets
            if t.role.lower().rstrip("s") == r or t.role.lower().rstrip("s").startswith(r)
        ]

    def resolve_individual(self, user_id: str) -> list[BroadcastTarget]:
        uid = user_id.lstrip("@").lower()
        return [t for t in self._targets if t.user_id.lower() == uid or t.name.lower() == uid]

    def resolve(self, intent: BroadcastIntent) -> list[BroadcastTarget]:
        """Dispatch to the appropriate resolver method based on intent scope."""

        if intent.audience_scope is AudienceScope.ALL:
            return self.resolve_all()
        if intent.audience_scope is AudienceScope.DEPARTMENT:
            return self.resolve_department(intent.audience_filter)
        if intent.audience_scope is AudienceScope.ROLE:
            return self.resolve_role(intent.audience_filter)
        if intent.audience_scope is AudienceScope.INDIVIDUAL:
            return self.resolve_individual(intent.audience_filter)
        return self.resolve_all()


# ---------------------------------------------------------------------------
# Broadcast service
# ---------------------------------------------------------------------------


class BroadcastService:
    """Orchestrates the full broadcast pipeline: parse → resolve → deliver → track.

    Example::

        engine = NotificationEngine()
        engine.register_channel(InAppChannel())
        resolver = StaticAudienceResolver(targets=[...])
        service = BroadcastService(engine=engine, resolver=resolver)

        broadcast = await service.broadcast_text(
            "notify all employees meeting tonight 7pm",
            created_by="manager-alice",
        )
        print(broadcast.status)  # BroadcastStatus.DELIVERED

        # Later, collect confirmations:
        confirmed = await service.get_receipts(broadcast.id, state=ConfirmationState.CONFIRMED)
        unconfirmed = await service.get_unconfirmed(broadcast.id)

        # Send reminders to unconfirmed recipients:
        await service.send_reminders(broadcast.id)
    """

    def __init__(
        self,
        *,
        engine: NotificationEngine,
        resolver: AudienceResolver,
        parser: BroadcastParser | None = None,
        reminder_interval: timedelta = timedelta(hours=1),
        max_reminders: int = 3,
    ) -> None:
        self._engine = engine
        self._resolver = resolver
        self._parser = parser or BroadcastParser()
        self._reminder_interval = reminder_interval
        self._max_reminders = max_reminders
        self._broadcasts: dict[str, BroadcastMessage] = {}
        self._receipts: dict[str, list[Receipt]] = {}
        self._status: dict[str, BroadcastStatus] = {}
        self._lock = asyncio.Lock()
        self._reminder_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self, text: str) -> BroadcastIntent:
        """Parse natural-language *text* into a :class:`BroadcastIntent`."""

        return self._parser.parse(text)

    # ------------------------------------------------------------------
    # Broadcast execution
    # ------------------------------------------------------------------

    async def broadcast_text(
        self,
        text: str,
        *,
        created_by: str = "",
        channels: list[NotificationChannel] | None = None,
        priority: NotificationPriority | None = None,
    ) -> BroadcastMessage:
        """Parse *text*, resolve audience, generate message and deliver.

        This is the main entry point. Returns the :class:`BroadcastMessage`
        with its final :attr:`BroadcastMessage.status`.
        """

        intent = self.parse(text)
        if channels:
            intent.channels = channels
        if priority is not None:
            intent.priority = priority
        return await self.broadcast_intent(intent, created_by=created_by)

    async def broadcast_intent(
        self,
        intent: BroadcastIntent,
        *,
        created_by: str = "",
    ) -> BroadcastMessage:
        """Execute a pre-parsed :class:`BroadcastIntent`."""

        targets = await asyncio.to_thread(self._resolver.resolve, intent)
        message = BroadcastMessage(
            intent=intent,
            title=intent.message_title or "Broadcast",
            body=intent.message_body or intent.raw_text,
            priority=intent.priority,
            channels=intent.channels or self._engine.registered_channels,
            created_by=created_by,
        )
        # Initialise receipts.
        receipts = [Receipt(broadcast_id=message.id, target=t) for t in targets]
        async with self._lock:
            self._broadcasts[message.id] = message
            self._receipts[message.id] = receipts

        logger.info(
            "Broadcast %s initiated: scope=%s targets=%d channels=%s",
            message.id,
            intent.audience_scope.value,
            len(targets),
            [c.value for c in message.channels],
        )

        await self._deliver(message, receipts)
        return message

    async def _deliver(self, message: BroadcastMessage, receipts: list[Receipt]) -> None:
        """Deliver the message to all recipients via the notification engine."""

        # Mark as sending.
        async with self._lock:
            self._status[message.id] = BroadcastStatus.SENDING
        # Deliver concurrently per-recipient.
        tasks = [self._deliver_to_recipient(message, receipt) for receipt in receipts]
        await asyncio.gather(*tasks)

        # Determine final status.
        delivered = sum(1 for r in receipts if r.delivered_channels)
        total = len(receipts)
        if total == 0:
            status = BroadcastStatus.FAILED
        elif delivered == total:
            status = BroadcastStatus.DELIVERED
        elif delivered > 0:
            status = BroadcastStatus.PARTIAL
        else:
            status = BroadcastStatus.FAILED
        logger.info(
            "Broadcast %s complete: status=%s delivered=%d/%d",
            message.id,
            status.value,
            delivered,
            total,
        )
        async with self._lock:
            self._status[message.id] = status

    async def _deliver_to_recipient(self, message: BroadcastMessage, receipt: Receipt) -> None:
        """Deliver the message to a single recipient across all channels."""

        target = receipt.target
        recipient_addr = target.email or target.phone or target.user_id
        records = await self._engine.notify(
            title=message.title,
            body=message.body,
            recipient=recipient_addr,
            channels=message.channels,
            priority=message.priority,
            metadata={"broadcast_id": message.id, "target_user_id": target.user_id},
        )
        any_delivered = False
        errors: list[str] = []
        for record in records:
            if record.status in (DeliveryStatus.SENT, DeliveryStatus.DELIVERED):
                receipt.mark_delivered(record.channel)
                any_delivered = True
            elif record.status is DeliveryStatus.FAILED:
                errors.append(f"{record.channel.value}: {record.error}")
        if not any_delivered:
            receipt.mark_failed("; ".join(errors) if errors else "All channels failed")
        elif receipt.state is ConfirmationState.PENDING:
            receipt.state = ConfirmationState.DELIVERED

    # ------------------------------------------------------------------
    # Receipt tracking
    # ------------------------------------------------------------------

    async def get_receipts(
        self,
        broadcast_id: str,
        *,
        state: ConfirmationState | None = None,
    ) -> list[Receipt]:
        """Return receipts for a broadcast, optionally filtered by state."""

        async with self._lock:
            receipts = list(self._receipts.get(broadcast_id, []))
        if state is not None:
            receipts = [r for r in receipts if r.state is state]
        return receipts

    async def get_unconfirmed(self, broadcast_id: str) -> list[Receipt]:
        """Return receipts that have not been confirmed yet."""

        async with self._lock:
            receipts = list(self._receipts.get(broadcast_id, []))
        return [
            r
            for r in receipts
            if r.state
            in (ConfirmationState.PENDING, ConfirmationState.DELIVERED, ConfirmationState.REMINDED)
        ]

    async def confirm_receipt(self, broadcast_id: str, user_id: str) -> Receipt | None:
        """Mark a receipt as confirmed by *user_id*."""

        async with self._lock:
            receipts = self._receipts.get(broadcast_id, [])
            for receipt in receipts:
                if receipt.target.user_id == user_id:
                    receipt.confirm()
                    logger.info(
                        "Receipt confirmed: broadcast=%s user=%s",
                        broadcast_id,
                        user_id,
                    )
                    return receipt
        return None

    async def get_status(self, broadcast_id: str) -> BroadcastStatus:
        """Return the current status of a broadcast."""

        async with self._lock:
            return self._status.get(broadcast_id, BroadcastStatus.DRAFT)

    async def get_broadcast(self, broadcast_id: str) -> BroadcastMessage | None:
        """Return the broadcast message by ID."""

        async with self._lock:
            return self._broadcasts.get(broadcast_id)

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    async def send_reminders(
        self,
        broadcast_id: str,
        *,
        force: bool = False,
    ) -> list[Receipt]:
        """Send reminder notifications to unconfirmed recipients.

        Skips recipients who have already received ``max_reminders`` unless
        *force* is True. Returns the list of recipients that were reminded.
        """

        broadcast = await self.get_broadcast(broadcast_id)
        if broadcast is None:
            raise BroadcastError(f"Broadcast not found: {broadcast_id}")

        unconfirmed = await self.get_unconfirmed(broadcast_id)
        reminded: list[Receipt] = []
        for receipt in unconfirmed:
            if not force and receipt.reminder_count >= self._max_reminders:
                logger.debug(
                    "Skipping reminder for %s: max reminders reached",
                    receipt.target.user_id,
                )
                continue
            # Only remind if enough time has passed since the last reminder
            # (or since delivery if no reminder yet).
            now = datetime.now(UTC)
            last_activity = receipt.last_reminder_at or receipt.confirmed_at
            if last_activity is not None:
                elapsed = now - last_activity
                if elapsed < self._reminder_interval and not force:
                    continue
            # Send a reminder via the notification engine.
            reminder_title = f"[Reminder] {broadcast.title}"
            reminder_body = (
                f"This is a reminder. You have not yet confirmed receipt of: {broadcast.body}"
            )
            recipient_addr = receipt.target.email or receipt.target.phone or receipt.target.user_id
            await self._engine.notify(
                title=reminder_title,
                body=reminder_body,
                recipient=recipient_addr,
                channels=broadcast.channels,
                priority=broadcast.priority,
                metadata={"broadcast_id": broadcast_id, "reminder": True},
            )
            receipt.record_reminder()
            reminded.append(receipt)
            logger.info(
                "Reminder #%d sent: broadcast=%s user=%s",
                receipt.reminder_count,
                broadcast_id,
                receipt.target.user_id,
            )
        return reminded

    async def schedule_reminders(
        self,
        broadcast_id: str,
        *,
        interval: timedelta | None = None,
    ) -> None:
        """Schedule periodic reminders as a background task until all confirm.

        This creates a long-running asyncio task. The task terminates when
        all recipients have confirmed or ``max_reminders`` is reached for
        every unconfirmed recipient.
        """

        effective_interval = interval or self._reminder_interval

        async def _reminder_loop() -> None:
            while True:
                await asyncio.sleep(effective_interval.total_seconds())
                try:
                    reminded = await self.send_reminders(broadcast_id)
                except BroadcastError:
                    break
                if not reminded:
                    unconfirmed = await self.get_unconfirmed(broadcast_id)
                    if not unconfirmed:
                        logger.info(
                            "Reminder loop complete: all confirmed for %s",
                            broadcast_id,
                        )
                        break
                    # Check if all have hit max reminders.
                    all_maxed = all(r.reminder_count >= self._max_reminders for r in unconfirmed)
                    if all_maxed:
                        logger.warning(
                            "Reminder loop stopped: max reminders reached for %s",
                            broadcast_id,
                        )
                        break

        task = asyncio.create_task(_reminder_loop())
        self._reminder_tasks.add(task)
        task.add_done_callback(self._reminder_tasks.discard)
        logger.info(
            "Reminder loop scheduled for broadcast %s (interval=%ss)",
            broadcast_id,
            effective_interval.total_seconds(),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def broadcast_count(self) -> int:
        """Total number of broadcasts initiated."""

        return len(self._broadcasts)

    @property
    def engine(self) -> NotificationEngine:
        """The underlying notification engine."""

        return self._engine


__all__ = [
    "AudienceResolver",
    "AudienceScope",
    "BroadcastAction",
    "BroadcastError",
    "BroadcastIntent",
    "BroadcastMessage",
    "BroadcastParser",
    "BroadcastService",
    "BroadcastStatus",
    "BroadcastTarget",
    "ConfirmationState",
    "Receipt",
    "StaticAudienceResolver",
]
