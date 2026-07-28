"""Meeting management — schedule, notify attendees, track RSVPs, generate minutes.

Provides the full meeting lifecycle for the Omniagent platform:

1. **Create** a meeting with attendees, agenda and optional location/link.
2. **Notify** all attendees via the :class:`NotificationEngine` (multi-channel).
3. **Track** RSVPs (accept / decline / tentative) per attendee.
4. **Update** meeting status as it progresses (scheduled → in-progress → completed).
5. **Generate** structured meeting minutes from agenda items and discussion notes.

Design:

* :class:`MeetingStatus` — lifecycle enum.
* :class:`RSVPStatus` — per-attendee response enum.
* :class:`MeetingType` — in-person, video, phone, hybrid.
* :class:`Attendee` — a meeting participant with RSVP state.
* :class:`AgendaItem` — a single agenda entry with status.
* :class:`Meeting` — the top-level meeting model.
* :class:`MeetingMinutes` — generated minutes document.
* :class:`MeetingService` — async orchestrator (create, notify, RSVP, minutes).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from myagent.communication.notification import (
    NotificationChannel,
    NotificationEngine,
    NotificationPriority,
)

logger = logging.getLogger("myagent.communication.meeting")


class MeetingError(Exception):
    """Raised when a meeting operation is invalid."""


class MeetingStatus(str, Enum):  # noqa: UP042 - match existing codebase style
    """Lifecycle status of a meeting."""

    DRAFT = "draft"
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RSVPStatus(str, Enum):  # noqa: UP042
    """Per-attendee response to a meeting invitation."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    TENTATIVE = "tentative"
    NO_RESPONSE = "no_response"


class MeetingType(str, Enum):  # noqa: UP042
    """The modality of a meeting."""

    IN_PERSON = "in_person"
    VIDEO = "video"
    PHONE = "phone"
    HYBRID = "hybrid"


class AgendaItemStatus(str, Enum):  # noqa: UP042
    """Status of a single agenda item during the meeting."""

    PENDING = "pending"
    DISCUSSED = "discussed"
    DEFERRED = "deferred"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Attendee(BaseModel):
    """A meeting participant with RSVP tracking.

    Attributes:
        user_id: Unique user identifier.
        name: Display name.
        email: Email address (for notifications).
        phone: Phone number (for SMS notifications).
        rsvp: Current RSVP status.
        rsvp_at: When the RSVP was last updated (UTC), or None.
        required: Whether attendance is mandatory.
        role: Role in the meeting (organizer, presenter, note-taker, etc.).
    """

    user_id: str
    name: str = ""
    email: str = ""
    phone: str = ""
    rsvp: RSVPStatus = RSVPStatus.PENDING
    rsvp_at: datetime | None = None
    required: bool = True
    role: str = "participant"

    @property
    def is_attending(self) -> bool:
        """True when the attendee has accepted or tentatively accepted."""

        return self.rsvp in (RSVPStatus.ACCEPTED, RSVPStatus.TENTATIVE)

    def respond(self, status: RSVPStatus) -> None:
        """Update the RSVP status and timestamp."""

        self.rsvp = status
        self.rsvp_at = datetime.now(UTC)


class AgendaItem(BaseModel):
    """A single agenda entry for a meeting.

    Attributes:
        id: Unique item identifier.
        title: Short description of the topic.
        presenter: User ID of the person leading this item.
        duration_minutes: Allocated time.
        status: Discussion status.
        notes: Free-text notes captured during discussion.
        order: Display/sorting order (lower = earlier).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str
    presenter: str = ""
    duration_minutes: int = 10
    status: AgendaItemStatus = AgendaItemStatus.PENDING
    notes: str = ""
    order: int = 0

    def mark_discussed(self, notes: str = "") -> None:
        """Mark this item as discussed, optionally appending notes."""

        self.status = AgendaItemStatus.DISCUSSED
        if notes:
            self.notes = notes

    def defer(self, reason: str = "") -> None:
        """Defer this item to a future meeting."""

        self.status = AgendaItemStatus.DEFERRED
        if reason:
            self.notes = reason

    def skip(self, reason: str = "") -> None:
        """Skip this item without discussion."""

        self.status = AgendaItemStatus.SKIPPED
        if reason:
            self.notes = reason


class Meeting(BaseModel):
    """A scheduled meeting with attendees and an agenda.

    Attributes:
        id: Unique meeting identifier.
        title: Meeting title / subject.
        description: Longer description or context.
        organizer: User ID of the organizer.
        attendees: List of attendees with RSVP state.
        agenda: Ordered list of agenda items.
        meeting_type: Modality (in-person, video, etc.).
        location: Physical room or address.
        meeting_url: Video conference URL.
        dial_in: Phone dial-in number.
        start_time: Scheduled start (UTC).
        end_time: Scheduled end (UTC).
        status: Current lifecycle status.
        created_at: UTC creation timestamp.
        cancelled_at: When the meeting was cancelled, or None.
        completed_at: When the meeting was completed, or None.
        metadata: Arbitrary extra key/value pairs.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str
    description: str = ""
    organizer: str = ""
    attendees: list[Attendee] = Field(default_factory=list)
    agenda: list[AgendaItem] = Field(default_factory=list)
    meeting_type: MeetingType = MeetingType.VIDEO
    location: str = ""
    meeting_url: str = ""
    dial_in: str = ""
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC) + timedelta(hours=1))
    end_time: datetime = Field(default_factory=lambda: datetime.now(UTC) + timedelta(hours=2))
    status: MeetingStatus = MeetingStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cancelled_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_minutes(self) -> int:
        """Scheduled duration in minutes."""

        return int((self.end_time - self.start_time).total_seconds() // 60)

    def get_attendee(self, user_id: str) -> Attendee | None:
        """Return the attendee matching *user_id*, or ``None``."""

        return next((a for a in self.attendees if a.user_id == user_id), None)

    @property
    def accepted_count(self) -> int:
        """Number of attendees who have accepted."""

        return sum(1 for a in self.attendees if a.rsvp is RSVPStatus.ACCEPTED)

    @property
    def declined_count(self) -> int:
        """Number of attendees who have declined."""

        return sum(1 for a in self.attendees if a.rsvp is RSVPStatus.DECLINED)

    @property
    def pending_count(self) -> int:
        """Number of attendees who have not yet responded."""

        return sum(
            1 for a in self.attendees if a.rsvp in (RSVPStatus.PENDING, RSVPStatus.NO_RESPONSE)
        )

    @property
    def is_upcoming(self) -> bool:
        """True when the meeting is scheduled and has not started yet."""

        return self.status in (MeetingStatus.SCHEDULED, MeetingStatus.CONFIRMED) and (
            self.start_time > datetime.now(UTC)
        )


class MeetingMinutes(BaseModel):
    """Generated minutes document for a completed meeting.

    Attributes:
        meeting_id: The parent meeting's ID.
        title: Meeting title (copied for standalone readability).
        date: The meeting start time.
        attendees_present: List of user IDs who attended.
        attendees_absent: List of user IDs who were absent.
        agenda_summary: Per-item summary of discussion outcomes.
        action_items: Extracted action items (owner, description, due_date).
        decisions: Key decisions made during the meeting.
        notes: Free-text notes captured during the meeting.
        generated_at: UTC generation timestamp.
        generated_by: User ID of the note-taker or system.
    """

    meeting_id: str
    title: str
    date: datetime
    attendees_present: list[str] = Field(default_factory=list)
    attendees_absent: list[str] = Field(default_factory=list)
    agenda_summary: list[dict[str, Any]] = Field(default_factory=list)
    action_items: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    notes: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    generated_by: str = ""

    def to_text(self) -> str:
        """Render the minutes as a plain-text document."""

        lines: list[str] = [
            f"Meeting Minutes: {self.title}",
            f"Date: {self.date.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            f"Attendees Present: {', '.join(self.attendees_present) or 'None'}",
            f"Attendees Absent: {', '.join(self.attendees_absent) or 'None'}",
            "",
            "Agenda Summary:",
        ]
        for item in self.agenda_summary:
            status = item.get("status", "pending")
            title = item.get("title", "(untitled)")
            notes = item.get("notes", "")
            lines.append(f"  - [{status}] {title}")
            if notes:
                lines.append(f"      Notes: {notes}")
        if self.action_items:
            lines.append("")
            lines.append("Action Items:")
            for action in self.action_items:
                owner = action.get("owner", "unassigned")
                desc = action.get("description", "")
                due = action.get("due_date", "")
                lines.append(f"  - [{owner}] {desc}" + (f" (due: {due})" if due else ""))
        if self.decisions:
            lines.append("")
            lines.append("Decisions:")
            for decision in self.decisions:
                lines.append(f"  - {decision}")
        if self.notes:
            lines.append("")
            lines.append("General Notes:")
            lines.append(f"  {self.notes}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MeetingService:
    """Async orchestrator for the full meeting lifecycle.

    Example::

        engine = NotificationEngine()
        engine.register_channel(InAppChannel())
        service = MeetingService(engine=engine)

        meeting = await service.create_meeting(
            title="Sprint Planning",
            organizer="alice",
            attendees=[
                Attendee(user_id="alice", name="Alice", email="alice@corp.com"),
                Attendee(user_id="bob", name="Bob", email="bob@corp.com"),
            ],
            agenda=[AgendaItem(title="Review backlog"), AgendaItem(title="Capacity")],
            start_time=datetime(2025, 1, 15, 14, 0, tzinfo=UTC),
            end_time=datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        )
        await service.notify_attendees(meeting.id)
        await service.respond_rsvp(meeting.id, "bob", RSVPStatus.ACCEPTED)
        minutes = await service.generate_minutes(meeting.id, generated_by="alice")
        print(minutes.to_text())
    """

    def __init__(
        self,
        *,
        engine: NotificationEngine | None = None,
        default_channels: list[NotificationChannel] | None = None,
    ) -> None:
        self._engine = engine
        self._default_channels = default_channels or []
        self._meetings: dict[str, Meeting] = {}
        self._minutes: dict[str, MeetingMinutes] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Meeting CRUD
    # ------------------------------------------------------------------

    async def create_meeting(
        self,
        *,
        title: str,
        organizer: str = "",
        attendees: list[Attendee] | None = None,
        agenda: list[AgendaItem] | None = None,
        meeting_type: MeetingType = MeetingType.VIDEO,
        location: str = "",
        meeting_url: str = "",
        dial_in: str = "",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Meeting:
        """Create a new meeting in ``DRAFT`` status. Returns the meeting.

        The organizer is automatically added to the attendee list if not
        already present.
        """

        now = datetime.now(UTC)
        start = start_time or (now + timedelta(hours=1))
        end = end_time or (start + timedelta(hours=1))
        if end <= start:
            raise MeetingError("Meeting end time must be after start time")

        attendee_list = list(attendees or [])
        if organizer and not any(a.user_id == organizer for a in attendee_list):
            attendee_list.insert(
                0,
                Attendee(
                    user_id=organizer,
                    name=organizer,
                    role="organizer",
                    required=True,
                ),
            )

        # Sort agenda by order field.
        agenda_list = sorted(agenda or [], key=lambda a: a.order)

        meeting = Meeting(
            title=title,
            description=description,
            organizer=organizer,
            attendees=attendee_list,
            agenda=agenda_list,
            meeting_type=meeting_type,
            location=location,
            meeting_url=meeting_url,
            dial_in=dial_in,
            start_time=start,
            end_time=end,
            status=MeetingStatus.DRAFT,
            metadata=metadata or {},
        )
        async with self._lock:
            self._meetings[meeting.id] = meeting
        logger.info("Meeting created: %s (%s) by %s", meeting.title, meeting.id, organizer)
        return meeting

    async def get_meeting(self, meeting_id: str) -> Meeting | None:
        """Return a meeting by ID, or ``None``."""

        async with self._lock:
            return self._meetings.get(meeting_id)

    async def list_meetings(
        self,
        *,
        organizer: str | None = None,
        status: MeetingStatus | None = None,
        participant: str | None = None,
    ) -> list[Meeting]:
        """List meetings, optionally filtered by organizer, status or participant."""

        async with self._lock:
            meetings = list(self._meetings.values())
        if organizer is not None:
            meetings = [m for m in meetings if m.organizer == organizer]
        if status is not None:
            meetings = [m for m in meetings if m.status is status]
        if participant is not None:
            meetings = [m for m in meetings if m.get_attendee(participant) is not None]
        return meetings

    async def update_meeting(
        self,
        meeting_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        location: str | None = None,
        meeting_url: str | None = None,
        meeting_type: MeetingType | None = None,
    ) -> Meeting:
        """Update mutable fields of a meeting.

        Raises :class:`MeetingError` if the meeting is cancelled or completed.
        """

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            if meeting.status in (MeetingStatus.CANCELLED, MeetingStatus.COMPLETED):
                raise MeetingError(f"Cannot update meeting in {meeting.status.value} status")
            if title is not None:
                meeting.title = title
            if description is not None:
                meeting.description = description
            if start_time is not None:
                meeting.start_time = start_time
            if end_time is not None:
                meeting.end_time = end_time
            if location is not None:
                meeting.location = location
            if meeting_url is not None:
                meeting.meeting_url = meeting_url
            if meeting_type is not None:
                meeting.meeting_type = meeting_type
            if meeting.end_time <= meeting.start_time:
                raise MeetingError("Meeting end time must be after start time")
            return meeting

    async def cancel_meeting(self, meeting_id: str, reason: str = "") -> Meeting:
        """Cancel a meeting. Raises if already completed."""

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            if meeting.status is MeetingStatus.COMPLETED:
                raise MeetingError("Cannot cancel a completed meeting")
            meeting.status = MeetingStatus.CANCELLED
            meeting.cancelled_at = datetime.now(UTC)
            if reason:
                meeting.metadata["cancellation_reason"] = reason
            return meeting

    # ------------------------------------------------------------------
    # Attendee management
    # ------------------------------------------------------------------

    async def add_attendee(self, meeting_id: str, attendee: Attendee) -> Meeting:
        """Add an attendee to a meeting."""

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            if meeting.get_attendee(attendee.user_id) is not None:
                raise MeetingError(f"Attendee {attendee.user_id} already in meeting {meeting_id}")
            meeting.attendees.append(attendee)
            return meeting

    async def remove_attendee(self, meeting_id: str, user_id: str) -> Meeting:
        """Remove an attendee from a meeting."""

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            meeting.attendees = [a for a in meeting.attendees if a.user_id != user_id]
            return meeting

    async def respond_rsvp(
        self,
        meeting_id: str,
        user_id: str,
        rsvp: RSVPStatus,
    ) -> Attendee:
        """Record an RSVP response from an attendee.

        Returns the updated attendee. Raises if the attendee is not found.
        """

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            attendee = meeting.get_attendee(user_id)
            if attendee is None:
                raise MeetingError(f"User {user_id} is not an attendee of meeting {meeting_id}")
            attendee.respond(rsvp)
            logger.info(
                "RSVP: meeting=%s user=%s response=%s",
                meeting_id,
                user_id,
                rsvp.value,
            )
            # Auto-confirm if all required attendees have responded.
            if rsvp is RSVPStatus.ACCEPTED and meeting.status is MeetingStatus.SCHEDULED:
                all_required_responded = all(
                    a.rsvp != RSVPStatus.PENDING for a in meeting.attendees if a.required
                )
                if all_required_responded:
                    meeting.status = MeetingStatus.CONFIRMED
            return attendee

    # ------------------------------------------------------------------
    # Agenda management
    # ------------------------------------------------------------------

    async def add_agenda_item(
        self,
        meeting_id: str,
        item: AgendaItem,
    ) -> Meeting:
        """Add an agenda item to a meeting."""

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            meeting.agenda.append(item)
            meeting.agenda.sort(key=lambda a: a.order)
            return meeting

    async def update_agenda_item(
        self,
        meeting_id: str,
        item_id: str,
        *,
        status: AgendaItemStatus | None = None,
        notes: str | None = None,
    ) -> AgendaItem:
        """Update an agenda item's status and/or notes during the meeting."""

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            item = next((a for a in meeting.agenda if a.id == item_id), None)
            if item is None:
                raise MeetingError(f"Agenda item not found: {item_id}")
            if status is not None:
                if status is AgendaItemStatus.DISCUSSED:
                    item.mark_discussed(notes or "")
                elif status is AgendaItemStatus.DEFERRED:
                    item.defer(notes or "")
                elif status is AgendaItemStatus.SKIPPED:
                    item.skip(notes or "")
                else:
                    item.status = status
            elif notes is not None:
                item.notes = notes
            return item

    # ------------------------------------------------------------------
    # Notification
    # ------------------------------------------------------------------

    async def notify_attendees(
        self,
        meeting_id: str,
        *,
        channels: list[NotificationChannel] | None = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> Meeting:
        """Send meeting invitations to all attendees via the notification engine.

        Sets the meeting status to ``SCHEDULED`` (from ``DRAFT``) and fires
        a notification per attendee. If no notification engine is configured,
        only the status is updated.
        """

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            if meeting.status is MeetingStatus.DRAFT:
                meeting.status = MeetingStatus.SCHEDULED
            meeting_snapshot = meeting.model_copy()

        if self._engine is not None:
            target_channels = channels or self._default_channels or self._engine.registered_channels
            tasks = [
                self._notify_one_attendee(meeting_snapshot, attendee, target_channels, priority)
                for attendee in meeting_snapshot.attendees
            ]
            await asyncio.gather(*tasks)
        logger.info(
            "Notified %d attendees for meeting %s",
            len(meeting_snapshot.attendees),
            meeting_id,
        )
        return meeting_snapshot

    async def _notify_one_attendee(
        self,
        meeting: Meeting,
        attendee: Attendee,
        channels: list[NotificationChannel],
        priority: NotificationPriority,
    ) -> None:
        """Send a meeting invitation to a single attendee."""

        if self._engine is None:
            return
        title = f"Meeting Invitation: {meeting.title}"
        body_parts = [
            f"You have been invited to: {meeting.title}",
            f"Organizer: {meeting.organizer or 'Unknown'}",
            f"Type: {meeting.meeting_type.value.replace('_', ' ').title()}",
            f"Start: {meeting.start_time.strftime('%Y-%m-%d %H:%M UTC')}",
            f"End: {meeting.end_time.strftime('%Y-%m-%d %H:%M UTC')}",
        ]
        if meeting.location:
            body_parts.append(f"Location: {meeting.location}")
        if meeting.meeting_url:
            body_parts.append(f"Join: {meeting.meeting_url}")
        if meeting.dial_in:
            body_parts.append(f"Dial-in: {meeting.dial_in}")
        if meeting.agenda:
            body_parts.append("Agenda:")
            for item in meeting.agenda:
                body_parts.append(f"  - {item.title}")
        body = "\n".join(body_parts)
        recipient = attendee.email or attendee.phone or attendee.user_id
        try:
            await self._engine.notify(
                title=title,
                body=body,
                recipient=recipient,
                channels=channels,
                priority=priority,
                metadata={"meeting_id": meeting.id, "attendee": attendee.user_id},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to notify attendee %s for meeting %s: %s",
                attendee.user_id,
                meeting.id,
                exc,
            )

    async def send_reminder(
        self,
        meeting_id: str,
        *,
        minutes_before: int = 15,
    ) -> Meeting:
        """Send a meeting-start reminder to all accepted attendees.

        Intended to be called *minutes_before* the scheduled start time.
        """

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            meeting_snapshot = meeting.model_copy()

        if self._engine is None:
            return meeting_snapshot

        accepted = [a for a in meeting_snapshot.attendees if a.is_attending]
        title = f"[Reminder] {meeting_snapshot.title} starts in {minutes_before} minutes"
        body = (
            f"Reminder: {meeting_snapshot.title} starts at "
            f"{meeting_snapshot.start_time.strftime('%Y-%m-%d %H:%M UTC')}."
        )
        if meeting_snapshot.meeting_url:
            body += f"\nJoin: {meeting_snapshot.meeting_url}"
        elif meeting_snapshot.location:
            body += f"\nLocation: {meeting_snapshot.location}"

        tasks: list[Any] = []
        for attendee in accepted:
            recipient = attendee.email or attendee.phone or attendee.user_id
            tasks.append(
                self._engine.notify(
                    title=title,
                    body=body,
                    recipient=recipient,
                    priority=NotificationPriority.HIGH,
                    metadata={"meeting_id": meeting_id, "reminder": True},
                )
            )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(
            "Sent reminders to %d attendees for meeting %s",
            len(accepted),
            meeting_id,
        )
        return meeting_snapshot

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    async def start_meeting(self, meeting_id: str) -> Meeting:
        """Transition a meeting to ``IN_PROGRESS``."""

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            if meeting.status not in (MeetingStatus.SCHEDULED, MeetingStatus.CONFIRMED):
                raise MeetingError(f"Cannot start meeting in {meeting.status.value} status")
            meeting.status = MeetingStatus.IN_PROGRESS
            return meeting

    async def complete_meeting(self, meeting_id: str) -> Meeting:
        """Transition a meeting to ``COMPLETED``."""

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")
            if meeting.status is not MeetingStatus.IN_PROGRESS:
                raise MeetingError(f"Cannot complete meeting in {meeting.status.value} status")
            meeting.status = MeetingStatus.COMPLETED
            meeting.completed_at = datetime.now(UTC)
            return meeting

    # ------------------------------------------------------------------
    # Minutes generation
    # ------------------------------------------------------------------

    async def generate_minutes(
        self,
        meeting_id: str,
        *,
        generated_by: str = "",
        notes: str = "",
        decisions: list[str] | None = None,
        action_items: list[dict[str, Any]] | None = None,
    ) -> MeetingMinutes:
        """Generate structured meeting minutes from the meeting state.

        Attendees are classified as present/absent based on their RSVP.
        Each agenda item's status and notes are summarised. Additional
        notes, decisions and action items can be supplied by the caller.

        Returns the generated :class:`MeetingMinutes`.
        """

        async with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                raise MeetingError(f"Meeting not found: {meeting_id}")

        present = [
            a.user_id
            for a in meeting.attendees
            if a.rsvp in (RSVPStatus.ACCEPTED, RSVPStatus.TENTATIVE)
        ]
        absent = [
            a.user_id
            for a in meeting.attendees
            if a.rsvp in (RSVPStatus.DECLINED, RSVPStatus.NO_RESPONSE, RSVPStatus.PENDING)
        ]

        agenda_summary: list[dict[str, Any]] = []
        for item in sorted(meeting.agenda, key=lambda a: a.order):
            agenda_summary.append(
                {
                    "id": item.id,
                    "title": item.title,
                    "presenter": item.presenter,
                    "status": item.status.value,
                    "notes": item.notes,
                    "duration_minutes": item.duration_minutes,
                }
            )

        minutes = MeetingMinutes(
            meeting_id=meeting.id,
            title=meeting.title,
            date=meeting.start_time,
            attendees_present=present,
            attendees_absent=absent,
            agenda_summary=agenda_summary,
            action_items=action_items or [],
            decisions=decisions or [],
            notes=notes,
            generated_by=generated_by,
        )
        async with self._lock:
            self._minutes[meeting.id] = minutes
        logger.info("Minutes generated for meeting %s", meeting_id)
        return minutes

    async def get_minutes(self, meeting_id: str) -> MeetingMinutes | None:
        """Return previously generated minutes for a meeting, or ``None``."""

        async with self._lock:
            return self._minutes.get(meeting_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def meeting_count(self) -> int:
        """Total number of meetings managed."""

        return len(self._meetings)

    @property
    def engine(self) -> NotificationEngine | None:
        """The underlying notification engine, if configured."""

        return self._engine


__all__ = [
    "AgendaItem",
    "AgendaItemStatus",
    "Attendee",
    "Meeting",
    "MeetingError",
    "MeetingMinutes",
    "MeetingService",
    "MeetingStatus",
    "MeetingType",
    "RSVPStatus",
]
