"""Enterprise internal communication module for the JustAgent platform.

This package provides two integrated subsystems:

* **Notification** (:mod:`justagent.communication.notification`) — a
  multi-channel notification engine with pluggable channels (desktop,
  email, SMS, IM, webhook, in-app), templates, priority routing and
  delivery tracking.
* **Meeting** (:mod:`justagent.communication.meeting`) — full meeting
  lifecycle management: create, schedule, notify attendees, track RSVPs,
  update agenda status in real time and generate structured minutes.

All subsystems use asyncio for concurrent operations and Pydantic v2 for
data models. Logging follows the ``justagent.communication.<submodule>``
namespace via the standard ``logging`` module.

Quick start::

    from justagent.communication import (
        NotificationEngine, InAppChannel, NotificationChannel,
        MeetingService, MeetingType, RSVPStatus,
    )

    # Notification engine
    engine = NotificationEngine()
    engine.register_channel(InAppChannel())

    # Meetings
    meetings = MeetingService(engine=engine)
    meeting = await meetings.create_meeting(title="Standup", organizer="alice")
    await meetings.notify_attendees(meeting.id)
"""

from __future__ import annotations

from justagent.communication.meeting import (
    AgendaItem,
    AgendaItemStatus,
    Attendee,
    Meeting,
    MeetingError,
    MeetingMinutes,
    MeetingService,
    MeetingStatus,
    MeetingType,
    RSVPStatus,
)
from justagent.communication.notification import (
    BaseNotificationChannel,
    DeliveryRecord,
    DeliveryStatus,
    DesktopChannel,
    EmailChannel,
    IMChannel,
    InAppChannel,
    Notification,
    NotificationChannel,
    NotificationEngine,
    NotificationError,
    NotificationPriority,
    NotificationTemplate,
    SMSChannel,
    WebhookChannel,
)

__all__ = [
    # meeting
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
    # notification
    "BaseNotificationChannel",
    "DeliveryRecord",
    "DeliveryStatus",
    "DesktopChannel",
    "EmailChannel",
    "IMChannel",
    "InAppChannel",
    "Notification",
    "NotificationChannel",
    "NotificationEngine",
    "NotificationError",
    "NotificationPriority",
    "NotificationTemplate",
    "SMSChannel",
    "WebhookChannel",
]
