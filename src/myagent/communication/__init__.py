"""Enterprise internal communication module for the Omniagent platform.

This package provides five integrated subsystems:

* **Messaging** (:mod:`myagent.communication.messaging`) — channels, direct
  messages, group chats, threaded replies and message history with read
  receipts.
* **Notification** (:mod:`myagent.communication.notification`) — a
  multi-channel notification engine with pluggable channels (desktop,
  email, SMS, IM, webhook, in-app), templates, priority routing and
  delivery tracking.
* **Broadcast** (:mod:`myagent.communication.broadcast`) — a one-to-many
  broadcast system that parses natural-language manager intent (e.g.
  *"notify all employees meeting tonight 7pm"*), resolves the target
  audience, delivers via multiple channels and tracks per-recipient
  confirmations with reminder scheduling.
* **Audit** (:mod:`myagent.communication.audit`) — an immutable,
  append-only audit log with SHA-256 hash chaining for tamper evidence,
  query filtering and export (JSONL / JSON / CSV).
* **Meeting** (:mod:`myagent.communication.meeting`) — full meeting
  lifecycle management: create, schedule, notify attendees, track RSVPs,
  update agenda status in real time and generate structured minutes.

All subsystems use asyncio for concurrent operations and Pydantic v2 for
data models. Logging follows the ``myagent.communication.<submodule>``
namespace via the standard ``logging`` module.

Quick start::

    from myagent.communication import (
        AuditStore, AuditCategory, AuditLevel,
        NotificationEngine, InAppChannel, NotificationChannel,
        MessagingService,
        BroadcastService, BroadcastParser, StaticAudienceResolver,
        MeetingService, MeetingType, RSVPStatus,
    )

    # Audit log
    audit = AuditStore()
    await audit.append(actor="system", action="startup",
                       category=AuditCategory.SYSTEM, level=AuditLevel.INFO)

    # Notification engine
    engine = NotificationEngine()
    engine.register_channel(InAppChannel())

    # Messaging
    messaging = MessagingService()
    channel = await messaging.create_channel("general", created_by="alice")

    # Broadcast
    resolver = StaticAudienceResolver()
    broadcast = BroadcastService(engine=engine, resolver=resolver)
    msg = await broadcast.broadcast_text("notify all employees town hall tomorrow 10am")

    # Meetings
    meetings = MeetingService(engine=engine)
    meeting = await meetings.create_meeting(title="Standup", organizer="alice")
    await meetings.notify_attendees(meeting.id)
"""

from __future__ import annotations

from myagent.communication.audit import (
    AuditCategory,
    AuditEntry,
    AuditLevel,
    AuditQuery,
    AuditStore,
    ChainVerification,
    create_audit_store,
)
from myagent.communication.broadcast import (
    AudienceResolver,
    AudienceScope,
    BroadcastAction,
    BroadcastError,
    BroadcastIntent,
    BroadcastMessage,
    BroadcastParser,
    BroadcastService,
    BroadcastStatus,
    BroadcastTarget,
    ConfirmationState,
    Receipt,
    StaticAudienceResolver,
)
from myagent.communication.meeting import (
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
from myagent.communication.messaging import (
    Channel,
    ChannelType,
    DirectConversation,
    Message,
    MessagePriority,
    MessageStatus,
    MessageType,
    MessagingError,
    MessagingService,
)
from myagent.communication.notification import (
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
    # audit
    "AuditCategory",
    "AuditEntry",
    "AuditLevel",
    "AuditQuery",
    "AuditStore",
    "ChainVerification",
    "create_audit_store",
    # broadcast
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
    # messaging
    "Channel",
    "ChannelType",
    "DirectConversation",
    "Message",
    "MessagePriority",
    "MessageStatus",
    "MessageType",
    "MessagingError",
    "MessagingService",
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
