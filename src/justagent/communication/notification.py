"""Multi-channel notification engine — pluggable channels, templates, delivery tracking.

Routes notifications to one or more pluggable channels (desktop, email,
SMS, IM, webhook, in-app). Each channel is an async subclass of
:class:`BaseNotificationChannel`. The engine fans a single notification out
to all configured channels concurrently, records per-channel delivery
status, and supports named templates for reusable message formats.

Design:

* :class:`NotificationChannel` — enum of supported channel types.
* :class:`NotificationPriority` — urgency ranking (drives escalation).
* :class:`Notification` — the Pydantic message model routed through channels.
* :class:`NotificationTemplate` — reusable, parameterised message skeleton.
* :class:`DeliveryRecord` — per-recipient, per-channel delivery outcome.
* :class:`BaseNotificationChannel` — ABC defining the async ``send`` contract.
* Channel implementations: :class:`DesktopChannel`, :class:`EmailChannel`,
  :class:`SMSChannel`, :class:`IMChannel`, :class:`WebhookChannel`,
  :class:`InAppChannel`.
* :class:`NotificationEngine` — orchestrates fan-out, templates, retries
  and delivery tracking.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from email.message import EmailMessage
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("justagent.communication.notification")


class NotificationError(Exception):
    """Raised when a notification cannot be constructed or delivered."""


class NotificationChannel(str, Enum):  # noqa: UP042 - match existing codebase style
    """Supported delivery channel types."""

    DESKTOP = "desktop"
    EMAIL = "email"
    SMS = "sms"
    IM = "im"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    MOBILE = "mobile"


class NotificationPriority(str, Enum):  # noqa: UP042
    """Urgency ranking for a notification.

    Higher priority may trigger additional escalation channels or
    reminder loops in the broadcast system.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

    @property
    def weight(self) -> int:
        """Numeric weight for sorting (higher = more urgent)."""

        order = {
            NotificationPriority.LOW: 0,
            NotificationPriority.NORMAL: 1,
            NotificationPriority.HIGH: 2,
            NotificationPriority.URGENT: 3,
            NotificationPriority.CRITICAL: 4,
        }
        return order[self]


class DeliveryStatus(str, Enum):  # noqa: UP042
    """Lifecycle status of a single delivery attempt."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    READ = "read"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class Notification(BaseModel):
    """An addressable notification message routed through one or more channels.

    Attributes:
        id: Unique identifier (UUID4 hex).
        title: Short headline.
        body: Full message text.
        priority: Urgency ranking.
        recipient: User identifier (email, user-id, phone, etc.).
        recipient_name: Optional display name.
        channels: Channels to attempt (empty = use engine defaults).
        metadata: Arbitrary key/value pairs for channel-specific routing.
        created_at: UTC creation timestamp.
        template_id: If generated from a template, its identifier.
        tags: Free-form labels for filtering/grouping.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str
    body: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    recipient: str = ""
    recipient_name: str = ""
    channels: list[NotificationChannel] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    template_id: str | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since :attr:`created_at`."""

        return (datetime.now(UTC) - self.created_at).total_seconds()


class NotificationTemplate(BaseModel):
    """A reusable, parameterised notification skeleton.

    The ``title_template`` and ``body_template`` strings are formatted with
    ``str.format_map`` using the parameters supplied at render time.
    """

    id: str
    name: str
    title_template: str
    body_template: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: list[NotificationChannel] = Field(default_factory=list)
    description: str = ""

    def render(self, **params: Any) -> Notification:
        """Produce a :class:`Notification` from this template + parameters."""

        safe_params = _SafeDict(params)
        title = self.title_template.format_map(safe_params)
        body = self.body_template.format_map(safe_params)
        return Notification(
            title=title,
            body=body,
            priority=self.priority,
            channels=list(self.channels),
            template_id=self.id,
        )


class _SafeDict(dict[str, Any]):
    """Dict subclass that returns ``{key}`` for missing keys (no KeyError)."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class DeliveryRecord(BaseModel):
    """Outcome of delivering one notification to one channel for one recipient."""

    notification_id: str
    channel: NotificationChannel
    recipient: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    delivered_at: datetime | None = None
    error: str = ""
    attempts: int = 0

    def mark_sent(self) -> None:
        self.status = DeliveryStatus.SENT
        self.delivered_at = datetime.now(UTC)

    def mark_delivered(self) -> None:
        self.status = DeliveryStatus.DELIVERED
        self.delivered_at = datetime.now(UTC)

    def mark_failed(self, error: str) -> None:
        self.status = DeliveryStatus.FAILED
        self.error = error


# ---------------------------------------------------------------------------
# Channel base class + implementations
# ---------------------------------------------------------------------------


class BaseNotificationChannel(ABC):
    """Abstract base for async notification delivery channels.

    Subclasses must implement :meth:`send`. The ``channel_type`` property
    identifies which :class:`NotificationChannel` enum value this
    implementation handles.
    """

    @property
    @abstractmethod
    def channel_type(self) -> NotificationChannel:
        """The :class:`NotificationChannel` this implementation serves."""

    @abstractmethod
    async def send(self, notification: Notification) -> DeliveryRecord:
        """Deliver *notification* and return a :class:`DeliveryRecord`."""

    async def close(self) -> None:
        """Release any resources (HTTP clients, sockets, etc.)."""

        return


class DesktopChannel(BaseNotificationChannel):
    """Best-effort desktop notification via ``notify-send`` (Linux) or
    ``osascript`` (macOS). Failures are recorded but never raised.
    """

    def __init__(self, *, urgency_override: str | None = None) -> None:
        self._urgency_override = urgency_override
        self._platform = sys.platform

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.DESKTOP

    async def send(self, notification: Notification) -> DeliveryRecord:
        record = DeliveryRecord(
            notification_id=notification.id,
            channel=self.channel_type,
            recipient=notification.recipient or "local-desktop",
            attempts=1,
        )
        try:
            await asyncio.to_thread(self._send_sync, notification)
            record.mark_sent()
        except Exception as exc:  # noqa: BLE001 - record, do not raise
            record.mark_failed(str(exc))
            logger.warning("Desktop notification %s failed: %s", notification.id, exc)
        return record

    def _send_sync(self, notification: Notification) -> None:
        urgency = self._urgency_override or self._urgency_for(notification.priority)
        if self._platform.startswith("linux"):
            subprocess.run(
                ["notify-send", "--urgency", urgency, notification.title, notification.body],
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        elif self._platform == "darwin":
            title = notification.title.replace('"', '\\"')
            body = notification.body.replace('"', '\\"')
            script = f'display notification "{body}" with title "{title}"'
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        else:
            logger.debug("Desktop notifications unsupported on %s", self._platform)

    @staticmethod
    def _urgency_for(priority: NotificationPriority) -> str:
        return {
            NotificationPriority.LOW: "low",
            NotificationPriority.NORMAL: "normal",
            NotificationPriority.HIGH: "normal",
            NotificationPriority.URGENT: "critical",
            NotificationPriority.CRITICAL: "critical",
        }.get(priority, "normal")


class EmailChannel(BaseNotificationChannel):
    """SMTP email delivery.

    Uses :mod:`smtplib` wrapped in ``asyncio.to_thread`` so the event loop
    is never blocked. TLS is used when ``use_tls`` is True (recommended).
    """

    def __init__(
        self,
        *,
        host: str,
        port: int = 587,
        username: str,
        password: str,
        from_addr: str,
        use_tls: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._from_addr = from_addr
        self._use_tls = use_tls
        self._timeout = timeout

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.EMAIL

    async def send(self, notification: Notification) -> DeliveryRecord:
        record = DeliveryRecord(
            notification_id=notification.id,
            channel=self.channel_type,
            recipient=notification.recipient,
            attempts=1,
        )
        if not notification.recipient:
            record.mark_failed("No recipient email address provided")
            return record
        try:
            await asyncio.to_thread(self._send_sync, notification)
            record.mark_sent()
        except Exception as exc:  # noqa: BLE001
            record.mark_failed(str(exc))
            logger.warning("Email notification %s failed: %s", notification.id, exc)
        return record

    def _send_sync(self, notification: Notification) -> None:
        msg = EmailMessage()
        msg["From"] = self._from_addr
        msg["To"] = notification.recipient
        msg["Subject"] = notification.title
        msg.set_content(notification.body)
        msg["X-Priority"] = str(notification.priority.weight + 1)
        with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as server:
            server.ehlo()
            if self._use_tls:
                server.starttls()
                server.ehlo()
            if self._username:
                server.login(self._username, self._password)
            server.send_message(msg)


class SMSChannel(BaseNotificationChannel):
    """SMS delivery via a configurable HTTP gateway (e.g. Twilio REST API).

    The gateway URL, auth header, and JSON body template are configurable.
    The ``{phone}`` and ``{message}`` placeholders in ``body_template``
    are substituted at send time.
    """

    def __init__(
        self,
        *,
        api_url: str,
        auth_header: str = "",
        body_template: str = '{"to": "{phone}", "body": "{message}"}',
        timeout: float = 15.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._api_url = api_url
        self._auth_header = auth_header
        self._body_template = body_template
        self._timeout = timeout
        self._extra_headers = extra_headers or {}
        self._client: httpx.AsyncClient | None = None

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.SMS

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = dict(self._extra_headers)
            if self._auth_header:
                headers["Authorization"] = self._auth_header
            self._client = httpx.AsyncClient(headers=headers, timeout=self._timeout)
        return self._client

    async def send(self, notification: Notification) -> DeliveryRecord:
        record = DeliveryRecord(
            notification_id=notification.id,
            channel=self.channel_type,
            recipient=notification.recipient,
            attempts=1,
        )
        if not notification.recipient:
            record.mark_failed("No recipient phone number provided")
            return record
        try:
            client = await self._ensure_client()
            body = self._body_template.format(
                phone=notification.recipient,
                message=notification.body,
                title=notification.title,
            )
            # body is JSON if the default template is used, otherwise raw.
            if body.strip().startswith("{"):
                resp = await client.post(
                    self._api_url, content=body, headers={"Content-Type": "application/json"}
                )
            else:
                resp = await client.post(self._api_url, content=body)
            resp.raise_for_status()
            record.mark_sent()
        except Exception as exc:  # noqa: BLE001
            record.mark_failed(str(exc))
            logger.warning("SMS notification %s failed: %s", notification.id, exc)
        return record

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class IMChannel(BaseNotificationChannel):
    """Instant-messaging delivery via an HTTP webhook (Slack, Lark, Teams, etc.).

    Posts a JSON payload to the configured webhook URL. The ``payload_builder``
    callable lets callers customise the request body per platform.
    """

    def __init__(
        self,
        *,
        webhook_url: str,
        timeout: float = 15.0,
        headers: dict[str, str] | None = None,
        payload_builder: _PayloadBuilder | None = None,
    ) -> None:
        self._webhook_url = webhook_url
        self._timeout = timeout
        self._headers = headers or {"Content-Type": "application/json"}
        self._payload_builder = payload_builder
        self._client: httpx.AsyncClient | None = None

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.IM

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(headers=self._headers, timeout=self._timeout)
        return self._client

    async def send(self, notification: Notification) -> DeliveryRecord:
        record = DeliveryRecord(
            notification_id=notification.id,
            channel=self.channel_type,
            recipient=notification.recipient or self._webhook_url,
            attempts=1,
        )
        try:
            client = await self._ensure_client()
            payload = self._build_payload(notification)
            resp = await client.post(self._webhook_url, json=payload)
            resp.raise_for_status()
            record.mark_sent()
        except Exception as exc:  # noqa: BLE001
            record.mark_failed(str(exc))
            logger.warning("IM notification %s failed: %s", notification.id, exc)
        return record

    def _build_payload(self, notification: Notification) -> dict[str, Any]:
        if self._payload_builder is not None:
            return self._payload_builder(notification)
        return {
            "text": f"*{notification.title}*\n{notification.body}",
            "title": notification.title,
            "priority": notification.priority.value,
            "recipient": notification.recipient,
        }

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class WebhookChannel(BaseNotificationChannel):
    """Generic HTTP webhook delivery.

    Posts the full notification as JSON to an arbitrary endpoint.
    """

    def __init__(
        self,
        *,
        url: str,
        timeout: float = 15.0,
        headers: dict[str, str] | None = None,
        secret: str = "",
    ) -> None:
        self._url = url
        self._timeout = timeout
        self._headers = headers or {"Content-Type": "application/json"}
        self._secret = secret
        self._client: httpx.AsyncClient | None = None

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.WEBHOOK

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(headers=self._headers, timeout=self._timeout)
        return self._client

    async def send(self, notification: Notification) -> DeliveryRecord:
        record = DeliveryRecord(
            notification_id=notification.id,
            channel=self.channel_type,
            recipient=notification.recipient or self._url,
            attempts=1,
        )
        try:
            client = await self._ensure_client()
            payload = notification.model_dump(mode="json")
            if self._secret:
                import hashlib
                import hmac

                raw = notification.model_dump_json().encode("utf-8")
                signature = hmac.new(self._secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
                resp = await client.post(
                    self._url,
                    content=raw,
                    headers={"Content-Type": "application/json", "X-Signature": signature},
                )
            else:
                resp = await client.post(self._url, json=payload)
            resp.raise_for_status()
            record.mark_sent()
        except Exception as exc:  # noqa: BLE001
            record.mark_failed(str(exc))
            logger.warning("Webhook notification %s failed: %s", notification.id, exc)
        return record

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class InAppChannel(BaseNotificationChannel):
    """In-app notification stored in an in-memory queue.

    Useful for testing and for single-process deployments. Consumers
    drain the queue via :meth:`drain`.
    """

    _queue: asyncio.Queue[Notification] = asyncio.Queue()

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.IN_APP

    async def send(self, notification: Notification) -> DeliveryRecord:
        record = DeliveryRecord(
            notification_id=notification.id,
            channel=self.channel_type,
            recipient=notification.recipient or "in-app",
            attempts=1,
        )
        try:
            await self._queue.put(notification)
            record.mark_delivered()
        except Exception as exc:  # noqa: BLE001
            record.mark_failed(str(exc))
        return record

    @classmethod
    async def drain(cls, timeout: float = 0.0) -> list[Notification]:
        """Remove and return all queued notifications (non-blocking by default)."""

        result: list[Notification] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    result.append(cls._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            else:
                try:
                    result.append(await asyncio.wait_for(cls._queue.get(), timeout=remaining))
                except TimeoutError:
                    break
        return result


# Type alias for the IM payload builder callable.
_PayloadBuilder = Callable[["Notification"], dict[str, Any]]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class NotificationEngine:
    """Orchestrates multi-channel notification fan-out, templates and tracking.

    Example::

        engine = NotificationEngine()
        engine.register_channel(DesktopChannel())
        engine.register_channel(WebhookChannel(url="https://hooks.example.com/notify"))

        record = await engine.notify(
            title="Deploy complete",
            body="v2.3.0 shipped to production",
            recipient="ops-team",
            channels=[NotificationChannel.DESKTOP, NotificationChannel.WEBHOOK],
            priority=NotificationPriority.HIGH,
        )
        print(record.status)  # DeliveryStatus.SENT
    """

    def __init__(self) -> None:
        self._channels: dict[NotificationChannel, BaseNotificationChannel] = {}
        self._templates: dict[str, NotificationTemplate] = {}
        self._records: list[DeliveryRecord] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Channel registration
    # ------------------------------------------------------------------

    def register_channel(self, channel: BaseNotificationChannel) -> None:
        """Register or replace a channel implementation."""

        self._channels[channel.channel_type] = channel
        logger.debug("Registered notification channel: %s", channel.channel_type.value)

    def unregister_channel(self, channel_type: NotificationChannel) -> bool:
        """Remove a channel by type. Returns True if it was registered."""

        return self._channels.pop(channel_type, None) is not None

    def get_channel(self, channel_type: NotificationChannel) -> BaseNotificationChannel | None:
        """Return the channel implementation for *channel_type*, if registered."""

        return self._channels.get(channel_type)

    @property
    def registered_channels(self) -> list[NotificationChannel]:
        """Channel types currently registered."""

        return list(self._channels.keys())

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def register_template(self, template: NotificationTemplate) -> None:
        """Register or replace a notification template by its ``id``."""

        self._templates[template.id] = template

    def get_template(self, template_id: str) -> NotificationTemplate | None:
        """Return a registered template by ID, or ``None``."""

        return self._templates.get(template_id)

    def render_template(self, template_id: str, **params: Any) -> Notification:
        """Render a registered template with the given parameters.

        Raises :class:`NotificationError` if the template is unknown.
        """

        template = self._templates.get(template_id)
        if template is None:
            raise NotificationError(f"Unknown notification template: {template_id}")
        return template.render(**params)

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    async def notify(
        self,
        *,
        title: str,
        body: str,
        recipient: str = "",
        channels: list[NotificationChannel] | None = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> list[DeliveryRecord]:
        """Build a notification and fan it out to the specified channels.

        If *channels* is ``None`` or empty, all registered channels are used.
        Returns one :class:`DeliveryRecord` per attempted channel.
        """

        notification = Notification(
            title=title,
            body=body,
            recipient=recipient,
            channels=channels or [],
            priority=priority,
            metadata=metadata or {},
            tags=tags or [],
        )
        return await self.deliver(notification)

    async def deliver(self, notification: Notification) -> list[DeliveryRecord]:
        """Fan *notification* out to its target channels concurrently.

        Uses the notification's own ``channels`` list, or all registered
        channels when that list is empty.
        """

        targets = notification.channels or self.registered_channels
        if not targets:
            logger.warning(
                "No channels configured for notification %s; nothing delivered",
                notification.id,
            )
            return []

        tasks = [self._deliver_one(notification, ct) for ct in targets]
        records = await asyncio.gather(*tasks)
        async with self._lock:
            self._records.extend(records)
        return records

    async def _deliver_one(
        self, notification: Notification, channel_type: NotificationChannel
    ) -> DeliveryRecord:
        channel = self._channels.get(channel_type)
        if channel is None:
            record = DeliveryRecord(
                notification_id=notification.id,
                channel=channel_type,
                recipient=notification.recipient,
            )
            record.mark_failed(f"No implementation registered for {channel_type.value}")
            return record
        try:
            return await channel.send(notification)
        except Exception as exc:  # noqa: BLE001
            record = DeliveryRecord(
                notification_id=notification.id,
                channel=channel_type,
                recipient=notification.recipient,
            )
            record.mark_failed(str(exc))
            logger.error(
                "Unexpected error delivering notification %s via %s: %s",
                notification.id,
                channel_type.value,
                exc,
            )
            return record

    async def notify_from_template(
        self,
        template_id: str,
        *,
        recipient: str = "",
        channels: list[NotificationChannel] | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[DeliveryRecord]:
        """Render a template and deliver it in one step."""

        notification = self.render_template(template_id, **(params or {}))
        notification.recipient = recipient or notification.recipient
        if channels:
            notification.channels = channels
        return await self.deliver(notification)

    # ------------------------------------------------------------------
    # Delivery tracking
    # ------------------------------------------------------------------

    async def get_records(
        self,
        *,
        notification_id: str | None = None,
        channel: NotificationChannel | None = None,
        status: DeliveryStatus | None = None,
    ) -> list[DeliveryRecord]:
        """Query stored delivery records by optional filters."""

        async with self._lock:
            snapshot = list(self._records)
        result = snapshot
        if notification_id is not None:
            result = [r for r in result if r.notification_id == notification_id]
        if channel is not None:
            result = [r for r in result if r.channel is channel]
        if status is not None:
            result = [r for r in result if r.status is status]
        return result

    async def delivery_summary(self) -> dict[str, int]:
        """Return a count of records grouped by :class:`DeliveryStatus`."""

        async with self._lock:
            snapshot = list(self._records)
        summary: dict[str, int] = {s.value: 0 for s in DeliveryStatus}
        for record in snapshot:
            summary[record.status.value] += 1
        return summary

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all registered channels that hold resources."""

        for channel in self._channels.values():
            await channel.close()


__all__ = [
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
