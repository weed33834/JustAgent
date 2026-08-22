"""Tests for communication/notification.py — engine, channels, templates."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from justagent.communication.notification import (
    DeliveryRecord,
    DeliveryStatus,
    DesktopChannel,
    EmailChannel,
    InAppChannel,
    Notification,
    NotificationChannel,
    NotificationEngine,
    NotificationError,
    NotificationPriority,
    NotificationTemplate,
    WebhookChannel,
)


def _notif(**kw) -> Notification:
    base: dict = {"title": "t", "body": "b", "recipient": "ops", "channels": [NotificationChannel.IN_APP]}
    base.update(kw)
    return Notification(**base)


class TestModels:
    def test_age_seconds(self) -> None:
        n = _notif(created_at=time.time() - 30)
        assert 25 <= n.age_seconds <= 40

    def test_priority_weights_ordered(self) -> None:
        order = [
            NotificationPriority.LOW,
            NotificationPriority.NORMAL,
            NotificationPriority.HIGH,
            NotificationPriority.URGENT,
        ]
        weights = [p.weight for p in order]
        assert weights == sorted(weights)

    def test_template_render_with_params(self) -> None:
        tpl = NotificationTemplate(
            id="deploy", name="Deploy notice",
            title_template="Deploy {env}",
            body_template="{service} v{ver} shipped",
            channels=[NotificationChannel.WEBHOOK],
        )
        n = tpl.render(env="prod", service="api", ver="2.0")
        assert n.title == "Deploy prod"
        assert n.body == "api v2.0 shipped"

    def test_delivery_record_lifecycle(self) -> None:
        r = DeliveryRecord(notification_id="n1", channel=NotificationChannel.EMAIL, recipient="a@b.c")
        assert r.status is DeliveryStatus.PENDING
        r.mark_sent()
        r.mark_delivered()
        assert r.status is DeliveryStatus.DELIVERED
        # failure can be recorded from any prior state (no guard by design)
        r.mark_failed("bounce")
        assert r.status is DeliveryStatus.FAILED


@pytest.mark.asyncio
class TestEngineFanOut:
    async def test_notify_fans_out_to_registered_channels(self) -> None:
        eng = NotificationEngine()
        eng.register_channel(InAppChannel())
        records = await eng.notify(title="hi", body="there")
        assert len(records) == 1
        assert records[0].status in (DeliveryStatus.SENT, DeliveryStatus.DELIVERED)
        assert all(r.notification_id == records[0].notification_id for r in records)

    async def test_no_channels_returns_empty(self) -> None:
        eng = NotificationEngine()
        assert await eng.notify(title="x", body="y") == []

    async def test_unregistered_channel_marks_failed(self) -> None:
        eng = NotificationEngine()
        records = await eng.notify(
            title="x", body="y", channels=[NotificationChannel.EMAIL],
        )
        assert records[0].status is DeliveryStatus.FAILED
        assert "No implementation" in (records[0].error or "")

    async def test_channel_exception_becomes_failed_record(self) -> None:
        eng = NotificationEngine()

        class Exploding(InAppChannel):
            async def send(self, notification):  # type: ignore[override]
                raise RuntimeError("smtp down")

        eng.register_channel(Exploding())
        records = await eng.notify(title="x", body="y")
        assert records[0].status is DeliveryStatus.FAILED
        assert "smtp down" in (records[0].error or "")

    async def test_template_flow_via_engine(self) -> None:
        eng = NotificationEngine()
        eng.register_channel(InAppChannel())
        eng.register_template(NotificationTemplate(
            id="alert", name="Alert",
            title_template="[ALERT] {kind}", body_template="{detail}",
            channels=[NotificationChannel.IN_APP],
        ))
        records = await eng.notify_from_template(
            "alert", recipient="ops", params={"kind": "disk", "detail": "91%"}
        )
        assert records and records[0].status is DeliveryStatus.DELIVERED
        with pytest.raises(NotificationError):
            eng.render_template("nope")

    async def test_records_history_accumulates(self) -> None:
        eng = NotificationEngine()
        eng.register_channel(InAppChannel())
        await eng.notify(title="1", body="b")
        await eng.notify(title="2", body="b")
        summary = await eng.delivery_summary()
        assert sum(summary.values()) == 2


@pytest.mark.asyncio
class TestChannels:
    async def test_in_app_send_marks_delivered(self) -> None:
        ch = InAppChannel()
        rec = await ch.send(_notif(channels=[NotificationChannel.IN_APP]))
        assert rec.status is DeliveryStatus.DELIVERED

    async def test_desktop_urgency_mapping(self) -> None:
        assert DesktopChannel._urgency_for(NotificationPriority.URGENT) == "critical"

    async def test_webhook_posts_payload(self) -> None:
        ch = WebhookChannel(url="https://hooks.test/x")
        sent: dict = {}

        class FakeResp:
            status_code = 200
            def raise_for_status(self): ...

        import httpx

        class FakeAsync(httpx.AsyncClient):
            async def post(self, url, json=None, **kw):  # type: ignore[override]
                sent["url"] = url
                sent["json"] = json
                return FakeResp()

        ch._client = FakeAsync()
        rec = await ch.send(_notif(channels=[NotificationChannel.WEBHOOK]))
        assert rec.status is DeliveryStatus.SENT
        assert sent["url"] == "https://hooks.test/x"

    async def test_email_send_uses_smtp_flow(self) -> None:
        """Mocked server: reaching SENT proves the full SMTP flow ran offline."""
        ch = EmailChannel(host="smtp.test", port=465, username="u", password="p", from_addr="j@x.y")
        smtp = MagicMock()
        with patch("smtplib.SMTP", return_value=smtp):
            rec = await ch.send(_notif(channels=[NotificationChannel.EMAIL], recipient="ops@x.y"))
        assert rec.status is DeliveryStatus.SENT
        # login/send_message are recorded on the context-manager child mock
        assert smtp.mock_calls, 'SMTP was never constructed'
        cm_calls = [c for c in smtp.mock_calls if '.login' in c[0]]
        assert any('.login' in c[0] for c in smtp.mock_calls), smtp.mock_calls
        assert any('.send_message' in c[0] for c in smtp.mock_calls), smtp.mock_calls
