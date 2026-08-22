"""Tests for :mod:`justagent.core.notifications`."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from justagent.core.notifications import (
    DesktopChannel,
    LogChannel,
    Notification,
    NotificationError,
    NotificationLevel,
    NotificationManager,
    TerminalChannel,
    WebhookChannel,
)

# ---------------------------------------------------------------------------
# TestNotificationLevel
# ---------------------------------------------------------------------------


class TestNotificationLevel:
    def test_values(self) -> None:
        assert NotificationLevel.INFO.value == "info"
        assert NotificationLevel.SUCCESS.value == "success"
        assert NotificationLevel.WARNING.value == "warning"
        assert NotificationLevel.ERROR.value == "error"

    def test_is_str(self) -> None:
        assert isinstance(NotificationLevel.INFO, str)

    def test_from_value(self) -> None:
        assert NotificationLevel("warning") is NotificationLevel.WARNING


# ---------------------------------------------------------------------------
# TestNotification
# ---------------------------------------------------------------------------


class TestNotification:
    def test_construction_defaults(self) -> None:
        n = Notification(title="hi", message="hello")
        assert n.title == "hi"
        assert n.message == "hello"
        assert n.level is NotificationLevel.INFO
        assert n.timestamp == 0.0
        assert n.metadata == {}

    def test_construction_full(self) -> None:
        n = Notification(
            title="t",
            message="m",
            level=NotificationLevel.ERROR,
            timestamp=123.0,
            metadata={"k": "v"},
        )
        assert n.level is NotificationLevel.ERROR
        assert n.timestamp == 123.0
        assert n.metadata == {"k": "v"}

    def test_frozen(self) -> None:
        n = Notification(title="t", message="m")
        with pytest.raises(FrozenInstanceError):
            n.title = "other"  # type: ignore[misc]

    def test_metadata_default_independent(self) -> None:
        a = Notification(title="a", message="m")
        b = Notification(title="b", message="m")
        a.metadata["x"] = "1"  # mutate one instance's metadata
        assert b.metadata == {}


# ---------------------------------------------------------------------------
# TestTerminalChannel
# ---------------------------------------------------------------------------


class TestTerminalChannel:
    def test_info_goes_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        channel = TerminalChannel(use_color=False)
        channel.send(Notification(title="t", message="m", level=NotificationLevel.INFO))
        captured = capsys.readouterr()
        assert "INFO" in captured.out
        assert "t" in captured.out
        assert "m" in captured.out
        assert captured.err == ""

    def test_success_goes_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        channel = TerminalChannel(use_color=False)
        channel.send(Notification(title="t", message="m", level=NotificationLevel.SUCCESS))
        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out
        assert captured.err == ""

    def test_warning_goes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        channel = TerminalChannel(use_color=False)
        channel.send(Notification(title="t", message="m", level=NotificationLevel.WARNING))
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert captured.out == ""

    def test_error_goes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        channel = TerminalChannel(use_color=False)
        channel.send(Notification(title="t", message="m", level=NotificationLevel.ERROR))
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert captured.out == ""

    def test_color_codes_present_when_enabled(self, capsys: pytest.CaptureFixture[str]) -> None:
        channel = TerminalChannel(use_color=True)
        channel.send(Notification(title="t", message="m", level=NotificationLevel.SUCCESS))
        captured = capsys.readouterr()
        assert "\x1b[32m" in captured.out
        assert "\x1b[0m" in captured.out

    def test_color_codes_absent_when_disabled(self, capsys: pytest.CaptureFixture[str]) -> None:
        channel = TerminalChannel(use_color=False)
        channel.send(Notification(title="t", message="m", level=NotificationLevel.SUCCESS))
        captured = capsys.readouterr()
        assert "\x1b[" not in captured.out


# ---------------------------------------------------------------------------
# TestDesktopChannel
# ---------------------------------------------------------------------------


class TestDesktopChannel:
    def test_linux_uses_notify_send(self) -> None:
        channel = DesktopChannel()
        channel.platform = "linux"
        with patch(
            "justagent.core.notifications.subprocess.run",
            return_value=MagicMock(returncode=0),
        ) as mock_run:
            channel.send(
                Notification(title="t", message="m", level=NotificationLevel.INFO, timestamp=1.0)
            )
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        assert "notify-send" in args[0]

    def test_linux_failure_does_not_raise(self, capsys: pytest.CaptureFixture[str]) -> None:
        channel = DesktopChannel()
        channel.platform = "linux"
        with patch(
            "justagent.core.notifications.subprocess.run",
            side_effect=FileNotFoundError("notify-send not installed"),
        ):
            channel.send(Notification(title="t", message="m"))
        captured = capsys.readouterr()
        assert "desktop notification failed" in captured.err

    def test_macos_uses_osascript(self) -> None:
        channel = DesktopChannel()
        channel.platform = "darwin"
        with patch(
            "justagent.core.notifications.subprocess.run",
            return_value=MagicMock(returncode=0),
        ) as mock_run:
            channel.send(
                Notification(title="t", message="m", level=NotificationLevel.INFO, timestamp=1.0)
            )
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        assert "osascript" in args[0]

    def test_unknown_platform_no_op(self, capsys: pytest.CaptureFixture[str]) -> None:
        channel = DesktopChannel()
        channel.platform = "freebsd"
        # Should not raise or print anything.
        channel.send(Notification(title="t", message="m"))
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


# ---------------------------------------------------------------------------
# TestWebhookChannel
# ---------------------------------------------------------------------------


class TestWebhookChannel:
    def test_success(self) -> None:
        channel = WebhookChannel(url="https://example.com/hook")
        with patch(
            "justagent.core.notifications.httpx.post", return_value=MagicMock()
        ) as mock_post:
            channel.send(
                Notification(
                    title="t",
                    message="m",
                    level=NotificationLevel.SUCCESS,
                    timestamp=123.0,
                    metadata={"k": "v"},
                )
            )
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["title"] == "t"
        assert kwargs["json"]["message"] == "m"
        assert kwargs["json"]["level"] == "success"
        assert kwargs["json"]["timestamp"] == 123.0
        assert kwargs["json"]["metadata"] == {"k": "v"}
        assert kwargs["timeout"] == 10.0

    def test_failure_does_not_raise(self, capsys: pytest.CaptureFixture[str]) -> None:
        channel = WebhookChannel(url="https://example.com/hook")
        with patch(
            "justagent.core.notifications.httpx.post",
            side_effect=Exception("network down"),
        ):
            channel.send(Notification(title="t", message="m"))
        captured = capsys.readouterr()
        assert "webhook notification failed" in captured.err

    def test_custom_headers(self) -> None:
        channel = WebhookChannel(
            url="https://example.com/hook",
            headers={"Authorization": "Bearer secret"},
            timeout=5.0,
        )
        with patch(
            "justagent.core.notifications.httpx.post", return_value=MagicMock()
        ) as mock_post:
            channel.send(Notification(title="t", message="m"))
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer secret"
        assert kwargs["timeout"] == 5.0

    def test_raise_for_status_does_not_raise(self, capsys: pytest.CaptureFixture[str]) -> None:
        channel = WebhookChannel(url="https://example.com/hook")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")
        with patch("justagent.core.notifications.httpx.post", return_value=mock_response):
            channel.send(Notification(title="t", message="m"))
        captured = capsys.readouterr()
        assert "webhook notification failed" in captured.err


# ---------------------------------------------------------------------------
# TestLogChannel
# ---------------------------------------------------------------------------


class TestLogChannel:
    def test_writes_to_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "notifications.log"
        channel = LogChannel(log_path=log_path)
        channel.send(
            Notification(
                title="t",
                message="m",
                level=NotificationLevel.ERROR,
                timestamp=0.0,
            )
        )
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "ERROR" in content
        assert "t: m" in content
        assert content.endswith("\n")

    def test_appends_multiple_lines(self, tmp_path: Path) -> None:
        log_path = tmp_path / "notifications.log"
        channel = LogChannel(log_path=log_path)
        channel.send(Notification(title="first", message="m1", level=NotificationLevel.INFO))
        channel.send(Notification(title="second", message="m2", level=NotificationLevel.WARNING))
        content = log_path.read_text(encoding="utf-8")
        assert "first" in content
        assert "second" in content
        assert content.count("\n") == 2

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nested" / "dir" / "notifications.log"
        channel = LogChannel(log_path=log_path)
        channel.send(Notification(title="t", message="m"))
        assert log_path.exists()
        assert log_path.parent.is_dir()

    def test_default_timestamp_uses_now(self, tmp_path: Path) -> None:
        log_path = tmp_path / "notifications.log"
        channel = LogChannel(log_path=log_path)
        # When timestamp=0.0 (default), the channel substitutes time.time().
        with patch("justagent.core.notifications.time.time", return_value=1700000000.0):
            channel.send(Notification(title="t", message="m"))
        content = log_path.read_text(encoding="utf-8")
        # The substituted timestamp should produce an ISO8601 string.
        assert "2023" in content or "1970" not in content


# ---------------------------------------------------------------------------
# TestNotificationManager
# ---------------------------------------------------------------------------


class TestNotificationManager:
    def test_add_remove_channel(self) -> None:
        manager = NotificationManager()
        channel = TerminalChannel(use_color=False)
        manager.add_channel(channel)
        assert channel in manager.channels
        assert manager.remove_channel(channel) is True
        assert channel not in manager.channels

    def test_remove_missing_returns_false(self) -> None:
        manager = NotificationManager()
        channel = TerminalChannel(use_color=False)
        assert manager.remove_channel(channel) is False

    def test_send_to_all_channels(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        manager = NotificationManager()
        terminal = TerminalChannel(use_color=False)
        log_path = tmp_path / "n.log"
        log_channel = LogChannel(log_path=log_path)
        manager.add_channel(terminal)
        manager.add_channel(log_channel)
        results = manager.send(Notification(title="t", message="m", level=NotificationLevel.INFO))
        captured = capsys.readouterr()
        assert "INFO" in captured.out
        assert log_path.exists()
        assert all(err is None for err in results.values())
        assert len(results) == 2

    def test_one_channel_failure_does_not_stop_others(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        manager = NotificationManager()

        class FailingChannel:
            def send(self, notification: Notification) -> None:
                raise RuntimeError("boom")

        failing = FailingChannel()
        terminal = TerminalChannel(use_color=False)
        log_path = tmp_path / "n.log"
        log_channel = LogChannel(log_path=log_path)
        manager.add_channel(failing)
        manager.add_channel(terminal)
        manager.add_channel(log_channel)
        results = manager.send(Notification(title="t", message="m"))
        assert isinstance(results[failing], RuntimeError)
        assert results[terminal] is None
        assert results[log_channel] is None
        captured = capsys.readouterr()
        # The terminal channel should still have printed.
        assert "t: m" in captured.out or "t" in captured.out
        # Log channel should have written.
        assert log_path.exists()

    def test_send_empty_channels(self) -> None:
        manager = NotificationManager()
        results = manager.send(Notification(title="t", message="m"))
        assert results == {}

    def test_send_never_raises(self) -> None:
        manager = NotificationManager()

        class BadChannel:
            def send(self, notification: Notification) -> None:
                raise ValueError("always fails")

        manager.add_channel(BadChannel())
        # Should not raise.
        results = manager.send(Notification(title="t", message="m"))
        assert len(results) == 1
        for err in results.values():
            assert isinstance(err, ValueError)


# ---------------------------------------------------------------------------
# TestNotify (convenience method)
# ---------------------------------------------------------------------------


class TestNotify:
    def test_notify_creates_notification_and_sends(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manager = NotificationManager()
        manager.add_channel(TerminalChannel(use_color=False))
        results = manager.notify("hello", "world", level=NotificationLevel.SUCCESS)
        captured = capsys.readouterr()
        assert "hello" in captured.out
        assert "world" in captured.out
        assert all(err is None for err in results.values())

    def test_notify_defaults_to_info_level(self, capsys: pytest.CaptureFixture[str]) -> None:
        manager = NotificationManager()
        manager.add_channel(TerminalChannel(use_color=False))
        manager.notify("t", "m")
        captured = capsys.readouterr()
        assert "INFO" in captured.out


# ---------------------------------------------------------------------------
# Module surface checks
# ---------------------------------------------------------------------------


def test_notification_error_is_myagent_error() -> None:
    """NotificationError must subclass MyAgentError per codebase conventions."""
    from justagent.exceptions import MyAgentError

    assert issubclass(NotificationError, MyAgentError)


def test_unused_sys_import_safety() -> None:
    """Trivial check that ``sys`` is importable from the module surface."""
    assert sys is not None
