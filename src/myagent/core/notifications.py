"""Notifications — pluggable notification channels.

Route notifications (task completion, errors, info) to one or more
channels: terminal (stdout/stderr), desktop (notify-send on Linux,
osascript on macOS, log on other platforms), or webhook (HTTP POST).

Used by the scheduler to notify on task completion, and by CLI commands
for user feedback.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

import httpx

from myagent.exceptions import MyAgentError


class NotificationError(MyAgentError):
    """Raised when a notification payload is invalid or a channel is misconfigured."""


class NotificationLevel(str, Enum):  # noqa: UP042 - match existing codebase style
    """Severity level for a :class:`Notification`."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Notification:
    """An addressable notification message routed through one or more channels."""

    title: str
    message: str
    level: NotificationLevel = NotificationLevel.INFO
    timestamp: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


class NotificationChannel(Protocol):
    """A sink for :class:`Notification` objects."""

    def send(self, notification: Notification) -> None:
        """Deliver ``notification`` to this channel. May raise on failure."""
        ...


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_RESET = "\x1b[0m"
_LEVEL_COLORS: dict[NotificationLevel, str] = {
    NotificationLevel.INFO: "\x1b[36m",  # cyan
    NotificationLevel.SUCCESS: "\x1b[32m",  # green
    NotificationLevel.WARNING: "\x1b[33m",  # yellow
    NotificationLevel.ERROR: "\x1b[31m",  # red
}

_STERR_LEVELS = frozenset({NotificationLevel.WARNING, NotificationLevel.ERROR})


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


class TerminalChannel:
    """Print notifications to stdout/stderr with optional ANSI colour."""

    def __init__(self, use_color: bool = True) -> None:
        self.use_color = use_color

    def send(self, notification: Notification) -> None:
        color = _LEVEL_COLORS.get(notification.level, "")
        prefix = notification.level.value.upper()
        body = f"[{prefix}] {notification.title}: {notification.message}"
        line = f"{color}{body}{_RESET}" if self.use_color and color else body
        if notification.level in _STERR_LEVELS:
            print(line, file=sys.stderr)
        else:
            print(line)


class DesktopChannel:
    """Best-effort desktop notification via ``notify-send`` or ``osascript``.

    Failures (missing binary, non-zero exit) are swallowed — the channel
    logs a warning to stderr and returns. Never raises.
    """

    def __init__(self) -> None:
        self.platform = sys.platform

    def send(self, notification: Notification) -> None:
        try:
            if self.platform.startswith("linux"):
                self._send_linux(notification)
            elif self.platform == "darwin":
                self._send_macos(notification)
            # Other platforms: silently no-op.
        except Exception as exc:  # noqa: BLE001 - never raise from a notify channel
            print(
                f"[myagent] desktop notification failed: {exc}",
                file=sys.stderr,
            )

    def _send_linux(self, notification: Notification) -> None:
        urgency = {
            NotificationLevel.ERROR: "critical",
            NotificationLevel.WARNING: "normal",
            NotificationLevel.INFO: "low",
            NotificationLevel.SUCCESS: "normal",
        }.get(notification.level, "normal")
        subprocess.run(
            [
                "notify-send",
                "--urgency",
                urgency,
                notification.title,
                notification.message,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )

    def _send_macos(self, notification: Notification) -> None:
        # Escape double quotes in arguments before interpolating into AppleScript.
        title = notification.title.replace('"', '\\"')
        message = notification.message.replace('"', '\\"')
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )


class WebhookChannel:
    """POST notifications as JSON to an HTTP endpoint.

    Failures (network errors, non-2xx status) are logged to stderr and
    swallowed. Never raises.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout

    def send(self, notification: Notification) -> None:
        payload = {
            "title": notification.title,
            "message": notification.message,
            "level": notification.level.value,
            "timestamp": notification.timestamp,
            "metadata": dict(notification.metadata),
        }
        try:
            response = httpx.post(
                self.url,
                json=payload,
                headers=dict(self.headers),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - never raise from a notify channel
            print(
                f"[myagent] webhook notification failed: {exc}",
                file=sys.stderr,
            )


class LogChannel:
    """Append notifications to a log file as ``[ISO8601] [LEVEL] title: message``."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def send(self, notification: Notification) -> None:
        ts = notification.timestamp or time.time()
        iso = datetime.fromtimestamp(ts, tz=UTC).isoformat()
        line = f"[{iso}] [{notification.level.value.upper()}] {notification.title}: {notification.message}\n"
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as exc:
            print(
                f"[myagent] log notification failed: {exc}",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# NotificationManager
# ---------------------------------------------------------------------------


class NotificationManager:
    """Fan-out notifications to a set of registered channels.

    ``send`` collects per-channel errors and never raises; the returned
    dict maps each channel to its exception (or None on success).
    """

    def __init__(self) -> None:
        self._channels: list[NotificationChannel] = []

    def add_channel(self, channel: NotificationChannel) -> None:
        """Register ``channel`` for future notifications."""
        self._channels.append(channel)

    def remove_channel(self, channel: NotificationChannel) -> bool:
        """Remove ``channel``; return True if it was registered."""
        try:
            self._channels.remove(channel)
            return True
        except ValueError:
            return False

    @property
    def channels(self) -> list[NotificationChannel]:
        """Return a defensive copy of the registered channels."""
        return list(self._channels)

    def send(
        self, notification: Notification
    ) -> dict[NotificationChannel, Exception | None]:
        """Send ``notification`` to every registered channel.

        Returns a map of ``channel -> exception`` (None on success).
        """
        results: dict[NotificationChannel, Exception | None] = {}
        for channel in self._channels:
            try:
                channel.send(notification)
                results[channel] = None
            except Exception as exc:  # noqa: BLE001 - collect, do not raise
                results[channel] = exc
        return results

    def notify(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
    ) -> dict[NotificationChannel, Exception | None]:
        """Convenience wrapper: build a :class:`Notification` and send it."""
        notification = Notification(
            title=title,
            message=message,
            level=level,
            timestamp=time.time(),
        )
        return self.send(notification)


__all__ = [
    "DesktopChannel",
    "LogChannel",
    "Notification",
    "NotificationChannel",
    "NotificationError",
    "NotificationLevel",
    "NotificationManager",
    "TerminalChannel",
    "WebhookChannel",
]
