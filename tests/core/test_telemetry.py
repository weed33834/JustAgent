"""Tests for telemetry collection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from autoship.core.telemetry import TelemetryCollector, TelemetryEvent


@pytest.fixture
def telemetry_log(tmp_path: Path) -> Path:
    return tmp_path / "telemetry.logl"


def test_collector_does_nothing_when_disabled(telemetry_log: Path) -> None:
    collector = TelemetryCollector(enabled=False, log_dir=telemetry_log.parent)
    event = collector.record("clean", time.perf_counter(), 0)
    assert event is None
    assert not telemetry_log.exists()


def test_collector_writes_local_log(telemetry_log: Path) -> None:
    collector = TelemetryCollector(enabled=True, log_dir=telemetry_log.parent)
    start = time.perf_counter()
    event = collector.record("commit", start, 0)
    assert event is not None
    assert event.command == "commit"
    assert event.exit_code == 0

    lines = telemetry_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["command"] == "commit"
    assert record["exit_code"] == 0
    assert "duration_ms" in record
    assert "python_version" in record


def test_collector_records_exception_info(telemetry_log: Path) -> None:
    collector = TelemetryCollector(enabled=True, log_dir=telemetry_log.parent)
    try:
        raise ValueError("secret detail")
    except ValueError as exc:
        event = collector.record("verify", time.perf_counter(), 1, exc=exc)

    assert event is not None
    assert event.exception_type == "ValueError"
    assert event.exception_lineno is not None
    lines = telemetry_log.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["exception_type"] == "ValueError"
    assert "secret detail" not in json.dumps(record)


def test_collector_sends_to_endpoint_when_configured(telemetry_log: Path) -> None:
    collector = TelemetryCollector(
        enabled=True,
        endpoint="https://telemetry.autoship.dev/v1/events",
        log_dir=telemetry_log.parent,
    )
    with patch("autoship.core.telemetry.httpx.post") as mock_post:
        collector.record("upload", time.perf_counter(), 0)
        collector.flush()
    mock_post.assert_called_once()
    call = mock_post.call_args
    assert call.kwargs["timeout"] == collector.timeout
    payload = json.loads(call.kwargs["content"])
    assert len(payload) == 1
    assert payload[0]["command"] == "upload"


def test_collector_accepts_untrusted_endpoint_when_allowed(telemetry_log: Path) -> None:
    collector = TelemetryCollector(
        enabled=True,
        endpoint="https://example.com/telemetry",
        log_dir=telemetry_log.parent,
        allow_untrusted=True,
    )
    assert collector.endpoint == "https://example.com/telemetry"


def test_collector_rejects_http_endpoint(telemetry_log: Path) -> None:
    collector = TelemetryCollector(
        enabled=True,
        endpoint="http://telemetry.autoship.dev/v1/events",
        log_dir=telemetry_log.parent,
    )
    assert collector.endpoint is None


def test_collector_rejects_untrusted_https_endpoint(telemetry_log: Path) -> None:
    collector = TelemetryCollector(
        enabled=True,
        endpoint="https://example.com/telemetry",
        log_dir=telemetry_log.parent,
    )
    assert collector.endpoint is None


def test_collector_timeout_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AUTOSHIP_TELEMETRY_TIMEOUT", "10")
    collector = TelemetryCollector()
    assert collector.timeout == 10.0


def test_collector_timeout_bounds(monkeypatch) -> None:
    monkeypatch.setenv("AUTOSHIP_TELEMETRY_TIMEOUT", "999")
    collector = TelemetryCollector()
    assert collector.timeout == 5.0

    monkeypatch.setenv("AUTOSHIP_TELEMETRY_TIMEOUT", "not-a-number")
    collector = TelemetryCollector()
    assert collector.timeout == 5.0


def test_event_to_dict() -> None:
    event = TelemetryEvent(command="clean", duration_ms=12.3, exit_code=0)
    data = event.to_dict()
    assert data["command"] == "clean"
    assert data["duration_ms"] == 12.3
    assert data["exit_code"] == 0


def test_collector_batches_events_and_flushes(telemetry_log: Path) -> None:
    collector = TelemetryCollector(
        enabled=True,
        endpoint="https://telemetry.autoship.dev/v1/events",
        log_dir=telemetry_log.parent,
        batch_size=3,
    )
    with patch("autoship.core.telemetry.httpx.post") as mock_post:
        for _ in range(3):
            collector.record("clean", time.perf_counter(), 0)
        # batch_size reached -> auto-flush happened once
        assert mock_post.call_count == 1
        call = mock_post.call_args
        payload = json.loads(call.kwargs["content"])
        assert len(payload) == 3
        assert payload[0]["command"] == "clean"

        # leftover events flush on explicit flush
        collector.record("verify", time.perf_counter(), 0)
        collector.flush()
        assert mock_post.call_count == 2
        last_payload = json.loads(mock_post.call_args.kwargs["content"])
        assert len(last_payload) == 1
        assert last_payload[0]["command"] == "verify"


def test_collector_scrubs_pii_from_arbitrary_event(telemetry_log: Path) -> None:
    collector = TelemetryCollector(
        enabled=True,
        log_dir=telemetry_log.parent,
    )
    raw = {
        "command": "upload",
        "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
        "user_email": "alice@example.com",
        "work_dir": "/home/alice/secret-project",
        "short_ok": "hello",
    }
    collector.record_event(raw)

    lines = telemetry_log.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["command"] == "upload"
    assert record["api_key"] == "<redacted>"
    assert record["user_email"] == "<redacted>"
    assert record["work_dir"] == "<path>"
    assert record["short_ok"] == "hello"
