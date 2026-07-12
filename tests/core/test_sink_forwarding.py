"""Tests for sink forwarding in AuditLogger and TelemetryCollector."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import httpx

from autoship.core.audit_logger import AuditLogger
from autoship.core.config_center import _deep_merge
from autoship.core.sink import SinkServer
from autoship.core.telemetry import TelemetryCollector
from autoship.models.config import AppConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_sink(
    tmp_path: Path, *, token: str | None = None
) -> tuple[SinkServer, int, threading.Thread]:
    server = SinkServer(tmp_path / "sink-store", port=0, token=token, retention_days=7)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, thread


def _stop(server: SinkServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _config_with_sink(tmp_path: Path, sink_url: str, *, token: str | None = None) -> AppConfig:
    """Build an AppConfig whose [sink] section points at the given URL."""
    from autoship.core.config_center import _default_config

    base = _default_config()
    override: dict[str, Any] = {
        "project_root": str(tmp_path),
        "audit_log_dir": str(tmp_path / "logs"),
        "sink": {
            "enabled": True,
            "url": sink_url,
            "max_failures": 2,
            "forward_audit": True,
            "forward_telemetry": True,
        },
    }
    if token is not None:
        override["sink"]["token"] = token
    merged = _deep_merge(base, override)
    return AppConfig.model_validate(merged)


# ---------------------------------------------------------------------------
# AuditLogger -> sink
# ---------------------------------------------------------------------------


def test_audit_logger_forwards_to_sink(tmp_path: Path) -> None:
    server, port, thread = _start_sink(tmp_path)
    try:
        config = _config_with_sink(tmp_path, f"http://127.0.0.1:{port}")
        logger = AuditLogger(config)
        logger.record("test.event", {"command": "clean"})
        # record() forwards synchronously, so the record is already on the
        # server by the time record() returns.
        status = httpx.get(f"http://127.0.0.1:{port}/status", timeout=3.0).json()
        assert status["audit_records"] == 1
    finally:
        _stop(server, thread)


def test_audit_logger_no_forward_when_disabled(tmp_path: Path) -> None:
    """When sink.enabled is False, no sink client is created."""
    config = AppConfig(project_root=tmp_path, audit_log_dir=tmp_path / "logs")
    logger = AuditLogger(config)
    assert logger._sink_client is None


def test_audit_logger_no_forward_when_forward_audit_false(tmp_path: Path) -> None:
    server, port, thread = _start_sink(tmp_path)
    try:
        from autoship.core.config_center import _default_config

        base = _default_config()
        merged = _deep_merge(
            base,
            {
                "project_root": str(tmp_path),
                "audit_log_dir": str(tmp_path / "logs"),
                "sink": {
                    "enabled": True,
                    "url": f"http://127.0.0.1:{port}",
                    "forward_audit": False,  # opt out of audit forwarding
                },
            },
        )
        config = AppConfig.model_validate(merged)
        logger = AuditLogger(config)
        assert logger._sink_client is None
    finally:
        _stop(server, thread)


def test_audit_logger_sink_auth_with_token(tmp_path: Path) -> None:
    server, port, thread = _start_sink(tmp_path, token="team-secret")
    try:
        config = _config_with_sink(tmp_path, f"http://127.0.0.1:{port}", token="team-secret")
        logger = AuditLogger(config)
        logger.record("test.event", {"command": "verify"})
        status = httpx.get(f"http://127.0.0.1:{port}/status", timeout=3.0).json()
        assert status["audit_records"] == 1
    finally:
        _stop(server, thread)


def test_audit_logger_sink_circuit_breaker(tmp_path: Path) -> None:
    """After max_failures consecutive failures, forwarding is disabled."""
    # Point at a port nobody listens on to force connection failures.
    config = _config_with_sink(tmp_path, "http://127.0.0.1:1")  # port 1: reserved, will fail
    logger = AuditLogger(config)
    assert logger._sink_client is not None
    # max_failures=2 (set in _config_with_sink)
    logger.record("a", {})
    logger.record("b", {})
    # After 2 failures, the circuit breaker should be tripped.
    assert logger._sink_disabled is True
    assert logger._sink_failures >= 2


# ---------------------------------------------------------------------------
# TelemetryCollector -> sink
# ---------------------------------------------------------------------------


def test_telemetry_forwards_to_sink(tmp_path: Path) -> None:
    server, port, thread = _start_sink(tmp_path)
    try:
        collector = TelemetryCollector(
            enabled=True,
            sink_endpoint=f"http://127.0.0.1:{port}",
            batch_size=1,
        )
        collector.record_event({"command": "clean", "exit_code": 0})
        # batch_size=1 triggers flush immediately.
        status = httpx.get(f"http://127.0.0.1:{port}/status", timeout=3.0).json()
        assert status["telemetry_records"] == 1
    finally:
        _stop(server, thread)


def test_telemetry_no_sink_when_not_set(tmp_path: Path) -> None:
    collector = TelemetryCollector(enabled=True)
    assert collector.sink_endpoint is None
    # flush should be a no-op, not raise.
    collector.flush()


def test_telemetry_sink_with_token(tmp_path: Path) -> None:
    server, port, thread = _start_sink(tmp_path, token="t-team")
    try:
        collector = TelemetryCollector(
            enabled=True,
            sink_endpoint=f"http://127.0.0.1:{port}",
            sink_token="t-team",
            batch_size=1,
        )
        collector.record_event({"command": "verify", "exit_code": 0})
        status = httpx.get(f"http://127.0.0.1:{port}/status", timeout=3.0).json()
        assert status["telemetry_records"] == 1
    finally:
        _stop(server, thread)


def test_telemetry_sink_failure_is_swallowed(tmp_path: Path) -> None:
    """A down sink must not raise — telemetry never breaks the CLI."""
    collector = TelemetryCollector(
        enabled=True,
        sink_endpoint="http://127.0.0.1:1",  # nothing listening
        batch_size=1,
    )
    collector.record_event({"command": "clean", "exit_code": 0})
    # flush should not raise despite the unreachable sink.
    collector.flush()


def test_telemetry_sink_appends_telemetry_path(tmp_path: Path) -> None:
    """The constructor appends /telemetry to the user-supplied sink URL."""
    collector = TelemetryCollector(enabled=True, sink_endpoint="http://127.0.0.1:9999/sink")
    assert collector.sink_endpoint == "http://127.0.0.1:9999/sink/telemetry"
