"""Tests for AuditLogger."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from autoship.core.audit_logger import AuditLogger
from autoship.models.config import AppConfig, AuditConfig, SinkConfig


def test_audit_logger_creates_log_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)
    audit.record("test.event", {"status": "value"})

    assert audit.log_file.exists()
    lines = audit.log_file.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "test.event"
    assert entry["trace_id"] == audit.trace_id
    assert entry["payload"]["status"] == "value"


def test_audit_logger_bind_context_returns_self(app_config: AppConfig) -> None:
    audit = AuditLogger(app_config)
    bound = audit.bind_context(extra="value")
    assert bound is audit


def test_audit_logger_bind_context_merges_into_records(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)
    audit.bind_context(user="alice", env="prod")
    audit.record("test.bound", {"action": "deploy"})
    audit.record("test.override", {"action": "verify", "user": "bob"})

    lines = audit.log_file.read_text().strip().splitlines()
    assert len(lines) == 2
    bound_entry = json.loads(lines[0])
    assert bound_entry["payload"]["user"] == "alice"
    assert bound_entry["payload"]["env"] == "prod"
    assert bound_entry["payload"]["action"] == "deploy"
    override_entry = json.loads(lines[1])
    assert override_entry["payload"]["user"] == "bob"
    assert override_entry["payload"]["env"] == "prod"


def test_audit_logger_export_filters_by_since(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)

    old_ts = "2020-01-01T00:00:00+00:00"
    new_ts = "2099-01-01T00:00:00+00:00"
    log_file = log_dir / "audit.2020-01-01.jsonl"
    log_file.write_text(json.dumps({"ts": old_ts, "event": "old"}) + "\n")
    audit.log_file.write_text(json.dumps({"ts": new_ts, "event": "new"}) + "\n")

    output = audit.export()
    lines = output.read_text().strip().splitlines()
    assert len(lines) == 2

    since = datetime(2090, 1, 1, tzinfo=timezone.utc)
    output_since = audit.export(since=since)
    lines_since = output_since.read_text().strip().splitlines()
    assert len(lines_since) == 1
    assert json.loads(lines_since[0])["event"] == "new"


def test_audit_logger_cleanup_removes_old_files(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)

    old_file = log_dir / "audit.2020-01-01.jsonl"
    old_file.write_text('{"ts": "2020-01-01T00:00:00+00:00", "event": "old"}\n')
    new_file = log_dir / "audit.2099-01-01.jsonl"
    new_file.write_text('{"ts": "2099-01-01T00:00:00+00:00", "event": "new"}\n')

    # Set mtime far in the past for the old file so cleanup removes it.
    os.utime(old_file, (1, 1))

    removed = audit.cleanup(retention_days=30)
    assert removed == 1
    assert not old_file.exists()
    assert new_file.exists()


def test_audit_logger_redacts_sensitive_fields(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)

    audit.record(
        "test.secrets",
        {
            "event_type": "command",
            "command": "deploy",
            "returncode": 0,
            "duration_ms": 42,
            "api_key": "super-secret",
            "nested": {"password": "nested-secret", "siem_token": "token123"},
            "items": [{"private": "private123"}, {"ok": "value"}],
        },
    )

    lines = audit.log_file.read_text().strip().splitlines()
    entry = json.loads(lines[0])
    payload = entry["payload"]
    assert payload["event_type"] == "command"
    assert payload["command"] == "deploy"
    assert payload["returncode"] == 0
    assert payload["duration_ms"] == 42
    assert payload["api_key"] == "***"
    assert payload["nested"]["password"] == "***"
    assert payload["nested"]["siem_token"] == "***"
    assert payload["items"][0]["private"] == "***"
    assert payload["items"][1]["ok"] == "value"


def test_audit_logger_forwards_redacted_record_to_siem(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)
    posted: list[dict[str, object]] = []

    class FakeClient:
        def post(self, _path: str, *, json: dict[str, object]) -> None:
            posted.append(json)

    audit._siem_client = FakeClient()

    audit.record(
        "test.siem",
        {"command": "deploy", "api_key": "secret", "nested": {"token": "tok"}},
    )

    assert len(posted) == 1
    forwarded = posted[0]
    assert forwarded["event"] == "test.siem"
    assert forwarded["payload"]["command"] == "deploy"
    assert forwarded["payload"]["api_key"] == "***"
    assert forwarded["payload"]["nested"]["token"] == "***"


def test_audit_logger_siem_failure_is_best_effort(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)

    class FailingClient:
        def post(self, _path: str, *, json: object) -> None:
            raise httpx.ConnectError("connection refused")

    audit._siem_client = FailingClient()

    # Should not raise despite SIEM being down.
    audit.record("test.siem_fail", {"status": "value"})

    assert audit.log_file.exists()


def test_audit_logger_redacts_secret_values_by_pattern(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)

    github_token = "ghp_" + "a" * 36
    openai_key = "sk-" + "b" * 48
    audit.record(
        "test.patterns",
        {
            "message": f"Authorization: Bearer {github_token}",
            "openai_api_key": openai_key,
            "aws_key": "AKIAIOSFODNN7EXAMPLE",
            "nested": {
                "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
            },
            "safe": "plain text without secrets",
        },
    )

    lines = audit.log_file.read_text().strip().splitlines()
    entry = json.loads(lines[0])
    payload = entry["payload"]
    assert payload["message"] == "***"
    assert payload["openai_api_key"] == "***"
    assert payload["aws_key"] == "***"
    assert payload["nested"]["jwt"] == "***"
    assert payload["safe"] == "plain text without secrets"


def test_audit_logger_exact_key_match_not_substring(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)

    audit.record(
        "test.exact",
        {
            "mytoken": "not-redacted-by-key",
            "api_key": "redacted-by-key",
            "token_value": "ghp_" + "c" * 36,
        },
    )

    lines = audit.log_file.read_text().strip().splitlines()
    payload = json.loads(lines[0])["payload"]
    assert payload["mytoken"] == "not-redacted-by-key"
    assert payload["api_key"] == "***"
    assert payload["token_value"] == "***"


def test_audit_logger_sets_restrictive_permissions(tmp_path: Path) -> None:
    """Audit log directory and file are only owner-readable/writable."""
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)

    audit.record("test.permissions", {"status": "ok"})

    assert audit.log_dir.exists()
    assert stat.S_IMODE(audit.log_dir.stat().st_mode) == 0o700
    assert audit.log_file.exists()
    assert stat.S_IMODE(audit.log_file.stat().st_mode) == 0o600


def test_audit_logger_export_sets_restrictive_permissions(tmp_path: Path, monkeypatch) -> None:
    """Exported audit files are only owner-readable/writable."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "audit.2099-01-01.jsonl"
    log_file.write_text(json.dumps({"ts": "2099-01-01T00:00:00+00:00", "event": "x"}) + "\n")

    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)
    output = audit.export()

    assert output.exists()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_audit_logger_redact_unknown_fields(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(
        audit_log_dir=log_dir,
        audit=AuditConfig(redact_unknown_fields=True),
    )
    audit = AuditLogger(config)

    audit.record(
        "test.unknown",
        {
            "command": "deploy",
            "returncode": 0,
            "unknown_field": "secret",
            "details": {"value": "kept", "another_unknown": "data"},
        },
    )

    lines = audit.log_file.read_text().strip().splitlines()
    payload = json.loads(lines[0])["payload"]
    assert payload["command"] == "deploy"
    assert payload["returncode"] == 0
    assert payload["unknown_field"] == "***"
    assert payload["details"]["value"] == "kept"
    assert payload["details"]["another_unknown"] == "***"


def test_audit_logger_disables_siem_after_consecutive_failures(tmp_path: Path, caplog) -> None:
    log_dir = tmp_path / "logs"
    config = AppConfig(
        audit_log_dir=log_dir,
        audit=AuditConfig(siem_max_failures=2),
    )
    audit = AuditLogger(config)

    class FailingClient:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, _path: str, *, json: object) -> None:
            self.calls += 1
            raise httpx.ConnectError("connection refused")

    audit._siem_client = FailingClient()

    with caplog.at_level("WARNING", logger="autoship"):
        audit.record("test.siem_1", {"status": "value"})
        audit.record("test.siem_2", {"status": "value"})
        # After 2 failures, forwarding should be disabled.
        audit.record("test.siem_3", {"status": "value"})

    assert audit._siem_disabled is True
    assert audit._siem_failures == 2
    assert "SIEM forwarding has failed 2 consecutive times" in caplog.text
    # The failing client should have been called exactly twice, not three times.
    assert audit._siem_client is not None
    assert audit._siem_client.calls == 2


# ---------------------------------------------------------------------------
# close() and context-manager lifecycle (H1)
# ---------------------------------------------------------------------------


def test_close_releases_siem_and_sink_clients(tmp_path: Path) -> None:
    """close() closes both the SIEM and sink HTTP clients and nulls them out."""
    log_dir = tmp_path / "logs"
    config = AppConfig(
        audit_log_dir=log_dir,
        audit=AuditConfig(siem_enabled=True, siem_url="https://siem.example.com"),
        sink=SinkConfig(enabled=True, url="http://127.0.0.1:8787", forward_audit=True),
    )
    audit = AuditLogger(config)
    # Replace the real httpx clients with mocks so we can assert close() was
    # called without depending on network state.
    siem_mock = MagicMock()
    sink_mock = MagicMock()
    audit._siem_client = siem_mock
    audit._sink_client = sink_mock

    audit.close()

    siem_mock.close.assert_called_once()
    sink_mock.close.assert_called_once()
    assert audit._siem_client is None
    assert audit._sink_client is None


def test_close_idempotent(tmp_path: Path) -> None:
    """Calling close() twice does not raise."""
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)

    audit.close()
    audit.close()  # should not raise

    assert audit._siem_client is None
    assert audit._sink_client is None


def test_close_tolerates_missing_attributes(monkeypatch) -> None:
    """close() does not raise even if __init__ did not set the client attributes.

    The ``getattr``-tolerant access in :meth:`AuditLogger.close` exists so that
    test doubles (or subclasses) that bypass ``__init__`` do not crash on
    cleanup. This is the one legitimate use of a stub ``__init__``.
    """

    def _stub_init(self: AuditLogger) -> None:
        pass  # Deliberately do not set _siem_client / _sink_client.

    monkeypatch.setattr(AuditLogger, "__init__", _stub_init)
    audit = AuditLogger()

    audit.close()  # should not raise


def test_context_manager_closes_on_exit(tmp_path: Path) -> None:
    """Using AuditLogger as a context manager closes clients on exit."""
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)
    audit._siem_client = MagicMock()
    audit._sink_client = MagicMock()

    with audit as ctx:
        assert ctx is audit

    assert audit._siem_client is None
    assert audit._sink_client is None


def test_context_manager_closes_on_exception(tmp_path: Path) -> None:
    """The context manager closes clients even when an exception propagates."""
    log_dir = tmp_path / "logs"
    config = AppConfig(audit_log_dir=log_dir)
    audit = AuditLogger(config)
    audit._siem_client = MagicMock()
    audit._sink_client = MagicMock()

    with pytest.raises(RuntimeError, match="boom"), audit:
        raise RuntimeError("boom")

    assert audit._siem_client is None
    assert audit._sink_client is None
