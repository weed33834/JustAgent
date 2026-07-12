"""Tests for the SCIM module."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoship.core.scim import (
    ScimError,
    StubScimClient,
    get_client,
    sync,
)
from autoship.models.config import AppConfig, ScimConfig


def _scim_config(local_path: Path, endpoint: str | None = None) -> ScimConfig:
    return ScimConfig(enabled=True, endpoint=endpoint, local_path=local_path)


def _write_users(path: Path, users: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(users), encoding="utf-8")
    path.chmod(0o600)


def test_stub_scim_reads_local_json(tmp_path: Path) -> None:
    local = tmp_path / "scim" / "users.json"
    _write_users(
        local,
        [
            {"user": "alice", "groups": ["eng", "staff"], "active": True},
            {"user": "bob", "groups": ["eng"], "active": False},
        ],
    )
    client = StubScimClient(_scim_config(local))
    users = client.list_users()
    assert len(users) == 2
    assert users[0].user == "alice"
    assert users[0].groups == ["eng", "staff"]
    assert users[1].active is False


def test_stub_scim_missing_file_returns_empty(tmp_path: Path) -> None:
    local = tmp_path / "missing.json"
    client = StubScimClient(_scim_config(local))
    assert client.list_users() == []


def test_stub_scim_malformed_json_raises(tmp_path: Path) -> None:
    local = tmp_path / "scim" / "users.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("{not json", encoding="utf-8")
    client = StubScimClient(_scim_config(local))
    with pytest.raises(ScimError, match="Failed to read SCIM local file"):
        client.list_users()


def test_sync_writes_cache_0600(tmp_path: Path) -> None:
    local = tmp_path / "scim" / "users.json"
    _write_users(
        local,
        [{"user": "alice", "groups": ["eng"], "active": True}],
    )
    config = AppConfig(project_root=tmp_path, scim=_scim_config(local))
    sync(config)
    assert local.exists()
    mode = stat.S_IMODE(local.stat().st_mode)
    assert mode == 0o600
    # Round-trip: file should still contain valid user records.
    data = json.loads(local.read_text(encoding="utf-8"))
    assert data[0]["user"] == "alice"


def test_sync_records_audit(tmp_path: Path) -> None:
    local = tmp_path / "scim" / "users.json"
    _write_users(
        local,
        [
            {"user": "alice", "groups": ["eng"], "active": True},
            {"user": "bob", "groups": ["eng", "ops"], "active": True},
        ],
    )
    config = AppConfig(project_root=tmp_path, scim=_scim_config(local))
    audit = MagicMock()
    sync(config, audit_logger=audit)
    audit.record.assert_called_once()
    assert audit.record.call_args.args[0] == "scim.synced"
    payload = audit.record.call_args.args[1]
    assert payload["user_count"] == 2
    assert payload["group_count"] == 2  # eng, ops


def test_sync_returns_counts(tmp_path: Path) -> None:
    local = tmp_path / "scim" / "users.json"
    _write_users(
        local,
        [
            {"user": "alice", "groups": ["eng", "staff"]},
            {"user": "bob", "groups": ["eng"]},
            {"user": "carol", "groups": []},
        ],
    )
    config = AppConfig(project_root=tmp_path, scim=_scim_config(local))
    user_count, group_count = sync(config)
    assert user_count == 3
    assert group_count == 2  # eng, staff


def test_unknown_client_raises_scim_error(tmp_path: Path) -> None:
    # An endpoint is set but no http client is registered.
    config = ScimConfig(enabled=True, endpoint="https://scim.example.com")
    with pytest.raises(ScimError, match="No SCIM client registered"):
        get_client(config)


def test_sync_failure_records_audit_and_raises(tmp_path: Path) -> None:
    local = tmp_path / "scim" / "users.json"
    _write_users(local, [{"user": "alice"}])
    config = AppConfig(
        project_root=tmp_path,
        # Force failure: endpoint set with no registered HTTP client.
        scim=ScimConfig(enabled=True, endpoint="https://scim.example.com", local_path=local),
    )
    audit = MagicMock()
    with pytest.raises(ScimError, match="SCIM sync failed"):
        sync(config, audit_logger=audit)
    audit.record.assert_called_once()
    assert audit.record.call_args.args[0] == "scim.sync_failed"
    payload = audit.record.call_args.args[1]
    assert "error" in payload
