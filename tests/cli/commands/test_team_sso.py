"""Tests for the ``team sso`` / ``team scim`` / ``team role`` subcommands.

These tests exercise the CLI surface end-to-end through the Typer ``CliRunner``,
so they cover the integration between ``main_callback`` (SSO identity binding),
the RBAC gate helpers, and the new team subcommands.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from autoship.cli.main import app
from autoship.core.sso import Identity
from autoship.exceptions import ExitCode, PermissionDeniedError
from autoship.utils.permissions import ensure_dir_permissions

runner = CliRunner()


def _clear_sso_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove AUTOSHIP_SSO_* env vars so they don't leak between tests."""
    for key in list(os.environ):
        if key.startswith("AUTOSHIP_SSO_"):
            monkeypatch.delenv(key, raising=False)


def _write_config(
    tmp_path: Path,
    *,
    sso_enabled: bool = False,
    rbac_enabled: bool = False,
    token_cache: Path | None = None,
    bindings: list[tuple[str, list[str], list[str]]] | None = None,
    roles: dict[str, list[str]] | None = None,
    scim_local_path: Path | None = None,
    audit_log_dir: Path | None = None,
) -> Path:
    """Write a .autoship.toml configured for SSO/RBAC/SCIM testing."""
    config_path = tmp_path / ".autoship.toml"
    lines: list[str] = [f'schema_version = 1\nproject_root = "{tmp_path}"\n']
    if audit_log_dir is not None:
        lines.append(f'\n[audit]\nlog_dir = "{audit_log_dir}"\n')
    if sso_enabled:
        tc = str(token_cache or tmp_path / "sso" / "token.json")
        lines.append(f'\n[sso]\nenabled = true\nprovider = "stub"\ntoken_cache = "{tc}"\n')
    if rbac_enabled:
        lines.append("\n[rbac]\nenabled = true\n")
        if roles:
            for role_name, perms in roles.items():
                lines.append(f"\n[rbac.roles.{role_name}]\npermissions = {json.dumps(perms)}\n")
        for role, users, groups in bindings or []:
            lines.append("\n[[rbac.bindings]]\n")
            lines.append(f'role = "{role}"\n')
            if users:
                lines.append(f"users = {json.dumps(users)}\n")
            if groups:
                lines.append(f"groups = {json.dumps(groups)}\n")
    if scim_local_path is not None:
        lines.append(f'\n[scim]\nenabled = true\nlocal_path = "{scim_local_path}"\n')
    config_path.write_text("".join(lines), encoding="utf-8")
    return config_path


def _write_identity_cache(cache_path: Path, identity: Identity) -> None:
    ensure_dir_permissions(cache_path.parent, 0o700)
    cache_path.write_text(identity.model_dump_json(), encoding="utf-8")
    cache_path.chmod(0o600)


def test_team_sso_status_sso_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    config_path = _write_config(tmp_path, sso_enabled=False, audit_log_dir=tmp_path / "logs")
    result = runner.invoke(app, ["--config", str(config_path), "team", "sso", "status"])
    assert result.exit_code == 0
    assert "SSO is not enabled" in result.output


def test_team_sso_status_no_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    config_path = _write_config(
        tmp_path,
        sso_enabled=True,
        token_cache=tmp_path / "sso" / "token.json",
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(app, ["--config", str(config_path), "team", "sso", "status"])
    assert result.exit_code == 0
    assert "Not logged in" in result.output


def test_team_sso_status_with_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    cache = tmp_path / "sso" / "token.json"
    _write_identity_cache(
        cache,
        Identity(
            user="alice@example.com",
            subject="alice-sub",
            groups=["eng"],
            provider="stub",
        ),
    )
    config_path = _write_config(
        tmp_path,
        sso_enabled=True,
        token_cache=cache,
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(app, ["--config", str(config_path), "team", "sso", "status"])
    assert result.exit_code == 0
    assert "alice@example.com" in result.output
    assert "alice-sub" in result.output
    assert "eng" in result.output


def test_team_sso_login_stub_writes_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    monkeypatch.setenv("AUTOSHIP_SSO_USER", "bob@example.com")
    monkeypatch.setenv("AUTOSHIP_SSO_GROUPS", "eng")
    cache = tmp_path / "sso" / "token.json"
    config_path = _write_config(
        tmp_path,
        sso_enabled=True,
        token_cache=cache,
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(app, ["--config", str(config_path), "team", "sso", "login"])
    assert result.exit_code == 0
    assert "Logged in" in result.output
    assert cache.exists()
    # Cache should be readable back as the same identity.
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["user"] == "bob@example.com"


def test_team_sso_logout_removes_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    cache = tmp_path / "sso" / "token.json"
    _write_identity_cache(cache, Identity(user="alice", subject="alice"))
    config_path = _write_config(
        tmp_path,
        sso_enabled=True,
        token_cache=cache,
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(app, ["--config", str(config_path), "team", "sso", "logout"])
    assert result.exit_code == 0
    assert "Logged out" in result.output
    assert not cache.exists()


def test_team_scim_sync_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    local = tmp_path / "scim" / "users.json"
    ensure_dir_permissions(local.parent, 0o700)
    local.write_text(
        json.dumps(
            [
                {"user": "alice", "groups": ["eng"], "active": True},
                {"user": "bob", "groups": ["ops"], "active": True},
            ]
        ),
        encoding="utf-8",
    )
    local.chmod(0o600)
    config_path = _write_config(
        tmp_path,
        scim_local_path=local,
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(app, ["--config", str(config_path), "team", "scim", "sync"])
    assert result.exit_code == 0
    assert "Synced 2 user" in result.output
    assert "2 group" in result.output


def test_team_role_list_shows_builtin_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_sso_env(monkeypatch)
    config_path = _write_config(tmp_path, audit_log_dir=tmp_path / "logs")
    result = runner.invoke(app, ["--config", str(config_path), "team", "role", "list"])
    assert result.exit_code == 0
    assert "viewer" in result.output
    assert "developer" in result.output
    assert "maintainer" in result.output
    assert "admin" in result.output
    assert "commit:run" in result.output


def test_team_role_list_includes_custom_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_sso_env(monkeypatch)
    config_path = _write_config(
        tmp_path,
        rbac_enabled=True,
        roles={"intern": ["clean:run"]},
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(app, ["--config", str(config_path), "team", "role", "list"])
    assert result.exit_code == 0
    assert "intern" in result.output
    assert "custom" in result.output


def test_team_role_check_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    monkeypatch.setenv("AUTOSHIP_SSO_USER", "alice@example.com")
    config_path = _write_config(
        tmp_path,
        sso_enabled=True,
        rbac_enabled=True,
        bindings=[("developer", ["alice@example.com"], [])],
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(
        app, ["--config", str(config_path), "team", "role", "check", "commit:run"]
    )
    assert result.exit_code == 0
    assert "ok" in result.output.lower()


def test_team_role_check_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    monkeypatch.setenv("AUTOSHIP_SSO_USER", "alice@example.com")
    config_path = _write_config(
        tmp_path,
        sso_enabled=True,
        rbac_enabled=True,
        bindings=[("viewer", ["alice@example.com"], [])],
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(
        app, ["--config", str(config_path), "team", "role", "check", "commit:run"]
    )
    # viewer cannot commit:run → PermissionDeniedError (code 3).
    assert isinstance(result.exception, PermissionDeniedError)
    assert int(result.exception.code) == int(ExitCode.PERMISSION_DENIED)


def test_team_role_check_unknown_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_sso_env(monkeypatch)
    config_path = _write_config(tmp_path, audit_log_dir=tmp_path / "logs")
    result = runner.invoke(
        app, ["--config", str(config_path), "team", "role", "check", "bogus:perm"]
    )
    assert result.exit_code != 0
    assert "Unknown permission" in result.output


def test_rbac_gated_upload_denied_without_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_sso_env(monkeypatch)
    monkeypatch.setenv("AUTOSHIP_SSO_USER", "alice@example.com")
    config_path = _write_config(
        tmp_path,
        sso_enabled=True,
        rbac_enabled=True,
        bindings=[("viewer", ["alice@example.com"], [])],
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(
        app,
        ["--config", str(config_path), "--yes", "upload", "--target", "pypi", "--dry-run"],
    )
    # viewer cannot upload → PermissionDeniedError (code 3).
    assert isinstance(result.exception, PermissionDeniedError)
    assert int(result.exception.code) == int(ExitCode.PERMISSION_DENIED)


def test_rbac_disabled_upload_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    config_path = _write_config(
        tmp_path,
        sso_enabled=False,
        rbac_enabled=False,
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(
        app,
        ["--config", str(config_path), "--yes", "upload", "--target", "pypi", "--dry-run"],
    )
    # RBAC disabled → no permission denial. Upload may succeed or fail for
    # other reasons (missing config, etc.), but must not be PERMISSION_DENIED.
    assert not isinstance(result.exception, PermissionDeniedError)


def test_rbac_gated_audit_cleanup_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    monkeypatch.setenv("AUTOSHIP_SSO_USER", "alice@example.com")
    config_path = _write_config(
        tmp_path,
        sso_enabled=True,
        rbac_enabled=True,
        bindings=[("viewer", ["alice@example.com"], [])],
        audit_log_dir=tmp_path / "logs",
    )
    result = runner.invoke(
        app, ["--config", str(config_path), "audit", "cleanup", "--retention-days", "30"]
    )
    # viewer cannot audit:cleanup → PermissionDeniedError (code 3).
    assert isinstance(result.exception, PermissionDeniedError)
    assert int(result.exception.code) == int(ExitCode.PERMISSION_DENIED)


def test_audit_record_includes_identity_after_sso_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_sso_env(monkeypatch)
    # Use a plain username (not an email) so the audit logger's email
    # redaction pattern does not mask the value we want to assert on.
    monkeypatch.setenv("AUTOSHIP_SSO_USER", "alice")
    monkeypatch.setenv("AUTOSHIP_SSO_GROUPS", "eng")
    log_dir = tmp_path / "logs"
    config_path = _write_config(
        tmp_path,
        sso_enabled=True,
        token_cache=tmp_path / "sso" / "token.json",
        audit_log_dir=log_dir,
    )
    result = runner.invoke(app, ["--config", str(config_path), "team", "sso", "login"])
    assert result.exit_code == 0

    # The audit log should contain a record carrying the bound SSO user.
    audit_files = list(log_dir.glob("audit.*.jsonl"))
    assert audit_files, "expected at least one audit log file"
    records: list[dict[str, Any]] = []
    for audit_file in audit_files:
        for line in audit_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # The sso.login record (and any record after context binding) should
    # carry the user field.
    sso_login_records = [r for r in records if r.get("event") == "sso.login"]
    assert sso_login_records, "expected an sso.login audit record"
    payload = sso_login_records[0].get("payload", {})
    assert payload.get("user") == "alice"
