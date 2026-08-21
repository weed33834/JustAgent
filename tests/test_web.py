"""Tests for the JustAgent Web interface."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from justagent.cli.commands import judicial
from justagent.models.config import AppConfig
from justagent.web.app import create_app
from justagent.web.users import PBKDF2_ITERATIONS, UserStore


@pytest.fixture
def isolated_stores(tmp_path: Path, monkeypatch) -> Path:
    """Point user/session stores at a temp dir so tests never touch $HOME."""
    monkeypatch.setenv("JUSTAGENT_WEB_USERS_FILE", str(tmp_path / "users.json"))
    monkeypatch.setenv("JUSTAGENT_WEB_SESSIONS_FILE", str(tmp_path / "sessions.json"))
    return tmp_path


def _make_state(tmp_path: Path) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": AppConfig(project_root=tmp_path),
        "dry_run": False,
        "verbose": False,
        "audit_logger": MagicMock(),
    }
    judicial.case_create(
        ctx,
        case_number="（2026）京01民初1号",
        cause="买卖合同纠纷",
        court="北京市第一中级人民法院",
        judge="张法官",
        description="测试",
        domain="civil",
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch, isolated_stores):
    monkeypatch.setenv("JUSTAGENT_JUDICIAL_STATE", str(tmp_path / ".justagent" / "judicial_state.json"))
    _make_state(tmp_path)
    config = AppConfig(project_root=tmp_path)
    return TestClient(create_app(config, no_auth=True))


@pytest.fixture
def secure_client(tmp_path: Path, monkeypatch, isolated_stores):
    """Auth-enforced app with a known admin password."""
    monkeypatch.setenv("JUSTAGENT_JUDICIAL_STATE", str(tmp_path / ".justagent" / "judicial_state.json"))
    _make_state(tmp_path)
    monkeypatch.setenv("JUSTAGENT_WEB_ADMIN_PASSWORD", "test-admin-pass")
    config = AppConfig(project_root=tmp_path)
    return TestClient(create_app(config))


def _login(client: TestClient, password: str = "test-admin-pass") -> dict:
    r = client.post("/api/auth/login", json={"username": "admin", "password": password})
    assert r.status_code == 200
    return r.json()


def test_index_served(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "JustAgent" in r.text


def test_health(client) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_state_returns_created_case(client) -> None:
    data = client.get("/api/state").json()
    assert len(data["cases"]) == 1
    assert data["cases"][0]["cause"] == "买卖合同纠纷"


def test_chat_without_llm_is_graceful(client) -> None:
    r = client.post("/api/chat", json={"message": "列出案件"})
    body = r.json()
    assert body["error"] == "no_llm"
    assert "No LLM backend" in body["reply"]


def test_create_case_via_web(client) -> None:
    r = client.post(
        "/api/judicial/case",
        json={"case_number": "（2026）沪01民初1号", "cause": "借款合同纠纷", "court": "上海法院"},
    )
    assert r.json()["ok"] is True
    assert len(client.get("/api/state").json()["cases"]) == 2


def test_login_invalid_returns_401(client) -> None:
    r = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_unknown_route_404(client) -> None:
    assert client.get("/api/nope").status_code == 404


# --- security: default-deny auth -------------------------------------------


def test_anonymous_requests_are_401(secure_client) -> None:
    assert secure_client.get("/api/config").status_code == 401
    assert secure_client.get("/api/state").status_code == 401
    assert secure_client.post("/api/chat", json={"message": "hi"}).status_code == 401


def test_health_and_login_stay_open(secure_client) -> None:
    assert secure_client.get("/api/health").json() == {"status": "ok"}
    assert secure_client.get("/api/auth/login").status_code in (405, 422)


def test_session_login_grants_access(secure_client) -> None:
    body = _login(secure_client)
    headers = {"Authorization": f"Bearer {body['token']}"}
    assert secure_client.get("/api/state", headers=headers).status_code == 200


def test_viewer_role_is_read_only(secure_client, isolated_stores, monkeypatch) -> None:
    users_file = isolated_stores / "users.json"
    data = json.loads(users_file.read_text(encoding="utf-8"))
    store = UserStore(path=users_file)
    data["viewer1"] = {
        "username": "viewer1",
        "role": "viewer",
        "password_hash": store._hash("viewer-pass"),
        "created_at": 0.0,
    }
    users_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    r = secure_client.post("/api/auth/login", json={"username": "viewer1", "password": "viewer-pass"})
    token = r.json()["token"]
    read = secure_client.get("/api/state", headers={"Authorization": f"Bearer {token}"})
    assert read.status_code == 200
    write = secure_client.post(
        "/api/judicial/case",
        headers={"Authorization": f"Bearer {token}"},
        json={"case_number": "X-1", "cause": "测试"},
    )
    assert write.status_code == 403


def test_shared_token_grants_full_access(tmp_path: Path, monkeypatch, isolated_stores) -> None:
    monkeypatch.setenv("JUSTAGENT_JUDICIAL_STATE", str(tmp_path / ".justagent" / "judicial_state.json"))
    _make_state(tmp_path)
    monkeypatch.setenv("JUSTAGENT_WEB_TOKEN", "shared-secret")
    config = AppConfig(project_root=tmp_path)
    c = TestClient(create_app(config))
    assert c.get("/api/state").status_code == 401
    assert c.get("/api/state", headers={"Authorization": "Bearer shared-secret"}).status_code == 200


# --- security: password hashing (PBKDF2 + legacy migration) ----------------


def test_new_hashes_use_pbkdf2(isolated_stores) -> None:
    monkey_pw = secrets.token_urlsafe(8)
    store = UserStore()
    stored = store._hash(monkey_pw)
    prefix, iterations, _, _ = stored.split("$", 3)
    assert prefix == "pbkdf2_sha256"
    assert int(iterations) >= PBKDF2_ITERATIONS


def test_legacy_hash_verifies_and_migrates(isolated_stores) -> None:
    users_file = isolated_stores / "users.json"
    salt = secrets.token_hex(8)
    digest = hmac.new(salt.encode(), b"old-pw", "sha256").hexdigest()
    users_file.write_text(
        json.dumps({"admin": {"username": "admin", "role": "admin",
                              "password_hash": f"{salt}${digest}", "created_at": 0.0}}),
        encoding="utf-8",
    )
    store = UserStore(path=users_file)
    user = store.authenticate("admin", "old-pw")
    assert user is not None
    # Transparent migration: the row is now PBKDF2 and still authenticates.
    migrated = json.loads(users_file.read_text(encoding="utf-8"))["admin"]["password_hash"]
    assert migrated.startswith("pbkdf2_sha256$")
    assert store.authenticate("admin", "old-pw") is not None
    assert store.authenticate("admin", "wrong") is None


def test_pbkdf2_known_vector() -> None:
    """PBKDF2 path matches hashlib directly (guards against format regressions)."""
    salt_b64 = base64.b64encode(b"0123456789abcdef").decode("ascii")
    expected = base64.b64encode(
        hashlib.pbkdf2_hmac("sha256", b"pw", b"0123456789abcdef", PBKDF2_ITERATIONS)
    ).decode("ascii")
    crafted = f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${expected}"
    assert UserStore()._check("pw", crafted) is True
    assert UserStore()._check("bad", crafted) is False


# --- security: session persistence ------------------------------------------


def test_tokens_survive_restart(isolated_stores) -> None:
    from justagent.web.users import TokenManager

    UserStore().ensure_admin()
    tm1 = TokenManager(ttl=3600)
    token = tm1.issue(type("U", (), {"username": "admin", "role": "admin"})())
    tm2 = TokenManager(ttl=3600)  # simulates a fresh process
    info = tm2.resolve(token)
    assert info is not None and info["username"] == "admin"
    tm2.revoke(token)
    assert TokenManager(ttl=3600).resolve(token) is None
