"""Tests for the JustAgent Web interface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from justagent.cli.commands import judicial
from justagent.models.config import AppConfig
from justagent.web.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    sp = tmp_path / ".justagent" / "judicial_state.json"
    monkeypatch.setenv("JUSTAGENT_JUDICIAL_STATE", str(sp))
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
    config = AppConfig(project_root=tmp_path)
    return TestClient(create_app(config))


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
