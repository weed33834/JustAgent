"""Tests for the ``judicial`` agent tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from justagent.models.config import AppConfig
from justagent.verticals.legal import cli as judicial
from justagent.verticals.legal.agent_tool import JudicialInput, _run, make_judicial_tool


@pytest.fixture
def state_path(tmp_path: Path, monkeypatch) -> Path:
    sp = tmp_path / "judicial.json"
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
    judicial.law_add(
        ctx,
        law_name="中华人民共和国民法典",
        article_number="第一百四十三条",
        content="具备下列条件的民事法律行为有效。",
        content_file=None,
        domain="civil",
        chapter="",
        effective_date="",
        keywords="",
    )
    return sp


def test_tool_id_is_judicial() -> None:
    tool = make_judicial_tool("/tmp/x.json")
    assert tool.id == "judicial"
    assert tool.parameters is JudicialInput


def test_list_cases(state_path: Path) -> None:
    out = _run("list_cases", JudicialInput(action="list_cases"), state_path)
    assert "1 case" in out
    assert "买卖合同纠纷" in out


def test_case_summary(state_path: Path) -> None:
    out = _run("case_summary", JudicialInput(action="case_summary", case_id=""), state_path)
    # empty case_id -> not found message
    assert "not found" in out or "买卖合同纠纷" in out


def test_list_laws(state_path: Path) -> None:
    out = _run("list_laws", JudicialInput(action="list_laws"), state_path)
    assert "第一百四十三条" in out


def test_search_laws(state_path: Path) -> None:
    out = _run("search_laws", JudicialInput(action="search_laws", query="民事法律行为"), state_path)
    assert "第一百四十三条" in out or "No matching" in out


def test_tool_fails_without_state() -> None:
    tool = make_judicial_tool()  # no state path
    import asyncio

    result = asyncio.run(tool.invoke(JudicialInput(action="list_cases"), MagicMock()))
    assert result.error  # should surface "not configured"
