"""Tests for judicial case summary / evidence export / law export."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from justagent.models.config import AppConfig
from justagent.verticals.legal import cli as judicial


@pytest.fixture
def jctx(tmp_path: Path, monkeypatch) -> MagicMock:
    state_path = tmp_path / "judicial.json"
    monkeypatch.setenv("JUSTAGENT_JUDICIAL_STATE", str(state_path))
    ctx = MagicMock()
    ctx.obj = {
        "config": AppConfig(project_root=tmp_path),
        "dry_run": False,
        "verbose": False,
        "audit_logger": MagicMock(),
    }
    return ctx


def _create_case(ctx: MagicMock, number: str = "（2026）京01民初1号") -> None:
    judicial.case_create(
        ctx,
        case_number=number,
        cause="买卖合同纠纷",
        court="北京市第一中级人民法院",
        judge="张法官",
        description="测试案件",
        domain="civil",
    )


def _add_law(ctx: MagicMock) -> None:
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


def test_case_summary_outputs_overview(jctx, capsys) -> None:
    from justagent.verticals.legal.cli import _JudicialState, _state_path

    _create_case(jctx)
    capsys.readouterr()
    state = _JudicialState.load(_state_path(jctx))
    case_id = state.case_manager.list_cases()[0].id[:8]
    judicial.case_summary(jctx, case_id)
    out = capsys.readouterr().out
    assert "案件摘要" in out
    assert "买卖合同纠纷" in out


def test_evidence_export_empty(jctx, capsys) -> None:
    judicial.evidence_export(jctx, case_id="", output=None, json_output=False)
    out = capsys.readouterr().out
    assert "证据清单" in out
    assert "共 0 项证据" in out


def test_evidence_export_json_empty(jctx, capsys) -> None:
    judicial.evidence_export(jctx, case_id="", output=None, json_output=True)
    out = capsys.readouterr().out
    import json as _json

    payload = _json.loads(out)
    assert payload["evidence"] == []


def test_law_export_lists_articles(jctx, capsys) -> None:
    _add_law(jctx)
    capsys.readouterr()
    judicial.law_export(jctx, output=None, json_output=False)
    out = capsys.readouterr().out
    assert "第一百四十三条" in out
    assert "法律知识库" in out


def test_law_export_writes_file(jctx, tmp_path) -> None:
    _add_law(jctx)
    out_file = tmp_path / "laws.md"
    judicial.law_export(jctx, output=out_file, json_output=False)
    assert out_file.exists()
    assert "第一百四十三条" in out_file.read_text(encoding="utf-8")


def test_evidence_audit_cli_reports_issues(jctx, capsys) -> None:
    from justagent.verticals.legal.case_manager import Claim
    from justagent.verticals.legal.cli import _JudicialState, _state_path

    _create_case(jctx)
    state = _JudicialState.load(_state_path(jctx))
    case = state.case_manager.list_cases()[0]
    state.case_manager.add_claim(case.id, Claim(description="判令被告支付货款100万元"))
    # 扣押收集但没有保管链条记录 → 审计必须报保管链条问题。
    state.evidence_chain.add_evidence(
        judicial.Evidence(
            name="扣押账本",
            collection_method="扣押",
            proving_object="被告欠付货款的事实",
            case_id=case.id,
        )
    )
    state.save()
    capsys.readouterr()

    judicial.evidence_audit(jctx, case.id[:8], fmt="rich", output=None)
    out = capsys.readouterr().out
    assert "审计结论" in out
    assert "无保管链条记录" in out
    assert "有瑕疵" in out or "严重缺陷" in out


def test_evidence_audit_cli_markdown_to_file(jctx, tmp_path: Path) -> None:
    from justagent.verticals.legal.cli import _JudicialState, _state_path

    _create_case(jctx)
    state = _JudicialState.load(_state_path(jctx))
    case_id = state.case_manager.list_cases()[0].id
    state.save()

    out_file = tmp_path / "audit.md"
    judicial.evidence_audit(jctx, case_id[:8], fmt="markdown", output=out_file)
    content = out_file.read_text(encoding="utf-8")
    assert "# 证据链审计报告" in content
    assert "审计结论" in content
