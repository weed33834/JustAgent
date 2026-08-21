"""Tests for the judicial ``law list`` / ``law show`` commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from justagent.models.config import AppConfig
from justagent.verticals.legal import cli as judicial


@pytest.fixture
def law_ctx(tmp_path: Path, monkeypatch) -> tuple[MagicMock, Path]:
    """A CLI ctx with a temp judicial state file; returns (ctx, state_path)."""
    state_path = tmp_path / "judicial.json"
    monkeypatch.setenv("JUSTAGENT_JUDICIAL_STATE", str(state_path))
    ctx = MagicMock()
    ctx.obj = {
        "config": AppConfig(project_root=tmp_path),
        "dry_run": False,
        "verbose": False,
        "audit_logger": MagicMock(),
    }
    return ctx, state_path


def _add_law(ctx: MagicMock, name: str, article: str, content: str) -> None:
    judicial.law_add(
        ctx,
        law_name=name,
        article_number=article,
        content=content,
        content_file=None,
        domain="civil",
        chapter="",
        effective_date="",
        keywords="",
    )


def test_law_list_empty_shows_hint(law_ctx, capsys) -> None:
    ctx, _ = law_ctx
    judicial.law_list(ctx, domain=None, law_name=None, status=None, json_output=False)
    assert "法律库为空" in capsys.readouterr().out


def test_law_list_lists_added_articles(law_ctx, capsys) -> None:
    ctx, _ = law_ctx
    _add_law(ctx, "中华人民共和国民法典", "第一百四十三条", "行为有效")
    _add_law(ctx, "中华人民共和国民法典", "第一百五十三条", "行为无效")
    judicial.law_list(ctx, domain=None, law_name=None, status=None, json_output=False)
    out = capsys.readouterr().out
    assert "第一百四十三条" in out
    assert "第一百五十三条" in out
    assert "2 条" in out


def test_law_list_json_output(law_ctx, capsys) -> None:
    ctx, _ = law_ctx
    _add_law(ctx, "中华人民共和国民法典", "第一百四十三条", "行为有效")
    capsys.readouterr()  # discard the "added" message
    judicial.law_list(ctx, domain=None, law_name=None, status=None, json_output=True)
    out = capsys.readouterr().out
    import json as _json

    rows = _json.loads(out)
    assert rows and rows[0]["article_number"] == "第一百四十三条"


def test_law_show_returns_article(law_ctx, capsys) -> None:
    ctx, _ = law_ctx
    _add_law(ctx, "中华人民共和国民法典", "第一百四十三条", "行为有效")
    judicial.law_show(ctx, "第一百四十三条")
    out = capsys.readouterr().out
    assert "第一百四十三条" in out
    assert "行为有效" in out


def test_law_show_unknown_raises(law_ctx) -> None:
    ctx, _ = law_ctx
    import typer

    with pytest.raises(typer.BadParameter):
        judicial.law_show(ctx, "不存在的法条")
