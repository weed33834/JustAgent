"""Tests for the init command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from autoship.cli.commands import init
from autoship.models.config import AppConfig


def test_init_creates_config(project_root: Path, app_config: AppConfig, monkeypatch) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
    }
    output = project_root / ".autoship.toml"
    init.init(ctx, output=output)
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "project_type" in content


def test_init_dry_run_does_not_write(
    project_root: Path, app_config: AppConfig, monkeypatch
) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": True,
        "yes": True,
    }
    output = project_root / ".autoship.toml"
    init.init(ctx, output=output)
    assert not output.exists()
