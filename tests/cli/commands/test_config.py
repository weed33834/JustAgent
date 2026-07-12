"""Tests for the config command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typer

from autoship.cli.commands import config
from autoship.core.i18n import get_i18n
from autoship.models.config import AppConfig


def _ctx(app_config: AppConfig, config_path: Path | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "config_path": config_path,
        "audit_logger": MagicMock(),
        "i18n": get_i18n("en"),
    }
    return ctx


def test_list_config_redacts_sensitive_values(app_config: AppConfig, capsys) -> None:
    app_config.llm.api_key = "secret-key"
    ctx = _ctx(app_config)
    config.list_config(ctx, json_output=True)
    captured = capsys.readouterr()
    assert "***" in captured.out
    assert "secret-key" not in captured.out


def test_get_config_existing_key(app_config: AppConfig, capsys) -> None:
    ctx = _ctx(app_config)
    config.get_config(ctx, "model.default_tier")
    captured = capsys.readouterr()
    assert captured.out.strip() == str(app_config.model.default_tier)


def test_get_config_missing_key(app_config: AppConfig, capsys) -> None:
    ctx = _ctx(app_config)
    with pytest.raises(typer.Exit) as exc_info:
        config.get_config(ctx, "model.nonexistent")
    assert exc_info.value.exit_code == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_telemetry_disable_writes_project_config(project_root: Path, app_config: AppConfig) -> None:
    config_file = project_root / ".autoship.toml"
    config_file.write_text("schema_version = 1\ntelemetry_enabled = true\n", encoding="utf-8")
    ctx = _ctx(app_config, config_path=config_file)
    config.telemetry_config(ctx, enable=False, disable=True, status=False)
    content = config_file.read_text(encoding="utf-8")
    assert "telemetry_enabled = false" in content


def test_telemetry_status_prints_current_state(app_config: AppConfig, capsys) -> None:
    ctx = _ctx(app_config)
    config.telemetry_config(ctx, enable=False, disable=False, status=True)
    captured = capsys.readouterr()
    assert "disabled" in captured.out


def test_legacy_telemetry_enabled_migrates_to_telemetry_config() -> None:
    from autoship.models.config import AppConfig

    cfg = AppConfig(telemetry_enabled=True)
    assert cfg.telemetry.enabled is True


def test_telemetry_disabled_by_default() -> None:
    from autoship.models.config import AppConfig

    cfg = AppConfig()
    assert cfg.telemetry.enabled is False
