"""Tests for the ``justagent info`` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from justagent.cli.commands import info
from justagent.models.config import AppConfig, ModelBackendConfig, Provider


def _ctx(app_config: AppConfig, config_path: Path | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "config_path": config_path,
        "audit_logger": MagicMock(),
        "i18n": MagicMock(),
    }
    return ctx


def test_info_prints_header_and_runtime(app_config: AppConfig, capsys) -> None:
    info.info(_ctx(app_config))
    captured = capsys.readouterr()
    assert "JustAgent" in captured.out
    assert "python" in captured.out
    assert "model backends" in captured.out


def test_info_lists_model_backends(app_config: AppConfig, capsys) -> None:
    cfg = app_config.model_copy(deep=True)
    cfg.model.backends = [
        ModelBackendConfig(
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/v1",
            model="qwen2.5:7b",
            tier=2,
        )
    ]
    info.info(_ctx(cfg))
    captured = capsys.readouterr()
    assert "ollama" in captured.out
    assert "qwen2.5:7b" in captured.out


def test_info_config_path_shown(app_config: AppConfig, capsys) -> None:
    info.info(_ctx(app_config, config_path=Path("/tmp/justagent.toml")))
    captured = capsys.readouterr()
    assert "/tmp/justagent.toml" in captured.out
