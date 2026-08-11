"""Tests for the ``justagent models`` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from justagent.cli.commands import models
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


def _with_backend(app_config: AppConfig) -> AppConfig:
    cfg = app_config.model_copy(deep=True)
    cfg.model.backends = [
        ModelBackendConfig(
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/v1",
            model="qwen2.5:7b",
            tier=2,
        )
    ]
    return cfg


def test_models_no_backends(app_config: AppConfig, capsys) -> None:
    models.list_backends(_ctx(app_config))
    captured = capsys.readouterr()
    assert "No model backends configured" in captured.out


def test_models_lists_backend(app_config: AppConfig, capsys) -> None:
    cfg = _with_backend(app_config)
    models.list_backends(_ctx(cfg))
    captured = capsys.readouterr()
    assert "ollama" in captured.out
    assert "qwen2.5:7b" in captured.out
    assert "localhost:11434" in captured.out


def test_models_check_does_not_crash(app_config: AppConfig, capsys) -> None:
    cfg = _with_backend(app_config)
    models.list_backends(_ctx(cfg), check=True)
    captured = capsys.readouterr()
    # Table header + summary line should be present; health may be OK/FAIL/ERROR
    # depending on whether a local Ollama is running, but never crash.
    assert "ENDPOINT" in captured.out
    assert "backend(s) healthy" in captured.out
