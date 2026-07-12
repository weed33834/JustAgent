"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoship.core.config_center import (
    _coerce_env_value,
    _deep_merge,
    _env_to_dict,
    _env_var_name,
    _filter_env_cfg,
    _find_project_root,
    load_config,
)
from autoship.exceptions import ConfigError
from autoship.models.config import AppConfig


def test_default_config(project_root: Path) -> None:
    cfg = load_config(project_root=project_root)
    assert isinstance(cfg, AppConfig)
    assert cfg.schema_version == 1
    assert cfg.log_level == "INFO"
    assert cfg.clean.tools == ["autoflake", "black"]


def test_load_from_file(project_root: Path) -> None:
    config_file = project_root / ".autoship.toml"
    config_file.write_text(
        'schema_version = 1\nlog_level = "DEBUG"\n[clean]\ntools = ["ruff"]\n',
        encoding="utf-8",
    )
    cfg = load_config(config_path=config_file)
    assert cfg.log_level == "DEBUG"
    assert cfg.clean.tools == ["ruff"]


def test_invalid_config_raises(project_root: Path) -> None:
    config_file = project_root / ".autoship.toml"
    config_file.write_text('schema_version = "not-an-int"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path=config_file)


def test_unsupported_clean_tool_raises(project_root: Path) -> None:
    config_file = project_root / ".autoship.toml"
    config_file.write_text('[clean]\ntools = ["unknown-tool"]\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="Unsupported clean tools"):
        load_config(config_path=config_file)


def test_supported_clean_tool_accepts(project_root: Path) -> None:
    config_file = project_root / ".autoship.toml"
    config_file.write_text('[clean]\ntools = ["ruff"]\n', encoding="utf-8")
    cfg = load_config(config_path=config_file)
    assert cfg.clean.tools == ["ruff"]


def test_find_project_root(project_root: Path) -> None:
    config_file = project_root / ".autoship.toml"
    config_file.write_text("schema_version = 1\n", encoding="utf-8")
    nested = project_root / "src" / "nested"
    nested.mkdir(parents=True)
    assert _find_project_root(nested) == project_root


def test_env_to_dict(monkeypatch) -> None:
    monkeypatch.setenv("AUTOSHIP_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AUTOSHIP_MODEL__DEFAULT_TIER", "1")
    env_cfg = _env_to_dict()
    assert env_cfg["log_level"] == "DEBUG"
    assert env_cfg["model"]["default_tier"] == 1


def test_coerce_env_value() -> None:
    assert _coerce_env_value("true") is True
    assert _coerce_env_value("false") is False
    assert _coerce_env_value("42") == 42
    assert _coerce_env_value("3.14") == 3.14
    assert _coerce_env_value("hello") == "hello"


def test_deep_merge() -> None:
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}}
    merged = _deep_merge(base, override)
    assert merged["a"] == 1
    assert merged["b"] == {"c": 99, "d": 3}


def test_env_var_name() -> None:
    assert _env_var_name(["log", "level"]) == "AUTOSHIP_LOG__LEVEL"


def test_filter_env_cfg_allows_allowlisted_keys() -> None:
    assert _filter_env_cfg({"log_level": "DEBUG", "clean": {"enabled": False}}) == {
        "log_level": "DEBUG",
        "clean": {"enabled": False},
    }


def test_filter_env_cfg_blocks_sensitive_keys() -> None:
    assert _filter_env_cfg({"api_key": "secret"}) == {}


def test_filter_env_cfg_blocks_unknown_keys(caplog) -> None:
    assert _filter_env_cfg({"unknown_field": "value"}) == {}
    assert "AUTOSHIP_UNKNOWN_FIELD" in caplog.text
    assert "not in allowlist" in caplog.text


def test_filter_env_cfg_blocks_nested_sensitive_keys(caplog) -> None:
    assert _filter_env_cfg({"model": {"backends": {"0": {"api_key": "secret"}}}}) == {}
    assert "AUTOSHIP_MODEL__BACKENDS__0__API_KEY" in caplog.text
    assert "sensitive" in caplog.text


def test_load_config_respects_allowlisted_env_vars(monkeypatch, project_root: Path) -> None:
    monkeypatch.setenv("AUTOSHIP_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AUTOSHIP_CLEAN__ENABLED", "false")
    cfg = load_config(project_root=project_root)
    assert cfg.log_level == "DEBUG"
    assert cfg.clean.enabled is False


def test_load_config_ignores_sensitive_env_vars(monkeypatch, project_root: Path) -> None:
    monkeypatch.setenv("AUTOSHIP_LLM__API_KEY", "secret-key")
    cfg = load_config(project_root=project_root)
    assert cfg.llm.api_key is None


def test_list_env_var_parsed_as_list(monkeypatch, project_root: Path) -> None:
    """Comma-separated env value reaches a ``list[str]`` field as a real list."""
    monkeypatch.setenv("AUTOSHIP_CLEAN__TOOLS", "ruff,isort,black")
    cfg = load_config(project_root=project_root)
    assert cfg.clean.tools == ["ruff", "isort", "black"]


def test_list_env_var_strips_whitespace(monkeypatch, project_root: Path) -> None:
    """Whitespace around comma-separated items is stripped."""
    monkeypatch.setenv("AUTOSHIP_CLEAN__TOOLS", " ruff , isort ")
    cfg = load_config(project_root=project_root)
    assert cfg.clean.tools == ["ruff", "isort"]


def test_list_env_var_empty_value(monkeypatch, project_root: Path) -> None:
    """An empty list-typed env value yields an empty list, not a single empty string."""
    monkeypatch.setenv("AUTOSHIP_CLEAN__TOOLS", "")
    cfg = load_config(project_root=project_root)
    assert cfg.clean.tools == []


def test_non_list_env_var_unchanged(monkeypatch, project_root: Path) -> None:
    """A non-list allowlisted env var keeps its coerced scalar type (bool, not split)."""
    monkeypatch.setenv("AUTOSHIP_CLEAN__DRY_RUN", "true")
    cfg = load_config(project_root=project_root)
    assert cfg.clean.dry_run is True
    assert isinstance(cfg.clean.dry_run, bool)
