"""Configuration loading, merging, and validation — powered by pydantic-settings."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

import structlog
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from autoship.exceptions import ConfigError
from autoship.models.config import AppConfig

logger = structlog.get_logger("autoship")

DEFAULT_CONFIG_NAME = ".autoship.toml"
TEAM_CONFIG_NAME = ".autoship.team.toml"
ENV_PREFIX = "AUTOSHIP_"


class _EnvSettings(BaseSettings):
    """Environment-variable bridge: maps AUTOSHIP_* vars to AppConfig fields."""

    model_config = SettingsConfigDict(
        env_prefix=f"{ENV_PREFIX}",
        env_nested_delimiter="__",
        extra="ignore",
    )
    # Top-level scalar fields
    log_level: str | None = None
    locale: str | None = None
    project_root: str | None = None

    # Nested sections — pydantic-settings handles AUTOSHIP_CLEAN__TOOLS etc.
    clean: dict[str, Any] | None = None
    commit: dict[str, Any] | None = None
    security: dict[str, Any] | None = None
    audit: dict[str, Any] | None = None
    sandbox: dict[str, Any] | None = None
    web_search: dict[str, Any] | None = None
    docker_ship: dict[str, Any] | None = None
    model: dict[str, Any] | None = None
    verify: dict[str, Any] | None = None
    cache: dict[str, Any] | None = None
    registry: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
    tools: dict[str, Any] | None = None
    hooks: dict[str, Any] | None = None


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Failed to load config from {path}: {exc}") from exc


def _env_settings_dict() -> dict[str, Any]:
    """Load env vars via pydantic-settings and drop None values."""
    settings = _EnvSettings()
    raw = settings.model_dump()
    return _strip_none(raw)


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove None values and empty dicts from a nested dict."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            cleaned = _strip_none(cast(dict[str, Any], v))
            if cleaned:
                result[k] = cleaned
        else:
            result[k] = v
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge *override* into *base* recursively. Lists are replaced, not extended."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _find_project_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    for candidate in [start, *start.parents]:
        if (candidate / DEFAULT_CONFIG_NAME).exists():
            return candidate
    return start


def _default_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project_root": ".",
        "log_level": "INFO",
        "clean": {
            "enabled": True,
            "tools": ["autoflake", "black"],
            "dry_run": False,
            "exclude": [],
        },
        "commit": {
            "enabled": True,
            "max_tokens": 512,
            "conventional_commits": True,
            "auto_push": False,
        },
        "model": {"default_tier": 2, "fallback": True, "backends": []},
    }


def load_config(
    config_path: Path | None = None,
    project_root: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load and validate the full application configuration.

    Priority (high → low):
        1. CLI overrides
        2. Environment variables (AUTOSHIP_*)
        3. Project-level ``.autoship.toml`` (including team overlay)
        4. Built-in defaults
    """
    merged = _default_config()

    # Resolve project root
    if config_path is not None:
        project_root = config_path.parent
    else:
        project_root = project_root or _find_project_root()

    # Project config
    project_config_path = (
        config_path if config_path is not None else (project_root / DEFAULT_CONFIG_NAME)
    )
    project_cfg = _load_toml(project_config_path)
    merged = _deep_merge(merged, project_cfg)

    # Team config overlay with optional signature verification
    team_path = project_root / TEAM_CONFIG_NAME
    team_section = cast(dict[str, Any], merged.get("team") or {})
    team_public_key: str | None = None
    if isinstance(team_section.get("public_key"), str):
        team_public_key = team_section["public_key"]
    require_signature = bool(team_section.get("require_signature"))

    from autoship.core.team_config import TeamConfigError, load_team_config

    try:
        team_cfg = load_team_config(
            team_path, public_key_b64=team_public_key, require_signature=require_signature
        )
    except TeamConfigError as exc:
        if require_signature:
            raise ConfigError(str(exc)) from exc
        team_cfg = {}
    merged = _deep_merge(merged, team_cfg)

    # Environment variables (pydantic-settings handles AUTOSHIP_* natively)
    env_cfg = _env_settings_dict()
    merged = _deep_merge(merged, env_cfg)

    # CLI overrides (highest priority)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    merged["project_root"] = str(project_root)

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
