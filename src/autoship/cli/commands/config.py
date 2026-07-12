"""``autoship config`` command."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, cast

import structlog
import tomli_w
import typer

from autoship.core.config_center import DEFAULT_CONFIG_NAME
from autoship.exceptions import ConfigError
from autoship.utils.json_io import atomic_write_text
from autoship.utils.redaction import is_sensitive_key

logger = structlog.get_logger("autoship")

_i18n = None  # i18n removed
app = typer.Typer(
    name="config",
    help="config.help",
    rich_markup_mode="rich",
)


def _redact(value: Any) -> Any:
    """Recursively redact sensitive dictionary values."""
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {k: "***" if is_sensitive_key(k) else _redact(v) for k, v in mapping.items()}
    if isinstance(value, list):
        sequence = cast(list[Any], value)
        return [_redact(item) for item in sequence]
    return value


def _drop_none(value: Any) -> Any:
    """Recursively drop ``None`` values so output is TOML serializable."""
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {k: _drop_none(v) for k, v in mapping.items() if v is not None}
    if isinstance(value, list):
        sequence = cast(list[Any], value)
        return [_drop_none(item) for item in sequence if item is not None]
    return value


def _dotted_get(cfg: dict[str, Any], dotted_key: str, i18n) -> Any:
    """Retrieve a nested configuration value by dotted key."""
    parts = dotted_key.split(".")
    target: Any = cfg
    for part in parts:
        if not isinstance(target, dict):
            raise ConfigError("config.key_not_found")
        mapping = cast(dict[str, Any], target)
        if part not in mapping:
            raise ConfigError("config.key_not_found")
        target = mapping[part]
    return target


def _target_path(ctx: typer.Context) -> Path:
    """Return the configuration file path to modify."""
    config_path: Path | None = ctx.obj.get("config_path") if ctx.obj else None
    if config_path is not None:
        return config_path
    project_root: Path = ctx.obj["config"].project_root
    return project_root / DEFAULT_CONFIG_NAME


def _load_toml_file(path: Path) -> dict[str, Any]:
    """Load a TOML file or return an empty dict if it does not exist."""
    if not path.exists():
        return {}
    return tomllib.loads(path.read_bytes())


@app.command("list", help="config.list_help")
def list_config(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="config.option.json"),
) -> None:
    """Show effective configuration (sensitive values are redacted)."""
    cfg = ctx.obj["config"].model_dump(mode="json")
    cfg = _redact(cfg)
    if json_output:
        typer.echo(json.dumps(cfg, indent=2))
    else:
        typer.echo(tomli_w.dumps(_drop_none(cfg)).strip())


@app.command("get", help="config.get_help")
def get_config(
    ctx: typer.Context,
    key: str = typer.Argument(..., help="config.option.key"),
) -> None:
    """Get a single configuration value."""

    cfg = ctx.obj["config"].model_dump(mode="json")
    try:
        value = _dotted_get(cfg, key, None)
    except ConfigError as exc:
        typer.secho("error.prefix", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    if isinstance(value, dict | list):
        typer.echo(json.dumps(value, indent=2))
    else:
        typer.echo(str(value))


@app.command("telemetry", help="config.telemetry_help")
def telemetry_config(
    ctx: typer.Context,
    enable: bool = typer.Option(False, "--enable", help="config.option.enable"),
    disable: bool = typer.Option(False, "--disable", help="config.option.disable"),
    status: bool = typer.Option(False, "--status", help="config.option.status"),
) -> None:
    """Enable, disable, or view telemetry setting."""

    cfg = ctx.obj["config"]
    if status or (not enable and not disable):
        _state = "enabled" if cfg.telemetry.enabled else "disabled"
        typer.echo("config.telemetry_status")
        return

    # RBAC gate on writes (status reads are left open).

    target = _target_path(ctx)
    data = _load_toml_file(target)
    data["telemetry_enabled"] = enable if enable else not disable
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, tomli_w.dumps(data))
    _state = "enabled" if data["telemetry_enabled"] else "disabled"
    typer.echo("config.telemetry_set")


def register(parent: typer.Typer) -> None:
    """Register the config command group."""
    parent.add_typer(app)
