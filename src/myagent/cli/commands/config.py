"""``myagent config`` command: inspect and manage configuration."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, cast

import tomli_w
import typer

from myagent.core.i18n import I18n, get_i18n_from_ctx
from myagent.exceptions import ConfigError
from myagent.models.config import AppConfig
from myagent.utils.redaction import is_sensitive_key

app = typer.Typer(
    name="config",
    help="Inspect and manage MyAgent configuration.",
    rich_markup_mode="rich",
)


def _redact(value: Any) -> Any:
    """Recursively redact sensitive dict values."""
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {k: "***" if is_sensitive_key(k) else _redact(v) for k, v in mapping.items()}
    if isinstance(value, list):
        sequence = cast(list[Any], value)
        return [_redact(item) for item in sequence]
    return value


def _drop_none(value: Any) -> Any:
    """Recursively drop ``None`` values so the output is TOML-serializable."""
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {k: _drop_none(v) for k, v in mapping.items() if v is not None}
    if isinstance(value, list):
        sequence = cast(list[Any], value)
        return [_drop_none(item) for item in sequence if item is not None]
    return value


def _dotted_get(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Retrieve a nested config value by dotted key path."""
    parts = dotted_key.split(".")
    target: Any = cfg
    for part in parts:
        if not isinstance(target, dict):
            raise ConfigError(f"Key '{dotted_key}' not found in configuration")
        mapping = cast(dict[str, Any], target)
        if part not in mapping:
            raise ConfigError(f"Key '{dotted_key}' not found in configuration")
        target = mapping[part]
    return target


@app.command("list", help="Show effective configuration (sensitive values are redacted).")
def list_config(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show effective configuration (sensitive values are redacted)."""
    cfg = ctx.obj["config"].model_dump(mode="json")
    cfg = _redact(cfg)
    if json_output:
        typer.echo(json.dumps(cfg, indent=2))
    else:
        typer.echo(tomli_w.dumps(_drop_none(cfg)).strip())


@app.command("get", help="Get a single configuration value.")
def get_config(
    ctx: typer.Context,
    key: str = typer.Argument(..., help="Dotted configuration key, e.g. model.default_tier"),
) -> None:
    """Get a single configuration value."""
    i18n: I18n = get_i18n_from_ctx(ctx)
    cfg = ctx.obj["config"].model_dump(mode="json")
    try:
        value = _dotted_get(cfg, key)
    except ConfigError:
        typer.secho(i18n._("config.key_not_found", key=key), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None
    if isinstance(value, dict | list):
        typer.echo(json.dumps(value, indent=2))
    else:
        typer.echo(str(value))


@app.command("telemetry", help="Enable, disable, or view telemetry setting.")
def telemetry_config(
    ctx: typer.Context,
    enable: bool = typer.Option(False, "--enable", help="Enable telemetry"),
    disable: bool = typer.Option(False, "--disable", help="Disable telemetry"),
    status: bool = typer.Option(False, "--status", help="Show current telemetry status"),
) -> None:
    """Enable, disable, or view the telemetry setting.

    Writes the change back to the project configuration file (``--config``)
    using the legacy ``telemetry_enabled`` key so it round-trips through
    :meth:`AppConfig._migrate_legacy_telemetry`.
    """
    i18n: I18n = get_i18n_from_ctx(ctx)
    config: AppConfig = ctx.obj["config"]

    if status:
        state = "enabled" if config.telemetry.enabled else "disabled"
        typer.echo(i18n._("config.telemetry_status", state=state))
        return

    if not enable and not disable:
        return

    new_value = enable  # True when --enable, False when --disable

    config_path = ctx.obj.get("config_path")
    if config_path is None:
        typer.secho(i18n._("config.read_error", exc="no config file path"), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    config_path = Path(config_path)
    if config_path.exists():
        try:
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            typer.secho(i18n._("config.read_error", exc=exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
    else:
        data = {}

    data["telemetry_enabled"] = new_value

    try:
        config_path.write_text(tomli_w.dumps(data), encoding="utf-8")
    except OSError as exc:
        typer.secho(i18n._("config.read_error", exc=exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    state = "enabled" if new_value else "disabled"
    typer.echo(i18n._("config.telemetry_set", state=state, target=config_path))


def register(parent: typer.Typer) -> None:
    """Register the config command group."""
    parent.add_typer(app)
