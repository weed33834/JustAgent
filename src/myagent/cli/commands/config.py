"""``myagent config`` 命令：查看与管理配置。

注意：``config telemetry`` 子命令已移除（依赖不存在的 ``config.telemetry``
字段，遥测能力属企业级，已从本地 AI 编码智能体定位中剔除）。
"""

from __future__ import annotations

import json
from typing import Any, cast

import tomli_w
import typer

from myagent.exceptions import ConfigError
from myagent.utils.redaction import is_sensitive_key

app = typer.Typer(
    name="config",
    help="查看和管理 MyAgent 配置。",
    rich_markup_mode="rich",
)


def _redact(value: Any) -> Any:
    """递归脱敏敏感字典值。"""
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {k: "***" if is_sensitive_key(k) else _redact(v) for k, v in mapping.items()}
    if isinstance(value, list):
        sequence = cast(list[Any], value)
        return [_redact(item) for item in sequence]
    return value


def _drop_none(value: Any) -> Any:
    """递归剔除 ``None`` 值，保证输出可被 TOML 序列化。"""
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {k: _drop_none(v) for k, v in mapping.items() if v is not None}
    if isinstance(value, list):
        sequence = cast(list[Any], value)
        return [_drop_none(item) for item in sequence if item is not None]
    return value


def _dotted_get(cfg: dict[str, Any], dotted_key: str) -> Any:
    """按点号分隔键取嵌套配置值。"""
    parts = dotted_key.split(".")
    target: Any = cfg
    for part in parts:
        if not isinstance(target, dict):
            raise ConfigError(f"配置中不存在键 '{dotted_key}'")
        mapping = cast(dict[str, Any], target)
        if part not in mapping:
            raise ConfigError(f"配置中不存在键 '{dotted_key}'")
        target = mapping[part]
    return target


@app.command("list", help="显示生效配置（敏感值会被脱敏）。")
def list_config(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="以 JSON 格式输出"),
) -> None:
    """显示生效配置（敏感值会被脱敏）。"""
    cfg = ctx.obj["config"].model_dump(mode="json")
    cfg = _redact(cfg)
    if json_output:
        typer.echo(json.dumps(cfg, indent=2))
    else:
        typer.echo(tomli_w.dumps(_drop_none(cfg)).strip())


@app.command("get", help="获取单个配置项的值。")
def get_config(
    ctx: typer.Context,
    key: str = typer.Argument(..., help="点号分隔的配置键，例如 model.default_tier"),
) -> None:
    """获取单个配置项的值。"""
    cfg = ctx.obj["config"].model_dump(mode="json")
    try:
        value = _dotted_get(cfg, key)
    except ConfigError as exc:
        typer.secho(f"错误：{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    if isinstance(value, dict | list):
        typer.echo(json.dumps(value, indent=2))
    else:
        typer.echo(str(value))


def register(parent: typer.Typer) -> None:
    """注册 config 命令组。"""
    parent.add_typer(app)
