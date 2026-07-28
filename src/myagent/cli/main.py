"""Typer CLI 入口与全局选项。"""

from __future__ import annotations

import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import structlog
import typer

from myagent.cli import commands
from myagent.core.audit_logger import AuditLogger
from myagent.core.config_center import load_config
from myagent.core.i18n import get_i18n
from myagent.core.logging_config import configure_structlog
from myagent.exceptions import ConfigError, ExitCode, MyAgentError

# AuditLogger 在 main_callback 中创建，在 cli_entrypoint 的 finally 中关闭。
# 两者不共享 ctx，故用模块级变量桥接。
_audit_logger: AuditLogger | None = None

app = typer.Typer(
    name="myagent",
    help="MyAgent：本地优先的 AI 编码智能体（agent / session / plan / act / yolo）",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool | None) -> None:
    """``--version`` 时打印版本并退出。is_eager 保证先于其它回调触发。"""
    if value:
        try:
            version = _pkg_version("myagent")
        except Exception:  # noqa: BLE001 - 永不让 --version 崩溃
            version = "unknown"
        typer.echo(f"myagent {version}")
        raise typer.Exit()


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="仅显示操作而不执行"),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过交互式确认"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="配置文件路径"),
    lang: str | None = typer.Option(
        None, "--lang", help="输出语言（保留兼容，当前默认中文）", show_default=False
    ),
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        expose_value=False,
        help="显示版本并退出。",
    ),
) -> None:
    """MyAgent 全局选项。"""
    global _audit_logger
    ctx.ensure_object(dict)

    config = load_config(config_path=config_path)
    audit_logger = AuditLogger(config)
    _audit_logger = audit_logger

    # SSO/RBAC/telemetry/sink 等企业级能力已移除，身份与角色恒为 None。
    identity = None
    role: str | None = None
    audit_logger.bind_context(user=None, role=role, sso_subject=None, sso_provider=None)

    audit_logger.record("cli.invoked", {"config_path": str(config_path) if config_path else None})

    ctx.obj["config"] = config
    ctx.obj["config_path"] = config_path
    ctx.obj["audit_logger"] = audit_logger
    ctx.obj["verbose"] = verbose
    ctx.obj["dry_run"] = dry_run
    ctx.obj["yes"] = yes
    ctx.obj["identity"] = identity
    ctx.obj["role"] = role
    # When main_callback is called directly (e.g. in tests), typer leaves
    # OptionInfo sentinel objects as default values instead of None.
    lang_str = lang if isinstance(lang, str) else None
    ctx.obj["i18n"] = get_i18n(lang_str)


commands.register_all(app)


def _command_name(cmd: Any) -> str | None:
    """返回已注册命令的字符串名称，否则 None。"""
    name = getattr(cmd, "name", None)
    return name if isinstance(name, str) and name else None


def _group_name(group: Any) -> str | None:
    """返回命令组的字符串名称。Typer 可能把名字存在父级或子 Typer.info 上。"""
    name = getattr(group, "name", None)
    if isinstance(name, str) and name:
        return name
    child = getattr(group, "typer_instance", None)
    if child is not None:
        child_name = getattr(getattr(child, "info", None), "name", None)
        if isinstance(child_name, str) and child_name:
            return child_name
    return None


# 在注册完成后立即快照顶层命令名，避免运行时依赖可变的 ``app`` 对象。
_KNOWN_COMMANDS: set[str] = set()
for _cmd in app.registered_commands:
    _name = _command_name(_cmd)
    if _name:
        _KNOWN_COMMANDS.add(_name)
for _group in app.registered_groups:
    _name = _group_name(_group)
    if _name:
        _KNOWN_COMMANDS.add(_name)


def _guess_command() -> str:
    """从 ``sys.argv`` 推断被调用的子命令。"""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    return "help"


def _is_unknown_command(command: str) -> bool:
    """Return True if ``command`` looks like a user-supplied but unknown subcommand."""
    return bool(command) and command not in _KNOWN_COMMANDS and command != "help"


def _print_suggestion(exc: MyAgentError) -> None:
    """针对常见错误类型打印下一步建议（明文中文）。"""
    message = str(exc).lower()
    details = getattr(exc, "details", {}) or {}
    suggestion: str | None = None

    if isinstance(exc, ConfigError):
        suggestion = "运行 `myagent init` 创建配置文件。"
    elif "api key" in message or (
        "model" in message and ("unreachable" in message or "backend" in message)
    ):
        suggestion = "编辑 `.myagent.toml` 配置模型后端，或运行 `myagent init`。"
    elif "command not found" in message or "not found on path" in message:
        suggestion = "安装所需工具，或将其加入 PATH。"
    elif "upload" in message:
        _target = details.get("target") or "<target>"
        typer.secho(
            f"\n💡 使用 `myagent upload --target {_target} --dry-run` 预览上传。",
            fg=typer.colors.CYAN,
            err=True,
        )
        return

    if suggestion:
        typer.secho(f"\n💡 {suggestion}", fg=typer.colors.CYAN, err=True)


def cli_entrypoint() -> int:
    """``myagent`` 控制台脚本的顶层入口。"""
    global _audit_logger
    configure_structlog()
    logger = structlog.get_logger("myagent")
    command = _guess_command()

    if _is_unknown_command(command):
        typer.secho(f"未知命令：{command}", fg=typer.colors.RED, err=True)
        typer.secho("💡 运行 `myagent --help` 查看可用命令。", fg=typer.colors.CYAN, err=True)
        # 用法错误，退出码 2 与 Unix "调用错误" 约定一致。
        return ExitCode.CONFIG_ERROR

    exit_code = 0
    try:
        app()
    except typer.Exit as exc:
        exit_code = exc.exit_code
    except MyAgentError as exc:
        exit_code = exc.code
        typer.secho(f"错误：{exc}", fg=typer.colors.RED, err=True)
        _print_suggestion(exc)
    except Exception as exc:
        exit_code = ExitCode.USAGE_ERROR
        logger.exception("Unhandled exception")
        typer.secho(f"意外错误：{exc}", fg=typer.colors.RED, err=True)
        typer.secho("\n💡 运行 `myagent doctor` 诊断环境。", fg=typer.colors.CYAN, err=True)
    finally:
        # 关闭 main_callback 创建的 AuditLogger，释放 HTTP 连接池。
        if _audit_logger is not None:
            try:
                _audit_logger.close()
            except Exception:
                logger.debug("Error closing audit logger", exc_info=True)
            _audit_logger = None

    return exit_code
