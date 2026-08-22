"""Typer CLI 入口与全局选项。"""

from __future__ import annotations

import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import structlog
import typer

from justagent.cli import commands
from justagent.core.audit_logger import AuditLogger
from justagent.core.config_center import load_config
from justagent.core.i18n import I18n, get_i18n
from justagent.core.logging_config import configure_structlog
from justagent.exceptions import ConfigError, ExitCode, MyAgentError

# AuditLogger 在 main_callback 中创建，在 cli_entrypoint 的 finally 中关闭。
# 两者不共享 ctx，故用模块级变量桥接。
_audit_logger: AuditLogger | None = None

# 模块级 i18n：用于绑定 help 文本（导入时）以及错误处理路径（运行时）。
# main_callback 会按 --lang 重新初始化以覆盖运行时输出。
_i18n: I18n = get_i18n()

app = typer.Typer(
    name="justagent",
    help=_i18n._("cli.help"),
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool | None) -> None:
    """``--version`` 时打印版本并退出。is_eager 保证先于其它回调触发。"""
    if value:
        try:
            version = _pkg_version("justagent")
        except Exception:  # noqa: BLE001 - 永不让 --version 崩溃
            version = "unknown"
        typer.echo(f"justagent {version}")
        raise typer.Exit()


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_i18n._("option.verbose")),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help=_i18n._("option.dry_run")),
    yes: bool = typer.Option(False, "--yes", "-y", help=_i18n._("option.yes")),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help=_i18n._("option.config_path")
    ),
    lang: str | None = typer.Option(
        None, "--lang", help=_i18n._("option.lang"), show_default=False
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
    """JustAgent 全局选项。"""
    global _audit_logger, _i18n
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
    _i18n = get_i18n(lang_str)
    ctx.obj["i18n"] = _i18n


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
    """针对常见错误类型打印下一步建议。"""
    message = str(exc).lower()
    details = getattr(exc, "details", {}) or {}
    suggestion: str | None = None

    if isinstance(exc, ConfigError):
        suggestion = _i18n._("error.suggestion.init")
    elif "api key" in message or (
        "model" in message and ("unreachable" in message or "backend" in message)
    ):
        suggestion = _i18n._("error.suggestion.model_config")
    elif "command not found" in message or "not found on path" in message:
        suggestion = _i18n._("error.suggestion.install_tool")
    elif "upload" in message:
        _target = details.get("target") or "<target>"
        typer.secho(
            f"\n💡 {_i18n._('error.suggestion.upload_dry_run', target=_target)}",
            fg=typer.colors.CYAN,
            err=True,
        )
        return

    if suggestion:
        typer.secho(f"\n💡 {suggestion}", fg=typer.colors.CYAN, err=True)


def cli_entrypoint() -> int:
    """``justagent`` 控制台脚本的顶层入口。"""
    global _audit_logger
    configure_structlog()
    logger = structlog.get_logger("justagent")
    command = _guess_command()

    if _is_unknown_command(command):
        typer.secho(_i18n._("cli.unknown_command", command=command), fg=typer.colors.RED, err=True)
        typer.secho(
            f"💡 {_i18n._('cli.unknown_command.suggestion')}", fg=typer.colors.CYAN, err=True
        )
        # 用法错误，退出码 2 与 Unix "调用错误" 约定一致。
        return ExitCode.CONFIG_ERROR

    exit_code = 0
    try:
        app()
    except typer.Exit as exc:
        exit_code = exc.exit_code
    except MyAgentError as exc:
        exit_code = exc.code
        typer.secho(_i18n._("error.prefix", exc=str(exc)), fg=typer.colors.RED, err=True)
        _print_suggestion(exc)
    except Exception as exc:
        exit_code = ExitCode.USAGE_ERROR
        logger.exception("Unhandled exception")
        typer.secho(_i18n._("unexpected_error.prefix", exc=str(exc)), fg=typer.colors.RED, err=True)
        typer.secho(f"\n💡 {_i18n._('error.suggestion.doctor')}", fg=typer.colors.CYAN, err=True)
    finally:
        # 关闭 main_callback 创建的 AuditLogger，释放 HTTP 连接池。
        if _audit_logger is not None:
            try:
                _audit_logger.close()
            except Exception:
                logger.debug("Error closing audit logger", exc_info=True)
            _audit_logger = None

    return exit_code
