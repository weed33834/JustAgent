"""Typer CLI entry point and global options."""

from __future__ import annotations

import sys
import time
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import structlog
import typer

from autoship.cli import commands
from autoship.core.audit_logger import AuditLogger
from autoship.core.config_center import load_config
from autoship.core.logging_config import configure_structlog
from autoship.exceptions import AutoShipError, ConfigError, ExitCode

_i18n = None  # i18n removed

#: Bridge the ``AuditLogger`` created in ``main_callback`` (which owns the
#: Typer ``ctx``) to ``cli_entrypoint``'s ``finally`` block (which does not).
#: Set on every invocation and reset to ``None`` once closed.
_audit_logger: AuditLogger | None = None

app = typer.Typer(
    name="autoship",
    help="cli.help",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool | None) -> None:
    """Print the installed ``autoship`` version and exit when ``--version`` is passed.

    Reads the version from the installed package metadata (``importlib.metadata``)
    so it stays in lockstep with ``pyproject.toml`` without a hardcoded literal.
    ``is_eager=True`` on the option ensures this fires before any other callback
    work — notably before the ``AuditLogger`` is created in ``main_callback``.
    """
    if value:
        try:
            version = _pkg_version("autoship")
        except Exception:  # noqa: BLE001 - never crash --version
            version = "unknown"
        typer.echo(f"autoship {version}")
        raise typer.Exit()


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="option.verbose"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="option.dry_run"),
    yes: bool = typer.Option(False, "--yes", "-y", help="option.yes"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="option.config_path"),
    lang: str | None = typer.Option(None, "--lang", help="option.lang", show_default=False),
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        expose_value=False,
        help="Show version and exit.",
    ),
) -> None:
    """AutoShip global options."""
    global _audit_logger
    ctx.ensure_object(dict)

    config = load_config(config_path=config_path)
    _selected_lang = lang if isinstance(lang, str) and lang.lower() != "auto" else config.locale
    i18n = None
    audit_logger = AuditLogger(config)
    _audit_logger = audit_logger

    # Bind SSO identity to the audit context before recording the first
    # event, so the first audit record of every invocation carries the
    # actor identity. Previously ``cli.invoked`` was recorded first and
    # ``bind_context`` after, leaving the first record without ``user`` /
    # ``role`` / ``sso_subject`` / ``sso_provider``.
    identity = None
    role: str | None = None
    if config.sso.enabled:
        try:
            identity = get_current_identity(config)  # noqa: F821
        except Exception as exc:  # noqa: BLE001 — soft-fail, never block CLI
            audit_logger.record("sso.identity_failed", {"error": str(exc)})
            identity = None
        if identity is not None and config.rbac.enabled:
            role = resolve_role(identity, config.rbac)  # noqa: F821
    audit_logger.bind_context(
        user=identity.user if identity else None,
        role=role,
        sso_subject=identity.subject if identity else None,
        sso_provider=identity.provider if identity else None,
    )

    audit_logger.record("cli.invoked", {"config_path": str(config_path) if config_path else None})

    ctx.obj["config"] = config
    ctx.obj["config_path"] = config_path
    ctx.obj["i18n"] = i18n
    ctx.obj["audit_logger"] = audit_logger
    ctx.obj["verbose"] = verbose
    ctx.obj["dry_run"] = dry_run
    ctx.obj["yes"] = yes
    ctx.obj["identity"] = identity
    ctx.obj["role"] = role


commands.register_all(app)


def _command_name(cmd: Any) -> str | None:
    """Return the string name of a registered command or ``None``."""
    name = getattr(cmd, "name", None)
    return name if isinstance(name, str) and name else None


def _group_name(group: Any) -> str | None:
    """Return the string name of a registered command group or ``None``.

    Typer stores the group name either directly on the parent ``TyperInfo`` or,
    when ``add_typer`` is called without an explicit ``name``, on the child
    ``Typer.info`` object. ``DefaultPlaceholder`` values are resolved best-effort.
    """
    name = getattr(group, "name", None)
    if isinstance(name, str) and name:
        return name
    child = getattr(group, "typer_instance", None)
    if child is not None:
        child_name = getattr(getattr(child, "info", None), "name", None)
        if isinstance(child_name, str) and child_name:
            return child_name
    return None


# Snapshot top-level command names immediately after registration. This avoids
# depending on the mutable ``app`` object at runtime, which matters for tests
# that patch ``main.app`` and for consistent error handling.
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
    """Infer the invoked subcommand from ``sys.argv``."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    return "help"


def _is_unknown_command(command: str) -> bool:
    """Return True if ``command`` looks like a user-supplied but unknown subcommand."""
    return bool(command) and command not in _KNOWN_COMMANDS and command != "help"


def _print_suggestion(i18n, exc: AutoShipError) -> None:
    """Print a contextual next-step suggestion for common error types."""
    message = str(exc).lower()
    details = getattr(exc, "details", {}) or {}
    suggestion_key: str | None = None

    if isinstance(exc, ConfigError):
        suggestion_key = "error.suggestion.init"
    elif "api key" in message or (
        "model" in message and ("unreachable" in message or "backend" in message)
    ):
        suggestion_key = "error.suggestion.model_config"
    elif "command not found" in message or "not found on path" in message:
        suggestion_key = "error.suggestion.install_tool"
    elif "upload" in message:
        _target = details.get("target") or "<target>"
        suggestion_key = "error.suggestion.upload_dry_run"
        typer.secho(f"\n💡 {suggestion_key}", fg=typer.colors.CYAN, err=True)
        return

    if suggestion_key:
        typer.secho(f"\n💡 {suggestion_key}", fg=typer.colors.CYAN, err=True)


def cli_entrypoint() -> int:
    """Top-level entry point used by ``autoship`` console script."""
    global _audit_logger
    configure_structlog()
    logger = structlog.get_logger("autoship")
    start = time.perf_counter()
    command = _guess_command()
    exit_code = 0
    exc_record: BaseException | None = None

    try:
        config = load_config()
        i18n = None
    except ConfigError as exc:
        i18n = None  # i18n removed
        typer.secho("error.prefix", fg=typer.colors.RED, err=True)
        _print_suggestion(i18n, exc)
        return exc.code

    telemetry = TelemetryCollector(  # noqa: F821
        enabled=config.telemetry.enabled,
        endpoint=str(config.telemetry.endpoint) if config.telemetry.endpoint else None,
        timeout=config.telemetry.timeout,
        allow_untrusted=config.telemetry.allow_untrusted_endpoint,
        batch_size=config.telemetry.batch_size,
        sink_endpoint=(
            str(config.sink.url)
            if config.sink.enabled and config.sink.url and config.sink.forward_telemetry
            else None
        ),
        sink_token=config.sink.token,
    )

    if _is_unknown_command(command):
        typer.secho("cli.unknown_command", fg=typer.colors.RED, err=True)
        typer.secho(
            f"💡 {'cli.unknown_command.suggestion'}",
            fg=typer.colors.CYAN,
            err=True,
        )
        # An unknown command is a usage error; exit 2 (CONFIG_ERROR) matches
        # the Unix convention for "incorrect invocation" and lets shell
        # scripts distinguish "bad command" (2) from "command ran but failed"
        # (1).
        telemetry.record(command, start, ExitCode.CONFIG_ERROR, exc=None)
        telemetry.flush()
        return ExitCode.CONFIG_ERROR

    try:
        app()
    except typer.Exit as exc:
        exit_code = exc.exit_code
    except AutoShipError as exc:
        exit_code = exc.code
        exc_record = exc
        typer.secho("error.prefix", fg=typer.colors.RED, err=True)
        _print_suggestion(i18n, exc)
    except Exception as exc:
        exit_code = ExitCode.USAGE_ERROR
        exc_record = exc
        logger.exception("Unhandled exception")
        typer.secho("unexpected_error.prefix", fg=typer.colors.RED, err=True)
        typer.secho(
            f"\n💡 {'error.suggestion.doctor'}",
            fg=typer.colors.CYAN,
            err=True,
        )
    finally:
        telemetry.record(command, start, exit_code, exc=exc_record)
        telemetry.flush()
        # Close the AuditLogger created by ``main_callback``. It owns SIEM/sink
        # HTTP clients whose connection pools must be released; without this
        # the pools leak for the lifetime of the process. Guarded so a failure
        # here cannot mask the original exit code.
        if _audit_logger is not None:
            try:
                _audit_logger.close()
            except Exception:
                logger.debug("Error closing audit logger", exc_info=True)
            _audit_logger = None

    return exit_code
