"""Tests for CommandContext."""

from __future__ import annotations

from autoship.core.context import CommandContext
from autoship.models.config import AppConfig


def test_command_context_defaults(app_config: AppConfig) -> None:
    ctx = CommandContext(
        command="init",
        project_root=app_config.project_root,
        config=app_config,
    )
    assert ctx.command == "init"
    assert ctx.dry_run is False
    assert ctx.yes is False
    assert ctx.verbose is False
    assert ctx.extras == {}


def test_command_context_extras(app_config: AppConfig) -> None:
    ctx = CommandContext(
        command="verify",
        project_root=app_config.project_root,
        config=app_config,
        extras={"command": "pytest"},
    )
    assert ctx.extras["command"] == "pytest"


def test_command_context_is_frozen(app_config: AppConfig) -> None:
    ctx = CommandContext(
        command="init",
        project_root=app_config.project_root,
        config=app_config,
    )
    try:
        ctx.command = "other"
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "cannot assign" in str(exc).lower()
    else:
        raise AssertionError("CommandContext should be immutable")
