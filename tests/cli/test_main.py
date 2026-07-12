"""Tests for the main CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoship.cli import main
from autoship.exceptions import ConfigError, ExitCode


class FakeContext:
    """Minimal stand-in for a Typer context."""

    def __init__(self) -> None:
        self.obj: dict = {}

    def ensure_object(self, typ: type) -> None:
        if not self.obj:
            self.obj = {}


def test_main_callback_loads_config(tmp_path: Path) -> None:
    ctx = FakeContext()
    config_path = tmp_path / "autoship.toml"
    config_path.write_text("[model]\ndefault_tier = 1\n")

    main.main_callback(ctx, config_path=config_path)

    assert "config" in ctx.obj
    assert ctx.obj["config"].model.default_tier == 1
    assert "audit_logger" in ctx.obj


def test_main_callback_without_config() -> None:
    ctx = FakeContext()
    main.main_callback(ctx, config_path=None)
    assert "config" in ctx.obj
    assert "audit_logger" in ctx.obj


def test_cli_entrypoint_handles_autoship_error() -> None:
    error = ConfigError("bad config")
    mock_app = MagicMock(side_effect=error)
    with (
        patch.object(sys, "argv", ["autoship", "fix"]),
        patch.object(main, "app", mock_app),
    ):
        exit_code = main.cli_entrypoint()
    assert exit_code == error.code


def test_cli_entrypoint_handles_unexpected_error() -> None:
    mock_app = MagicMock(side_effect=RuntimeError("boom"))
    with (
        patch.object(sys, "argv", ["autoship", "fix"]),
        patch.object(main, "app", mock_app),
    ):
        exit_code = main.cli_entrypoint()
    assert exit_code == ExitCode.USAGE_ERROR


def test_cli_entrypoint_help_does_not_traceback(capsys) -> None:
    """--help must exit cleanly without an exception traceback."""
    with patch.object(sys, "argv", ["autoship", "--help"]), pytest.raises(SystemExit) as exc_info:
        main.cli_entrypoint()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Usage:" in captured.out


def test_cli_entrypoint_unknown_command_gives_friendly_message(capsys) -> None:
    """An unknown command must print a friendly hint instead of crashing."""
    with patch.object(sys, "argv", ["autoship", "not-a-command"]):
        exit_code = main.cli_entrypoint()
    # Unknown commands exit with CONFIG_ERROR (2) rather than
    # USAGE_ERROR (1) so shell scripts can distinguish "user misuse" from
    # "autoship rejected the invocation".
    assert exit_code == ExitCode.CONFIG_ERROR
    captured = capsys.readouterr()
    assert "not-a-command" in captured.err
    assert "--help" in captured.err


def test_cli_entrypoint_config_error_shows_next_step_suggestion(capsys) -> None:
    """ConfigError should be followed by an actionable next-step suggestion."""
    error = ConfigError("missing config")
    mock_app = MagicMock(side_effect=error)
    with (
        patch.object(sys, "argv", ["autoship", "fix"]),
        patch.object(main, "app", mock_app),
    ):
        exit_code = main.cli_entrypoint()
    assert exit_code == error.code
    captured = capsys.readouterr()
    assert "autoship init" in captured.err
