"""Smoke tests: every registered subcommand's ``--help`` must exit cleanly.

This catches registration errors, broken imports, and typer help-text binding
issues that would only surface when a user runs ``autoship <cmd> --help``.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from autoship.cli.main import _KNOWN_COMMANDS, app

runner = CliRunner()

# Every top-level command discovered at registration time. Using the snapshot
# in ``_KNOWN_COMMANDS`` keeps the test deterministic even if ``app`` is later
# patched by other tests in the same process.
_TOP_LEVEL_COMMANDS = sorted(_KNOWN_COMMANDS)


@pytest.mark.parametrize("command", _TOP_LEVEL_COMMANDS)
def test_subcommand_help_exits_zero(command: str) -> None:
    """``autoship <command> --help`` exits 0 with no traceback."""
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, f"{command} --help exited {result.exit_code}:\n{result.output}"
    combined = result.output + (result.stderr or "")
    assert "Traceback (most recent call last)" not in combined, (
        f"{command} --help produced a traceback:\n{combined}"
    )


def test_top_level_help_exits_zero() -> None:
    """``autoship --help`` itself exits 0 with no traceback."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    combined = result.output + (result.stderr or "")
    assert "Traceback (most recent call last)" not in combined
