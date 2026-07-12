"""Tests for ToolChain adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from autoship.adapters.tool_adapter import ToolChain


def test_toolchain_dry_run_returns_zero_exit(project_root: Path) -> None:
    toolchain = ToolChain(["black"], project_root, dry_run=True)
    result = toolchain._run(["black", "."])
    assert result.returncode == 0


def test_toolchain_preview_returns_empty_when_no_black(project_root: Path) -> None:
    toolchain = ToolChain(["black"], project_root)
    with patch("shutil.which", return_value=None):
        preview = toolchain.preview([Path(".")])
    assert preview == ""


def test_toolchain_apply_skips_missing_tools(project_root: Path) -> None:
    toolchain = ToolChain(["autoflake", "black"], project_root)
    with patch("shutil.which", return_value=None):
        toolchain.apply([Path(".")])


def test_toolchain_verbose_prints_command(project_root: Path, capsys) -> None:
    toolchain = ToolChain(["black"], project_root, dry_run=True, verbose=True)
    toolchain._run(["black", "."])
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out


def test_toolchain_preview_uses_black(project_root: Path) -> None:
    toolchain = ToolChain(["black"], project_root)
    with (
        patch("shutil.which", return_value="/usr/bin/black"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "--- diff ---"
        preview = toolchain.preview([Path(".")])
    assert preview == "--- diff ---"


def test_toolchain_apply_runs_autoflake_and_black(project_root: Path) -> None:
    toolchain = ToolChain(["autoflake", "black"], project_root)
    with (
        patch("shutil.which", return_value="/usr/bin/black"),
        patch("subprocess.run") as mock_run,
    ):
        toolchain.apply([Path(".")])
    assert mock_run.call_count == 2


def test_toolchain_preview_raises_on_tool_failure(project_root: Path) -> None:
    toolchain = ToolChain(["black"], project_root)
    with (
        patch("shutil.which", return_value="/usr/bin/black"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["black", "--diff", "."], stderr="syntax error"
        )
        with pytest.raises(subprocess.CalledProcessError):
            toolchain.preview([Path(".")])


def test_toolchain_apply_raises_on_tool_failure(project_root: Path) -> None:
    toolchain = ToolChain(["black"], project_root)
    with (
        patch("shutil.which", return_value="/usr/bin/black"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.side_effect = subprocess.CalledProcessError(returncode=123, cmd=["black", "."])
        with pytest.raises(subprocess.CalledProcessError):
            toolchain.apply([Path(".")])
