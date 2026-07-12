"""Tests for the built-in typecheck plugin."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoship.core.context import CommandContext
from autoship.exceptions import VerifyError
from autoship.models.config import AppConfig
from autoship.plugins.typecheck import TypecheckPlugin


@pytest.fixture
def plugin() -> TypecheckPlugin:
    return TypecheckPlugin()


def test_pre_commit_skips_when_pyright_missing(
    plugin: TypecheckPlugin, app_config: AppConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    context = CommandContext(
        command="commit",
        project_root=tmp_path,
        config=app_config,
    )
    with patch("autoship.plugins.typecheck._tool_executable", return_value=None):
        plugin.pre_commit(context)
    assert "pyright not found on PATH; skipping type check" in caplog.text


def test_pre_commit_passes_when_pyright_succeeds(
    plugin: TypecheckPlugin, app_config: AppConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    context = CommandContext(
        command="commit",
        project_root=tmp_path,
        config=app_config,
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""
    with (
        patch("autoship.plugins.typecheck._tool_executable", return_value="/usr/bin/pyright"),
        patch("subprocess.run", return_value=mock_result) as mock_run,
    ):
        plugin.pre_commit(context)
    mock_run.assert_called_once_with(
        ["/usr/bin/pyright"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert "Type check passed" in caplog.text


def test_pre_commit_fails_when_pyright_reports_errors(
    plugin: TypecheckPlugin, app_config: AppConfig, tmp_path: Path
) -> None:
    context = CommandContext(
        command="commit",
        project_root=tmp_path,
        config=app_config,
    )
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = "1 error, 0 warnings"
    mock_result.stderr = ""
    with (
        patch("autoship.plugins.typecheck._tool_executable", return_value="/usr/bin/pyright"),
        patch("subprocess.run", return_value=mock_result),
        pytest.raises(VerifyError, match="Type check failed"),
    ):
        plugin.pre_commit(context)


def test_pre_commit_dry_run_does_not_execute(
    plugin: TypecheckPlugin, app_config: AppConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    context = CommandContext(
        command="commit",
        project_root=tmp_path,
        config=app_config,
        dry_run=True,
    )
    with (
        patch("autoship.plugins.typecheck._tool_executable", return_value="/usr/bin/pyright"),
        patch("subprocess.run") as mock_run,
    ):
        plugin.pre_commit(context)
    mock_run.assert_not_called()
    assert "[dry-run] Would run pyright type check" in caplog.text
