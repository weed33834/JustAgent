"""Tests for the project-guard plugin."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from myagent_custom_plugin.plugin import ProjectGuardPlugin

from justagent.core.context import CommandContext
from justagent.exceptions import VerifyError


@pytest.fixture
def plugin() -> ProjectGuardPlugin:
    return ProjectGuardPlugin()


def test_on_error_returns_suggestion_when_fix_enabled(
    plugin: ProjectGuardPlugin, tmp_path: Path
) -> None:
    context = MagicMock(spec=CommandContext)
    context.extras = {"fix": True}
    suggestion = plugin.on_error(context, VerifyError("failed"))
    assert suggestion is not None
    assert "justagent clean" in suggestion.description


def test_on_error_returns_none_when_fix_disabled(
    plugin: ProjectGuardPlugin, tmp_path: Path
) -> None:
    context = MagicMock(spec=CommandContext)
    context.extras = {"fix": False}
    assert plugin.on_error(context, VerifyError("failed")) is None


def test_staged_python_files_filters_non_python_files(
    plugin: ProjectGuardPlugin, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello\n", encoding="utf-8")
    staged = plugin._staged_python_files(tmp_path)
    assert staged == []
