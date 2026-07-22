"""Tests for the go-ship plugin."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from myagent_go_ship.plugin import GoShipPlugin

from myagent.core.context import CommandContext


@pytest.fixture
def go_plugin() -> GoShipPlugin:
    return GoShipPlugin()


def test_constants_match_spec() -> None:
    assert GoShipPlugin.ARTIFACTS == ("bin/", "*.test", "*.out")
    assert GoShipPlugin.TEST_COMMAND == "go test ./..."
    assert GoShipPlugin.LINT_COMMAND == "golangci-lint run"


def test_hooks_are_callable(go_plugin: GoShipPlugin, tmp_path: Path) -> None:
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    context.trace_id = "test-trace"
    go_plugin.pre_clean(context)
    go_plugin.post_clean(context)
    go_plugin.pre_verify(context)


def test_pre_clean_detects_bin_dir(go_plugin: GoShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "app").write_bytes(b"\x7fELF")
    (tmp_path / "foo.test").write_text("go test binary\n", encoding="utf-8")
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    context.trace_id = "test-trace"
    go_plugin.pre_clean(context)
    found = go_plugin._found["test-trace"]
    assert "bin/" in found
    assert "*.test" in found


def test_post_clean_reports_removed(go_plugin: GoShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "foo.out").write_text("coverage\n", encoding="utf-8")
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    context.trace_id = "test-trace"
    go_plugin.pre_clean(context)
    (tmp_path / "foo.out").unlink()  # simulate removal between the two hooks
    go_plugin.post_clean(context)
    assert go_plugin._found.get("test-trace") is None


def test_pre_verify_suggests_go_test(go_plugin: GoShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example\n\ngo 1.21\n", encoding="utf-8")
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    go_plugin.pre_verify(context)  # smoke: should not raise


def test_pre_verify_noop_without_go_mod(go_plugin: GoShipPlugin, tmp_path: Path) -> None:
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    go_plugin.pre_verify(context)  # smoke: should not raise


def test_register_factory_returns_plugin() -> None:
    from myagent_go_ship.plugin import register

    assert isinstance(register(), GoShipPlugin)
