"""Tests for the node-ship plugin."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from myagent_node_ship.plugin import NodeShipPlugin, plugin

from myagent.core.context import CommandContext


@pytest.fixture
def node_plugin() -> NodeShipPlugin:
    return NodeShipPlugin()


def test_module_plugin_instance_exists() -> None:
    assert isinstance(plugin, NodeShipPlugin)


def test_constants_match_spec() -> None:
    assert NodeShipPlugin.ARTIFACTS == ("dist/", "build/", "*.tsbuildinfo", "coverage/")
    assert NodeShipPlugin.TEST_COMMAND == "npm test"
    assert NodeShipPlugin.LINT_COMMAND == "npm run lint"


def test_hooks_are_callable(node_plugin: NodeShipPlugin, tmp_path: Path) -> None:
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    context.trace_id = "test-trace"
    node_plugin.pre_clean(context)
    node_plugin.post_clean(context)
    node_plugin.pre_verify(context)


def test_pre_clean_detects_dist_and_coverage(node_plugin: NodeShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "index.js").write_text("console.log(1)\n", encoding="utf-8")
    (tmp_path / "coverage").mkdir()
    (tmp_path / "app.tsbuildinfo").write_text("{}", encoding="utf-8")
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    context.trace_id = "test-trace"
    node_plugin.pre_clean(context)
    found = node_plugin._found["test-trace"]
    assert "dist/" in found
    assert "coverage/" in found
    assert "*.tsbuildinfo" in found


def test_detect_test_command_defaults_to_npm(node_plugin: NodeShipPlugin, tmp_path: Path) -> None:
    assert node_plugin._detect_test_command(tmp_path) == "npm test"


def test_detect_test_command_picks_pnpm(node_plugin: NodeShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: 1\n", encoding="utf-8")
    assert node_plugin._detect_test_command(tmp_path) == "pnpm test"


def test_detect_test_command_picks_yarn(node_plugin: NodeShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "yarn.lock").write_text("# yarn\n", encoding="utf-8")
    assert node_plugin._detect_test_command(tmp_path) == "yarn test"


def test_pre_verify_suggests_pnpm_test(node_plugin: NodeShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name": "example"}\n', encoding="utf-8")
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: 1\n", encoding="utf-8")
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    node_plugin.pre_verify(context)  # smoke: should not raise


def test_pre_verify_noop_without_package_json(node_plugin: NodeShipPlugin, tmp_path: Path) -> None:
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    node_plugin.pre_verify(context)  # smoke: should not raise


def test_register_factory_returns_plugin() -> None:
    from myagent_node_ship.plugin import register

    assert isinstance(register(), NodeShipPlugin)
