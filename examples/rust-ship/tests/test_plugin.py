"""Tests for the rust-ship plugin."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from autoship_rust_ship.plugin import RustShipPlugin

from autoship.core.context import CommandContext


@pytest.fixture
def rust_plugin() -> RustShipPlugin:
    return RustShipPlugin()


def test_constants_match_spec() -> None:
    assert RustShipPlugin.ARTIFACTS == ("target/debug/", "target/release/")
    assert RustShipPlugin.TEST_COMMAND == "cargo test"
    assert RustShipPlugin.LINT_COMMAND == "cargo clippy"


def test_target_root_is_not_an_artifact() -> None:
    """The bare ``target/`` directory must not be matched wholesale."""
    assert not any(item.rstrip("/") == "target" for item in RustShipPlugin.ARTIFACTS)


def test_hooks_are_callable(rust_plugin: RustShipPlugin, tmp_path: Path) -> None:
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    context.trace_id = "test-trace"
    rust_plugin.pre_clean(context)
    rust_plugin.post_clean(context)
    rust_plugin.pre_verify(context)


def test_pre_clean_detects_profile_dirs(rust_plugin: RustShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "target" / "debug").mkdir(parents=True)
    (tmp_path / "target" / "release").mkdir(parents=True)
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    context.trace_id = "test-trace"
    rust_plugin.pre_clean(context)
    found = rust_plugin._found["test-trace"]
    assert "target/debug/" in found
    assert "target/release/" in found


def test_post_clean_reports_removed(rust_plugin: RustShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "target" / "debug").mkdir(parents=True)
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    context.trace_id = "test-trace"
    rust_plugin.pre_clean(context)
    # simulate `cargo clean` running between the two hooks
    shutil.rmtree(tmp_path / "target")
    rust_plugin.post_clean(context)
    assert rust_plugin._found.get("test-trace") is None


def test_pre_verify_suggests_cargo_test(rust_plugin: RustShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "example"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    rust_plugin.pre_verify(context)  # smoke: should not raise


def test_pre_verify_noop_without_cargo_toml(rust_plugin: RustShipPlugin, tmp_path: Path) -> None:
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    rust_plugin.pre_verify(context)  # smoke: should not raise


def test_register_factory_returns_plugin() -> None:
    from autoship_rust_ship.plugin import register

    assert isinstance(register(), RustShipPlugin)
