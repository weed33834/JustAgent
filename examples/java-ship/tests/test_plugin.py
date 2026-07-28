"""Tests for the java-ship plugin."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from myagent_java_ship.plugin import JavaShipPlugin, plugin

from justagent.core.context import CommandContext


@pytest.fixture
def java_plugin() -> JavaShipPlugin:
    return JavaShipPlugin()


def test_module_plugin_instance_exists() -> None:
    assert isinstance(plugin, JavaShipPlugin)


def test_constants_match_spec() -> None:
    assert JavaShipPlugin.ARTIFACTS == ("target/classes/", "target/test-classes/", "*.class")
    assert JavaShipPlugin.TEST_COMMAND == "mvn test"
    assert JavaShipPlugin.LINT_COMMAND == "mvn checkstyle:check"


def test_hooks_are_callable(java_plugin: JavaShipPlugin, tmp_path: Path) -> None:
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    context.trace_id = "test-trace"
    java_plugin.pre_clean(context)
    java_plugin.post_clean(context)
    java_plugin.pre_verify(context)


def test_pre_clean_detects_classes_dirs(java_plugin: JavaShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "target" / "classes").mkdir(parents=True)
    (tmp_path / "target" / "test-classes").mkdir(parents=True)
    (tmp_path / "Loose.class").write_bytes(b"\xca\xfe\xba\xbe")
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    context.trace_id = "test-trace"
    java_plugin.pre_clean(context)
    found = java_plugin._found["test-trace"]
    assert "target/classes/" in found
    assert "target/test-classes/" in found
    assert "*.class" in found


def test_detect_test_command_defaults_to_mvn(java_plugin: JavaShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    assert java_plugin._detect_test_command(tmp_path) == "mvn test"


def test_detect_test_command_picks_gradle(java_plugin: JavaShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")
    assert java_plugin._detect_test_command(tmp_path) == "gradle test"


def test_detect_test_command_picks_gradle_kotlin(
    java_plugin: JavaShipPlugin, tmp_path: Path
) -> None:
    (tmp_path / "build.gradle.kts").write_text("plugins { java }\n", encoding="utf-8")
    assert java_plugin._detect_test_command(tmp_path) == "gradle test"


def test_is_java_project_recognises_pom(java_plugin: JavaShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    assert java_plugin._is_java_project(tmp_path) is True


def test_is_java_project_recognises_gradle(java_plugin: JavaShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "build.gradle.kts").write_text("plugins { java }\n", encoding="utf-8")
    assert java_plugin._is_java_project(tmp_path) is True


def test_is_java_project_false_for_empty_dir(java_plugin: JavaShipPlugin, tmp_path: Path) -> None:
    assert java_plugin._is_java_project(tmp_path) is False


def test_pre_verify_suggests_gradle_test(java_plugin: JavaShipPlugin, tmp_path: Path) -> None:
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    java_plugin.pre_verify(context)  # smoke: should not raise


def test_pre_verify_noop_without_build_file(java_plugin: JavaShipPlugin, tmp_path: Path) -> None:
    context = MagicMock(spec=CommandContext)
    context.project_root = tmp_path
    context.extras = {}
    context.dry_run = False
    java_plugin.pre_verify(context)  # smoke: should not raise


def test_register_factory_returns_plugin() -> None:
    from myagent_java_ship.plugin import register

    assert isinstance(register(), JavaShipPlugin)
