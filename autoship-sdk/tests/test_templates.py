"""Tests for autoship_sdk.templates."""

from __future__ import annotations

import pytest

from autoship_sdk import TemplateError, create_plugin


def test_create_plugin_scaffolds_project(tmp_path) -> None:
    project_root = create_plugin(
        tmp_path / "my-plugin",
        plugin_name="security-scan",
        description="Security scan plugin",
    )
    assert project_root.exists()
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "README.md").exists()
    assert (project_root / "src" / "security_scan" / "plugin.py").exists()
    assert (project_root / "src" / "security_scan" / "__init__.py").exists()


def test_create_plugin_rejects_invalid_name(tmp_path) -> None:
    with pytest.raises(TemplateError):
        create_plugin(tmp_path / "bad", plugin_name="123invalid")


def test_create_plugin_rejects_non_empty_target(tmp_path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    (target / "file.txt").write_text("x")
    with pytest.raises(TemplateError):
        create_plugin(target, plugin_name="my-plugin")


def test_create_plugin_renders_metadata(tmp_path) -> None:
    project_root = create_plugin(
        tmp_path / "my-plugin",
        plugin_name="lint-helper",
        description="Lint helper plugin",
        repository_url="https://example.com/repo",
    )
    pyproject = (project_root / "pyproject.toml").read_text()
    assert 'name = "autoship-lint-helper"' in pyproject
    assert 'description = "Lint helper plugin"' in pyproject
    assert 'Repository = "https://example.com/repo"' in pyproject
    plugin_code = (project_root / "src" / "lint_helper" / "plugin.py").read_text()
    assert "class LintHelper(Plugin):" in plugin_code
