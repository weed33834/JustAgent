"""Tests for configuration template rendering."""

from __future__ import annotations

from autoship.utils.template import render_default_config


def test_render_default_config_contains_project_type() -> None:
    rendered = render_default_config("python")
    assert 'project_type = "python"' in rendered
    assert "[model]" in rendered
    assert "[clean]" in rendered
    assert "[commit]" in rendered


def test_render_default_config_is_valid_toml(project_root) -> None:
    rendered = render_default_config("node")
    config_file = project_root / ".autoship.toml"
    config_file.write_text(rendered, encoding="utf-8")
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with config_file.open("rb") as f:
        data = tomllib.load(f)
    assert data["project_type"] == "node"
    assert data["schema_version"] == 1
