"""Tests for project type detection."""

from __future__ import annotations

from autoship.utils.project_detector import PROJECT_MARKERS, detect_project_type


def test_detect_python(project_root) -> None:
    (project_root / "pyproject.toml").write_text("", encoding="utf-8")
    assert detect_project_type(project_root) == "python"


def test_detect_node(project_root) -> None:
    (project_root / "package.json").write_text("{}", encoding="utf-8")
    assert detect_project_type(project_root) == "node"


def test_detect_rust(project_root) -> None:
    (project_root / "Cargo.toml").write_text("", encoding="utf-8")
    assert detect_project_type(project_root) == "rust"


def test_detect_generic(project_root) -> None:
    assert detect_project_type(project_root) == "generic"


def test_project_markers_are_nonempty() -> None:
    for ptype, markers in PROJECT_MARKERS.items():
        assert markers
        assert ptype in ("python", "node", "rust", "go", "java")
