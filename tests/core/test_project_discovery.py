"""Tests for :mod:`justagent.core.project_discovery`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from justagent.core.project_discovery import (
    DiscoveredProject,
    DiscoveryConfig,
    ProjectDiscovery,
    ProjectDiscoveryError,
    ProjectType,
)
from justagent.core.project_store import ProjectStore
from justagent.models.project import ManagedProject

# ---------------------------------------------------------------------------
# TestProjectType
# ---------------------------------------------------------------------------


class TestProjectType:
    def test_project_type_values(self) -> None:
        assert ProjectType.PYTHON.value == "python"
        assert ProjectType.NODE.value == "node"
        assert ProjectType.RUST.value == "rust"
        assert ProjectType.GO.value == "go"
        assert ProjectType.GENERIC.value == "generic"

    def test_project_type_is_str(self) -> None:
        assert isinstance(ProjectType.PYTHON, str)
        assert isinstance(ProjectType.GENERIC, str)


# ---------------------------------------------------------------------------
# TestDiscoveredProject
# ---------------------------------------------------------------------------


class TestDiscoveredProject:
    def test_construction(self, tmp_path: Path) -> None:
        project = DiscoveredProject(
            path=tmp_path,
            name="demo",
            project_type=ProjectType.PYTHON,
            markers=["pyproject.toml"],
            has_git=True,
        )
        assert project.name == "demo"
        assert project.project_type == ProjectType.PYTHON
        assert project.has_git is True
        assert project.markers == ["pyproject.toml"]

    def test_frozen(self, tmp_path: Path) -> None:
        project = DiscoveredProject(
            path=tmp_path,
            name="demo",
            project_type=ProjectType.PYTHON,
            markers=[],
            has_git=False,
        )
        with pytest.raises(FrozenInstanceError):
            project.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestDiscoveryConfig
# ---------------------------------------------------------------------------


class TestDiscoveryConfig:
    def test_defaults(self) -> None:
        config = DiscoveryConfig()
        assert config.max_depth == 4
        assert config.max_projects == 200
        assert "pyproject.toml" in config.markers
        assert config.markers["pyproject.toml"] == ProjectType.PYTHON
        assert ".git" in config.ignore_dirs
        assert "node_modules" in config.ignore_dirs

    def test_custom_markers(self) -> None:
        config = DiscoveryConfig(markers={"foo.txt": ProjectType.GENERIC})
        assert config.markers == {"foo.txt": ProjectType.GENERIC}

    def test_custom_ignore_dirs(self) -> None:
        config = DiscoveryConfig(ignore_dirs=["foo"])
        assert config.ignore_dirs == ["foo"]

    def test_defaults_are_independent(self) -> None:
        first = DiscoveryConfig()
        first.ignore_dirs.append("custom")
        second = DiscoveryConfig()
        assert "custom" not in second.ignore_dirs


# ---------------------------------------------------------------------------
# TestDetectType
# ---------------------------------------------------------------------------


class TestDetectType:
    def test_detect_python(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        ptype, markers, has_git = ProjectDiscovery().detect_type(tmp_path)
        assert ptype == ProjectType.PYTHON
        assert "pyproject.toml" in markers
        assert has_git is False

    def test_detect_node(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        ptype, markers, has_git = ProjectDiscovery().detect_type(tmp_path)
        assert ptype == ProjectType.NODE
        assert "package.json" in markers
        assert has_git is False

    def test_detect_rust(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        ptype, _, _ = ProjectDiscovery().detect_type(tmp_path)
        assert ptype == ProjectType.RUST

    def test_detect_go(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
        ptype, _, _ = ProjectDiscovery().detect_type(tmp_path)
        assert ptype == ProjectType.GO

    def test_detect_generic_makefile(self, tmp_path: Path) -> None:
        (tmp_path / "Makefile").write_text("all:\n", encoding="utf-8")
        ptype, markers, has_git = ProjectDiscovery().detect_type(tmp_path)
        assert ptype == ProjectType.GENERIC
        assert "Makefile" in markers
        assert has_git is False

    def test_detect_git_only(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        ptype, markers, has_git = ProjectDiscovery().detect_type(tmp_path)
        assert ptype == ProjectType.GENERIC
        assert has_git is True
        assert ".git" in markers

    def test_detect_empty_dir(self, tmp_path: Path) -> None:
        ptype, markers, has_git = ProjectDiscovery().detect_type(tmp_path)
        assert ptype == ProjectType.GENERIC
        assert markers == []
        assert has_git is False

    def test_detect_multiple_markers_python_wins(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        (tmp_path / "Makefile").write_text("all:\n", encoding="utf-8")
        ptype, markers, has_git = ProjectDiscovery().detect_type(tmp_path)
        assert ptype == ProjectType.PYTHON
        assert "pyproject.toml" in markers
        assert "Makefile" in markers
        assert has_git is False

    def test_detect_python_with_git(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        (tmp_path / ".git").mkdir()
        ptype, markers, has_git = ProjectDiscovery().detect_type(tmp_path)
        assert ptype == ProjectType.PYTHON
        assert has_git is True
        assert ".git" in markers


# ---------------------------------------------------------------------------
# TestDiscover
# ---------------------------------------------------------------------------


class TestDiscover:
    def test_discover_single_project(self, tmp_path: Path) -> None:
        (tmp_path / "myproj").mkdir()
        (tmp_path / "myproj" / "pyproject.toml").write_text(
            "[project]\nname='x'\n", encoding="utf-8"
        )
        results = ProjectDiscovery().discover(tmp_path)
        assert len(results) == 1
        assert results[0].project_type == ProjectType.PYTHON
        assert results[0].name == "myproj"

    def test_discover_nested_projects_with_git(self, tmp_path: Path) -> None:
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        nested = parent / "nested"
        nested.mkdir()
        (nested / ".git").mkdir()
        (nested / "package.json").write_text("{}", encoding="utf-8")
        results = ProjectDiscovery().discover(tmp_path)
        names = {r.name for r in results}
        assert "parent" in names
        assert "nested" in names

    def test_does_not_descend_into_project_subdirs(self, tmp_path: Path) -> None:
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        sub = parent / "sub"
        sub.mkdir()
        (sub / "package.json").write_text("{}", encoding="utf-8")
        results = ProjectDiscovery().discover(tmp_path)
        names = {r.name for r in results}
        assert "parent" in names
        assert "sub" not in names

    def test_ignored_dirs_skipped(self, tmp_path: Path) -> None:
        proj = tmp_path / "realproj"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "pkg").mkdir()
        (node_modules / "pkg" / "package.json").write_text("{}", encoding="utf-8")
        results = ProjectDiscovery().discover(tmp_path)
        names = {r.name for r in results}
        assert "realproj" in names
        assert "pkg" not in names

    def test_max_depth_respected(self, tmp_path: Path) -> None:
        # Branch found at depth 2.
        a = tmp_path / "a"
        a.mkdir()
        b = a / "b"
        b.mkdir()
        (b / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        # Branch found at depth 3 (should be skipped with max_depth=2).
        p = tmp_path / "p"
        p.mkdir()
        q = p / "q"
        q.mkdir()
        r = q / "r"
        r.mkdir()
        (r / "pyproject.toml").write_text("[project]\nname='y'\n", encoding="utf-8")
        discovery = ProjectDiscovery(DiscoveryConfig(max_depth=2))
        results = discovery.discover(tmp_path)
        names = {found.name for found in results}
        assert "b" in names
        assert "r" not in names

    def test_max_projects_limit(self, tmp_path: Path) -> None:
        for i in range(5):
            d = tmp_path / f"proj{i}"
            d.mkdir()
            (d / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        discovery = ProjectDiscovery(DiscoveryConfig(max_projects=3))
        results = discovery.discover(tmp_path)
        assert len(results) == 3

    def test_no_projects_found(self, tmp_path: Path) -> None:
        (tmp_path / "empty1").mkdir()
        (tmp_path / "empty2").mkdir()
        results = ProjectDiscovery().discover(tmp_path)
        assert results == []

    def test_sorted_by_path(self, tmp_path: Path) -> None:
        z_dir = tmp_path / "zproj"
        z_dir.mkdir()
        (z_dir / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        a_dir = tmp_path / "aproj"
        a_dir.mkdir()
        (a_dir / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        results = ProjectDiscovery().discover(tmp_path)
        assert len(results) == 2
        assert results[0].path == a_dir
        assert results[1].path == z_dir

    def test_discover_root_itself_is_project(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        results = ProjectDiscovery().discover(tmp_path)
        assert len(results) == 1
        assert results[0].path == tmp_path


# ---------------------------------------------------------------------------
# TestDiscoverAndRegister
# ---------------------------------------------------------------------------


class TestDiscoverAndRegister:
    def test_registers_discovered_projects(self, tmp_path: Path) -> None:
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        store = ProjectStore(store_path=tmp_path / "projects.json")
        added = ProjectDiscovery().discover_and_register(tmp_path, store)
        assert len(added) == 1
        assert added[0].name == "myproj"
        fetched = store.get("myproj")
        assert fetched is not None
        assert Path(fetched.path) == proj

    def test_dry_run_does_not_persist(self, tmp_path: Path) -> None:
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        store = ProjectStore(store_path=tmp_path / "projects.json")
        added = ProjectDiscovery().discover_and_register(tmp_path, store, dry_run=True)
        assert len(added) == 1
        assert store.list_all() == []

    def test_tags_applied(self, tmp_path: Path) -> None:
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        store = ProjectStore(store_path=tmp_path / "projects.json")
        added = ProjectDiscovery().discover_and_register(tmp_path, store, tags=["web", "py"])
        assert added[0].tags == ["web", "py"]
        fetched = store.get("myproj")
        assert fetched is not None
        assert fetched.tags == ["web", "py"]

    def test_existing_project_updated_in_place(self, tmp_path: Path) -> None:
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        store = ProjectStore(store_path=tmp_path / "projects.json")
        store.add(
            ManagedProject(
                name="myproj",
                path="/old/path",
                added_at=1000.0,
                tags=["old"],
            )
        )
        added = ProjectDiscovery().discover_and_register(tmp_path, store, tags=["new"])
        # Existing project is not "newly added".
        assert added == []
        fetched = store.get("myproj")
        assert fetched is not None
        assert Path(fetched.path) == proj
        assert fetched.tags == ["new"]
        assert fetched.added_at == 1000.0
        # Not duplicated.
        assert len(store.list_all()) == 1


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nonexistent_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectDiscoveryError):
            ProjectDiscovery().discover(tmp_path / "does-not-exist")

    def test_root_is_file_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        file_path.write_text("hi", encoding="utf-8")
        with pytest.raises(ProjectDiscoveryError):
            ProjectDiscovery().discover(file_path)

    def test_permission_denied_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_iterdir = Path.iterdir
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        good = tmp_path / "good"
        good.mkdir()
        (good / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

        def fake_iterdir(self: Path) -> object:
            if self == blocked:
                raise PermissionError("denied")
            return real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", fake_iterdir)
        results = ProjectDiscovery().discover(tmp_path)
        names = [r.name for r in results]
        assert "good" in names
        assert "blocked" not in names

    def test_symlinks_skipped(self, tmp_path: Path) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(tmp_path)
        results = ProjectDiscovery().discover(tmp_path)
        names = [r.name for r in results]
        assert "proj" in names
        assert "link" not in names
        assert len(results) == 1
