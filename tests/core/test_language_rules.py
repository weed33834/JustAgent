"""Tests for the language_rules module."""

from __future__ import annotations

from pathlib import Path

import pytest

from myagent.core.language_rules import (
    RULES,
    apply_artifact_removal,
    plan_artifact_removal,
    primary_language,
    rules_for_project,
)


def test_rules_table_covers_expected_languages() -> None:
    assert set(RULES.keys()) == {"go", "rust", "node", "java", "python"}


def test_rust_rule_does_not_remove_target_wholesale() -> None:
    """Cargo uses target/ as a cache; MyAgent must not list it as artifact_dirs."""
    assert "target" not in RULES["rust"].artifact_dirs
    assert "target/debug" not in RULES["rust"].artifact_dirs
    assert "target/release" not in RULES["rust"].artifact_dirs


def test_node_rule_does_not_remove_node_modules() -> None:
    assert "node_modules" not in RULES["node"].artifact_dirs


def test_rules_for_project_returns_empty_for_empty_dir(tmp_path: Path) -> None:
    assert rules_for_project(tmp_path) == []
    assert primary_language(tmp_path) is None


def test_rules_for_project_returns_go_when_go_mod_present(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/test\n", encoding="utf-8")
    matches = rules_for_project(tmp_path)
    assert len(matches) == 1
    assert matches[0].language == "go"
    assert primary_language(tmp_path) == "go"


def test_rules_for_project_returns_multiple_for_polyglot(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/test\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name": "frontend"}\n', encoding="utf-8")
    matches = rules_for_project(tmp_path)
    languages = [r.language for r in matches]
    assert "go" in languages
    assert "node" in languages


def test_plan_artifact_removal_empty_project(tmp_path: Path) -> None:
    plan = plan_artifact_removal(tmp_path)
    assert plan.total == 0
    assert plan.by_language == {}


def test_plan_artifact_removal_finds_go_bin(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/test\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "compiled-binary").write_bytes(b"\x7fELF")
    (tmp_path / "main.test").write_bytes(b"test binary")
    (tmp_path / "trace.out").write_text("trace", encoding="utf-8")

    plan = plan_artifact_removal(tmp_path)
    assert "go" in plan.by_language
    paths = plan.by_language["go"]
    # The bin/ directory and the two loose globs should be picked up.
    assert bin_dir.resolve() in [p.resolve() for p in paths]
    assert (tmp_path / "main.test").resolve() in [p.resolve() for p in paths]
    assert (tmp_path / "trace.out").resolve() in [p.resolve() for p in paths]


def test_plan_artifact_removal_finds_node_dist_and_tsbuildinfo(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name": "frontend"}\n', encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_text("// bundle\n", encoding="utf-8")
    (tmp_path / "app.tsbuildinfo").write_text("{}", encoding="utf-8")

    plan = plan_artifact_removal(tmp_path)
    assert "node" in plan.by_language
    paths = plan.by_language["node"]
    assert (tmp_path / "dist").resolve() in [p.resolve() for p in paths]
    assert (tmp_path / "app.tsbuildinfo").resolve() in [p.resolve() for p in paths]


def test_plan_artifact_removal_finds_java_class_files(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    (tmp_path / "target" / "classes").mkdir(parents=True)
    (tmp_path / "target" / "classes" / "Main.class").write_bytes(b"\xca\xfe\xba\xbe")
    (tmp_path / "loose.class").write_bytes(b"\xca\xfe\xba\xbe")

    plan = plan_artifact_removal(tmp_path)
    assert "java" in plan.by_language
    paths = plan.by_language["java"]
    # target/classes is listed as a directory; loose .class files are listed individually.
    assert any(p.name == "classes" for p in paths)
    assert (tmp_path / "loose.class").resolve() in [p.resolve() for p in paths]


def test_plan_skips_node_modules_contents(tmp_path: Path) -> None:
    """node_modules contents must NEVER be flagged for removal."""
    (tmp_path / "package.json").write_text('{"name": "frontend"}\n', encoding="utf-8")
    nm = tmp_path / "node_modules" / "some-pkg"
    nm.mkdir(parents=True)
    (nm / "build.tsbuildinfo").write_text("{}", encoding="utf-8")

    plan = plan_artifact_removal(tmp_path)
    # The tsbuildinfo inside node_modules must NOT be in the removal list.
    for paths in plan.by_language.values():
        for p in paths:
            assert "node_modules" not in p.parts


def test_apply_artifact_removal_dry_run_does_not_delete(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "binary").write_bytes(b"x")
    plan = plan_artifact_removal(tmp_path)

    removed = apply_artifact_removal(plan, dry_run=True)
    assert removed == 1
    # Nothing was actually deleted.
    assert (tmp_path / "bin" / "binary").exists()


def test_apply_artifact_removal_deletes_files_and_dirs(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "binary").write_bytes(b"x")
    (tmp_path / "main.test").write_bytes(b"x")
    plan = plan_artifact_removal(tmp_path)

    removed = apply_artifact_removal(plan, dry_run=False)
    assert removed == 2
    assert not (tmp_path / "bin").exists()
    assert not (tmp_path / "main.test").exists()


def test_apply_artifact_removal_handles_missing_paths_gracefully(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    plan = plan_artifact_removal(tmp_path)
    # Inject a path that doesn't exist — apply should silently skip.
    plan.by_language["go"].append(tmp_path / "does-not-exist")
    removed = apply_artifact_removal(plan, dry_run=False)
    assert removed == 0


def test_language_rule_is_immutable() -> None:
    rule = RULES["go"]
    with pytest.raises(Exception):  # noqa: B017 — frozen dataclass
        rule.language = "python"  # type: ignore[misc]


def test_artifact_removal_plan_total_counts_all_languages(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name": "x"}\n', encoding="utf-8")
    (tmp_path / "bin").mkdir()
    (tmp_path / "dist").mkdir()

    plan = plan_artifact_removal(tmp_path)
    assert plan.total >= 2  # at least bin/ (go) + dist/ (node)


def test_plan_skips_symlink_escape(tmp_path: Path) -> None:
    """A symlink that points outside project_root must be skipped, not removed."""
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    (outside / "dangerous").write_text("do not touch", encoding="utf-8")

    # Symlink bin -> ../outside-target
    (tmp_path / "bin").symlink_to(outside, target_is_directory=True)
    plan = plan_artifact_removal(tmp_path)

    # The symlink target resolves outside project_root and must be skipped.
    # The plan should NOT include the symlink (because the resolved path is outside).
    for paths in plan.by_language.values():
        for p in paths:
            assert "outside-target" not in str(p)
