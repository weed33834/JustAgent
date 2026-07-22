"""Tests for the instructions auto-discovery module."""

from __future__ import annotations

from pathlib import Path

from autoship.context import InstructionConfig, InstructionDiscovery


def test_discover_finds_agents_md_in_cwd(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")

    discovery = InstructionDiscovery()
    files = discovery.discover(tmp_path)

    assert len(files) == 1
    assert files[0].filename == "AGENTS.md"
    assert files[0].level == 0
    assert files[0].content == "# Rules\n"
    assert files[0].path == tmp_path.resolve() / "AGENTS.md"


def test_discover_finds_agents_md_in_parent(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Parent rules\n", encoding="utf-8")
    child = tmp_path / "sub"
    child.mkdir()

    discovery = InstructionDiscovery()
    files = discovery.discover(child)

    assert len(files) == 1
    assert files[0].filename == "AGENTS.md"
    assert files[0].level == 1
    assert files[0].content == "# Parent rules\n"


def test_discover_multiple_files_at_different_levels(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Parent rules\n", encoding="utf-8")
    child = tmp_path / "sub"
    child.mkdir()
    (child / "AGENTS.md").write_text("# Child rules\n", encoding="utf-8")

    discovery = InstructionDiscovery()
    files = discovery.discover(child)

    assert len(files) == 2
    assert files[0].level == 0
    assert files[0].content == "# Child rules\n"
    assert files[1].level == 1
    assert files[1].content == "# Parent rules\n"


def test_discover_priority_multiple_files_same_level(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# Claude\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")

    discovery = InstructionDiscovery()
    files = discovery.discover(tmp_path)

    assert len(files) == 2
    assert files[0].filename == "AGENTS.md"
    assert files[1].filename == "CLAUDE.md"


def test_discover_respects_max_levels(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Too far up\n", encoding="utf-8")
    level1 = tmp_path / "a"
    level1.mkdir()
    level2 = level1 / "b"
    level2.mkdir()
    level3 = level2 / "c"
    level3.mkdir()
    (level2 / "AGENTS.md").write_text("# Within reach\n", encoding="utf-8")

    discovery = InstructionDiscovery(InstructionConfig(max_levels=1))
    files = discovery.discover(level3)

    assert len(files) == 1
    assert files[0].level == 1
    assert files[0].content == "# Within reach\n"


def test_load_truncates_to_max_total_chars(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("x" * 500, encoding="utf-8")

    discovery = InstructionDiscovery(InstructionConfig(max_total_chars=100))
    result = discovery.load(tmp_path)

    assert len(result) == 100


def test_discover_returns_empty_when_no_files(tmp_path: Path) -> None:
    discovery = InstructionDiscovery()
    assert discovery.discover(tmp_path) == []


def test_load_concatenation_format(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Rules", encoding="utf-8")

    discovery = InstructionDiscovery()
    result = discovery.load(tmp_path)

    assert result == "# From ./AGENTS.md\n# Rules\n\n"


def test_missing_directory_handling(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    discovery = InstructionDiscovery()

    assert discovery.discover(missing) == []
    assert discovery.load(missing) == ""
