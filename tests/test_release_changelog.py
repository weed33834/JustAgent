"""Tests for the release CHANGELOG.md generator (scripts/release_changelog.py)."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "release_changelog.py"

BASE_CHANGELOG = """\
# Changelog

All notable changes will be documented in this file.

## [Unreleased]

### Added

- Something pending.

## [1.0.0] - 2026-06-19

### Added

- Initial stable release.
"""


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("release_changelog", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rc() -> object:
    return _load_module()


def test_render_entry_includes_version_and_date(rc: object) -> None:
    entry = rc.render_entry("1.1.0", "2026-07-02", "### Added\n\n- New thing\n")  # type: ignore[attr-defined]
    assert entry.startswith("## [1.1.0] - 2026-07-02")
    assert "- New thing" in entry


def test_render_entry_without_date(rc: object) -> None:
    entry = rc.render_entry("1.1.0", "", "### Added\n\n- X\n")  # type: ignore[attr-defined]
    assert entry.startswith("## [1.1.0]")


def test_render_entry_strips_full_changelog_footer(rc: object) -> None:
    body = "### Added\n\n- X\n\n## Full Changelog\nhttps://github.com/compare/abc...def"
    entry = rc.render_entry("1.1.0", "2026-07-02", body)  # type: ignore[attr-defined]
    assert "## Full Changelog" not in entry
    assert "https://github.com/compare" not in entry
    assert "- X" in entry


def test_insert_entry_places_after_unreleased_and_before_previous(rc: object) -> None:
    entry = rc.render_entry("1.1.0", "2026-07-02", "### Added\n\n- X")  # type: ignore[attr-defined]
    new = rc.insert_entry(BASE_CHANGELOG, entry, "1.1.0")  # type: ignore[attr-defined]
    assert new.index("## [Unreleased]") < new.index("## [1.1.0]") < new.index("## [1.0.0]")


def test_insert_entry_without_unreleased(rc: object) -> None:
    content = "# Changelog\n\nIntro.\n\n## [1.0.0] - 2026-06-19\n\n- init\n"
    entry = rc.render_entry("1.1.0", "2026-07-02", "- X")  # type: ignore[attr-defined]
    new = rc.insert_entry(content, entry, "1.1.0")  # type: ignore[attr-defined]
    assert new.index("## [1.1.0]") < new.index("## [1.0.0]")


def test_find_version_detects_existing(rc: object) -> None:
    assert rc.find_version(BASE_CHANGELOG, "1.0.0")  # type: ignore[attr-defined]
    assert not rc.find_version(BASE_CHANGELOG, "9.9.9")  # type: ignore[attr-defined]


def test_main_is_idempotent(
    rc: object, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(BASE_CHANGELOG, encoding="utf-8")
    notes = tmp_path / "notes.md"
    notes.write_text("### Added\n\n- X\n", encoding="utf-8")

    rc.main(  # type: ignore[attr-defined]
        [
            "--version",
            "1.1.0",
            "--date",
            "2026-07-02",
            "--notes-file",
            str(notes),
            "--changelog",
            str(changelog),
        ]
    )
    first = changelog.read_text(encoding="utf-8")
    assert "## [1.1.0] - 2026-07-02" in first

    rc.main(  # type: ignore[attr-defined]
        [
            "--version",
            "1.1.0",
            "--date",
            "2026-07-02",
            "--notes-file",
            str(notes),
            "--changelog",
            str(changelog),
        ]
    )
    assert changelog.read_text(encoding="utf-8") == first
    assert "already contains" in capsys.readouterr().out


def test_main_dry_run_does_not_write(rc: object, tmp_path: pathlib.Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(BASE_CHANGELOG, encoding="utf-8")
    notes = tmp_path / "notes.md"
    notes.write_text("### Added\n\n- X\n", encoding="utf-8")

    rc.main(  # type: ignore[attr-defined]
        [
            "--version",
            "2.0.0",
            "--date",
            "2026-07-02",
            "--notes-file",
            str(notes),
            "--changelog",
            str(changelog),
            "--dry-run",
        ]
    )
    assert "## [2.0.0]" not in changelog.read_text(encoding="utf-8")


def test_main_rejects_invalid_version(rc: object, tmp_path: pathlib.Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(BASE_CHANGELOG, encoding="utf-8")
    notes = tmp_path / "notes.md"
    notes.write_text("### Added\n\n- X\n", encoding="utf-8")

    code = rc.main(  # type: ignore[attr-defined]
        [
            "--version",
            "1.1.0;rm -rf /",
            "--date",
            "2026-07-02",
            "--notes-file",
            str(notes),
            "--changelog",
            str(changelog),
        ]
    )
    assert code == 2
