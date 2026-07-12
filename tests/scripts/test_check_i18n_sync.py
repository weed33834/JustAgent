"""Unit tests for scripts/check_i18n_sync.py.

Covers the locale-catalog key-drift check added in 1.1.1 and the blog
directory inclusion in collect_default_files. Stdlib-only, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_i18n_sync as mod  # noqa: E402  # type: ignore[import-not-found]


def test_collect_default_files_includes_blog_dir(tmp_path: Path) -> None:
    """collect_default_files must pick up docs/blog/*.md, not just docs/*.md and docs/commands/*.md."""
    (tmp_path / "root.md").write_text("# root\n", encoding="utf-8")
    (tmp_path / "commands").mkdir()
    (tmp_path / "commands" / "cmd.md").write_text("# cmd\n", encoding="utf-8")
    (tmp_path / "blog").mkdir()
    (tmp_path / "blog" / "post.md").write_text("# post\n", encoding="utf-8")

    files = mod.collect_default_files(tmp_path)
    names = [f.name for f in files]
    assert "root.md" in names
    assert "cmd.md" in names
    assert "post.md" in names


def test_collect_default_files_includes_community_dir(tmp_path: Path) -> None:
    """collect_default_files must pick up docs/community/*.md too."""
    (tmp_path / "root.md").write_text("# root\n", encoding="utf-8")
    (tmp_path / "community").mkdir()
    (tmp_path / "community" / "page.md").write_text("# page\n", encoding="utf-8")

    files = mod.collect_default_files(tmp_path)
    names = [f.name for f in files]
    assert "root.md" in names
    assert "page.md" in names


def test_collect_default_files_ignores_nonexistent_subdirs(tmp_path: Path) -> None:
    """Missing commands/ or blog/ subdirs must not raise."""
    (tmp_path / "root.md").write_text("# root\n", encoding="utf-8")
    files = mod.collect_default_files(tmp_path)
    assert [f.name for f in files] == ["root.md"]


def _write_locale(repo_root: Path, lang: str, keys: list[str]) -> None:
    locales = repo_root / "src" / "autoship" / "locales"
    locales.mkdir(parents=True, exist_ok=True)
    (locales / f"{lang}.json").write_text(
        json.dumps(dict.fromkeys(keys, f"{lang}-value")), encoding="utf-8"
    )


def test_check_locale_catalogs_aligned(tmp_path: Path) -> None:
    """All three catalogs sharing the same key set -> no errors."""
    keys = ["a.b", "c.d", "e.f"]
    for lang in ("en", "zh", "ja"):
        _write_locale(tmp_path, lang, keys)
    assert mod.check_locale_catalogs(tmp_path) == []


def test_check_locale_catalogs_missing_key_in_zh(tmp_path: Path) -> None:
    """zh missing a key present in en -> one error mentioning zh and the key."""
    _write_locale(tmp_path, "en", ["a", "b", "c"])
    _write_locale(tmp_path, "zh", ["a", "b"])  # missing c
    _write_locale(tmp_path, "ja", ["a", "b", "c"])
    errors = mod.check_locale_catalogs(tmp_path)
    assert len(errors) == 1
    assert "zh.json" in errors[0]
    assert "missing" in errors[0]
    assert "'c'" in errors[0]


def test_check_locale_catalogs_extra_key_in_ja(tmp_path: Path) -> None:
    """ja having an extra key not in en -> one error mentioning ja and the key."""
    _write_locale(tmp_path, "en", ["a"])
    _write_locale(tmp_path, "zh", ["a"])
    _write_locale(tmp_path, "ja", ["a", "orphan"])
    errors = mod.check_locale_catalogs(tmp_path)
    assert len(errors) == 1
    assert "ja.json" in errors[0]
    assert "extra" in errors[0]
    assert "'orphan'" in errors[0]


def test_check_locale_catalogs_en_missing(tmp_path: Path) -> None:
    """Missing en.json -> one error, no further checks."""
    errors = mod.check_locale_catalogs(tmp_path)
    assert errors == ["ERROR [locales] en.json missing"]


def test_check_locale_catalogs_invalid_json(tmp_path: Path) -> None:
    """Malformed en.json -> one error mentioning JSON decode failure."""
    locales = tmp_path / "src" / "autoship" / "locales"
    locales.mkdir(parents=True)
    (locales / "en.json").write_text("{not valid json", encoding="utf-8")
    errors = mod.check_locale_catalogs(tmp_path)
    assert len(errors) == 1
    assert "en.json invalid JSON" in errors[0]


def test_check_locale_catalogs_lang_file_missing(tmp_path: Path) -> None:
    """en exists but zh.json missing -> error for zh, then ja checked."""
    _write_locale(tmp_path, "en", ["a"])
    _write_locale(tmp_path, "ja", ["a"])
    # zh.json deliberately not written
    errors = mod.check_locale_catalogs(tmp_path)
    assert any("zh.json missing" in e for e in errors)
    assert not any("ja.json" in e for e in errors)


def test_check_locale_catalogs_truncates_large_drift(tmp_path: Path) -> None:
    """When >10 keys drift, the list is truncated with '...'."""
    en_keys = [f"k{i}" for i in range(15)]
    _write_locale(tmp_path, "en", en_keys)
    _write_locale(tmp_path, "zh", [])  # all 15 missing
    _write_locale(tmp_path, "ja", en_keys)
    errors = mod.check_locale_catalogs(tmp_path)
    assert len(errors) == 1
    assert "..." in errors[0]
    assert "15 key(s)" in errors[0]


def test_main_exit_code_clean_on_real_repo() -> None:
    """Running the script against the real repo must exit 0 (no drift)."""
    rc = mod.main()
    assert rc == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
