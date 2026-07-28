"""Tests for the i18n module."""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from justagent.core.i18n import I18n, _detect_locale, get_i18n


def test_i18n_returns_translation() -> None:
    i18n = get_i18n("en")
    assert i18n._("clean.noop") == "Already clean."


def test_i18n_formats_arguments() -> None:
    i18n = get_i18n("en")
    assert i18n._("clean.complete") == "Clean complete."
    assert i18n._("init.created", output=".justagent.toml") == "Created .justagent.toml"


def test_i18n_falls_back_to_key() -> None:
    i18n = get_i18n("en")
    assert i18n._("missing.key") == "missing.key"


def test_i18n_chinese_catalog() -> None:
    i18n = get_i18n("zh")
    assert i18n._("clean.noop") == "已经是干净的。"
    assert i18n._("doctor.title") == "JustAgent 环境诊断"


def test_i18n_unknown_language_falls_back_to_english() -> None:
    """An unknown language code falls back to English rather than
    surfacing raw translation keys to the user.
    """
    i18n = get_i18n("xx")
    assert i18n.lang == "en"
    assert i18n._("clean.noop") == "Already clean."


def test_detect_locale_from_myagent_lang() -> None:
    with patch.dict(os.environ, {"MYAGENT_LANG": "zh_CN.UTF-8"}, clear=False):
        assert _detect_locale() == "zh"


def test_detect_locale_defaults_to_english() -> None:
    with (
        patch.dict(os.environ, {"MYAGENT_LANG": "", "LANG": ""}, clear=True),
        patch("locale.getlocale", side_effect=ValueError("no locale")),
    ):
        assert _detect_locale() == "en"


def test_get_i18n_uses_environment_when_no_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYAGENT_LANG", "zh")
    i18n = get_i18n()
    assert i18n.lang == "zh"
    assert i18n._("clean.noop") == "已经是干净的。"


def test_locale_files_are_shiped() -> None:
    locales_dir = Path(__file__).resolve().parents[2] / "src" / "justagent" / "locales"
    assert (locales_dir / "en.json").exists()
    assert (locales_dir / "zh.json").exists()
    assert (locales_dir / "ja.json").exists()


def test_locale_files_have_no_duplicate_keys() -> None:
    """Each locale file must not declare the same key twice (JSON silently keeps the last)."""
    locales_dir = Path(__file__).resolve().parents[2] / "src" / "justagent" / "locales"
    for name in ("en.json", "zh.json", "ja.json"):
        text = (locales_dir / name).read_text(encoding="utf-8")
        # Count key occurrences by scanning the raw text for top-level ``"key":`` lines.
        keys = re.findall(r'^\s*"([^"]+)"\s*:', text, flags=re.MULTILINE)
        duplicates = {k for k in keys if keys.count(k) > 1}
        assert not duplicates, f"{name} has duplicate keys: {sorted(duplicates)}"


def test_locale_files_share_identical_key_sets() -> None:
    """zh / en / ja locales must expose the same set of translation keys."""
    import json

    locales_dir = Path(__file__).resolve().parents[2] / "src" / "justagent" / "locales"
    keys: dict[str, set[str]] = {}
    for name in ("en.json", "zh.json", "ja.json"):
        data = json.loads((locales_dir / name).read_text(encoding="utf-8"))
        keys[name] = set(data.keys())
    assert keys["en.json"] == keys["zh.json"] == keys["ja.json"]


def test_i18n_class_basic_usage() -> None:
    translator = I18n("test", {"greeting": "Hello {name}"})
    assert translator._("greeting", name="World") == "Hello World"


def test_i18n_returns_template_on_format_error() -> None:
    translator = I18n("test", {"greeting": "Hello {name}"})
    assert translator._("greeting") == "Hello {name}"


def test_i18n_returns_template_on_missing_kwarg() -> None:
    translator = I18n("test", {"greeting": "Hello {name}"})
    assert translator._("greeting", unused="value") == "Hello {name}"
