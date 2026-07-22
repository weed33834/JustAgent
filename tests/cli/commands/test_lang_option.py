"""End-to-end tests for the global ``--lang`` option.

The ``--lang`` option selects the runtime i18n catalogue that is stored in
the Typer context (``ctx.obj["i18n"]``) and used for all command output.
These tests verify that ``zh`` and ``ja`` select the correct catalogue and
that ``auto`` falls back gracefully without crashing.
"""

from __future__ import annotations

from typer.testing import CliRunner

from myagent.cli import main
from myagent.cli.main import app
from myagent.core.i18n import I18n


class _FakeContext:
    """Minimal stand-in for a Typer context (mirrors tests/cli/test_main.py)."""

    def __init__(self) -> None:
        self.obj: dict = {}

    def ensure_object(self, typ: type) -> None:
        if not self.obj:
            self.obj = {}


# ---------------------------------------------------------------------------
# Direct callback tests: verify ctx.obj["i18n"].lang
# ---------------------------------------------------------------------------


def test_lang_zh_sets_i18n_to_chinese() -> None:
    """``--lang zh`` stores a Chinese i18n instance in the context."""
    ctx = _FakeContext()
    main.main_callback(ctx, config_path=None, lang="zh")
    i18n = ctx.obj["i18n"]
    assert isinstance(i18n, I18n)
    assert i18n.lang == "zh"
    # The Chinese catalogue must actually be loaded (not empty fallback).
    assert i18n.catalog, "zh catalogue should not be empty"


def test_lang_ja_sets_i18n_to_japanese() -> None:
    """``--lang ja`` stores a Japanese i18n instance in the context."""
    ctx = _FakeContext()
    main.main_callback(ctx, config_path=None, lang="ja")
    i18n = ctx.obj["i18n"]
    assert isinstance(i18n, I18n)
    assert i18n.lang == "ja"
    assert i18n.catalog, "ja catalogue should not be empty"


def test_lang_auto_falls_back_without_crashing() -> None:
    """``--lang auto`` defers to config.locale / detection and must not crash."""
    ctx = _FakeContext()
    main.main_callback(ctx, config_path=None, lang="auto")
    i18n = ctx.obj["i18n"]
    assert isinstance(i18n, I18n)
    # The detected language is one of the supported codes.
    assert i18n.lang in {"en", "zh", "ja"}


def test_lang_none_defaults_to_config_locale() -> None:
    """When ``--lang`` is not passed, config.locale governs (default ``auto``)."""
    ctx = _FakeContext()
    main.main_callback(ctx, config_path=None, lang=None)
    i18n = ctx.obj["i18n"]
    assert isinstance(i18n, I18n)
    assert i18n.lang in {"en", "zh", "ja"}


# ---------------------------------------------------------------------------
# End-to-end via CliRunner: --lang zh <cmd> --help exits cleanly
# ---------------------------------------------------------------------------


def test_lang_zh_with_subcommand_help_exits_cleanly() -> None:
    """``myagent --lang zh clean --help`` must not crash."""
    runner = CliRunner()
    result = runner.invoke(app, ["--lang", "zh", "clean", "--help"])
    assert result.exit_code == 0
    assert "Traceback" not in (result.output + result.stderr)


def test_lang_ja_with_subcommand_help_exits_cleanly() -> None:
    """``myagent --lang ja init --help`` must not crash."""
    runner = CliRunner()
    result = runner.invoke(app, ["--lang", "ja", "init", "--help"])
    assert result.exit_code == 0
    assert "Traceback" not in (result.output + result.stderr)
