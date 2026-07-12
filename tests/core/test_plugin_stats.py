"""Tests for plugin usage statistics."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoship.core.plugin_stats import PluginStats


@pytest.fixture
def stats_file(tmp_path: Path) -> Path:
    return tmp_path / "plugin_stats.json"


def test_record_install(stats_file: Path) -> None:
    stats = PluginStats(stats_file=stats_file)
    stats.record_install("my-plugin")
    assert stats.get("my-plugin").installs == 1


def test_record_uninstall(stats_file: Path) -> None:
    stats = PluginStats(stats_file=stats_file)
    stats.record_uninstall("my-plugin")
    assert stats.get("my-plugin").uninstalls == 1


def test_record_rate(stats_file: Path) -> None:
    stats = PluginStats(stats_file=stats_file)
    stats.record_rate("my-plugin", 4.0)
    stats.record_rate("my-plugin", 5.0)
    rating = stats.get("my-plugin").rating
    assert rating.count == 2
    assert rating.score == 4.5


def test_record_rate_out_of_range(stats_file: Path) -> None:
    stats = PluginStats(stats_file=stats_file)
    with pytest.raises(ValueError):
        stats.record_rate("my-plugin", 6.0)


def test_persistence(stats_file: Path) -> None:
    stats = PluginStats(stats_file=stats_file)
    stats.record_install("my-plugin")
    stats.record_rate("my-plugin", 5.0)

    stats2 = PluginStats(stats_file=stats_file)
    assert stats2.get("my-plugin").installs == 1
    assert stats2.get("my-plugin").rating.count == 1
    assert stats2.get("my-plugin").rating.score == 5.0


def test_summary(stats_file: Path) -> None:
    stats = PluginStats(stats_file=stats_file)
    stats.record_install("a")
    stats.record_rate("a", 3.0)
    summary = stats.summary()
    assert "a" in summary
    assert summary["a"]["installs"] == 1


def test_load_ignores_non_object_json(stats_file: Path) -> None:
    stats_file.write_text("[1, 2, 3]", encoding="utf-8")
    stats = PluginStats(stats_file=stats_file)
    assert stats.summary() == {}
