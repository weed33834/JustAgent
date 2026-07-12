"""Tests for the ``autoship metrics`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoship.cli.main import app
from autoship.core.metrics import MetricsRegistry, get_registry, set_registry


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def clean_registry() -> MetricsRegistry:
    original = get_registry()
    reg = MetricsRegistry()
    set_registry(reg)
    yield reg
    reg.reset()
    set_registry(original)


def test_metrics_show_empty(
    runner: CliRunner,
    clean_registry: MetricsRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Point the audit log at an empty temp dir so ``_collect_audit_counters``
    # (which aggregates audit JSONL files) has nothing to aggregate. Without
    # this the metrics command would surface audit events written by previous
    # test runs / the current ``cli.invoked`` record.
    monkeypatch.setenv("AUTOSHIP_AUDIT__LOG_DIR", str(tmp_path / "audit"))
    result = runner.invoke(app, ["metrics", "show"])
    assert result.exit_code == 0
    assert "No metrics collected" in result.output


def test_metrics_show_json(runner: CliRunner, clean_registry: MetricsRegistry) -> None:
    clean_registry.inc("test_metric", description="desc")
    result = runner.invoke(app, ["metrics", "show", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["test_metric"]["type"] == "counter"
    assert data["test_metric"]["value"] == 1


def test_metrics_export(runner: CliRunner, clean_registry: MetricsRegistry, tmp_path: Path) -> None:
    clean_registry.inc("export_metric", description="desc")
    output = tmp_path / "metrics.json"
    result = runner.invoke(app, ["metrics", "export", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["export_metric"]["value"] == 1


def test_metrics_show_reset(runner: CliRunner, clean_registry: MetricsRegistry) -> None:
    clean_registry.inc("reset_metric", description="desc")
    result = runner.invoke(app, ["metrics", "show", "--reset"])
    assert result.exit_code == 0
    assert clean_registry.snapshot() == {}
