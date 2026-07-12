"""Tests for the artifacts command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from autoship.cli.main import app

runner = CliRunner()


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / ".autoship.toml"
    config_path.write_text(
        f'schema_version = 1\nproject_root = "{tmp_path}"\n',
        encoding="utf-8",
    )
    return config_path


def test_artifacts_dry_run_lists_planned_removal(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / "go.mod").write_text("module example.com/test\n", encoding="utf-8")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "binary").write_bytes(b"\x7fELF")

    result = runner.invoke(
        app,
        ["--config", str(config_path), "artifacts", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "Planned removal" in result.output
    assert "bin" in result.output
    assert "[dry-run]" in result.output


def test_artifacts_no_rules_for_empty_project(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["--config", str(config_path), "artifacts", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "No language rules apply" in result.output


def test_artifacts_nothing_to_remove_when_clean(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["--config", str(config_path), "artifacts", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "No language-native build artifacts to remove" in result.output


def test_artifacts_list_shows_rule_table(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name": "frontend"}\n', encoding="utf-8")

    result = runner.invoke(
        app,
        ["--config", str(config_path), "artifacts", "--list"],
    )
    assert result.exit_code == 0
    assert "[go]" in result.output
    assert "[node]" in result.output
    assert "go test ./..." in result.output
    assert "npm test" in result.output
    assert "manifests" in result.output


def test_artifacts_remove_with_yes_flag(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "binary").write_bytes(b"\x7fELF")

    result = runner.invoke(
        app,
        ["--config", str(config_path), "artifacts", "--yes"],
    )
    assert result.exit_code == 0
    assert "Removed" in result.output
    assert not bin_dir.exists()


def test_artifacts_filter_by_language(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name": "frontend"}\n', encoding="utf-8")
    (tmp_path / "bin").mkdir()  # go
    (tmp_path / "dist").mkdir()  # node

    result = runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "artifacts",
            "--language",
            "go",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "bin" in result.output
    assert "dist" not in result.output


def test_artifacts_list_with_unknown_language_yields_no_rules(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "artifacts",
            "--list",
            "--language",
            "python",  # not in the rule table
        ],
    )
    assert result.exit_code == 0
    assert "No language rules apply" in result.output


def test_artifacts_aborts_without_yes(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "binary").write_bytes(b"\x7fELF")

    # Decline the confirmation prompt by sending "n".
    result = runner.invoke(
        app,
        ["--config", str(config_path), "artifacts"],
        input="n\n",
    )
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert (tmp_path / "bin").exists()
