"""End-to-end tests using Typer's CliRunner."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from autoship.cli.main import app

runner = CliRunner()


def test_help_prints_usage() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AutoShip" in result.output


def test_init_creates_config(tmp_path: Path) -> None:
    config_path = tmp_path / ".autoship.toml"
    result = runner.invoke(
        app,
        ["--yes", "init", "--output", str(config_path)],
    )
    assert result.exit_code == 0
    assert config_path.exists()
    assert "project_type" in config_path.read_text(encoding="utf-8")


def test_clean_dry_run(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--dry-run", "--yes", "clean", str(tmp_path)],
    )
    assert result.exit_code == 0


def test_verify_runs_command(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--yes", "verify", "python -c \"print('ok')\""],
    )
    assert result.exit_code == 0
    assert "Verified" in result.output


def test_commit_with_yes(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "hello.txt").write_text("world")

    result = runner.invoke(
        app,
        ["--config", str(tmp_path / ".autoship.toml"), "--yes", "commit"],
    )
    assert result.exit_code == 0
    assert "Committed" in result.output


def test_upload_dry_run(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--dry-run", "--yes", "upload", "--target", "pypi"],
    )
    assert result.exit_code == 0
    assert "Would upload to pypi" in result.output
    assert "repository: testpypi" in result.output


def test_init_uses_hardware_tier(tmp_path: Path) -> None:
    config_path = tmp_path / ".autoship.toml"
    with patch("autoship.cli.commands.init.detect_hardware") as mock_hw:
        mock_hw.return_value.recommended_tier = 1
        result = runner.invoke(
            app,
            ["--yes", "init", "--output", str(config_path)],
        )
    assert result.exit_code == 0
    assert config_path.exists()
    assert "default_tier = 1" in config_path.read_text(encoding="utf-8")


def test_plugin_subcommand_list_empty() -> None:
    with patch("autoship.cli.commands.plugin.PluginRegistry") as mock_cls:
        mock_cls.return_value.list.return_value = []
        result = runner.invoke(app, ["plugin", "list"])
    assert result.exit_code == 0
    assert "No plugins registered" in result.output


def test_upload_docker_dry_run(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--dry-run", "--yes", "upload", "--target", "docker", "--image", "myapp", "--tag", "v1"],
    )
    assert result.exit_code == 0
    assert "Would upload to docker" in result.output
    assert "image: myapp:v1" in result.output
