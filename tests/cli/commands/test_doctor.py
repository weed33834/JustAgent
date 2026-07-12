"""Tests for the doctor command."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from autoship.cli.commands.doctor import CheckResult, Status, check_directories
from autoship.cli.main import app
from autoship.models.config import AppConfig

runner = CliRunner()


def test_doctor_runs_and_reports_summary() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "AutoShip Environment Diagnostics" in result.output
    assert "Summary:" in result.output


def test_doctor_json_output() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code in (0, 1)
    assert '"summary"' in result.output
    assert '"python"' in result.output
    assert '"git"' in result.output
    assert '"tools"' in result.output
    assert '"model"' in result.output
    assert '"dirs"' in result.output
    assert '"cache"' in result.output
    assert '"observability"' in result.output


def test_doctor_detects_missing_git() -> None:
    with patch("autoship.cli.commands.doctor._run_cmd", return_value=(False, "not found")):
        result = runner.invoke(app, ["doctor", "--fail-on-error"])
    assert result.exit_code == 1
    assert "Git not found" in result.output


def test_doctor_detects_old_python() -> None:
    with patch(
        "autoship.cli.commands.doctor.check_python",
        return_value=CheckResult(
            name="python",
            status=Status.ERROR,
            message="Python 3.9.0",
            suggestion="Upgrade to Python 3.10 or later.",
        ),
    ):
        result = runner.invoke(app, ["doctor", "--fail-on-error"])
    assert result.exit_code == 1
    assert "Python 3.9" in result.output


def test_doctor_defaults_to_zero_exit_with_errors() -> None:
    with patch("autoship.cli.commands.doctor._run_cmd", return_value=(False, "not found")):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Git not found" in result.output


def test_doctor_json_fail_on_error() -> None:
    with patch("autoship.cli.commands.doctor._run_cmd", return_value=(False, "not found")):
        result = runner.invoke(app, ["doctor", "--json", "--fail-on-error"])
    assert result.exit_code == 1
    assert '"error"' in result.output


def test_check_directories_writable(i18n, tmp_path) -> None:
    config = AppConfig(project_root=tmp_path)
    result = check_directories(config, i18n)
    assert result.status == Status.OK
    assert "accessible" in result.message


def test_check_directories_not_writable(i18n, tmp_path) -> None:
    config = AppConfig(project_root=tmp_path)
    with patch("autoship.cli.commands.doctor.os.access", return_value=False):
        result = check_directories(config, i18n)
    assert result.status == Status.WARNING
    assert "not writable" in result.message


def test_check_directories_missing(i18n, tmp_path) -> None:
    config = AppConfig(project_root=tmp_path / "does" / "not" / "exist")
    result = check_directories(config, i18n)
    assert result.status == Status.WARNING
    assert "Cannot access" in result.message
