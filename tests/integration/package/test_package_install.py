"""Integration tests for installing autoship from a built wheel.

These tests build the ``autoship`` wheel, install it into a clean virtual
environment, and verify that the entry point and non-AI commands work without
the source tree or the ``ai`` extras.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import install_wheel, run_in_venv, venv_bin, venv_python

pytestmark = pytest.mark.integration


def test_build_wheel(autoship_wheel: Path) -> None:
    """The wheel artifact was produced."""
    assert autoship_wheel.exists()
    assert autoship_wheel.suffix == ".whl"


def test_entry_point_available(venv_dir: Path, autoship_wheel: Path) -> None:
    """The ``autoship`` console script is installed and executable."""
    install_wheel(venv_dir, autoship_wheel)

    autoship = venv_bin(venv_dir, "autoship")
    result = run_in_venv(venv_dir, [str(autoship), "--help"])

    assert result.returncode == 0
    assert "AutoShip" in result.stdout


def test_python_module_entry_point(venv_dir: Path, autoship_wheel: Path) -> None:
    """``python -m autoship`` works in the installed wheel."""
    install_wheel(venv_dir, autoship_wheel)

    result = run_in_venv(
        venv_dir,
        [str(venv_python(venv_dir)), "-m", "autoship", "--help"],
    )

    assert result.returncode == 0
    assert "AutoShip" in result.stdout


def test_init_command_works(venv_dir: Path, autoship_wheel: Path, tmp_path: Path) -> None:
    """``autoship init --yes`` runs in a clean project after pip install."""
    install_wheel(venv_dir, autoship_wheel)

    project = tmp_path / "new_project"
    project.mkdir()
    autoship = venv_bin(venv_dir, "autoship")
    result = run_in_venv(
        venv_dir,
        [str(autoship), "init", "--yes"],
        cwd=project,
    )

    assert result.returncode == 0
    assert (project / ".autoship.toml").exists()


def test_doctor_command_works(venv_dir: Path, autoship_wheel: Path, tmp_path: Path) -> None:
    """``autoship doctor --json`` runs after pip install."""
    install_wheel(venv_dir, autoship_wheel)

    autoship = venv_bin(venv_dir, "autoship")
    result = run_in_venv(
        venv_dir,
        [str(autoship), "doctor", "--json"],
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "model-backend" in result.stdout or "ok" in result.stdout.lower()


def test_upload_dry_run_works(venv_dir: Path, autoship_wheel: Path, tmp_path: Path) -> None:
    """``autoship upload --dry-run --yes --target pypi`` runs after pip install."""
    install_wheel(venv_dir, autoship_wheel)

    autoship = venv_bin(venv_dir, "autoship")
    result = run_in_venv(
        venv_dir,
        [str(autoship), "--dry-run", "--yes", "upload", "--target", "pypi"],
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "Would upload to pypi" in result.stdout


def test_plugin_list_works(venv_dir: Path, autoship_wheel: Path, tmp_path: Path) -> None:
    """``autoship plugin list`` runs after pip install."""
    install_wheel(venv_dir, autoship_wheel)

    autoship = venv_bin(venv_dir, "autoship")
    result = run_in_venv(
        venv_dir,
        [str(autoship), "plugin", "list"],
        cwd=tmp_path,
    )

    assert result.returncode == 0


def test_verify_command_works(venv_dir: Path, autoship_wheel: Path, tmp_path: Path) -> None:
    """``autoship verify`` runs an allowed command after pip install."""
    install_wheel(venv_dir, autoship_wheel)

    autoship = venv_bin(venv_dir, "autoship")
    result = run_in_venv(
        venv_dir,
        [str(autoship), "verify", "python --version"],
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "Verified" in result.stdout


def test_sdist_installs_and_runs(venv_dir: Path, autoship_sdist: Path, tmp_path: Path) -> None:
    """The source distribution installs and the entry point works."""
    run_in_venv(
        venv_dir,
        [str(venv_python(venv_dir)), "-m", "pip", "install", str(autoship_sdist)],
    )

    autoship = venv_bin(venv_dir, "autoship")
    result = run_in_venv(venv_dir, [str(autoship), "--help"], cwd=tmp_path)

    assert result.returncode == 0
    assert "AutoShip" in result.stdout
