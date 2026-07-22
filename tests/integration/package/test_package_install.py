"""Integration tests for installing myagent from a built wheel.

These tests build the ``myagent`` wheel, install it into a clean virtual
environment, and verify that the entry point and non-AI commands work without
the source tree or the ``ai`` extras.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import install_wheel, run_in_venv, venv_bin, venv_python

pytestmark = pytest.mark.integration


def test_build_wheel(myagent_wheel: Path) -> None:
    """The wheel artifact was produced."""
    assert myagent_wheel.exists()
    assert myagent_wheel.suffix == ".whl"


def test_entry_point_available(venv_dir: Path, myagent_wheel: Path) -> None:
    """The ``myagent`` console script is installed and executable."""
    install_wheel(venv_dir, myagent_wheel)

    myagent = venv_bin(venv_dir, "myagent")
    result = run_in_venv(venv_dir, [str(myagent), "--help"])

    assert result.returncode == 0
    assert "MyAgent" in result.stdout


def test_python_module_entry_point(venv_dir: Path, myagent_wheel: Path) -> None:
    """``python -m myagent`` works in the installed wheel."""
    install_wheel(venv_dir, myagent_wheel)

    result = run_in_venv(
        venv_dir,
        [str(venv_python(venv_dir)), "-m", "myagent", "--help"],
    )

    assert result.returncode == 0
    assert "MyAgent" in result.stdout


def test_init_command_works(venv_dir: Path, myagent_wheel: Path, tmp_path: Path) -> None:
    """``myagent init --yes`` runs in a clean project after pip install."""
    install_wheel(venv_dir, myagent_wheel)

    project = tmp_path / "new_project"
    project.mkdir()
    myagent = venv_bin(venv_dir, "myagent")
    result = run_in_venv(
        venv_dir,
        [str(myagent), "init", "--yes"],
        cwd=project,
    )

    assert result.returncode == 0
    assert (project / ".myagent.toml").exists()


def test_doctor_command_works(venv_dir: Path, myagent_wheel: Path, tmp_path: Path) -> None:
    """``myagent doctor --json`` runs after pip install."""
    install_wheel(venv_dir, myagent_wheel)

    myagent = venv_bin(venv_dir, "myagent")
    result = run_in_venv(
        venv_dir,
        [str(myagent), "doctor", "--json"],
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "model-backend" in result.stdout or "ok" in result.stdout.lower()


def test_upload_dry_run_works(venv_dir: Path, myagent_wheel: Path, tmp_path: Path) -> None:
    """``myagent upload --dry-run --yes --target pypi`` runs after pip install."""
    install_wheel(venv_dir, myagent_wheel)

    myagent = venv_bin(venv_dir, "myagent")
    result = run_in_venv(
        venv_dir,
        [str(myagent), "--dry-run", "--yes", "upload", "--target", "pypi"],
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "Would upload to pypi" in result.stdout


def test_plugin_list_works(venv_dir: Path, myagent_wheel: Path, tmp_path: Path) -> None:
    """``myagent plugin list`` runs after pip install."""
    install_wheel(venv_dir, myagent_wheel)

    myagent = venv_bin(venv_dir, "myagent")
    result = run_in_venv(
        venv_dir,
        [str(myagent), "plugin", "list"],
        cwd=tmp_path,
    )

    assert result.returncode == 0


def test_verify_command_works(venv_dir: Path, myagent_wheel: Path, tmp_path: Path) -> None:
    """``myagent verify`` runs an allowed command after pip install."""
    install_wheel(venv_dir, myagent_wheel)

    myagent = venv_bin(venv_dir, "myagent")
    result = run_in_venv(
        venv_dir,
        [str(myagent), "verify", "python --version"],
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "Verified" in result.stdout


def test_sdist_installs_and_runs(venv_dir: Path, myagent_sdist: Path, tmp_path: Path) -> None:
    """The source distribution installs and the entry point works."""
    run_in_venv(
        venv_dir,
        [str(venv_python(venv_dir)), "-m", "pip", "install", str(myagent_sdist)],
    )

    myagent = venv_bin(venv_dir, "myagent")
    result = run_in_venv(venv_dir, [str(myagent), "--help"], cwd=tmp_path)

    assert result.returncode == 0
    assert "MyAgent" in result.stdout
