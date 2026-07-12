"""Integration tests for the autoship-sdk package distribution.

These tests build the ``autoship-sdk`` wheel, install it into a clean virtual
environment, and verify that it imports cleanly and has no dependency cycles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import install_wheel, run_in_venv, venv_python

pytestmark = pytest.mark.integration


def test_sdk_build_wheel(sdk_wheel: Path) -> None:
    """The autoship-sdk wheel artifact was produced."""
    assert sdk_wheel.exists()
    assert sdk_wheel.suffix == ".whl"


def test_sdk_installs_and_imports(venv_dir: Path, autoship_wheel: Path, sdk_wheel: Path) -> None:
    """autoship-sdk installs and imports in a clean venv with autoship present."""
    install_wheel(venv_dir, autoship_wheel)
    install_wheel(venv_dir, sdk_wheel)

    result = run_in_venv(
        venv_dir,
        [str(venv_python(venv_dir)), "-c", "import autoship_sdk; print('ok')"],
    )

    assert result.returncode == 0
    assert "ok" in result.stdout


def test_sdk_no_dependency_cycles(venv_dir: Path, autoship_wheel: Path, sdk_wheel: Path) -> None:
    """``pip check`` reports no broken dependencies after installing both packages."""
    install_wheel(venv_dir, autoship_wheel)
    install_wheel(venv_dir, sdk_wheel)

    result = run_in_venv(
        venv_dir,
        [str(venv_python(venv_dir)), "-m", "pip", "check"],
    )

    assert result.returncode == 0
    assert "No broken requirements" in result.stdout
