"""Shared helpers for package distribution integration tests."""

from __future__ import annotations

import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def autoship_wheel(project_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the autoship wheel once per session and return its path."""
    if not shutil.which("python"):
        pytest.skip("python is not available")
    output_dir = tmp_path_factory.mktemp("autoship-dist")
    wheels = build_wheel(project_root, output_dir)
    if not wheels:
        pytest.skip("wheel build produced no artifacts")
    return wheels[0]


@pytest.fixture(scope="session")
def autoship_sdist(project_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the autoship sdist once per session and return its path."""
    if not shutil.which("python"):
        pytest.skip("python is not available")
    output_dir = tmp_path_factory.mktemp("autoship-sdist")
    sdist = build_sdist(project_root, output_dir)
    if not sdist:
        pytest.skip("sdist build produced no artifacts")
    return sdist[0]


@pytest.fixture(scope="session")
def sdk_wheel(project_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the autoship-sdk wheel once per session and return its path."""
    sdk_root = project_root / "autoship-sdk"
    if not sdk_root.exists():
        pytest.skip("autoship-sdk workspace member not found")
    if not shutil.which("python"):
        pytest.skip("python is not available")
    output_dir = tmp_path_factory.mktemp("autoship-sdk-dist")
    wheels = build_wheel(sdk_root, output_dir)
    if not wheels:
        pytest.skip("autoship-sdk wheel build produced no artifacts")
    return wheels[0]


def build_wheel(project_root: Path, output_dir: Path) -> list[Path]:
    """Build wheel artifacts for ``project_root`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(output_dir)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(output_dir.glob("*.whl"))


def build_sdist(project_root: Path, output_dir: Path) -> list[Path]:
    """Build sdist artifacts for ``project_root`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(output_dir)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(output_dir.glob("*.tar.gz"))


@pytest.fixture
def venv_dir(tmp_path: Path) -> Path:
    """Create a clean virtual environment and return its path."""
    venv_path = tmp_path / "venv"
    venv.create(venv_path, with_pip=True)
    return venv_path


def venv_python(venv_dir: Path) -> Path:
    """Return the Python executable path inside the venv."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_bin(venv_dir: Path, name: str) -> Path:
    """Return a binary path inside the venv."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def run_in_venv(
    venv_dir: Path,
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command using the venv Python/binary."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def install_wheel(venv_dir: Path, wheel: Path) -> None:
    """Install a wheel into the venv."""
    run_in_venv(
        venv_dir,
        [str(venv_python(venv_dir)), "-m", "pip", "install", str(wheel)],
        check=True,
    )
