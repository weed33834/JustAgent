"""Shared helpers for upload integration tests."""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

import pytest


def find_free_port(host: str = "127.0.0.1") -> int:
    """Return a free TCP port on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def tool_available(name: str) -> bool:
    """Return True if ``name`` is on PATH."""
    return shutil.which(name) is not None


def run_cmd(
    cmd: list[str], cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the completed process."""
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


@pytest.fixture
def minimal_python_package(tmp_path: Path) -> Path:
    """Create a minimal publishable Python package in a temp directory."""
    pkg_dir = tmp_path / "demo_pkg"
    pkg_dir.mkdir()

    (pkg_dir / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = ['hatchling']\n"
        "build-backend = 'hatchling.build'\n\n"
        "[project]\n"
        "name = 'demo-pkg-for-autoship-upload'\n"
        "version = '0.0.1'\n"
        "description = 'Demo package for upload integration tests'\n"
        "requires-python = '>=3.10'\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        "packages = ['src/demo_pkg']\n",
        encoding="utf-8",
    )

    src = pkg_dir / "src" / "demo_pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text('__version__ = "0.0.1"\n', encoding="utf-8")

    return pkg_dir
