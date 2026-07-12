"""Integration tests for real Docker build/push against a local registry.

These tests require a working Docker daemon and the ``docker`` CLI. When
Docker is unavailable, every test is skipped.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from autoship.adapters.upload.docker import DockerUploader

from .conftest import find_free_port, run_cmd, tool_available

pytestmark = pytest.mark.integration


def _docker_available() -> bool:
    """Return True if docker daemon responds to ``docker version``."""
    if not tool_available("docker"):
        return False
    try:
        subprocess.run(
            ["docker", "version"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture
def local_registry():
    """Start a local Docker registry on a free port and yield its address."""
    if not _docker_available():
        pytest.skip("Docker daemon is not available")

    port = find_free_port()
    registry = f"127.0.0.1:{port}"
    name = f"autoship-test-registry-{port}"

    run_cmd(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"{port}:5000",
            "registry:2",
        ]
    )

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            run_cmd(["docker", "exec", name, "wget", "-qO-", "http://localhost:5000/v2/_catalog"])
            break
        except subprocess.CalledProcessError:
            time.sleep(0.5)
    else:
        run_cmd(["docker", "stop", name], check=False)
        pytest.skip("local Docker registry failed to start")

    try:
        yield registry
    finally:
        run_cmd(["docker", "stop", name], check=False)


@pytest.fixture
def minimal_docker_project(tmp_path: Path) -> Path:
    """Create a minimal Docker project in a temp directory."""
    (tmp_path / "Dockerfile").write_text(
        "FROM scratch\nCOPY hello.txt /hello.txt\n",
        encoding="utf-8",
    )
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    return tmp_path


def test_docker_build_and_push_to_local_registry(
    minimal_docker_project: Path, local_registry: str
) -> None:
    """Build and push a real image to a local Docker registry."""
    image = "autoship-upload-test"
    tag = "v1"
    uploader = DockerUploader(
        minimal_docker_project,
        image=image,
        tag=tag,
        registry=local_registry,
    )

    result = uploader.upload()

    assert result.success is True
    assert result.target == "docker"
    assert local_registry in result.details["image"]
    catalog = run_cmd(
        [
            "docker",
            "exec",
            f"autoship-test-registry-{local_registry.split(':')[1]}",
            "wget",
            "-qO-",
            "http://localhost:5000/v2/_catalog",
        ]
    )
    assert image in catalog.stdout
