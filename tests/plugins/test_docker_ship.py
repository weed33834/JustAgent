"""Tests for the built-in docker-ship plugin."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from autoship.core.context import CommandContext
from autoship.exceptions import UploadError
from autoship.models.config import AppConfig, DockerShipConfig
from autoship.plugins import docker_ship


@pytest.fixture
def docker_context(app_config: AppConfig) -> CommandContext:
    """Return a CommandContext for a Docker upload."""
    app_config.docker_ship = DockerShipConfig(enabled=True, default_image="myapp")
    return CommandContext(
        command="upload",
        project_root=app_config.project_root,
        config=app_config,
        extras={"target": "docker", "tag": "v1"},
    )


def test_docker_ship_skips_non_docker_target(docker_context: CommandContext) -> None:
    docker_context.extras["target"] = "pypi"
    with patch("subprocess.run") as mock_run:
        docker_ship.plugin.pre_upload(docker_context)
    mock_run.assert_not_called()


def test_docker_ship_disabled(docker_context: CommandContext) -> None:
    docker_context.config.docker_ship.enabled = False
    with patch("subprocess.run") as mock_run:
        docker_ship.plugin.pre_upload(docker_context)
    mock_run.assert_not_called()


def test_docker_ship_missing_image(app_config: AppConfig) -> None:
    app_config.docker_ship = DockerShipConfig(enabled=True)
    ctx = CommandContext(
        command="upload",
        project_root=app_config.project_root,
        config=app_config,
        extras={"target": "docker"},
    )
    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        pytest.raises(UploadError, match="Docker image name is required"),
    ):
        docker_ship.plugin.pre_upload(ctx)


def test_docker_ship_builds_image(docker_context: CommandContext) -> None:
    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run") as mock_run,
    ):
        docker_ship.plugin.pre_upload(docker_context)

    cmd = mock_run.call_args[0][0]
    assert cmd[:4] == ["/usr/bin/docker", "build", "-t", "myapp:v1"]


def test_docker_ship_no_docker_command(docker_context: CommandContext) -> None:
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(UploadError, match="docker command not found"),
    ):
        docker_ship.plugin.pre_upload(docker_context)


def test_docker_ship_pushes_when_configured(docker_context: CommandContext) -> None:
    docker_context.config.docker_ship.push = True
    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run") as mock_run,
    ):
        docker_ship.plugin.post_upload(docker_context)

    cmd = mock_run.call_args[0][0]
    assert cmd == ["/usr/bin/docker", "push", "myapp:v1"]


def test_docker_ship_build_args_passed(docker_context: CommandContext) -> None:
    docker_context.config.docker_ship.build_args = {
        "VERSION": "1.0.0",
        "BUILD_DATE": "2026-06-21",
    }
    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run") as mock_run,
    ):
        docker_ship.plugin.pre_upload(docker_context)

    cmd = mock_run.call_args[0][0]
    assert "VERSION=1.0.0" in cmd
    assert "BUILD_DATE=2026-06-21" in cmd


def test_docker_ship_invalid_build_arg_key_skipped(
    docker_context: CommandContext, caplog: pytest.LogCaptureFixture
) -> None:
    docker_context.config.docker_ship.build_args = {"9bad": "value", "GOOD": "value"}
    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run") as mock_run,
    ):
        docker_ship.plugin.pre_upload(docker_context)

    cmd = " ".join(mock_run.call_args[0][0])
    assert "GOOD=value" in cmd
    assert "9bad" not in cmd
    assert "invalid key" in caplog.text


def test_docker_ship_dangerous_build_arg_value_skipped(
    docker_context: CommandContext, caplog: pytest.LogCaptureFixture
) -> None:
    docker_context.config.docker_ship.build_args = {"CMD": "foo$(bar)", "SAFE": "ok"}
    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run") as mock_run,
    ):
        docker_ship.plugin.pre_upload(docker_context)

    cmd = " ".join(mock_run.call_args[0][0])
    assert "SAFE=ok" in cmd
    assert "foo$(bar)" not in cmd
    assert "shell metacharacters" in caplog.text
