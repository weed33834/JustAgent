"""Integration tests for the Docker upload adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from autoship.adapters.upload.docker import DockerUploader
from autoship.exceptions import UploadError


def _write_docker_project(root: Path) -> None:
    """Create a minimal Docker project in the temporary root."""
    (root / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")


def test_docker_dry_run(tmp_path: Path) -> None:
    uploader = DockerUploader(tmp_path, image="myapp", tag="latest")
    result = uploader.upload(dry_run=True)
    assert result.success is True
    assert result.target == "docker"
    assert result.details["dry_run"] is True
    assert result.details["image"] == "myapp:latest"


def test_docker_validate_missing_cli(tmp_path: Path) -> None:
    uploader = DockerUploader(tmp_path, image="myapp")
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(UploadError, match="docker` CLI not found"),
    ):
        uploader.validate()


def test_docker_upload_success(tmp_path: Path) -> None:
    _write_docker_project(tmp_path)
    uploader = DockerUploader(tmp_path, image="myapp", tag="v1")
    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run") as mock_run,
    ):
        result = uploader.upload()

    assert result.success is True
    assert result.target == "docker"
    assert result.details["image"] == "myapp:v1"
    assert mock_run.call_count == 2

    build_call, push_call = mock_run.call_args_list
    assert build_call.args[0] == ["docker", "build", "-t", "myapp:v1", "."]
    assert build_call.kwargs["cwd"] == tmp_path
    assert build_call.kwargs["check"] is True
    assert push_call.args[0] == ["docker", "push", "myapp:v1"]
    assert push_call.kwargs["cwd"] == tmp_path
    assert push_call.kwargs["check"] is True


def test_docker_upload_with_registry(tmp_path: Path) -> None:
    _write_docker_project(tmp_path)
    uploader = DockerUploader(tmp_path, image="myapp", tag="v1", registry="localhost:5000")
    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run") as mock_run,
    ):
        result = uploader.upload()

    assert result.success is True
    assert result.details["image"] == "localhost:5000/myapp:v1"
    assert mock_run.call_count == 2

    build_call, push_call = mock_run.call_args_list
    assert build_call.args[0] == ["docker", "build", "-t", "localhost:5000/myapp:v1", "."]
    assert push_call.args[0] == ["docker", "push", "localhost:5000/myapp:v1"]


def test_docker_upload_failure_raises_upload_error(tmp_path: Path) -> None:
    _write_docker_project(tmp_path)
    uploader = DockerUploader(tmp_path, image="myapp", tag="v1")

    def _fail_build(*_args, **_kwargs) -> None:
        raise subprocess.CalledProcessError(1, ["docker", "build"])

    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run", side_effect=_fail_build),
        pytest.raises(UploadError, match="Docker upload failed"),
    ):
        uploader.upload()


def test_docker_upload_failure_redacts_and_tails_output(tmp_path: Path) -> None:
    """Failed-build details must contain redacted, tailed stderr/stdout.

    The implementation wraps the captured output through ``_tail`` (so only
    the last ``_MAX_OUTPUT_CHARS`` characters are retained) and then
    ``redact_text`` (so secret-like values such as GitHub tokens are masked).
    The two behaviours are exercised on separate streams because
    ``redact_text`` replaces the *entire* string with ``"***"`` when any
    secret pattern matches, which would otherwise hide the tail marker.
    """
    from autoship.adapters.upload.docker import _MAX_OUTPUT_CHARS

    _write_docker_project(tmp_path)
    uploader = DockerUploader(tmp_path, image="myapp", tag="v1")

    secret = "ghp_" + "a" * 36
    # stdout carries a secret token -> redact_text masks the whole string.
    stdout = f"using token {secret} to pull base image"
    # stderr is long but secret-free -> _tail keeps only the last chunk, with
    # a recognisable marker at the end.
    long_prefix = "x" * (_MAX_OUTPUT_CHARS + 200)
    stderr_tail_marker = "error: build failed near end"
    stderr = long_prefix + " " + stderr_tail_marker

    def _fail_build(*_args, **_kwargs) -> None:
        raise subprocess.CalledProcessError(1, ["docker", "build"], output=stdout, stderr=stderr)

    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run", side_effect=_fail_build),
        pytest.raises(UploadError) as exc_info,
    ):
        uploader.upload()

    details = exc_info.value.details
    assert isinstance(details, dict)
    assert "stderr" in details
    assert "stdout" in details

    # stdout: the secret was found -> whole string replaced with "***".
    assert secret not in details["stdout"]
    assert details["stdout"] == "***"

    # stderr: no secret -> _tail truncated to the last _MAX_OUTPUT_CHARS chars.
    assert secret not in details["stderr"]
    assert stderr_tail_marker in details["stderr"]
    assert len(details["stderr"]) <= _MAX_OUTPUT_CHARS
    # The long prefix was truncated away (only its tail end remains).
    assert details["stderr"].count("x") < len(long_prefix)


def test_docker_upload_verbose_prints_commands(tmp_path: Path, capsys) -> None:
    _write_docker_project(tmp_path)
    uploader = DockerUploader(tmp_path, image="myapp", tag="v1")
    with (
        patch("shutil.which", return_value="/usr/bin/docker"),
        patch("subprocess.run") as mock_run,
    ):
        uploader.upload(verbose=True)

    captured = capsys.readouterr()
    assert "docker build -t myapp:v1 ." in captured.out
    assert "docker push myapp:v1" in captured.out
    assert mock_run.call_count == 2
