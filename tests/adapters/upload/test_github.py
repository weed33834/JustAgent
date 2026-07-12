"""Integration tests for the GitHub release upload adapter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from autoship.adapters.upload.github import GitHubUploader
from autoship.exceptions import UploadError


def _write_artifacts(root: Path, names: list[str]) -> list[Path]:
    """Create temporary release artifacts in the project root."""
    artifacts: list[Path] = []
    for name in names:
        artifact = root / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"artifact {name}", encoding="utf-8")
        artifacts.append(artifact)
    return artifacts


def _mock_run_for_github(cmd, **_kwargs):
    """Return fake output for gh repo view; otherwise behave as success."""
    if cmd[:4] == ["gh", "repo", "view", "--json"]:
        mock = type(
            "CompletedProcess", (), {"stdout": json.dumps({"url": "https://github.com/owner/repo"})}
        )()
        return mock
    return type("CompletedProcess", (), {"returncode": 0})()


def test_github_dry_run(tmp_path: Path) -> None:
    uploader = GitHubUploader(tmp_path, tag="v1.0.0")
    result = uploader.upload(dry_run=True)
    assert result.success is True
    assert result.target == "github"
    assert result.details["dry_run"] is True
    assert result.details["tag"] == "v1.0.0"


def test_github_validate_missing_cli(tmp_path: Path) -> None:
    uploader = GitHubUploader(tmp_path, tag="v1.0.0")
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(UploadError, match="gh` CLI not found"),
    ):
        uploader.validate()


def test_github_upload_success(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, ["dist/package.tar.gz", "dist/package.whl"])
    uploader = GitHubUploader(tmp_path, tag="v1.0.0", artifacts=["dist/*"])
    with (
        patch("shutil.which", return_value="/usr/bin/gh"),
        patch("subprocess.run", side_effect=_mock_run_for_github) as mock_run,
    ):
        result = uploader.upload()

    assert result.success is True
    assert result.target == "github"
    assert result.url == "https://github.com/owner/repo/releases/tag/v1.0.0"
    assert result.details["tag"] == "v1.0.0"
    assert result.details["artifacts"] == ["dist/*"]
    assert mock_run.call_count == 3

    repo_call, create_call, upload_call = mock_run.call_args_list
    assert repo_call.args[0] == ["gh", "repo", "view", "--json", "url"]
    assert create_call.args[0] == [
        "gh",
        "release",
        "create",
        "v1.0.0",
        "--generate-notes",
    ]
    assert create_call.kwargs["cwd"] == tmp_path
    assert create_call.kwargs["check"] is True
    assert upload_call.args[0] == ["gh", "release", "upload", "v1.0.0", "dist/*"]
    assert upload_call.kwargs["cwd"] == tmp_path
    assert upload_call.kwargs["check"] is True


def test_github_upload_failure_raises_upload_error(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, ["dist/package.tar.gz"])
    uploader = GitHubUploader(tmp_path, tag="v1.0.0", artifacts=["dist/*"])

    def _fail_create(cmd, **_kwargs) -> None:
        if cmd[:4] == ["gh", "repo", "view", "--json"]:
            return type(
                "CompletedProcess",
                (),
                {"stdout": json.dumps({"url": "https://github.com/owner/repo"})},
            )()
        raise subprocess.CalledProcessError(1, cmd)

    with (
        patch("shutil.which", return_value="/usr/bin/gh"),
        patch("subprocess.run", side_effect=_fail_create),
        pytest.raises(UploadError, match="GitHub release failed"),
    ):
        uploader.upload()


def test_github_upload_verbose_prints_commands(tmp_path: Path, capsys) -> None:
    _write_artifacts(tmp_path, ["dist/package.tar.gz"])
    uploader = GitHubUploader(tmp_path, tag="v1.0.0", artifacts=["dist/*"])
    with (
        patch("shutil.which", return_value="/usr/bin/gh"),
        patch("subprocess.run", side_effect=_mock_run_for_github) as mock_run,
    ):
        uploader.upload(verbose=True)

    captured = capsys.readouterr()
    assert "gh release create v1.0.0 --generate-notes" in captured.out
    assert "gh release upload v1.0.0 dist/*" in captured.out
    assert mock_run.call_count == 3


def test_github_upload_succeeds_when_repo_view_returns_invalid_json(tmp_path: Path) -> None:
    """A non-JSON ``gh repo view`` stdout must not abort the upload.

    The implementation catches :class:`json.JSONDecodeError`, logs a warning,
    leaves ``repo_url`` empty, and still proceeds to create the release. The
    resulting ``UploadResult.url`` is an empty string because no repo URL could
    be parsed.
    """
    _write_artifacts(tmp_path, ["dist/package.tar.gz"])

    def _run_with_bad_json(cmd, **_kwargs):
        if cmd[:4] == ["gh", "repo", "view", "--json"]:
            # Simulate `gh` emitting a warning line followed by no JSON.
            return type("CompletedProcess", (), {"stdout": "warning: not a json\n"})()
        return type("CompletedProcess", (), {"returncode": 0})()

    uploader = GitHubUploader(tmp_path, tag="v1.0.0", artifacts=["dist/*"])
    with (
        patch("shutil.which", return_value="/usr/bin/gh"),
        patch("subprocess.run", side_effect=_run_with_bad_json),
    ):
        result = uploader.upload()

    assert result.success is True
    assert result.target == "github"
    # No repo URL could be parsed -> empty release URL, not an exception.
    assert result.url == ""
    assert result.details["tag"] == "v1.0.0"
