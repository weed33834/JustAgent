"""Integration tests for the PyPI upload adapter.

Note:
    The current ``PyPIUploader`` delegates authentication to the ``twine``
    CLI (e.g. via ``.pypirc`` or environment variables such as
    ``TWINE_USERNAME`` / ``TWINE_PASSWORD``). It does not read a
    ``PYPI_TOKEN`` environment variable itself, so the tests below verify
    the command invocation only. Explicit token handling is a documented
    implementation gap.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from autoship.adapters.upload.pypi import PyPIUploader
from autoship.exceptions import UploadError


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://test.pypi.org/legacy/", True),
        ("http://localhost:8080/simple/", True),
        ("http://127.0.0.1:8080/simple/", True),
        ("http://example.com/simple/", False),
        ("ftp://example.com/simple/", False),
    ],
)
def test_is_safe_repository_url(url: str, expected: bool) -> None:
    assert PyPIUploader.is_safe_repository_url(url) is expected


def _write_dist_artifacts(root: Path) -> None:
    """Create temporary wheel/sdist artifacts in ``dist/``."""
    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "autoship-1.0.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
    (dist / "autoship-1.0.0.tar.gz").write_text("sdist", encoding="utf-8")


def test_pypi_dry_run(tmp_path: Path) -> None:
    uploader = PyPIUploader(tmp_path)
    result = uploader.upload(dry_run=True)
    assert result.success is True
    assert result.target == "pypi"
    assert result.details["repository"] == "testpypi"
    assert result.details["dry_run"] is True


def test_pypi_validate_missing_tools(tmp_path: Path) -> None:
    uploader = PyPIUploader(tmp_path)
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(UploadError, match="Required tool"),
    ):
        uploader.validate()


def test_pypi_upload_success(tmp_path: Path) -> None:
    _write_dist_artifacts(tmp_path)
    uploader = PyPIUploader(tmp_path, repository="pypi")
    with (
        patch("shutil.which", return_value="/usr/bin/twine"),
        patch("subprocess.run") as mock_run,
    ):
        result = uploader.upload()

    assert result.success is True
    assert result.target == "pypi"
    assert result.details["repository"] == "pypi"
    assert mock_run.call_count == 2

    build_call, upload_call = mock_run.call_args_list
    assert build_call.args[0] == ["python", "-m", "build", "--sdist", "--wheel"]
    assert build_call.kwargs["cwd"] == tmp_path
    assert build_call.kwargs["check"] is True
    assert upload_call.args[0][:4] == [
        "twine",
        "upload",
        "--repository",
        "pypi",
    ]
    assert len(upload_call.args[0]) == 6
    assert all(str(tmp_path / "dist") in arg for arg in upload_call.args[0][4:])
    assert upload_call.kwargs["cwd"] == tmp_path
    assert upload_call.kwargs["check"] is True
    assert upload_call.kwargs["shell"] is False


def test_pypi_upload_with_repository_url(tmp_path: Path) -> None:
    _write_dist_artifacts(tmp_path)
    uploader = PyPIUploader(
        tmp_path, repository="pypi", repository_url="https://test.pypi.org/legacy/"
    )
    with (
        patch("shutil.which", return_value="/usr/bin/twine"),
        patch("subprocess.run") as mock_run,
    ):
        result = uploader.upload()

    assert result.success is True
    assert result.details["repository_url"] == "https://test.pypi.org/legacy/"

    _build_call, upload_call = mock_run.call_args_list
    assert upload_call.args[0][:4] == [
        "twine",
        "upload",
        "--repository-url",
        "https://test.pypi.org/legacy/",
    ]


def test_pypi_upload_failure_raises_upload_error(tmp_path: Path) -> None:
    _write_dist_artifacts(tmp_path)
    uploader = PyPIUploader(tmp_path)

    def _fail_upload(*_args, **_kwargs) -> None:
        raise subprocess.CalledProcessError(1, ["twine", "upload"])

    with (
        patch("shutil.which", return_value="/usr/bin/twine"),
        patch("subprocess.run", side_effect=_fail_upload),
        pytest.raises(UploadError, match="PyPI upload failed"),
    ):
        uploader.upload()


def test_pypi_upload_verbose_prints_command(tmp_path: Path, capsys) -> None:
    _write_dist_artifacts(tmp_path)
    uploader = PyPIUploader(tmp_path)
    with (
        patch("shutil.which", return_value="/usr/bin/twine"),
        patch("subprocess.run") as mock_run,
    ):
        uploader.upload(verbose=True)

    captured = capsys.readouterr()
    assert "twine upload --repository testpypi" in captured.out
    assert "dist" in captured.out
    assert mock_run.call_count == 2
