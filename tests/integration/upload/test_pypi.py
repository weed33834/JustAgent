"""Integration tests for real PyPI uploads against a local upload server.

These tests start a local HTTP server that mimics the legacy PyPI upload
endpoint (``POST /legacy/``) and use ``twine upload --repository-url`` to push
a freshly built wheel/sdist. When ``twine`` or ``python -m build`` is
unavailable, tests are skipped.
"""

from __future__ import annotations

import re
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from autoship.adapters.upload.pypi import PyPIUploader

from .conftest import find_free_port, run_cmd, tool_available

pytestmark = pytest.mark.integration


def _build_artifacts(project_root: Path) -> None:
    """Build wheel/sdist artifacts for ``project_root``."""
    python = shutil.which("python") or sys.executable
    run_cmd([python, "-m", "build", "--sdist", "--wheel"], cwd=project_root)


def _extract_filename(content_disposition: str) -> str | None:
    """Extract filename from a Content-Disposition header value."""
    match = re.search(r'filename="([^"]+)"', content_disposition)
    if match:
        return match.group(1)
    match = re.search(r"filename='([^']+)'", content_disposition)
    if match:
        return match.group(1)
    return None


class _UploadHandler(BaseHTTPRequestHandler):
    """Minimal legacy PyPI upload endpoint handler."""

    def __init__(self, packages_dir: Path, *args, **kwargs) -> None:
        self.packages_dir = packages_dir
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/legacy/":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)

        # Simple multipart parsing: look for Content-Disposition with filename.
        content_type = self.headers.get("Content-Type", "")
        boundary = b""
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[len("boundary=") :].strip('"').encode()
                break

        saved = False
        if boundary:
            for section in data.split(b"--" + boundary):
                header_end = section.find(b"\r\n\r\n")
                if header_end == -1:
                    continue
                headers = section[:header_end].decode("utf-8", errors="ignore")
                body = section[header_end + 4 :]
                if body.endswith(b"\r\n"):
                    body = body[:-2]
                filename = _extract_filename(headers)
                if filename and body:
                    (self.packages_dir / filename).write_bytes(body)
                    saved = True

        if saved:
            self.send_response(200)
        else:
            self.send_response(400)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logging during tests."""


@pytest.fixture
def local_pypi(tmp_path: Path):
    """Start a local upload server and yield its upload URL and packages dir."""
    if not tool_available("twine"):
        pytest.skip("twine is not available")
    if not tool_available("python"):
        pytest.skip("python is not available")

    packages_dir = tmp_path / "packages"
    packages_dir.mkdir()
    port = find_free_port()
    url = f"http://127.0.0.1:{port}/"
    upload_url = f"{url}legacy/"

    def handler(*args, **kwargs):
        return _UploadHandler(packages_dir, *args, **kwargs)

    server = HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield {"upload_url": upload_url, "packages_dir": packages_dir}
    finally:
        server.shutdown()


def test_pypi_upload_to_local_server(minimal_python_package: Path, local_pypi, monkeypatch) -> None:
    """Build and upload a real package to a local upload server."""
    _build_artifacts(minimal_python_package)

    # The local server accepts any credentials; twine still requires them.
    monkeypatch.setenv("TWINE_USERNAME", "user")
    monkeypatch.setenv("TWINE_PASSWORD", "pass")

    uploader = PyPIUploader(
        minimal_python_package,
        repository="testpypi",
        repository_url=local_pypi["upload_url"],
    )
    result = uploader.upload()

    assert result.success is True
    assert result.target == "pypi"
    packages_dir: Path = local_pypi["packages_dir"]
    uploaded = list(packages_dir.glob("*"))
    assert uploaded, "expected artifacts to be uploaded to local server"
    assert any("demo_pkg" in p.name for p in uploaded)


def test_pypi_upload_rejects_unsafe_http_url() -> None:
    """Non-local HTTP repository URLs are rejected by the CLI helper."""
    assert PyPIUploader.is_safe_repository_url("http://example.com/legacy/") is False
