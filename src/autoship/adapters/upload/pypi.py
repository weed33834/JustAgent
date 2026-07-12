"""PyPI upload adapter."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from autoship.adapters.upload.base import UploadAdapter, UploadResult
from autoship.exceptions import UploadError
from autoship.models.config import ToolsConfig
from autoship.utils.hashing import ToolVerifier


class PyPIUploader(UploadAdapter):
    """Publish Python packages to PyPI or TestPyPI."""

    name = "pypi"

    def __init__(
        self,
        project_root: Path,
        repository: str = "testpypi",
        repository_url: str | None = None,
        *,
        tool_verifier: ToolVerifier | None = None,
    ) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9\-]+", repository):
            raise ValueError(
                f"Invalid repository name: {repository!r}. "
                "Only alphanumeric characters and hyphens are allowed."
            )
        self.project_root = project_root
        self.repository = repository
        self.repository_url = repository_url
        self._verifier = tool_verifier or ToolVerifier(ToolsConfig())

    def validate(self) -> None:
        """Ensure build tooling is installed."""
        for tool in ("python", "twine"):
            if not self._verifier.check(tool):
                raise UploadError(f"Required tool `{tool}` not found for PyPI upload")

    def upload(self, *, dry_run: bool = False, verbose: bool = False) -> UploadResult:
        """Build and upload distribution artifacts."""
        details: dict[str, object] = {
            "repository": self.repository,
        }
        if self.repository_url:
            details["repository_url"] = self.repository_url
        if dry_run:
            details["dry_run"] = True
            return UploadResult(
                success=True,
                target=self.name,
                details=details,
            )

        self.validate()

        try:
            python = self._verifier.resolve("python")
            twine = self._verifier.resolve("twine")
            subprocess.run(
                [python, "-m", "build", "--sdist", "--wheel"],
                cwd=self.project_root,
                check=True,
            )
            dist_dir = self.project_root / "dist"
            artifacts = sorted(dist_dir.glob("*"))
            if not artifacts:
                raise UploadError("No distribution artifacts found in dist/")
            cmd = [twine, "upload"]
            if self.repository_url:
                cmd.extend(["--repository-url", self.repository_url])
            else:
                cmd.extend(["--repository", self.repository])
            cmd.extend(str(path) for path in artifacts)
            if verbose:
                print(f"[exec] {' '.join(cmd)}")
            subprocess.run(cmd, cwd=self.project_root, check=True, shell=False)
        except subprocess.CalledProcessError as exc:
            raise UploadError(f"PyPI upload failed: {exc}") from exc

        return UploadResult(
            success=True,
            target=self.name,
            details=details,
        )

    @staticmethod
    def is_safe_repository_url(url: str) -> bool:
        """Return True if ``url`` is HTTPS or points to localhost/loopback."""
        parsed = urlparse(url)
        if parsed.scheme == "https":
            return True
        if parsed.scheme == "http":
            hostname = parsed.hostname or ""
            return hostname in ("localhost", "127.0.0.1", "::1")
        return False
