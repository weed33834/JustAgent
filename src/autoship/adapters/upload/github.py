"""GitHub release adapter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import structlog

from autoship.adapters.upload.base import UploadAdapter, UploadResult
from autoship.exceptions import UploadError
from autoship.models.config import ToolsConfig
from autoship.utils.hashing import ToolVerifier

logger = structlog.get_logger("autoship")


class GitHubUploader(UploadAdapter):
    """Create a GitHub release and upload artifacts."""

    name = "github"

    def __init__(
        self,
        project_root: Path,
        tag: str,
        artifacts: list[str] | None = None,
        *,
        tool_verifier: ToolVerifier | None = None,
    ) -> None:
        self.project_root = project_root
        self.tag = tag
        self.artifacts = artifacts or ["dist/*"]
        self._verifier = tool_verifier or ToolVerifier(ToolsConfig())

    def validate(self) -> None:
        """Ensure GitHub CLI is available."""
        if not self._verifier.check("gh"):
            raise UploadError("`gh` CLI not found for GitHub release upload")

    def upload(self, *, dry_run: bool = False, verbose: bool = False) -> UploadResult:
        """Create a GitHub release and attach artifacts."""
        if dry_run:
            return UploadResult(
                success=True,
                target=self.name,
                details={"tag": self.tag, "artifacts": self.artifacts, "dry_run": True},
            )

        self.validate()

        try:
            gh = self._verifier.resolve("gh")
            repo_info = subprocess.run(
                [gh, "repo", "view", "--json", "url"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise UploadError(f"GitHub release failed: {exc}") from exc

        repo_url = ""
        try:
            repo_url = json.loads(repo_info.stdout).get("url", "").rstrip("/")
        except json.JSONDecodeError as exc:
            # ``gh repo view --json url`` may emit warnings or partial output
            # that is not valid JSON. The release can still be created; we just
            # cannot construct a friendly release URL.
            logger.warning("failed to parse gh repo view output: %s", exc)

        try:
            create_cmd = [gh, "release", "create", self.tag, "--generate-notes"]
            upload_cmd = [gh, "release", "upload", self.tag, *self.artifacts]
            if verbose:
                print(f"[exec] {' '.join(create_cmd)}")
                print(f"[exec] {' '.join(upload_cmd)}")
            subprocess.run(create_cmd, cwd=self.project_root, check=True)
            subprocess.run(upload_cmd, cwd=self.project_root, check=True)
        except subprocess.CalledProcessError as exc:
            raise UploadError(f"GitHub release failed: {exc}") from exc

        release_url = f"{repo_url}/releases/tag/{self.tag}" if repo_url else ""
        return UploadResult(
            success=True,
            target=self.name,
            url=release_url,
            details={"tag": self.tag, "artifacts": self.artifacts},
        )
