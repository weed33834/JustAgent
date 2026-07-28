"""Docker build/push adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path

import structlog

from justagent.adapters.upload.base import UploadAdapter, UploadResult
from justagent.exceptions import UploadError
from justagent.models.config import ToolsConfig
from justagent.utils.hashing import ToolVerifier
from justagent.utils.redaction import redact_text

logger = structlog.get_logger("justagent")

# Cap on the number of characters of captured docker output retained in error
# details so failed-build diagnostics stay useful without dumping full logs.
_MAX_OUTPUT_CHARS = 4000


def _tail(text: str) -> str:
    """Return the last ``_MAX_OUTPUT_CHARS`` characters of *text*."""
    if len(text) > _MAX_OUTPUT_CHARS:
        return text[-_MAX_OUTPUT_CHARS:]
    return text


class DockerUploader(UploadAdapter):
    """Build and push a Docker image."""

    name = "docker"

    def __init__(
        self,
        project_root: Path,
        image: str,
        tag: str = "latest",
        *,
        registry: str | None = None,
        tool_verifier: ToolVerifier | None = None,
    ) -> None:
        self.project_root = project_root
        self.image = image
        self.tag = tag
        self.registry = registry
        self._verifier = tool_verifier or ToolVerifier(ToolsConfig())

    @property
    def full_image(self) -> str:
        """Return the fully qualified image name including registry prefix."""
        if self.registry:
            return f"{self.registry.rstrip('/')}/{self.image}:{self.tag}"
        return f"{self.image}:{self.tag}"

    def validate(self) -> None:
        """Ensure Docker CLI is available."""
        if not self._verifier.check("docker"):
            raise UploadError("`docker` CLI not found for Docker upload")

    def upload(self, *, dry_run: bool = False, verbose: bool = False) -> UploadResult:
        """Build and push the configured image."""
        full_image = self.full_image
        details: dict[str, object] = {"image": full_image}
        if dry_run:
            details["dry_run"] = True
            return UploadResult(
                success=True,
                target=self.name,
                details=details,
            )

        self.validate()

        try:
            docker = self._verifier.resolve("docker")
            build_cmd = [docker, "build", "-t", full_image, "."]
            push_cmd = [docker, "push", full_image]
            if verbose:
                print(f"[exec] {' '.join(build_cmd)}")
                print(f"[exec] {' '.join(push_cmd)}")
            build_result = subprocess.run(
                build_cmd,
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug("docker build stdout: %s", build_result.stdout)
            push_result = subprocess.run(
                push_cmd,
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug("docker push stdout: %s", push_result.stdout)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            raise UploadError(
                f"Docker upload failed: {exc}",
                details={
                    "stderr": redact_text(_tail(stderr)),
                    "stdout": redact_text(_tail(stdout)),
                },
            ) from exc

        return UploadResult(
            success=True,
            target=self.name,
            details=details,
        )
