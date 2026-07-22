"""Official docker-ship plugin for MyAgent-CLI.

Automatically builds a Docker image before ``myagent upload --target docker``
and optionally pushes it afterwards. The plugin degrades gracefully if Docker
is not installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any

import structlog

from myagent.core.context import CommandContext
from myagent.exceptions import UploadError
from myagent.hookspec import hookimpl

logger = structlog.get_logger("myagent")

_BUILD_ARG_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BUILD_ARG_DANGEROUS_RE = re.compile(r"[`$;&|<>(){}[\]\\!\n\r]")


class DockerShipPlugin:
    """Build and push Docker images around the upload command."""

    @hookimpl
    def pre_upload(self, context: CommandContext) -> None:
        """Build the Docker image when uploading to Docker."""
        if not _should_handle(context):
            return

        executable = shutil.which("docker")
        if executable is None:
            raise UploadError("docker command not found on PATH")

        image, tag = _resolve_image_and_tag(context)
        full_tag = f"{image}:{tag}"

        if context.dry_run:
            logger.info("[dry-run] Would build Docker image %s", full_tag)
            return

        cmd = [executable, "build", "-t", full_tag]
        for key, value in _sanitize_build_args(context.config.docker_ship.build_args).items():
            cmd.extend(["--build-arg", f"{key}={value}"])
        cmd.append(str(context.project_root))

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            raise UploadError(f"Docker build failed for {full_tag}: {exc}") from exc

        logger.info("Built Docker image %s", full_tag)

    @hookimpl
    def post_upload(self, context: CommandContext) -> None:
        """Push the Docker image after a successful upload if configured."""
        if not _should_handle(context):
            return

        if not context.config.docker_ship.push:
            return

        executable = shutil.which("docker")
        if executable is None:
            raise UploadError("docker command not found on PATH")

        image, tag = _resolve_image_and_tag(context)
        full_tag = f"{image}:{tag}"

        if context.dry_run:
            logger.info("[dry-run] Would push Docker image %s", full_tag)
            return

        try:
            subprocess.run(
                [executable, "push", full_tag],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            raise UploadError(f"Docker push failed for {full_tag}: {exc}") from exc

        logger.info("Pushed Docker image %s", full_tag)


def _should_handle(context: CommandContext) -> bool:
    """Return True when this plugin should act on the upload context."""
    if not context.config.docker_ship.enabled:
        return False
    return context.extras.get("target") == "docker"


def _resolve_image_and_tag(context: CommandContext) -> tuple[str, str]:
    """Resolve the image name and tag from CLI options or configuration."""
    config = context.config.docker_ship
    image: Any = context.extras.get("image") or config.default_image
    tag: Any = context.extras.get("tag") or config.default_tag
    if not image:
        raise UploadError(
            "Docker image name is required (use --image or set docker_ship.default_image)"
        )
    return str(image), str(tag)


def _sanitize_build_args(build_args: dict[str, str]) -> dict[str, str]:
    """Return build args with unsafe keys or values removed.

    Keys must be valid shell-style identifiers. Values containing shell
    metacharacters are dropped to prevent command substitution or chaining
    inside Docker ``RUN`` instructions.
    """
    safe: dict[str, str] = {}
    for key, value in build_args.items():
        if not _BUILD_ARG_KEY_RE.fullmatch(key):
            logger.warning("Skipping docker build-arg with invalid key: %r", key)
            continue
        if _BUILD_ARG_DANGEROUS_RE.search(value):
            logger.warning("Skipping docker build-arg %r: value contains shell metacharacters", key)
            continue
        safe[key] = value
    return safe


plugin = DockerShipPlugin()
