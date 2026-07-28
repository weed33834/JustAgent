"""Upload adapter registry/factory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from justagent.adapters.upload.base import UploadAdapter
from justagent.adapters.upload.docker import DockerUploader
from justagent.adapters.upload.github import GitHubUploader
from justagent.adapters.upload.pypi import PyPIUploader
from justagent.exceptions import ConfigError
from justagent.models.config import ToolsConfig
from justagent.utils.hashing import ToolVerifier

_Registry = dict[
    str,
    Callable[..., UploadAdapter],
]


def _docker_factory(
    root: Path,
    cfg: dict[str, Any],
    *,
    tool_verifier: ToolVerifier | None = None,
) -> DockerUploader:
    image = cfg.get("image")
    if not image:
        raise ConfigError("Upload target 'docker' requires '--image'")
    return DockerUploader(
        root,
        image=image,
        tag=cfg.get("tag", "latest"),
        registry=cfg.get("registry"),
        tool_verifier=tool_verifier,
    )


def _github_factory(
    root: Path,
    cfg: dict[str, Any],
    *,
    tool_verifier: ToolVerifier | None = None,
) -> GitHubUploader:
    tag = cfg.get("tag")
    if not tag:
        raise ConfigError("Upload target 'github' requires '--tag'")
    return GitHubUploader(
        root,
        tag=tag,
        artifacts=cfg.get("artifacts"),
        tool_verifier=tool_verifier,
    )


def _pypi_factory(
    root: Path,
    cfg: dict[str, Any],
    *,
    tool_verifier: ToolVerifier | None = None,
) -> PyPIUploader:
    return PyPIUploader(
        root,
        repository=cfg.get("repository", "testpypi"),
        repository_url=cfg.get("repository_url"),
        tool_verifier=tool_verifier,
    )


_UPLOADERS: _Registry = {
    "pypi": _pypi_factory,
    "docker": _docker_factory,
    "github": _github_factory,
}


def register_uploader(name: str, factory: Callable[..., UploadAdapter]) -> None:
    """Register a custom upload adapter factory."""
    _UPLOADERS[name] = factory


def get_uploader(
    name: str,
    project_root: Path,
    cfg: dict[str, Any] | None = None,
    *,
    tools: ToolsConfig | None = None,
) -> UploadAdapter:
    """Return an upload adapter instance for the named target.

    When ``tools`` is provided the matching ``ToolVerifier`` is forwarded to
    each uploader so that pinned binary paths / hashes from ``config.tools``
    are honoured by the actual subprocess invocations.
    """
    if name not in _UPLOADERS:
        raise ConfigError(f"Unknown upload target: {name}")
    verifier = ToolVerifier(tools) if tools else None
    factory = _UPLOADERS[name]
    try:
        return factory(project_root, cfg or {}, tool_verifier=verifier)
    except TypeError:
        # Backwards-compatible path for custom factories registered without
        # the ``tool_verifier`` keyword argument.
        return factory(project_root, cfg or {})
