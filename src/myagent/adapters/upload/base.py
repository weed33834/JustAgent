"""Abstract base class for upload/publish adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class UploadResult:
    """Result of an upload operation."""

    success: bool
    target: str
    url: str | None = None
    details: dict[str, Any] | None = None


class UploadAdapter(ABC):
    """Abstract adapter for publishing artifacts."""

    name: str = ""

    @abstractmethod
    def validate(self) -> None:
        """Validate that required configuration and tooling are present."""

    @abstractmethod
    def upload(self, *, dry_run: bool = False, verbose: bool = False) -> UploadResult:
        """Execute the upload/publish action."""

    def _artifact_paths(self, patterns: list[str], project_root: Path) -> list[Path]:
        """Resolve glob patterns to concrete file paths."""
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(project_root.glob(pattern))
        return sorted(paths)
