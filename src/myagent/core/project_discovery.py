"""Project discovery — scan the filesystem for development projects.

Walks a directory tree looking for "project markers" (``.git``,
``pyproject.toml``, ``package.json``, ``Cargo.toml``, ``go.mod``, etc.)
and returns :class:`DiscoveredProject` records. Used by the
``myagent project scan`` command to let users find and register
projects without manually adding each one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from myagent.core.project_store import ProjectStore
from myagent.exceptions import MyAgentError
from myagent.models.project import ManagedProject


class ProjectDiscoveryError(MyAgentError):
    """Raised when project discovery fails (bad root, IO errors, etc.)."""


class ProjectType(str, Enum):  # noqa: UP042 - match existing codebase style
    """The kind of project detected from filesystem markers."""

    PYTHON = "python"
    NODE = "node"
    RUST = "rust"
    GO = "go"
    GENERIC = "generic"


# Priority ordering: a more specific language marker wins over a generic one
# when a directory contains several markers at once (e.g. ``pyproject.toml``
# beats ``Makefile``).
_TYPE_PRIORITY: dict[ProjectType, int] = {
    ProjectType.PYTHON: 0,
    ProjectType.NODE: 1,
    ProjectType.RUST: 2,
    ProjectType.GO: 3,
    ProjectType.GENERIC: 4,
}


def _default_markers() -> dict[str, ProjectType]:
    """Return the default marker filename -> project type mapping."""
    return {
        ".git": ProjectType.GENERIC,
        "pyproject.toml": ProjectType.PYTHON,
        "setup.py": ProjectType.PYTHON,
        "package.json": ProjectType.NODE,
        "Cargo.toml": ProjectType.RUST,
        "go.mod": ProjectType.GO,
        "pom.xml": ProjectType.GENERIC,
        "build.gradle": ProjectType.GENERIC,
        "CMakeLists.txt": ProjectType.GENERIC,
        "Makefile": ProjectType.GENERIC,
    }


def _default_ignore_dirs() -> list[str]:
    """Return the default list of directory names to skip while scanning."""
    return [
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        "target",
        ".next",
        ".myagent",
    ]


@dataclass(frozen=True)
class DiscoveredProject:
    """A project found during a filesystem scan."""

    path: Path
    name: str
    project_type: ProjectType
    markers: list[str]
    has_git: bool


@dataclass
class DiscoveryConfig:
    """Configuration for :class:`ProjectDiscovery`."""

    markers: dict[str, ProjectType] = field(default_factory=_default_markers)
    ignore_dirs: list[str] = field(default_factory=_default_ignore_dirs)
    max_depth: int = 4
    max_projects: int = 200


class ProjectDiscovery:
    """Scan the filesystem for development projects.

    The scanner walks a directory tree looking for project markers. Once a
    directory is identified as a project (it contains at least one marker),
    the scanner does **not** descend into that directory's subdirectories —
    they are assumed to belong to the project. The one exception is a
    subdirectory that has its own ``.git``: that is treated as a separate
    nested repository and scanned independently.

    Symlinks are skipped to avoid infinite loops, and directories that
    cannot be read (``PermissionError`` / ``OSError``) are skipped
    gracefully.
    """

    def __init__(self, config: DiscoveryConfig | None = None) -> None:
        self.config: DiscoveryConfig = config or DiscoveryConfig()

    def detect_type(
        self, directory: Path
    ) -> tuple[ProjectType, list[str], bool]:
        """Detect the project type from markers in ``directory``.

        Returns a tuple of ``(project_type, markers_found, has_git)``. When
        several markers are present the most specific language type wins
        (e.g. Python beats Generic). ``markers_found`` is sorted
        alphabetically. A directory with no markers returns
        ``(ProjectType.GENERIC, [], False)``.
        """
        markers_found: list[str] = []
        has_git = False
        best_type: ProjectType | None = None
        for marker, ptype in self.config.markers.items():
            try:
                if (directory / marker).exists():
                    markers_found.append(marker)
                    if marker == ".git":
                        has_git = True
                    if (
                        best_type is None
                        or _TYPE_PRIORITY[ptype] < _TYPE_PRIORITY[best_type]
                    ):
                        best_type = ptype
            except OSError:
                # exists() normally swallows errors, but be defensive.
                continue
        if best_type is None:
            return (ProjectType.GENERIC, [], False)
        return (best_type, sorted(markers_found), has_git)

    def discover(self, root: str | Path) -> list[DiscoveredProject]:
        """Scan ``root`` recursively and return discovered projects.

        Results are sorted by path. Raises :class:`ProjectDiscoveryError`
        if ``root`` does not exist or is not a directory.
        """
        root_path = Path(root)
        if not root_path.exists():
            raise ProjectDiscoveryError(
                f"scan root does not exist: {root_path}"
            )
        if not root_path.is_dir():
            raise ProjectDiscoveryError(
                f"scan root is not a directory: {root_path}"
            )
        results: list[DiscoveredProject] = []
        self._scan(root_path, 0, results)
        results.sort(key=lambda d: d.path)
        return results

    def _scan(
        self, directory: Path, depth: int, results: list[DiscoveredProject]
    ) -> None:
        """Recursively scan ``directory`` up to ``max_depth``."""
        if depth > self.config.max_depth:
            return
        if len(results) >= self.config.max_projects:
            return

        try:
            ptype, markers, has_git = self.detect_type(directory)
            children = list(directory.iterdir())
        except (PermissionError, OSError):
            return

        if markers:
            results.append(
                DiscoveredProject(
                    path=directory,
                    name=directory.name,
                    project_type=ptype,
                    markers=markers,
                    has_git=has_git,
                )
            )
            if len(results) >= self.config.max_projects:
                return
            # Do not descend into a project's own subdirectories, except to
            # find nested repositories (a subdirectory with its own .git).
            for child in children:
                if child.is_symlink():
                    continue
                if not child.is_dir():
                    continue
                if child.name in self.config.ignore_dirs:
                    continue
                if (child / ".git").exists():
                    self._scan(child, depth + 1, results)
            return

        # Not a project: descend into children to look for projects.
        for child in children:
            if child.is_symlink():
                continue
            if not child.is_dir():
                continue
            if child.name in self.config.ignore_dirs:
                continue
            self._scan(child, depth + 1, results)

    def discover_and_register(
        self,
        root: str | Path,
        store: ProjectStore,
        *,
        tags: list[str] | None = None,
        dry_run: bool = False,
    ) -> list[ManagedProject]:
        """Discover projects under ``root`` and add them to ``store``.

        Returns the list of **newly added** projects (projects that were not
        already present in the store). Projects that already exist in the
        store are updated in place (their path is refreshed and tags are
        applied) but are not included in the returned list.

        When ``dry_run`` is true, nothing is written to the store; the
        returned list reflects the projects that *would* be newly added.
        """
        applied_tags = list(tags) if tags else []
        discovered = self.discover(root)
        added: list[ManagedProject] = []
        for discovered_proj in discovered:
            existing = store.get(discovered_proj.name)
            if existing is None:
                project = ManagedProject(
                    name=discovered_proj.name,
                    path=str(discovered_proj.path),
                    added_at=time.time(),
                    tags=list(applied_tags),
                )
                if not dry_run:
                    store.add(project)
                added.append(project)
            else:
                # Update in place: refresh path, preserve added_at, apply tags.
                updated = ManagedProject(
                    name=existing.name,
                    path=str(discovered_proj.path),
                    added_at=existing.added_at,
                    tags=list(applied_tags) if applied_tags else list(existing.tags),
                )
                if not dry_run:
                    store.add(updated)
        return added
