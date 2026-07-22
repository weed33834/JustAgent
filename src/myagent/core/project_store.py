"""Persistent JSON store for managed projects.

Projects are persisted to ``~/.myagent/projects.json`` so the ``myagent
project`` command can recall them across invocations. A missing or corrupt
store file degrades gracefully to an empty project set rather than crashing
the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

from myagent.models.project import ManagedProject
from myagent.utils.atomic_write import atomic_write_text

DEFAULT_STORE_PATH = Path.home() / ".myagent" / "projects.json"


class ProjectStore:
    """Manage a JSON store of :class:`ManagedProject` records."""

    def __init__(self, store_path: Path | None = None) -> None:
        self.store_path: Path = store_path or DEFAULT_STORE_PATH

    def add(self, project: ManagedProject) -> None:
        """Add or update a project by name."""
        data = self._load()
        data[project.name] = project
        self._save(data)

    def remove(self, name: str) -> bool:
        """Remove a project by name; return True if it was present."""
        data = self._load()
        if name not in data:
            return False
        del data[name]
        self._save(data)
        return True

    def get(self, name: str) -> ManagedProject | None:
        """Return a project by name, or None if not present."""
        return self._load().get(name)

    def list_all(self) -> list[ManagedProject]:
        """Return all projects sorted by name."""
        return sorted(self._load().values(), key=lambda p: p.name)

    def _load(self) -> dict[str, ManagedProject]:
        """Load the store from disk; missing or corrupt files yield an empty dict."""
        if not self.store_path.exists():
            return {}
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        result: dict[str, ManagedProject] = {}
        for key, item in raw.items():
            if not isinstance(item, dict):
                continue
            try:
                project = ManagedProject(
                    name=str(item.get("name", key)),
                    path=str(item["path"]),
                    added_at=float(item["added_at"]),
                    tags=list(item.get("tags") or []),
                    description=str(item.get("description") or ""),
                )
            except (KeyError, TypeError, ValueError):
                continue
            result[project.name] = project
        return result

    def _save(self, data: dict[str, ManagedProject]) -> None:
        """Persist the store to disk as indented JSON."""
        payload = {
            p.name: {
                "name": p.name,
                "path": p.path,
                "added_at": p.added_at,
                "tags": list(p.tags),
                "description": p.description,
            }
            for p in data.values()
        }
        atomic_write_text(self.store_path, json.dumps(payload, indent=2))
