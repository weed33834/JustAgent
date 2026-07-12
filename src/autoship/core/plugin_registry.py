"""Local plugin registry and trust level management."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from autoship.utils.json_io import load_json, save_json

logger = structlog.get_logger("autoship")


DEFAULT_REGISTRY_DIR = Path.home() / ".config" / "autoship"


class TrustLevel(str, Enum):
    """Trust levels for installed plugins."""

    BUILTIN = "builtin"
    VERIFIED = "verified"
    COMMUNITY = "community"
    UNTRUSTED = "untrusted"


class CapabilityManifest(BaseModel):
    """Permission/capability manifest for a plugin.

    Plugins declare what resources they need so the CLI can warn users
    and, in the future, enforce sandbox restrictions.
    """

    filesystem: str = "read-only"
    network: bool = False
    shell: bool = False
    git: bool = False
    env: list[str] = Field(default_factory=list)

    def summary(self) -> list[str]:
        """Return a human-readable list of declared capabilities."""
        items = [f"filesystem={self.filesystem}"]
        if self.network:
            items.append("network=yes")
        if self.shell:
            items.append("shell=yes")
        if self.git:
            items.append("git=yes")
        if self.env:
            items.append(f"env={','.join(self.env)}")
        return items


class PluginSpec(BaseModel):
    """Metadata for a registered plugin."""

    name: str
    version: str = "0.0.0"
    source: str
    package: str | None = None
    entry_point: str | None = None
    hooks: list[str] = Field(default_factory=list)
    trust_level: TrustLevel = TrustLevel.COMMUNITY
    capabilities: CapabilityManifest = Field(default_factory=CapabilityManifest)
    sha256: str | None = None
    signature: str | None = None
    maintainer: str | None = None
    license: str | None = None


class PluginRegistry:
    """Manage a local JSON registry of installed plugins."""

    def __init__(self, registry_dir: Path | None = None) -> None:
        self.registry_dir = registry_dir or DEFAULT_REGISTRY_DIR
        self.registry_file = self.registry_dir / "registry.json"
        self._plugins: dict[str, PluginSpec] = {}
        self._load()

    def list(self) -> list[PluginSpec]:
        """Return all registered plugins sorted by name."""
        return sorted(self._plugins.values(), key=lambda p: p.name)

    def get(self, name: str) -> PluginSpec | None:
        """Return a plugin by name, or None if not registered."""
        return self._plugins.get(name)

    def add(self, spec: PluginSpec) -> None:
        """Add or update a plugin in the registry."""
        self._plugins[spec.name] = spec
        self._save()

    def remove(self, name: str) -> bool:
        """Remove a plugin from the registry."""
        if name not in self._plugins:
            return False
        del self._plugins[name]
        self._save()
        return True

    def trust(self, name: str, level: TrustLevel) -> bool:
        """Update the trust level of a registered plugin."""
        plugin = self._plugins.get(name)
        if plugin is None:
            return False
        self._plugins[name] = plugin.model_copy(update={"trust_level": level})
        self._save()
        return True

    def _load(self) -> None:
        """Load the registry from disk."""
        data = load_json(self.registry_file, label="plugin registry", check_perms=True)
        if not data:
            return
        for item in data.get("plugins", []):
            try:
                spec = PluginSpec.model_validate(item)
                self._plugins[spec.name] = spec
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid registry entry: %s", exc)

    def _save(self) -> None:
        """Persist the registry to disk."""
        payload = {"plugins": [spec.model_dump(mode="json") for spec in self._plugins.values()]}
        save_json(self.registry_file, payload, label="plugin registry")
