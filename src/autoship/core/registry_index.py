"""Official plugin registry index loader and search."""

from __future__ import annotations

import threading
import time
from typing import Any, cast

from autoship.core.registry_client import RegistryClient
from autoship.models.config import AppConfig

# Module-level lock guarding ``RegistryIndex._process_cache`` (the cache is
# shared across all instances, so the lock is module-level too).
_LOCK: threading.Lock = threading.Lock()


class RegistryIndex:
    """Load and query the official plugin registry index.

    A process-level cache (``_process_cache``) shares parsed index data across
    instances created within the same CLI invocation, avoiding redundant file
    reads and JSON parsing when multiple commands query the registry.
    """

    # Process-level cache: url -> (data, cached_at_timestamp)
    _process_cache: dict[str, tuple[dict[str, Any], float]] = {}

    def __init__(self, config: AppConfig | None = None) -> None:
        self.client = RegistryClient(config.registry if config else None)
        self._data: dict[str, Any] | None = None

    def _cache_key(self) -> str:
        return str(self.client.config.url)

    def load(self, *, no_cache: bool = False) -> dict[str, Any]:
        """Load the registry index via the caching client."""
        if self._data is not None and not no_cache:
            return self._data

        cache_key = self._cache_key()

        # Fast path: check the process-level cache without the lock.
        if not no_cache:
            cached = RegistryIndex._process_cache.get(cache_key)
            if cached is not None:
                data, cached_at = cached
                ttl = self.client.config.cache_ttl_seconds
                if time.time() - cached_at < ttl:
                    self._data = data
                    return data

        # Miss (or bypassed): acquire the lock and re-check before performing
        # the expensive fetch, so concurrent callers don't duplicate the work.
        with _LOCK:
            if not no_cache:
                cached = RegistryIndex._process_cache.get(cache_key)
                if cached is not None:
                    data, cached_at = cached
                    ttl = self.client.config.cache_ttl_seconds
                    if time.time() - cached_at < ttl:
                        self._data = data
                        return data

            data = self.client.get(no_cache=no_cache)
            self._data = data

            if not no_cache:
                RegistryIndex._process_cache[cache_key] = (data, time.time())

            return data

    @classmethod
    def invalidate_process_cache(cls) -> None:
        """Clear the process-level cache (e.g. after a registry sync)."""
        with _LOCK:
            cls._process_cache.clear()

    def list_plugins(self) -> list[dict[str, Any]]:
        """Return all plugins in the index."""
        return cast(list[dict[str, Any]], self.load().get("plugins", []))

    def search(self, keyword: str | None = None) -> list[dict[str, Any]]:
        """Search plugins by keyword in name or description."""
        plugins = self.list_plugins()
        if not keyword:
            return plugins
        keyword_lower = keyword.lower()
        return [
            plugin
            for plugin in plugins
            if keyword_lower in plugin.get("name", "").lower()
            or keyword_lower in plugin.get("description", "").lower()
        ]

    def get(self, name: str) -> dict[str, Any] | None:
        """Return a plugin by exact name."""
        for plugin in self.list_plugins():
            if plugin.get("name") == name:
                return plugin
        return None
