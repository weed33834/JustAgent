"""Disk cache wrapper around diskcache.Cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diskcache import Cache


class DiskCache:
    """Thread/process-safe disk cache backed by diskcache."""

    def __init__(self, cache_dir: Path | None = None, default_ttl: int = 3600) -> None:
        d = str(cache_dir or Path.home() / ".autoship" / "cache")
        self._cache = Cache(d)
        self.default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._cache.set(key, value, expire=ttl if ttl is not None else self.default_ttl)

    def invalidate(self, key: str) -> None:
        self._cache.delete(key)

    def clear(self) -> None:
        self._cache.clear()
