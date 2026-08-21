"""Tests for the disk-cache-backed Cache (diskcache implementation)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from justagent.core.cache import Cache


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    """Return a Cache rooted in a temporary directory."""
    return Cache(cache_dir=tmp_path)


def test_get_returns_none_for_missing_key(cache: Cache) -> None:
    assert cache.get("missing") is None


def test_get_returns_default_for_missing_key(cache: Cache) -> None:
    assert cache.get("missing", default="fallback") == "fallback"


def test_set_and_get_roundtrip(cache: Cache) -> None:
    cache.set("key", {"data": "value"})
    assert cache.get("key") == {"data": "value"}


def test_set_overwrites_existing_value(cache: Cache) -> None:
    cache.set("key", "first")
    cache.set("key", "second")
    assert cache.get("key") == "second"


def test_set_with_ttl_stores_value(cache: Cache) -> None:
    cache.set("key", "value", ttl=3600)
    assert cache.get("key") == "value"


def test_invalidate(cache: Cache) -> None:
    cache.set("key", "value")
    assert cache.get("key") == "value"
    cache.invalidate("key")
    assert cache.get("key") is None


def test_invalidate_missing_key_is_noop(cache: Cache) -> None:
    cache.invalidate("never-set")


def test_clear(cache: Cache) -> None:
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_concurrent_access(cache: Cache) -> None:
    """diskcache.Cache is thread-safe; parallel workers must not corrupt state."""
    errors: list[Exception] = []
    results: list[Any] = []

    def worker(index: int) -> None:
        try:
            key = f"key-{index % 5}"
            cache.set(key, index)
            value = cache.get(key)
            results.append(value)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == 100


def test_directory_property(cache: Cache, tmp_path: Path) -> None:
    assert cache.directory == str(tmp_path)


def test_size_is_non_negative_int(cache: Cache) -> None:
    cache.set("key", "payload")
    assert isinstance(cache.size, int)
    assert cache.size >= 0


def test_default_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = Path("/tmp/fake-home")
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    default_cache = Cache()
    assert default_cache.directory == str(fake_home / ".cache" / "justagent")
