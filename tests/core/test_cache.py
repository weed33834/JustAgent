"""Tests for the disk cache."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from justagent.core.cache import DiskCache


@pytest.fixture
def cache(tmp_path: Path) -> DiskCache:
    """Return a DiskCache rooted in a temporary directory."""
    return DiskCache(cache_dir=tmp_path, default_ttl=3600)


def test_get_returns_none_for_missing_key(cache: DiskCache) -> None:
    assert cache.get("missing") is None


def test_set_and_get_roundtrip(cache: DiskCache) -> None:
    cache.set("key", {"data": "value"})
    assert cache.get("key") == {"data": "value"}


def test_set_overwrites_existing_value(cache: DiskCache) -> None:
    cache.set("key", "first")
    cache.set("key", "second")
    assert cache.get("key") == "second"


def test_ttl_expiration(cache: DiskCache, monkeypatch) -> None:
    """A TTL of 0 means the entry is immediately expired on the next read.

    Uses a monkeypatched ``time.time`` to deterministically advance the clock
    past the expiry timestamp, avoiding flaky ``time.sleep`` based tests.
    """
    now = [time.time()]

    def advancing_time() -> float:
        return now[0]

    monkeypatch.setattr("justagent.core.cache.time.time", advancing_time)
    cache.set("key", "value", ttl=0)
    # Advance the clock so get() sees a timestamp strictly greater than the
    # expiry written by set().
    now[0] += 1.0
    assert cache.get("key") is None


def test_ttl_override_uses_default(cache: DiskCache) -> None:
    cache.set("key", "value")
    assert cache.get("key") == "value"


def test_invalidate(cache: DiskCache) -> None:
    cache.set("key", "value")
    assert cache.get("key") == "value"
    cache.invalidate("key")
    assert cache.get("key") is None


def test_invalidate_missing_key_is_noop(cache: DiskCache) -> None:
    cache.invalidate("never-set")


def test_clear(cache: DiskCache) -> None:
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_corrupted_entry_is_treated_as_missing(cache: DiskCache) -> None:
    cache.set("key", "value")
    path = cache._cache_path("key")
    path.write_text("not json", encoding="utf-8")
    assert cache.get("key") is None


def test_concurrent_access(cache: DiskCache) -> None:
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


def test_default_cache_dir_is_myagent_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = Path("/tmp/fake-home")
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    default_cache = DiskCache()
    assert default_cache.cache_dir == fake_home / ".justagent" / "cache"
