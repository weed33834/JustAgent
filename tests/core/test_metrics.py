"""Tests for the metrics registry."""

from __future__ import annotations

import threading
import time

import pytest

from autoship.core.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    get_registry,
    set_registry,
)


@pytest.fixture
def registry() -> MetricsRegistry:
    """Provide a clean registry for each test."""
    reg = MetricsRegistry()
    set_registry(reg)
    yield reg
    reg.reset()


def test_counter_increments(registry: MetricsRegistry) -> None:
    counter = registry.counter("test_counter", "A test counter")
    counter.inc()
    counter.inc(4)
    assert counter.value == 5


def test_gauge_set_and_inc(registry: MetricsRegistry) -> None:
    gauge = registry.gauge("test_gauge", "A test gauge")
    gauge.set(10.0)
    gauge.inc(2.5)
    gauge.dec(1.0)
    assert gauge.value == 11.5


def test_histogram_percentiles(registry: MetricsRegistry) -> None:
    hist = registry.histogram("test_hist", "A test histogram")
    for value in range(1, 101):
        hist.observe(float(value))
    assert hist.count == 100
    assert hist.percentile(50.0) == 50.5
    assert hist.percentile(95.0) == 95.05
    assert hist.percentile(99.0) == 99.01


def test_registry_snapshot(registry: MetricsRegistry) -> None:
    registry.inc("snapshot_counter", description="desc")
    registry.set("snapshot_gauge", 3.14, description="desc")
    registry.record("snapshot_hist", 1.0, description="desc")

    snapshot = registry.snapshot()
    assert snapshot["snapshot_counter"]["type"] == "counter"
    assert snapshot["snapshot_counter"]["value"] == 1
    assert snapshot["snapshot_gauge"]["type"] == "gauge"
    assert snapshot["snapshot_gauge"]["value"] == 3.14
    assert snapshot["snapshot_hist"]["type"] == "histogram"
    assert snapshot["snapshot_hist"]["count"] == 1


def test_timer_records_elapsed(registry: MetricsRegistry) -> None:
    with registry.time("timed_op", "A timed operation"):
        time.sleep(0.001)
    snapshot = registry.snapshot()
    assert snapshot["timed_op"]["count"] == 1
    assert snapshot["timed_op"]["mean"] >= 0.0


def test_global_registry_swap() -> None:
    original = get_registry()
    new_reg = MetricsRegistry()
    set_registry(new_reg)
    assert get_registry() is new_reg
    set_registry(original)


def test_counter_to_dict() -> None:
    counter = Counter("c", "d", value=7)
    assert counter.to_dict() == {"type": "counter", "value": 7, "description": "d"}


def test_gauge_to_dict() -> None:
    gauge = Gauge("g", "d", value=2.5)
    assert gauge.to_dict() == {"type": "gauge", "value": 2.5, "description": "d"}


def test_histogram_to_dict_empty() -> None:
    hist = Histogram("h", "d")
    assert hist.to_dict()["count"] == 0
    assert hist.to_dict()["mean"] == 0.0


def test_histogram_respects_max_samples() -> None:
    hist = Histogram("h", "d", max_samples=5)
    for value in range(10):
        hist.observe(float(value))
    assert hist.count == 5
    assert list(hist._values) == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_counter_concurrent_increments_are_accurate(registry: MetricsRegistry) -> None:
    counter = registry.counter("concurrent_counter", "A test counter")

    def _increment_many() -> None:
        for _ in range(1000):
            counter.inc()

    threads = [threading.Thread(target=_increment_many) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counter.value == 10000


def test_histogram_percentiles_batch_matches_individual(registry: MetricsRegistry) -> None:
    """percentiles() must return the same values as individual percentile() calls."""
    hist = registry.histogram("batch_hist", "Batch percentile test")
    for value in range(1, 101):
        hist.observe(float(value))
    ps = [50.0, 95.0, 99.0]
    batch = hist.percentiles(ps)
    individual = [hist.percentile(p) for p in ps]
    assert batch == individual


def test_histogram_sorted_cache_invalidated_on_observe(registry: MetricsRegistry) -> None:
    """After observe(), the sorted cache must be rebuilt."""
    hist = registry.histogram("cache_hist", "Cache invalidation test")
    hist.observe(1.0)
    hist.observe(3.0)
    # Prime the cache
    assert hist.percentile(50.0) == 2.0
    # Add a new value — cache should be invalidated
    hist.observe(2.0)
    assert hist.percentile(50.0) == 2.0  # median of [1,2,3] is 2


def test_histogram_percentiles_empty(registry: MetricsRegistry) -> None:
    """percentiles() on an empty histogram returns zeros."""
    hist = registry.histogram("empty_hist", "Empty histogram")
    assert hist.percentiles([50.0, 95.0, 99.0]) == [0.0, 0.0, 0.0]
