"""Metrics wrappers around prometheus_client."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram


class MetricsRegistry:
    """Thin wrapper around prometheus_client's default registry."""

    def __init__(self) -> None:
        self._registry: CollectorRegistry = REGISTRY

    def inc(self, name: str, amount: int = 1, description: str = "") -> None:
        c = Counter(name, description, registry=self._registry)
        c.inc(amount)

    def record(self, name: str, value: float, description: str = "") -> None:
        h = Histogram(name, description, registry=self._registry)
        h.observe(value)

    def set(self, name: str, value: float, description: str = "") -> None:
        g = Gauge(name, description, registry=self._registry)
        g.set(value)

    @contextmanager
    def time(self, name: str, description: str = "") -> Generator[None, None, None]:
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000
        self.record(name, elapsed, description)


_registry: MetricsRegistry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    return _registry


def set_registry(registry: MetricsRegistry) -> MetricsRegistry:
    global _registry
    _registry = registry
    return _registry
