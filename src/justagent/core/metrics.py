"""轻量指标收集（不依赖 prometheus_client）。

使用进程内 dict 存储计数器和直方图，满足 model_router 和 doctor 的基本需求。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


class _Gauge:
    """Thread-safe gauge metric supporting inc/dec/set operations."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"type": "gauge", "value": self._value, "description": self.description}


class _Histogram:
    """Thread-safe histogram metric with bounded sample storage."""

    def __init__(self, name: str, description: str = "", max_samples: int = 1000) -> None:
        self.name = name
        self.description = description
        self._max_samples = max_samples
        self._values: list[float] = []
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            if len(self._values) < self._max_samples:
                self._values.append(float(value))

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            values = list(self._values)
            count = self._count
        p50 = 0.0
        if values:
            sorted_vals = sorted(values)
            p50 = sorted_vals[len(sorted_vals) // 2]
        return {
            "type": "histogram",
            "count": count,
            "values": values,
            "p50": p50,
            "description": self.description,
        }


class MetricsRegistry:
    """进程内指标注册表，替代 prometheus_client。"""

    def __init__(self) -> None:
        self._metrics: dict[str, dict[str, Any]] = {}

    def inc(self, name: str, amount: int = 1, description: str = "") -> None:
        entry = self._metrics.setdefault(name, {"type": "counter", "value": 0, "description": description})
        entry["value"] = entry.get("value", 0) + amount

    def record(self, name: str, value: float, description: str = "") -> None:
        entry = self._metrics.setdefault(name, {"type": "histogram", "values": [], "description": description})
        entry.setdefault("values", []).append(value)

    def set(self, name: str, value: float, description: str = "") -> None:
        self._metrics[name] = {"type": "gauge", "value": value, "description": description}

    def gauge(self, name: str, description: str = "") -> _Gauge:
        """Create and return a thread-safe gauge metric."""
        return _Gauge(name, description)

    def histogram(self, name: str, description: str = "", max_samples: int = 1000) -> _Histogram:
        """Create and return a thread-safe histogram metric."""
        return _Histogram(name, description, max_samples=max_samples)

    @contextmanager
    def time(self, name: str, description: str = "") -> Generator[None, None, None]:
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000
        self.record(name, elapsed, description)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return dict(self._metrics)

    def reset(self) -> None:
        self._metrics.clear()


_registry: MetricsRegistry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    return _registry
