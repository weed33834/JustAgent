"""轻量指标收集（不依赖 prometheus_client）。

使用进程内 dict 存储计数器和直方图，满足 model_router 和 doctor 的基本需求。
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


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
