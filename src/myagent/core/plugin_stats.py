"""Local plugin usage statistics and ratings.

Stats are stored locally in ``~/.config/myagent/plugin_stats.json`` and never
include project paths or personal information. Anonymous telemetry emission is
only performed when the user has explicitly enabled telemetry.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from myagent.utils.json_io import load_json, save_json

if TYPE_CHECKING:
    from myagent.core.telemetry import TelemetryCollector

logger = logging.getLogger("myagent")

DEFAULT_STATS_DIR = Path.home() / ".config" / "myagent"
DEFAULT_STATS_FILE = DEFAULT_STATS_DIR / "plugin_stats.json"


@dataclass
class PluginRating:
    """Aggregate rating for a plugin."""

    score: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        # Not a dataclass field, so ``asdict`` does not serialize it. PluginRating
        # is reachable via ``PluginStats.get(...).rating``, so its mutation path is
        # independently exposed and must be self-protected.
        self._lock: threading.Lock = threading.Lock()

    def add(self, score: float) -> None:
        """Add a new rating and recompute the average."""
        with self._lock:
            total = self.score * self.count + score
            self.count += 1
            self.score = total / self.count


@dataclass
class PluginStat:
    """Local statistics for a single plugin."""

    installs: int = 0
    uninstalls: int = 0
    rating: PluginRating = field(default_factory=PluginRating)


class PluginStats:
    """Manage local plugin usage statistics and optional telemetry."""

    def __init__(
        self,
        stats_file: Path | None = None,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        self.stats_file = stats_file or DEFAULT_STATS_FILE
        self._stats: dict[str, PluginStat] = {}
        self._telemetry = telemetry
        self._lock: threading.Lock = threading.Lock()
        self._load()

    def record_install(self, plugin_name: str) -> None:
        """Record a plugin installation."""
        with self._lock:
            self._touch(plugin_name).installs += 1
            self._save()
        self._emit("install", plugin_name)

    def record_uninstall(self, plugin_name: str) -> None:
        """Record a plugin uninstallation."""
        with self._lock:
            self._touch(plugin_name).uninstalls += 1
            self._save()
        self._emit("uninstall", plugin_name)

    def record_rate(self, plugin_name: str, score: float) -> None:
        """Record a user rating for a plugin."""
        if not 1 <= score <= 5:
            raise ValueError("Rating must be between 1 and 5")
        with self._lock:
            self._touch(plugin_name).rating.add(score)
            self._save()
        self._emit("rate", plugin_name, score=score)

    def get(self, plugin_name: str) -> PluginStat:
        """Return stats for a plugin, defaulting to zeros.

        Read-only: a missing plugin yields a fresh empty :class:`PluginStat`
        without inserting it, so concurrent ``get`` calls cannot race on
        ``_touch`` or mutate ``_stats`` mid-iteration by another reader.
        """
        with self._lock:
            return self._stats.get(plugin_name, PluginStat())

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary of all recorded stats.

        The snapshot is built under ``_lock`` so a concurrent ``record_*``
        adding a key cannot trigger ``dictionary changed size during iteration``.
        """
        with self._lock:
            return {
                name: {
                    "installs": stat.installs,
                    "uninstalls": stat.uninstalls,
                    "rating": {"score": stat.rating.score, "count": stat.rating.count},
                }
                for name, stat in sorted(self._stats.items())
            }

    def _touch(self, plugin_name: str) -> PluginStat:
        if plugin_name not in self._stats:
            self._stats[plugin_name] = PluginStat()
        return self._stats[plugin_name]

    def _emit(self, action: str, plugin_name: str, **kwargs: Any) -> None:
        if self._telemetry is None or not self._telemetry.enabled:
            return
        event = {
            "type": "plugin_stat",
            "action": action,
            "plugin": plugin_name,
            "timestamp": time.time(),
            **kwargs,
        }
        self._telemetry.record_event(event)

    def _load(self) -> None:
        raw = load_json(self.stats_file, label="plugin stats")
        if not isinstance(raw, dict):
            if raw is not None:
                logger.warning("Plugin stats file is not a JSON object; ignoring")
            return

        raw = cast(dict[str, Any], raw)
        for name, data in raw.items():
            if not isinstance(data, dict):
                logger.warning("Skipping invalid plugin stat entry %s: not an object", name)
                continue
            data = cast(dict[str, Any], data)
            try:
                rating_data = data.get("rating", {})
                if not isinstance(rating_data, dict):
                    rating_data = {}
                rating_data = cast(dict[str, Any], rating_data)
                self._stats[name] = PluginStat(
                    installs=int(data.get("installs", 0)),
                    uninstalls=int(data.get("uninstalls", 0)),
                    rating=PluginRating(
                        score=float(rating_data.get("score", 0.0)),
                        count=int(rating_data.get("count", 0)),
                    ),
                )
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping invalid plugin stat entry %s: %s", name, exc)

    def _save(self) -> None:
        payload = {
            name: {
                "installs": stat.installs,
                "uninstalls": stat.uninstalls,
                "rating": asdict(stat.rating),
            }
            for name, stat in self._stats.items()
        }
        save_json(self.stats_file, payload, label="plugin stats")
