"""Resource monitor — real-time metrics, threshold alerts and health checks.

Collects host-level utilisation via :mod:`psutil` (CPU, memory, disk, network,
load average, process count) and optional GPU stats via ``nvidia-smi`` when
available. Each collected :class:`MetricSnapshot` is evaluated against a set
of configurable :class:`AlertThreshold` rules; transitions fire
:class:`Alert` objects through a callback (and are kept in an alert log).
Pluggable :class:`HealthCheck` callables produce an overall
:class:`HealthStatus`, and a background thread can drive continuous
monitoring.

Design:

* :class:`MetricType` / :class:`AlertLevel` / :class:`HealthStatus` — enums.
* :class:`DiskUsage` / :class:`NetworkIO` / :class:`DiskIO` / :class:`GpuStats` —
  Pydantic sub-models.
* :class:`MetricSnapshot` — a point-in-time host sample.
* :class:`AlertThreshold` / :class:`Alert` — rule + fired-alert models.
* :class:`HealthCheck` — protocol for pluggable checks.
* :class:`ResourceMonitor` — collect, evaluate, alert, report and (optionally)
  push load back into a :class:`~justagent.resources.registry.ResourceRegistry`
  and run a background polling loop.

The monitor has no hard dependency on the registry; when a registry +
resource id are bound via :meth:`ResourceMonitor.bind_resource`, each
collection updates that resource's :class:`~justagent.resources.registry.ResourceLoad`
so the scheduler's load balancing reflects live host utilisation.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger("justagent.resources")

#: Default polling interval for the background loop, in seconds.
DEFAULT_INTERVAL = 5.0

#: Maximum number of snapshots retained in the history ring buffer.
DEFAULT_HISTORY = 288  # 24h at 5s intervals


class MetricType(str, Enum):  # noqa: UP042
    """Categories of metrics the monitor can collect and alert on."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    LOAD = "load"
    GPU = "gpu"
    PROCESS = "process"


class AlertLevel(str, Enum):  # noqa: UP042
    """Severity of a fired alert."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric severity (higher = more severe) for ordering."""

        return _ALERT_RANK[self]


_ALERT_RANK: dict[AlertLevel, int] = {
    AlertLevel.INFO: 1,
    AlertLevel.WARNING: 2,
    AlertLevel.ERROR: 3,
    AlertLevel.CRITICAL: 4,
}


class HealthStatus(str, Enum):  # noqa: UP042
    """Overall health of a monitored subject."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

    @property
    def rank(self) -> int:
        """Numeric severity (higher = worse) for worst-of aggregation."""

        return _HEALTH_RANK[self]


_HEALTH_RANK: dict[HealthStatus, int] = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.UNKNOWN: 1,
    HealthStatus.DEGRADED: 2,
    HealthStatus.UNHEALTHY: 3,
}

#: A metric value worse than this rank means the host is not fully healthy.
_UNHEALTHY_RANK = HealthStatus.DEGRADED.rank


class MonitorError(Exception):
    """Raised for monitor configuration or collection failures."""


# ---------------------------------------------------------------------------
# Metric models
# ---------------------------------------------------------------------------


class DiskUsage(BaseModel):
    """Usage of a single mounted filesystem."""

    device: str = ""
    mountpoint: str = ""
    fstype: str = ""
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    percent: float = 0.0


class NetworkIO(BaseModel):
    """Cumulative network counters across all interfaces."""

    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0


class DiskIO(BaseModel):
    """Cumulative disk I/O counters."""

    read_bytes: int = 0
    write_bytes: int = 0
    read_count: int = 0
    write_count: int = 0


class GpuStats(BaseModel):
    """Stats for a single GPU (best-effort, via ``nvidia-smi``)."""

    index: int = 0
    name: str = ""
    utilization_percent: float = 0.0
    memory_used_mib: float = 0.0
    memory_total_mib: float = 0.0
    memory_percent: float = 0.0
    temperature_c: float = 0.0


class MetricSnapshot(BaseModel):
    """A point-in-time sample of host utilisation.

    Percent fields are 0–100. ``network_bytes_per_sec`` and
    ``disk_io_*_per_sec`` are rates derived from the previous snapshot;
    they are 0 on the first collection.
    """

    timestamp: float = Field(default_factory=time.time)
    cpu_percent: float = 0.0
    per_cpu_percent: list[float] = Field(default_factory=list)
    memory_percent: float = 0.0
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0
    memory_available_bytes: int = 0
    swap_percent: float = 0.0
    disks: list[DiskUsage] = Field(default_factory=list)
    disk_io: DiskIO = Field(default_factory=DiskIO)
    disk_read_bytes_per_sec: float = 0.0
    disk_write_bytes_per_sec: float = 0.0
    network: NetworkIO = Field(default_factory=NetworkIO)
    network_bytes_per_sec: float = 0.0
    load_avg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    process_count: int = 0
    gpus: list[GpuStats] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def metric_value(self, metric: str) -> float | None:
        """Return the scalar value for a named metric, or None if unknown.

        Supports the metric keys used by :class:`AlertThreshold`:
        ``cpu_percent``, ``memory_percent``, ``swap_percent``,
        ``disk_percent`` (max across mounts), ``network_bytes_per_sec``,
        ``load_avg_1m``, ``process_count``, ``gpu_percent`` (max across
        GPUs).
        """

        if metric == "cpu_percent":
            return self.cpu_percent
        if metric == "memory_percent":
            return self.memory_percent
        if metric == "swap_percent":
            return self.swap_percent
        if metric == "disk_percent":
            return max((d.percent for d in self.disks), default=0.0)
        if metric == "network_bytes_per_sec":
            return self.network_bytes_per_sec
        if metric == "load_avg_1m":
            return self.load_avg[0] if self.load_avg else 0.0
        if metric == "process_count":
            return float(self.process_count)
        if metric == "gpu_percent":
            return max((g.utilization_percent for g in self.gpus), default=0.0)
        return None


# ---------------------------------------------------------------------------
# Alert models
# ---------------------------------------------------------------------------


class AlertThreshold(BaseModel):
    """A threshold rule that turns a metric reading into an alert.

    Attributes:
        metric: Metric key understood by :meth:`MetricSnapshot.metric_value`.
        warning: Value at/above which a WARNING alert fires (None = disabled).
        critical: Value at/above which a CRITICAL alert fires (None = disabled).
        operator: ``"gt"`` fires when ``value >= threshold``;
            ``"lt"`` fires when ``value <= threshold`` (e.g. for free space).
    """

    metric: str
    warning: float | None = None
    critical: float | None = None
    operator: str = "gt"

    def classify(self, value: float) -> AlertLevel | None:
        """Return the :class:`AlertLevel` for ``value``, or None if clear."""

        if self.operator == "lt":
            if self.critical is not None and value <= self.critical:
                return AlertLevel.CRITICAL
            if self.warning is not None and value <= self.warning:
                return AlertLevel.WARNING
        else:
            if self.critical is not None and value >= self.critical:
                return AlertLevel.CRITICAL
            if self.warning is not None and value >= self.warning:
                return AlertLevel.WARNING
        return None


class Alert(BaseModel):
    """A fired alert.

    Attributes:
        id: Unique alert id.
        metric: Metric key that triggered the alert.
        level: :class:`AlertLevel`.
        value: Observed metric value.
        threshold: Threshold that was breached.
        message: Human-readable description.
        triggered_at: Unix timestamp.
        resource_id: Resource the alert concerns (empty for the local host).
        resolved: True once the condition cleared.
        resolved_at: Timestamp of resolution (0 while active).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    metric: str
    level: AlertLevel
    value: float
    threshold: float
    message: str = ""
    triggered_at: float = Field(default_factory=time.time)
    resource_id: str = ""
    resolved: bool = False
    resolved_at: float = 0.0


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


class HealthCheckResult(BaseModel):
    """The outcome of a single health check."""

    name: str
    status: HealthStatus
    detail: str = ""


@runtime_checkable
class HealthCheck(Protocol):
    """Protocol for a pluggable health check.

    A health check is any callable returning either a :class:`HealthStatus`
    or a :class:`HealthCheckResult`. Implementations may wrap external
    probes (database ping, storage reachability, custom service checks).
    """

    def __call__(self) -> HealthStatus | HealthCheckResult: ...


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def default_thresholds(cpu_count: int | None = None) -> dict[str, AlertThreshold]:
    """Return a sensible default threshold set.

    ``load_avg_1m`` thresholds scale with the CPU count so a 32-core host
    does not alert at a load that is healthy for it.
    """

    cores = cpu_count or _safe_cpu_count() or 1
    return {
        "cpu_percent": AlertThreshold(metric="cpu_percent", warning=80.0, critical=95.0),
        "memory_percent": AlertThreshold(metric="memory_percent", warning=85.0, critical=95.0),
        "disk_percent": AlertThreshold(metric="disk_percent", warning=85.0, critical=95.0),
        "load_avg_1m": AlertThreshold(
            metric="load_avg_1m",
            warning=float(cores),
            critical=float(cores * 2),
        ),
    }


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------


AlertCallback = Callable[[Alert], None]


class ResourceMonitor:
    """Collects host metrics, evaluates thresholds and reports health.

    Example:

        >>> mon = ResourceMonitor()
        >>> snap = mon.collect()
        >>> alerts = mon.evaluate(snap)
        >>> mon.health()["overall"]
        <HealthStatus.HEALTHY: 'healthy'>
    """

    def __init__(
        self,
        *,
        thresholds: Mapping[str, AlertThreshold] | None = None,
        history_size: int = DEFAULT_HISTORY,
        interval: float = DEFAULT_INTERVAL,
        alert_callback: AlertCallback | None = None,
    ) -> None:
        if history_size < 1:
            raise MonitorError("history_size must be >= 1")
        if interval <= 0:
            raise MonitorError("interval must be > 0")
        self._thresholds: dict[str, AlertThreshold] = dict(thresholds or default_thresholds())
        self._history: deque[MetricSnapshot] = deque(maxlen=history_size)
        self._interval = interval
        self._alert_callback = alert_callback
        self._alerts: deque[Alert] = deque(maxlen=512)
        self._active: dict[str, Alert] = {}
        self._health_checks: dict[str, Callable[[], HealthStatus | HealthCheckResult]] = {}
        self._lock = threading.RLock()

        # Registry binding (optional).
        self._registry: Any = None
        self._bound_resource_id: str = ""

        # Rate-derivation state.
        self._prev_snapshot: MetricSnapshot | None = None
        self._prev_ts: float = 0.0

        # Background loop state.
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def set_threshold(self, threshold: AlertThreshold) -> AlertThreshold:
        """Register or replace a threshold rule."""

        with self._lock:
            self._thresholds[threshold.metric] = threshold
        logger.info(
            "Threshold set: %s warning=%s critical=%s op=%s",
            threshold.metric,
            threshold.warning,
            threshold.critical,
            threshold.operator,
        )
        return threshold

    def remove_threshold(self, metric: str) -> bool:
        with self._lock:
            return self._thresholds.pop(metric, None) is not None

    def thresholds(self) -> dict[str, AlertThreshold]:
        with self._lock:
            return dict(self._thresholds)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def add_health_check(
        self,
        name: str,
        check: Callable[[], HealthStatus | HealthCheckResult],
    ) -> None:
        """Register a named health check callable."""

        if not name:
            raise MonitorError("health check name must not be empty")
        with self._lock:
            self._health_checks[name] = check

    def remove_health_check(self, name: str) -> bool:
        with self._lock:
            return self._health_checks.pop(name, None) is not None

    def health(self) -> dict[str, Any]:
        """Run every registered health check and return an aggregate view.

        The ``overall`` status is the worst per-check status; when there are
        no registered checks it is :attr:`HealthStatus.UNKNOWN` (nothing was
        probed). Each check contributes a :class:`HealthCheckResult`.
        """

        with self._lock:
            checks = dict(self._health_checks)
        results: dict[str, HealthCheckResult] = {}
        worst: HealthStatus | None = None
        for name, check in checks.items():
            try:
                outcome = check()
            except Exception as exc:  # noqa: BLE001
                result = HealthCheckResult(
                    name=name, status=HealthStatus.UNHEALTHY, detail=f"error: {exc}"
                )
            else:
                if isinstance(outcome, HealthCheckResult):
                    result = outcome
                else:
                    result = HealthCheckResult(name=name, status=outcome)
            results[name] = result
            if worst is None or result.status.rank > worst.rank:
                worst = result.status
        overall = worst if worst is not None else HealthStatus.UNKNOWN
        return {"overall": overall, "checks": results}

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(self) -> MetricSnapshot:
        """Collect a single :class:`MetricSnapshot` from the local host."""

        snapshot = self._collect_raw()
        self._derive_rates(snapshot)
        with self._lock:
            self._history.append(snapshot)
            self._prev_snapshot = snapshot
            self._prev_ts = snapshot.timestamp
        self._push_to_registry(snapshot)
        return snapshot

    def _collect_raw(self) -> MetricSnapshot:
        """Build a snapshot from :mod:`psutil` (and optional nvidia-smi)."""

        import psutil

        snapshot = MetricSnapshot(timestamp=time.time())
        # CPU (interval=None for non-blocking since first call; we pre-seed).
        snapshot.cpu_percent = _safe_call(lambda: psutil.cpu_percent(interval=None), 0.0)
        snapshot.per_cpu_percent = list(
            _safe_call(lambda: psutil.cpu_percent(interval=None, percpu=True), [])
        )
        # Pre-warm cpu_percent so the next reading is meaningful.
        with contextlib.suppress(Exception):
            psutil.cpu_percent(interval=None)

        mem = _safe_call(psutil.virtual_memory, None)
        if mem is not None:
            snapshot.memory_percent = float(mem.percent)
            snapshot.memory_used_bytes = int(mem.used)
            snapshot.memory_total_bytes = int(mem.total)
            snapshot.memory_available_bytes = int(getattr(mem, "available", 0))
        swap = _safe_call(psutil.swap_memory, None)
        if swap is not None:
            snapshot.swap_percent = float(swap.percent)

        for part in _safe_call(lambda: psutil.disk_partitions(all=False), []):
            usage = _safe_call(lambda p=part: psutil.disk_usage(p.mountpoint), None)
            if usage is None:
                continue
            snapshot.disks.append(
                DiskUsage(
                    device=part.device,
                    mountpoint=part.mountpoint,
                    fstype=part.fstype,
                    total_bytes=int(usage.total),
                    used_bytes=int(usage.used),
                    free_bytes=int(usage.free),
                    percent=float(usage.percent),
                )
            )

        dio = _safe_call(psutil.disk_io_counters, None)
        if dio is not None:
            snapshot.disk_io = DiskIO(
                read_bytes=int(getattr(dio, "read_bytes", 0)),
                write_bytes=int(getattr(dio, "write_bytes", 0)),
                read_count=int(getattr(dio, "read_count", 0)),
                write_count=int(getattr(dio, "write_count", 0)),
            )

        nio = _safe_call(psutil.net_io_counters, None)
        if nio is not None:
            snapshot.network = NetworkIO(
                bytes_sent=int(getattr(nio, "bytes_sent", 0)),
                bytes_recv=int(getattr(nio, "bytes_recv", 0)),
                packets_sent=int(getattr(nio, "packets_sent", 0)),
                packets_recv=int(getattr(nio, "packets_recv", 0)),
            )

        load = _safe_call(psutil.getloadavg, None)
        if load is not None:
            snapshot.load_avg = (float(load[0]), float(load[1]), float(load[2]))

        try:
            snapshot.process_count = len(psutil.pids())
        except Exception:  # noqa: BLE001
            snapshot.process_count = 0

        snapshot.gpus = _collect_gpu_stats()
        return snapshot

    def _derive_rates(self, snapshot: MetricSnapshot) -> None:
        """Fill in per-second rates using the previous snapshot."""

        prev = self._prev_snapshot
        prev_ts = self._prev_ts
        if prev is None or prev_ts <= 0:
            return
        elapsed = max(snapshot.timestamp - prev_ts, 1e-6)
        snapshot.network_bytes_per_sec = (
            max(
                0.0,
                (snapshot.network.bytes_sent + snapshot.network.bytes_recv)
                - (prev.network.bytes_sent + prev.network.bytes_recv),
            )
            / elapsed
        )
        snapshot.disk_read_bytes_per_sec = (
            max(0.0, snapshot.disk_io.read_bytes - prev.disk_io.read_bytes) / elapsed
        )
        snapshot.disk_write_bytes_per_sec = (
            max(0.0, snapshot.disk_io.write_bytes - prev.disk_io.write_bytes) / elapsed
        )

    # ------------------------------------------------------------------
    # Evaluation & alerts
    # ------------------------------------------------------------------

    def evaluate(self, snapshot: MetricSnapshot) -> list[Alert]:
        """Evaluate thresholds against ``snapshot``; returns newly-fired alerts.

        Uses level-transition deduplication: a metric only produces a new
        alert when its level changes (including transitioning to "clear",
        which resolves the active alert). Resolved alerts are also returned
        so callers can record the resolution.
        """

        fired: list[Alert] = []
        with self._lock:
            thresholds = dict(self._thresholds)
            active = dict(self._active)
        now = snapshot.timestamp
        for metric, rule in thresholds.items():
            value = snapshot.metric_value(metric)
            if value is None:
                continue
            level = rule.classify(value)
            prev_alert = active.get(metric)
            prev_level = prev_alert.level if prev_alert is not None else None
            if level is None:
                # Condition cleared — resolve any active alert.
                if prev_alert is not None and not prev_alert.resolved:
                    resolved = prev_alert.model_copy(update={"resolved": True, "resolved_at": now})
                    with self._lock:
                        self._active.pop(metric, None)
                        self._alerts.append(resolved)
                    fired.append(resolved)
                continue
            if level is prev_level:
                continue  # no transition
            threshold_value = rule.critical if level is AlertLevel.CRITICAL else rule.warning
            threshold_value = threshold_value if threshold_value is not None else 0.0
            alert = Alert(
                metric=metric,
                level=level,
                value=value,
                threshold=float(threshold_value),
                message=_alert_message(metric, level, value, float(threshold_value), rule),
                triggered_at=now,
                resource_id=self._bound_resource_id,
            )
            with self._lock:
                self._active[metric] = alert
                self._alerts.append(alert)
            fired.append(alert)
            self._fire_callback(alert)
        return fired

    def alerts(self, *, include_resolved: bool = True) -> list[Alert]:
        """Return the alert log (most-recent last)."""

        with self._lock:
            alerts = list(self._alerts)
        if not include_resolved:
            alerts = [a for a in alerts if not a.resolved]
        return alerts

    def active_alerts(self) -> list[Alert]:
        """Return alerts that are currently firing (unresolved)."""

        with self._lock:
            return list(self._active.values())

    def clear_alerts(self) -> None:
        """Clear the alert log and any active alerts."""

        with self._lock:
            self._alerts.clear()
            self._active.clear()

    # ------------------------------------------------------------------
    # History & reporting
    # ------------------------------------------------------------------

    def history(self) -> list[MetricSnapshot]:
        """Return a copy of the collected snapshots (oldest-first)."""

        with self._lock:
            return list(self._history)

    def latest(self) -> MetricSnapshot | None:
        """Return the most recent snapshot, or None."""

        with self._lock:
            return self._history[-1] if self._history else None

    def report(self) -> dict[str, Any]:
        """Return a utilisation summary derived from the latest snapshot.

        Includes per-disk and per-network breakdowns plus the overall
        :class:`HealthStatus` derived from active alerts.
        """

        snap = self.latest()
        if snap is None:
            return {"status": HealthStatus.UNKNOWN.value, "available": False}
        active = self.active_alerts()
        if any(a.level is AlertLevel.CRITICAL for a in active):
            status = HealthStatus.UNHEALTHY
        elif active:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        return {
            "available": True,
            "status": status.value,
            "timestamp": snap.timestamp,
            "cpu_percent": snap.cpu_percent,
            "memory_percent": snap.memory_percent,
            "memory_used_bytes": snap.memory_used_bytes,
            "memory_total_bytes": snap.memory_total_bytes,
            "swap_percent": snap.swap_percent,
            "load_avg": list(snap.load_avg),
            "process_count": snap.process_count,
            "network_bytes_per_sec": snap.network_bytes_per_sec,
            "disks": [d.model_dump() for d in snap.disks],
            "gpus": [g.model_dump() for g in snap.gpus],
            "active_alerts": len(active),
        }

    # ------------------------------------------------------------------
    # Registry binding
    # ------------------------------------------------------------------

    def bind_resource(self, registry: Any, resource_id: str) -> None:
        """Bind the monitor to a registry resource.

        After binding, every :meth:`collect` updates that resource's
        :class:`~justagent.resources.registry.ResourceLoad` with the live CPU
        and memory utilisation so the scheduler's load balancing reflects
        real host pressure.
        """

        self._registry = registry
        self._bound_resource_id = resource_id

    def _push_to_registry(self, snapshot: MetricSnapshot) -> None:
        """Update the bound resource's load snapshot (if any)."""

        if self._registry is None or not self._bound_resource_id:
            return
        try:
            from justagent.resources.registry import ResourceLoad  # local import

            load = ResourceLoad(
                running_tasks=0,
                cpu_usage_percent=snapshot.cpu_percent,
                memory_usage_percent=snapshot.memory_percent,
                gpu_usage_percent=max((g.utilization_percent for g in snapshot.gpus), default=0.0),
                updated_at=snapshot.timestamp,
            )
            self._registry.update_load(self._bound_resource_id, load)
        except Exception as exc:  # noqa: BLE001
            logger.debug("registry load push failed: %s", exc)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def start(self, interval: float | None = None) -> None:
        """Start a daemon thread that collects and evaluates periodically."""

        with self._lock:
            if self._running:
                return
            self._interval = interval or self._interval
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop, name="justagent-monitor", daemon=True
            )
            self._running = True
            self._thread.start()
        logger.info("Monitor started (interval=%.1fs)", self._interval)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the background loop to stop and wait briefly."""

        with self._lock:
            thread = self._thread
            self._running = False
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=timeout)
        logger.info("Monitor stopped")

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _loop(self) -> None:
        """Collect -> evaluate -> fire, sleeping ``interval`` between cycles."""

        while not self._stop_event.is_set():
            try:
                snapshot = self.collect()
                self.evaluate(snapshot)
            except Exception as exc:  # noqa: BLE001
                logger.error("monitor cycle failed: %s", exc)
            self._stop_event.wait(self._interval)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire_callback(self, alert: Alert) -> None:
        cb = self._alert_callback
        if cb is None:
            return
        try:
            cb(alert)
        except Exception as exc:  # noqa: BLE001
            logger.error("alert callback failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alert_message(
    metric: str,
    level: AlertLevel,
    value: float,
    threshold: float,
    rule: AlertThreshold,
) -> str:
    """Compose a human-readable alert message."""

    op = ">=" if rule.operator == "gt" else "<="
    return f"{metric} {level.value}: {value:.1f} {op} {threshold:.1f}"


def _safe_call(func: Callable[..., Any], default: Any) -> Any:
    """Call ``func`` and return its result, or ``default`` on failure."""

    try:
        return func()
    except Exception:  # noqa: BLE001
        return default


def _safe_cpu_count() -> int | None:
    try:
        return os.cpu_count()
    except Exception:  # noqa: BLE001
        return None


def _collect_gpu_stats() -> list[GpuStats]:
    """Best-effort GPU stats via ``nvidia-smi`` (empty when unavailable)."""

    if not shutil.which("nvidia-smi"):
        return []
    try:
        completed = subprocess.run(  # noqa: S603, S607 - trusted binary
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    stats: list[GpuStats] = []
    for line in completed.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            used = float(parts[3])
            total = float(parts[4])
            stats.append(
                GpuStats(
                    index=int(parts[0]),
                    name=parts[1],
                    utilization_percent=float(parts[2]),
                    memory_used_mib=used,
                    memory_total_mib=total,
                    memory_percent=(used / total * 100.0) if total else 0.0,
                    temperature_c=float(parts[5]),
                )
            )
        except (ValueError, IndexError):
            continue
    return stats


__all__ = [
    "Alert",
    "AlertCallback",
    "AlertLevel",
    "AlertThreshold",
    "DEFAULT_HISTORY",
    "DEFAULT_INTERVAL",
    "DiskIO",
    "DiskUsage",
    "GpuStats",
    "HealthCheck",
    "HealthCheckResult",
    "HealthStatus",
    "MetricSnapshot",
    "MetricType",
    "MonitorError",
    "NetworkIO",
    "ResourceMonitor",
    "default_thresholds",
]
