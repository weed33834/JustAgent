"""Concurrency tests for the metrics registry and audit logger.

These exercise the hot paths under real threads so a missing lock shows up as
either a torn write (corrupt JSON lines) or a lost update (final count below
the expected total). They are deterministic by construction: every thread
does a known number of operations, so the aggregate is fully predictable.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from myagent.core.audit_logger import AuditLogger
from myagent.core.metrics import MetricsRegistry
from myagent.models.config import AppConfig

THREADS = 8
ITERS = 500


def test_gauge_concurrent_inc_dec_balances_to_zero() -> None:
    """Each thread increments then decrements the same amount: net must be 0."""
    registry = MetricsRegistry()
    gauge = registry.gauge("balance", "inc/dec gauge")

    def _worker() -> None:
        for _ in range(ITERS):
            gauge.inc(1.0)
            gauge.dec(1.0)

    threads = [threading.Thread(target=_worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert gauge.to_dict()["value"] == 0.0


def test_gauge_concurrent_set_keeps_last_value_type() -> None:
    """Concurrent sets never corrupt the float; the value stays a float."""
    registry = MetricsRegistry()
    gauge = registry.gauge("setpoint", "concurrent set")

    def _worker() -> None:
        for i in range(ITERS):
            gauge.set(float(i))

    threads = [threading.Thread(target=_worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snapshot = gauge.to_dict()
    assert isinstance(snapshot["value"], float)
    assert 0.0 <= snapshot["value"] < float(ITERS)


def test_histogram_concurrent_observe_preserves_count() -> None:
    """Every observe must be counted: total == threads * iters."""
    registry = MetricsRegistry()
    expected = THREADS * ITERS
    # Allow room for every observation so the count reflects lock correctness,
    # not the default 1000-sample cap.
    hist = registry.histogram("latency", "concurrent observe", max_samples=expected)

    def _worker() -> None:
        for i in range(ITERS):
            hist.observe(float(i))

    threads = [threading.Thread(target=_worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert hist.count == expected
    # Percentiles must compute without raising on the full concurrent set.
    snap = hist.to_dict()
    assert snap["count"] == expected
    assert snap["p50"] >= 0.0


def test_audit_logger_concurrent_records_are_atomic(tmp_path: Path) -> None:
    """Concurrent writes never interleave: every line is valid JSON, count matches."""
    config = AppConfig(project_root=tmp_path, audit_log_dir=tmp_path / "logs")
    audit = AuditLogger(config)

    per_thread = 200
    expected_total = THREADS * per_thread

    def _worker(tid: int) -> None:
        for i in range(per_thread):
            audit.record("concurrent.event", {"thread": tid, "step": i})

    threads = [threading.Thread(target=_worker, args=(tid,)) for tid in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    text = audit.log_file.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == expected_total
    # Every line must be independently parseable — a torn write would fail here.
    parsed = [json.loads(ln) for ln in lines]
    assert all(e["event"] == "concurrent.event" for e in parsed)
    assert len({(e["payload"]["thread"], e["payload"]["step"]) for e in parsed}) == expected_total


def test_audit_logger_concurrent_bind_and_record(tmp_path: Path) -> None:
    """bind_context racing with record must not raise or lose records."""
    config = AppConfig(project_root=tmp_path, audit_log_dir=tmp_path / "logs")
    audit = AuditLogger(config)

    stop = threading.Event()

    def _binder() -> None:
        i = 0
        while not stop.is_set():
            audit.bind_context(seq=i)
            i += 1

    writers = [
        threading.Thread(target=_binder),
        *[
            threading.Thread(target=lambda n=n: audit.record("racing.event", {"k": n}))
            for n in range(THREADS)
        ],
    ]
    # Let writers run briefly while the binder mutates context.
    for t in writers[1:]:
        t.start()
    _binder_thread = writers[0]
    _binder_thread.start()

    import time

    time.sleep(0.05)
    stop.set()
    for t in writers[1:]:
        t.join()
    _binder_thread.join()

    # All records landed as valid JSON lines.
    text = audit.log_file.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == THREADS
    assert all(json.loads(ln)["event"] == "racing.event" for ln in lines)


def test_snapshot_during_concurrent_writes_does_not_raise() -> None:
    """A reader calling snapshot() while writers mutate must not blow up."""
    registry = MetricsRegistry()
    failures: list[Exception] = []

    stop = threading.Event()

    def _writer(metric: str) -> None:
        try:
            while not stop.is_set():
                registry.inc(metric)
                registry.record(metric + ".dur", 1.0)
                registry.set(metric + ".g", 1.0)
        except Exception as exc:  # noqa: BLE001 — surface any race to the assertion
            failures.append(exc)

    writers = [threading.Thread(target=_writer, args=(f"m{i}",)) for i in range(6)]
    for t in writers:
        t.start()

    snapshots: list[dict] = []
    for _ in range(200):
        snapshots.append(registry.snapshot())

    stop.set()
    for t in writers:
        t.join()

    assert failures == []
    # Every snapshot must be a plain dict with string keys.
    for snap in snapshots:
        assert isinstance(snap, dict)
        assert all(isinstance(k, str) for k in snap)
