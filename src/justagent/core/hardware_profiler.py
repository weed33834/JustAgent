"""Hardware profiling to recommend a default model tier."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Literal, cast

import structlog

logger = structlog.get_logger("justagent")


@dataclass
class HardwareProfile:
    """Summary of local compute resources relevant to model inference."""

    cpu_cores: int
    memory_gb: float
    has_gpu: bool
    recommended_tier: Literal[1, 2, 3]


def detect_hardware() -> HardwareProfile:
    """Detect CPU, RAM and GPU availability and recommend a model tier."""
    cpu_cores = _cpu_count()
    memory_gb = _memory_gb()
    has_gpu = _has_gpu()

    # Tier 3 (deep thinking) needs significant resources.
    if cpu_cores >= 8 and memory_gb >= 16 and has_gpu:
        recommended_tier: Literal[1, 2, 3] = 3
    elif cpu_cores >= 4 and memory_gb >= 8:
        recommended_tier = 2
    else:
        recommended_tier = 1

    return HardwareProfile(
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        has_gpu=has_gpu,
        recommended_tier=recommended_tier,
    )


def _cgroup_cpu_count() -> int | None:
    """Return the CPU quota imposed by cgroups, or ``None`` if unset/unreadable.

    Containers share the host's ``/proc/cpuinfo`` so ``os.cpu_count()``
    reports the host's CPUs; the cgroup CPU quota (CFS) is the real limit.
    We read cgroup v2 first (``cpu.max``) and fall back to v1
    (``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``). ``"max"`` quota, a
    missing file, or any read/parse error all yield ``None`` so the caller
    falls back to the host value.
    """
    # cgroup v2: /sys/fs/cgroup/cpu.max is "<quota> <period>" or "max <period>".
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:  # noqa: PTH123
            parts = f.read().split()
        if len(parts) == 2 and parts[0] != "max":
            quota = int(parts[0])
            period = int(parts[1])
            if quota > 0 and period > 0:
                return (quota + period - 1) // period  # ceil(quota / period)
    except (OSError, ValueError):
        pass
    # cgroup v1 fallback.
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:  # noqa: PTH123
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:  # noqa: PTH123
            period = int(f.read().strip())
        if quota > 0 and period > 0:
            return (quota + period - 1) // period
    except (OSError, ValueError):
        pass
    return None


def _cpu_count() -> int:
    """Return the effective number of logical CPUs, cgroup-aware.

    In a container ``os.cpu_count()`` reports the host's CPUs, which would
    over-recommend the model tier. We take ``min(host, cgroup_quota)`` so
    the recommendation reflects the actually-available CPU budget.
    """
    try:
        import os

        host = os.cpu_count() or 2
    except Exception:  # noqa: BLE001
        host = 2
    cgroup = _cgroup_cpu_count()
    return min(host, cgroup) if cgroup is not None else host


def _cgroup_memory_gb() -> float | None:
    """Return the cgroup memory limit in GB, or ``None`` if unset/unreadable.

    Mirrors :func:`_cgroup_cpu_count`: cgroup v2 ``memory.max`` first (a byte
    count or ``"max"``), then v1 ``memory.limit_in_bytes``. ``"max"`` or a
    huge v1 sentinel effectively means "no limit" but is still finite; the
    caller's ``min(host, cgroup)`` then keeps the host value, which is the
    correct outcome.
    """
    # cgroup v2: /sys/fs/cgroup/memory.max is a byte count or "max".
    try:
        with open("/sys/fs/cgroup/memory.max") as f:  # noqa: PTH123
            raw = f.read().strip()
        if raw != "max":
            bytes_limit = int(raw)
            if bytes_limit > 0:
                return bytes_limit / (1024**3)
    except (OSError, ValueError):
        pass
    # cgroup v1 fallback.
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:  # noqa: PTH123
            bytes_limit = int(f.read().strip())
        if bytes_limit > 0:
            return bytes_limit / (1024**3)
    except (OSError, ValueError):
        pass
    return None


def _host_memory_gb() -> float:
    """Return the host's total system memory in gigabytes."""
    try:
        import psutil

        return float(psutil.virtual_memory().total) / (1024**3)
    except Exception as exc:  # noqa: BLE001
        logger.debug("psutil unavailable or failed: %s", exc)

    # Fallback for Linux systems without psutil.
    try:
        with open("/proc/meminfo") as f:  # noqa: PTH123
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except (OSError, ValueError):
        pass

    logger.warning("Could not detect system memory; assuming 8 GB")
    return 8.0


def _memory_gb() -> float:
    """Return effective total memory in gigabytes, cgroup-aware.

    In a container ``psutil.virtual_memory()`` reports the host's memory;
    the cgroup memory limit may be lower. We take ``min(host, cgroup)`` so
    the recommended tier reflects the actually-available memory.
    """
    host = _host_memory_gb()
    cgroup = _cgroup_memory_gb()
    return min(host, cgroup) if cgroup is not None else host


def _has_gpu() -> bool:
    """Return True if a GPU runtime appears to be available."""
    if shutil.which("nvidia-smi"):
        return True
    try:
        import torch  # pyright: ignore[reportMissingImports]

        return cast(bool, torch.cuda.is_available())  # pyright: ignore[reportUnknownMemberType]
    except Exception:  # noqa: BLE001
        return False
