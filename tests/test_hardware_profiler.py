"""Tests for hardware profiling."""

from __future__ import annotations

import sys
from unittest.mock import patch

from myagent.core.hardware_profiler import HardwareProfile, detect_hardware


def test_detect_hardware_recommends_tier_3() -> None:
    with (
        patch("myagent.core.hardware_profiler._cpu_count", return_value=16),
        patch("myagent.core.hardware_profiler._memory_gb", return_value=32.0),
        patch("myagent.core.hardware_profiler._has_gpu", return_value=True),
    ):
        profile = detect_hardware()
    assert profile.recommended_tier == 3


def test_detect_hardware_recommends_tier_2() -> None:
    with (
        patch("myagent.core.hardware_profiler._cpu_count", return_value=4),
        patch("myagent.core.hardware_profiler._memory_gb", return_value=8.0),
        patch("myagent.core.hardware_profiler._has_gpu", return_value=False),
    ):
        profile = detect_hardware()
    assert profile.recommended_tier == 2


def test_detect_hardware_recommends_tier_1() -> None:
    with (
        patch("myagent.core.hardware_profiler._cpu_count", return_value=2),
        patch("myagent.core.hardware_profiler._memory_gb", return_value=4.0),
        patch("myagent.core.hardware_profiler._has_gpu", return_value=False),
    ):
        profile = detect_hardware()
    assert profile.recommended_tier == 1


def test_hardware_profile_fields() -> None:
    profile = HardwareProfile(cpu_cores=4, memory_gb=16.0, has_gpu=False, recommended_tier=2)
    assert profile.cpu_cores == 4
    assert profile.memory_gb == 16.0
    assert profile.has_gpu is False


def test_memory_gb_fallback_from_proc(tmp_path, monkeypatch) -> None:
    from myagent.core.hardware_profiler import _memory_gb

    monkeypatch.setitem(sys.modules, "psutil", None)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16384000 kB\n")

    def _open(*args, **kwargs):
        return meminfo.open()

    with patch("builtins.open", _open):
        assert abs(_memory_gb() - 15.625) < 0.01


def test_memory_gb_assumes_default_on_failure(monkeypatch) -> None:
    from myagent.core.hardware_profiler import _memory_gb

    monkeypatch.setitem(sys.modules, "psutil", None)
    with patch("builtins.open", side_effect=OSError("no /proc")):
        assert _memory_gb() == 8.0


def test_has_gpu_detects_nvidia_smi() -> None:
    from myagent.core.hardware_profiler import _has_gpu

    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
        assert _has_gpu() is True


def test_has_gpu_no_torch() -> None:
    from myagent.core.hardware_profiler import _has_gpu

    with patch("shutil.which", return_value=None), patch.dict("sys.modules", {"torch": None}):
        assert _has_gpu() is False


# ============================================================
# Tests for cgroup CPU/memory detection (M1)
# ============================================================


def _make_open(files: dict[str, str]):
    """Build a fake ``open`` that maps cgroup paths to in-memory file contents.

    Any path not in ``files`` raises :class:`FileNotFoundError` so the
    production code's ``except OSError`` branch is exercised.
    """

    import io

    def _open(path, *args, **kwargs):  # noqa: ANN002, ANN003
        try:
            content = files[str(path)]
        except KeyError as exc:
            raise FileNotFoundError(str(path)) from exc
        return io.StringIO(content)

    return _open


def test_cgroup_cpu_count_v2_quota(tmp_path, monkeypatch) -> None:
    """cgroup v2 ``cpu.max`` with a finite quota yields the ceiled CPU count."""
    from myagent.core.hardware_profiler import _cgroup_cpu_count

    # quota=400000 period=100000 -> ceil(400000/100000) = 4 CPUs.
    files = {"/sys/fs/cgroup/cpu.max": "400000 100000\n"}
    with patch("builtins.open", _make_open(files)):
        assert _cgroup_cpu_count() == 4


def test_cgroup_cpu_count_v2_max_returns_none(tmp_path, monkeypatch) -> None:
    """cgroup v2 ``cpu.max`` set to ``"max"`` means unlimited -> None."""
    from myagent.core.hardware_profiler import _cgroup_cpu_count

    files = {"/sys/fs/cgroup/cpu.max": "max 100000\n"}
    with patch("builtins.open", _make_open(files)):
        assert _cgroup_cpu_count() is None


def test_cgroup_cpu_count_v1_fallback(tmp_path, monkeypatch) -> None:
    """When v2 ``cpu.max`` is missing, the v1 fallback path is used."""
    from myagent.core.hardware_profiler import _cgroup_cpu_count

    # No cpu.max file -> v1 reads succeed. quota=200000 period=100000 -> 2 CPUs.
    files = {
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "200000\n",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
    }
    with patch("builtins.open", _make_open(files)):
        assert _cgroup_cpu_count() == 2


def test_cgroup_cpu_count_unreadable_returns_none(tmp_path, monkeypatch) -> None:
    """When no cgroup files are readable, ``None`` is returned."""
    from myagent.core.hardware_profiler import _cgroup_cpu_count

    with patch("builtins.open", _make_open({})):
        assert _cgroup_cpu_count() is None


def test_cgroup_memory_gb_v2(tmp_path, monkeypatch) -> None:
    """cgroup v2 ``memory.max`` byte count is converted to GB."""
    from myagent.core.hardware_profiler import _cgroup_memory_gb

    # 4 GiB = 4 * 1024**3 bytes.
    files = {"/sys/fs/cgroup/memory.max": str(4 * 1024**3) + "\n"}
    with patch("builtins.open", _make_open(files)):
        assert abs(_cgroup_memory_gb() - 4.0) < 0.001


def test_cgroup_memory_gb_v1(tmp_path, monkeypatch) -> None:
    """When v2 ``memory.max`` is missing, the v1 fallback path is used."""
    from myagent.core.hardware_profiler import _cgroup_memory_gb

    # 8 GiB = 8 * 1024**3 bytes.
    files = {"/sys/fs/cgroup/memory/memory.limit_in_bytes": str(8 * 1024**3) + "\n"}
    with patch("builtins.open", _make_open(files)):
        assert abs(_cgroup_memory_gb() - 8.0) < 0.001
