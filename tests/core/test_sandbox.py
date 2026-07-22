"""Tests for the sandbox runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from myagent.core import sandbox as sandbox_module
from myagent.core.sandbox import SandboxError, SandboxRunner, _decode_stream


@pytest.fixture(autouse=True)
def _clear_probe_cache() -> None:
    """Reset the network-isolation probe cache between tests."""
    sandbox_module._probe_cache.clear()


def test_sandbox_runs_command() -> None:
    runner = SandboxRunner(network=True)
    result = runner.run(["python", "-c", "print('hello')"])
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_sandbox_env_isolation() -> None:
    runner = SandboxRunner(network=True, env_whitelist=["PATH"])
    result = runner.run(
        ["python", "-c", "import os; print(os.environ.get('MYAGENT_TEST', 'missing'))"]
    )
    assert result.returncode == 0
    assert "missing" in result.stdout


def test_sandbox_uses_custom_working_dir(tmp_path: Path) -> None:
    runner = SandboxRunner(network=True, working_dir=tmp_path)
    result = runner.run(["python", "-c", "import os; print(os.getcwd())"])
    assert result.returncode == 0
    assert str(tmp_path) in result.stdout


def test_sandbox_returns_error_for_missing_command() -> None:
    runner = SandboxRunner(network=True)
    result = runner.run(["this-command-does-not-exist-12345"])
    assert result.returncode == -1
    assert result.stderr != ""


def test_sandbox_wraps_network_when_tool_available() -> None:
    runner = SandboxRunner(network=False)
    with (
        patch("shutil.which", side_effect=["unshare", None]),
        patch.object(sandbox_module, "_network_isolation_works", return_value=True),
    ):
        wrapped = runner._wrap_network(["python", "-c", "pass"])
    assert wrapped[0] == "unshare"
    assert wrapped[1] == "--net"


def test_sandbox_degrades_when_unshare_unusable_required_false() -> None:
    """H4: unshare on PATH but not usable in this env → required=False degrades."""
    runner = SandboxRunner(network=False, required=False)
    with (
        patch("shutil.which", return_value="/usr/bin/unshare"),
        patch.object(sandbox_module, "_network_isolation_works", return_value=False),
    ):
        wrapped = runner._wrap_network(["python", "-c", "pass"])
    # Falls through to the no-tool branch: command runs unwrapped + warning logged.
    assert wrapped == ["python", "-c", "pass"]


def test_sandbox_required_raises_when_unshare_unusable() -> None:
    """H4: unshare on PATH but not usable + required=True → raise SandboxError."""
    runner = SandboxRunner(network=False, required=True)
    with (
        patch("shutil.which", return_value="/usr/bin/unshare"),
        patch.object(sandbox_module, "_network_isolation_works", return_value=False),
        pytest.raises(SandboxError),
    ):
        runner._wrap_network(["python", "-c", "pass"])


def test_sandbox_falls_back_without_tool_when_explicitly_optional() -> None:
    runner = SandboxRunner(network=False, required=False)
    with patch("shutil.which", return_value=None):
        wrapped = runner._wrap_network(["python", "-c", "pass"])
    assert wrapped == ["python", "-c", "pass"]


def test_sandbox_default_raises_when_tool_missing() -> None:
    runner = SandboxRunner(network=False)
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(SandboxError),
    ):
        runner._wrap_network(["python", "-c", "pass"])


def test_sandbox_required_uses_tool_when_available() -> None:
    runner = SandboxRunner(network=False, required=True)
    with (
        patch("shutil.which", side_effect=[None, "firejail"]),
        patch.object(sandbox_module, "_network_isolation_works", return_value=True),
    ):
        wrapped = runner._wrap_network(["python", "-c", "pass"])
    assert wrapped[0] == "firejail"


def test_sandbox_available_reports_capabilities() -> None:
    runner = SandboxRunner(network=False)
    caps = runner.available()
    assert "network_unshare" in caps
    assert "network_firejail" in caps
    assert caps["directory_isolation"] is True
    assert caps["env_isolation"] is True


# ---------------------------------------------------------------------------
# _decode_stream
# ---------------------------------------------------------------------------


def test_decode_stream_returns_none_for_none() -> None:
    assert _decode_stream(None) is None


def test_decode_stream_decodes_bytes() -> None:
    assert _decode_stream(b"hello world") == "hello world"


def test_decode_stream_passes_through_str() -> None:
    assert _decode_stream("already str") == "already str"


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


def test_sandbox_timeout_produces_error_result() -> None:
    import subprocess

    runner = SandboxRunner(network=True)
    with patch.object(
        subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1)
    ):
        result = runner.run(["sleep", "10"], timeout=0.1)
    assert result.returncode == -1
    assert "timed out" in result.stderr.lower()


def test_sandbox_oserror_produces_error_result() -> None:
    import subprocess

    runner = SandboxRunner(network=True)
    with patch.object(subprocess, "run", side_effect=OSError("bad fd")):
        result = runner.run(["python", "-c", "pass"])
    assert result.returncode == -1
    assert "bad fd" in result.stderr
