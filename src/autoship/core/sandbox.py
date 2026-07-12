"""Sandbox execution helpers for untrusted/community plugins.

This module provides a ``SandboxRunner`` that executes a command inside a
restricted subprocess environment. It is a first-phase implementation: it
limits the environment and working directory, and optionally blocks network
access when ``unshare`` or ``firejail`` is available.

Roadmap:

* Phase 2: filesystem isolation via read-only root mounts and tmpfs overlay.
* Phase 3: cgroup-based CPU, memory and I/O limits.
* Phase 4: seccomp-bpf syscall filtering and user namespace support.
* Phase 5: declarative sandbox profiles per plugin type.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from autoship.exceptions import SandboxError

logger = structlog.get_logger("autoship")


def _decode_stream(value: str | bytes | None) -> str | None:
    """Normalize subprocess output to a string."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


# Probe commands that exercise the same code path as the real network
# isolation wrapper without actually running a payload. `unshare --net` in a
# container without CAP_SYS_ADMIN fails at the syscall, not at flag parsing,
# so `unshare --help` is not enough; we use `unshare --net -- true` which
# round-trips the namespace creation. firejail exposes `--version` which does
# not require any capabilities.
_PROBE_COMMANDS: dict[str, list[str]] = {
    "unshare": ["unshare", "--net", "--", "true"],
    "firejail": ["firejail", "--version"],
}

# Cache probe results for the process lifetime so we don't re-fork on every
# sandboxed invocation. A tool either works in this environment or it does
# not; that does not change mid-process.
_probe_cache: dict[str, bool] = {}

# Serializes probe execution so two threads do not fork the probe command
# concurrently for the same tool. Double-checked locking is used below: the
# cache is consulted once without the lock (fast path) and re-checked inside
# the lock before running the probe (slow path).
_LOCK = threading.Lock()


def _network_isolation_works(tool: str) -> bool:
    """Return True if ``tool`` can actually create a network namespace.

    ``shutil.which`` only checks that the binary is on PATH. In containers
    (CI without CAP_SYS_ADMIN, Docker default seccomp profile) ``unshare
    --net`` exists on PATH but the syscall is denied at runtime. We probe
    the real capability once per process and cache the result.
    """
    # Fast path: a previous probe already populated the cache.
    if tool in _probe_cache:
        return _probe_cache[tool]
    probe = _PROBE_COMMANDS.get(tool)
    if probe is None:
        _probe_cache[tool] = False
        return False
    # Slow path: probe under the lock, re-checking the cache first so the
    # probe runs at most once per tool even under concurrent callers.
    with _LOCK:
        if tool in _probe_cache:
            return _probe_cache[tool]
        try:
            result = subprocess.run(
                probe,
                capture_output=True,
                text=True,
                timeout=5,
            )
            works = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            works = False
        _probe_cache[tool] = works
    if not works:
        logger.info(
            "Network isolation tool %r found on PATH but not usable in this environment; "
            "required=False callers will degrade to running without network isolation.",
            tool,
        )
    return works


@dataclass
class SandboxResult:
    """Result of a sandboxed execution."""

    returncode: int
    stdout: str
    stderr: str


class SandboxRunner:
    """Run a command in a restricted subprocess environment.

    Restrictions applied:

    - A fresh temporary working directory is used by default.
    - Only whitelisted environment variables are inherited.
    - If ``network`` is ``False`` and a supported network namespace tool is
      available, the command is wrapped to block network access.

    **Important**: When ``network`` is ``True``, network access is **not**
    restricted in any way.  The command runs with the same network access as
    the parent process.  Full network isolation (e.g. egress filtering) is
    planned for a future phase; until then, ``network=True`` provides no
    network-level sandboxing.

    See the module docstring for the planned roadmap (filesystem isolation,
    cgroup limits, seccomp-bpf, etc.).

    The runner does **not** degrade to un-sandboxed execution by default. When
    ``required`` is ``True`` (the default) and no network isolation tool is
    available, a ``SandboxError`` is raised instead of running the command
    without network restrictions. Callers that explicitly want graceful
    degradation must pass ``required=False``; in that case the runner still
    applies the environment and directory restrictions and logs a warning that
    network isolation could not be enforced.
    """

    def __init__(
        self,
        *,
        network: bool = False,
        env_whitelist: list[str] | None = None,
        working_dir: Path | None = None,
        required: bool = True,
    ) -> None:
        self.network = network
        self.env_whitelist = env_whitelist or [
            "PATH",
            "HOME",
            "USER",
            "LANG",
            "LC_ALL",
            "VIRTUAL_ENV",
            "XDG_CACHE_HOME",
            "UV_CACHE_DIR",
        ]
        self.working_dir = working_dir
        self.required = required

    def run(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        """Run ``command`` inside the sandbox constraints."""
        owns_cwd = self.working_dir is None
        cwd = self.working_dir or Path(tempfile.mkdtemp(prefix="autoship-sandbox-"))
        cwd.mkdir(parents=True, exist_ok=True)

        try:
            env = self._build_env()
            wrapped = self._wrap_network(command)

            logger.debug("Sandbox run: %s in %s (network=%s)", wrapped, cwd, self.network)
            try:
                proc = subprocess.run(
                    wrapped,
                    cwd=cwd,
                    env=env,
                    input=input_text,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    returncode=-1,
                    stdout=_decode_stream(exc.stdout) or "",
                    stderr=_decode_stream(exc.stderr) or "Sandbox execution timed out",
                )
            except (OSError, FileNotFoundError) as exc:
                return SandboxResult(returncode=-1, stdout="", stderr=str(exc))

            return SandboxResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        finally:
            # Only clean up temp dirs we created; caller-owned working_dir
            # lifetime is the caller's responsibility.
            if owns_cwd:
                shutil.rmtree(cwd, ignore_errors=True)

    def _build_env(self) -> dict[str, str]:
        """Return a minimal environment containing only whitelisted variables.

        Credentials embedded in ``PIP_INDEX_URL`` (e.g.
        ``https://user:pass@pypi.example.com/simple``) are stripped before
        the value is passed into the sandbox so that third-party code cannot
        exfiltrate them.
        """
        env = {key: value for key, value in os.environ.items() if key in self.env_whitelist}
        if "PIP_INDEX_URL" in env and "@" in env["PIP_INDEX_URL"]:
            parsed = urllib.parse.urlparse(env["PIP_INDEX_URL"])
            if parsed.password and parsed.hostname:
                safe_url = parsed._replace(
                    netloc=parsed.hostname + (f":{parsed.port}" if parsed.port else "")
                )
                env["PIP_INDEX_URL"] = urllib.parse.urlunparse(safe_url)
                logger.warning(
                    "Private PyPI index credentials stripped from PIP_INDEX_URL; "
                    "pip inside the sandbox may be unable to access private packages "
                    "(expected 401). Pre-install dependencies outside the sandbox if "
                    "you need them there."
                )
        return env

    def _wrap_network(self, command: list[str]) -> list[str]:
        """Wrap ``command`` with a network namespace tool when appropriate."""
        if self.network:
            # network=True → no network restriction at all; the command runs
            # with the same network access as the parent process.
            return command

        # Probe unshare/firejail with a cheap no-op (`--help` / `--version`)
        # before using them to wrap a real command. In containers (e.g. CI
        # without CAP_SYS_ADMIN) `unshare --net` exists on PATH but fails at
        # runtime; without this probe the wrapped command would silently exit
        # non-zero. When `required=False` we degrade to running without the
        # network namespace and log a warning; when `required=True` we raise.
        if shutil.which("unshare") and _network_isolation_works("unshare"):
            return ["unshare", "--net", "--", *command]
        if shutil.which("firejail") and _network_isolation_works("firejail"):
            return ["firejail", "--net=none", "--quiet", "--", *command]

        if self.required:
            raise SandboxError(
                "Sandbox is required but no network isolation tool is available (unshare/firejail)"
            )

        logger.warning("No network sandbox tool available (unshare/firejail); network not blocked")
        return command

    def available(self) -> dict[str, Any]:
        """Report sandbox capability availability."""
        return {
            "network_unshare": shutil.which("unshare") is not None,
            "network_firejail": shutil.which("firejail") is not None,
            "directory_isolation": True,
            "env_isolation": True,
        }
