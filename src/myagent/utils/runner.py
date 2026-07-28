"""Unified subprocess execution helpers."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("myagent.runner")


@dataclass(frozen=True)
class RunResult:
    """Result of a command execution."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    cmd: Sequence[str],
    *,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    check: bool = False,
    env: dict[str, str] | None = None,
    capture: bool = True,
    text: bool = True,
) -> RunResult:
    """Run a command with standardized error handling and logging.

    Args:
        cmd: Command and arguments as a list.
        cwd: Working directory.
        timeout: Timeout in seconds.
        check: If True, raise CalledProcessError on non-zero exit.
        env: Additional environment variables.
        capture: If True, capture stdout/stderr.
        text: If True, decode output as text.

    Returns:
        RunResult with returncode, stdout, stderr.

    Raises:
        subprocess.CalledProcessError: If check=True and command fails.
        subprocess.TimeoutExpired: If timeout is reached.
    """
    logger.debug("Running command: %s (cwd=%s)", " ".join(cmd), cwd)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        check=check,
        capture_output=capture,
        text=text,
        env=env,
    )
    return RunResult(
        returncode=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


__all__ = ["RunResult", "run_command"]
