"""Run-on-save hook execution.

Run-on-save hooks are **configuration-driven** user hooks (not pluggy
hookspecs): when a file is saved, every entry in ``[hooks] on_save`` whose
``include`` globs match the saved path is executed by invoking the
corresponding MyAgent command (``clean`` or ``verify``) in a subprocess.

The runner is intentionally command-agnostic: it builds the subprocess
command, applies per-hook debouncing, records audit events, and returns
structured results. Two front-ends drive it:

* :func:`myagent.cli.commands.hooks` — the ``myagent hooks`` command
  (``list`` / ``run`` / ``watch``), the universal mechanism that works with
  any editor via filesystem events.
* :class:`myagent.cli.commands.lsp._LSPServer` — the ``textDocument/didSave``
  LSP notification, for editors that speak LSP.

Both reuse the same runner so debounce, glob filtering and audit semantics
are identical.
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from myagent.core.audit_logger import AuditLogger
    from myagent.models.config import AppConfig, HookConfig

logger = structlog.get_logger("myagent")


def _matches_any(path_str: str, patterns: Sequence[str]) -> bool:
    """Return True if ``path_str`` matches any of the glob ``patterns``."""
    return any(fnmatch.fnmatch(path_str, p) for p in patterns)


@dataclass(frozen=True)
class HookResult:
    """The outcome of running a single on-save hook."""

    hook: HookConfig
    command: list[str]
    exit_code: int
    duration_ms: float
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        """True when the hook completed without error or timeout."""
        return self.exit_code == 0 and not self.timed_out


class OnSaveHookRunner:
    """Execute configured ``[hooks] on_save`` entries for a saved file.

    The runner is stateful only for debouncing: it remembers the last run
    timestamp of each hook instance so rapid successive saves do not fire
    the same hook more often than ``debounce_ms`` allows. It is safe to
    reuse a single runner across many saves; create a fresh one per session.
    """

    def __init__(
        self,
        config: AppConfig,
        audit_logger: AuditLogger | None = None,
        *,
        python_executable: str | None = None,
        timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.audit_logger = audit_logger
        self.python_executable = python_executable or sys.executable
        self.timeout = float(timeout)
        self._clock = clock
        self._last_run: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _relative_path(self, path: Path) -> str:
        """Return ``path`` as a posix string relative to the project root.

        Falls back to the path's posix form when it cannot be made relative
        (e.g. it lives outside the project root), so glob filtering still
        has something deterministic to match against.
        """
        try:
            root = Path(self.config.project_root).resolve()
            rel = Path(path).resolve().relative_to(root)
            return rel.as_posix()
        except (ValueError, OSError):
            return Path(path).as_posix()

    def matching_hooks(self, path: Path) -> list[tuple[int, HookConfig]]:
        """Return ``(index, hook)`` pairs whose globs match ``path``.

        Returns an empty list when ``[hooks] enabled`` is false.
        """
        if not self.config.hooks.enabled:
            return []
        rel_str = self._relative_path(path)
        matches: list[tuple[int, HookConfig]] = []
        for idx, hook in enumerate(self.config.hooks.on_save):
            include = hook.include or ["**/*"]
            if not _matches_any(rel_str, include):
                continue
            if _matches_any(rel_str, hook.exclude):
                continue
            matches.append((idx, hook))
        return matches

    # ------------------------------------------------------------------
    # Debounce
    # ------------------------------------------------------------------

    def _is_debounced(self, hook: HookConfig) -> bool:
        last = self._last_run.get(id(hook))
        if last is None:
            return False
        return (self._clock() - last) * 1000.0 < hook.debounce_ms

    def reset(self) -> None:
        """Clear all debounce state (e.g. between unrelated runs)."""
        self._last_run.clear()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def build_command(self, hook: HookConfig, path: Path) -> list[str] | None:
        """Build the subprocess command for ``hook`` against ``path``.

        Returns ``None`` when a ``verify`` hook has no ``verify_command``;
        the caller should skip such a hook rather than run an empty command
        (which the ``verify`` entrypoint rejects with ``empty_command``).
        """
        cmd: list[str] = [self.python_executable, "-m", "myagent", hook.command]
        if hook.command == "clean":
            cmd.append("--yes")
            cmd.append(str(path))
        else:  # verify
            if not hook.verify_command:
                logger.warning(
                    "verify hook (include=%s) configured without verify_command, skipping",
                    hook.include,
                )
                return None
            cmd.append(hook.verify_command)
        cmd.extend(hook.args)
        return cmd

    def run_hook(self, hook: HookConfig, path: Path) -> HookResult | None:
        """Execute one hook synchronously and record audit events.

        Returns ``None`` when the hook is misconfigured (e.g. a ``verify``
        hook without ``verify_command``) so the caller can skip it without
        spawning a subprocess.
        """
        command = self.build_command(hook, path)
        if command is None:
            return None
        audit = self.audit_logger
        rel_str = self._relative_path(path)
        if audit is not None:
            audit.record(
                "hook.run.start",
                {"command": hook.command, "path": rel_str, "hook_index": id(hook)},
            )

        start = self._clock()
        exit_code = 0
        stdout = ""
        stderr = ""
        timed_out = False
        try:
            proc = subprocess.run(
                command,
                cwd=str(self.config.project_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            exit_code = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            out = exc.stdout
            err = exc.stderr
            stdout = out.decode("utf-8", "replace") if isinstance(out, bytes) else (out or "")
            stderr = err.decode("utf-8", "replace") if isinstance(err, bytes) else (err or "")
        except OSError as exc:
            exit_code = 127
            stderr = str(exc)

        duration_ms = (self._clock() - start) * 1000.0
        self._last_run[id(hook)] = self._clock()

        if audit is not None:
            audit.record(
                "hook.run.done" if exit_code == 0 else "hook.run.error",
                {
                    "command": hook.command,
                    "path": rel_str,
                    "exit_code": exit_code,
                    "duration_ms": round(duration_ms, 2),
                    "timed_out": timed_out,
                },
            )
        return HookResult(
            hook=hook,
            command=command,
            exit_code=exit_code,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )

    def run_for_path(self, path: Path) -> list[HookResult]:
        """Run every matching, non-debounced hook for ``path`` in order."""
        results: list[HookResult] = []
        for _idx, hook in self.matching_hooks(path):
            if self._is_debounced(hook):
                continue
            result = self.run_hook(hook, path)
            if result is None:
                continue
            results.append(result)
        return results
