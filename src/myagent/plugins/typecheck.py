"""Official typecheck plugin for MyAgent-CLI.

Runs lightweight static type checks during the ``pre_commit`` hook. The plugin
uses the ``pyright`` executable when available and degrades gracefully if Pyright
is not installed in the current environment.
"""

from __future__ import annotations

import shutil
import subprocess

import structlog

from myagent.core.context import CommandContext
from myagent.exceptions import VerifyError
from myagent.hookspec import hookimpl

logger = structlog.get_logger("myagent")


def _tool_executable() -> str | None:
    """Return the Pyright executable if it is available on PATH."""
    return shutil.which("pyright")


class TypecheckPlugin:
    """Built-in type checker invoked before each commit."""

    @hookimpl
    def pre_commit(self, context: CommandContext) -> None:
        """Run Pyright on the project if it is installed."""
        executable = _tool_executable()
        if executable is None:
            logger.warning("pyright not found on PATH; skipping type check.")
            return

        if context.dry_run:
            logger.info("[dry-run] Would run pyright type check")
            return

        result = subprocess.run(
            [executable],
            cwd=context.project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise VerifyError(
                "Type check failed. Run `pyright` locally for details.",
                details={"stdout": result.stdout, "stderr": result.stderr},
            )

        logger.info("Type check passed")


plugin = TypecheckPlugin()
