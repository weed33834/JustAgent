"""Adapter for Git operations."""

from __future__ import annotations

import subprocess
from pathlib import Path

from myagent.exceptions import GitError
from myagent.models.config import ToolsConfig
from myagent.utils.hashing import ToolVerifier


class GitAdapter:
    """Thin wrapper around Git subprocess calls."""

    def __init__(self, repo_root: Path, tool_verifier: ToolVerifier | None = None) -> None:
        self.repo_root = repo_root
        self._verifier = tool_verifier or ToolVerifier(ToolsConfig())

    def _git_cmd(self, *args: str) -> list[str]:
        """Build a git command list using the verified git executable."""
        return [self._verifier.resolve("git"), *args]

    def _run(
        self,
        cmd: list[str],
        *,
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                cmd,
                cwd=self.repo_root,
                check=check,
                capture_output=capture_output,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise GitError(
                f"Git command failed: {' '.join(cmd)}",
                details={"cmd": cmd, "returncode": exc.returncode, "stderr": exc.stderr},
            ) from exc

    def has_changes(self) -> bool:
        """Return True if the working tree has staged or unstaged changes."""
        result = self._run(self._git_cmd("status", "--porcelain"), check=False)
        return result.stdout.strip() != ""

    def is_git_repo(self) -> bool:
        """Return True if ``repo_root`` is inside a git working tree."""
        result = self._run(self._git_cmd("rev-parse", "--is-inside-work-tree"), check=False)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def diff(self) -> str:
        """Return the full diff of staged and unstaged changes.

        Combines ``git diff --cached`` and ``git diff`` so that both staged and
        unstaged changes are visible, and avoids ``HEAD`` which does not exist in
        freshly initialized repositories.
        """
        staged = self._run(self._git_cmd("diff", "--cached")).stdout
        unstaged = self._run(self._git_cmd("diff")).stdout
        return f"{staged}\n{unstaged}".strip()

    def stats(self) -> str:
        """Return a short diff stat summary for staged and unstaged changes."""
        staged = self._run(self._git_cmd("diff", "--stat", "--cached")).stdout
        unstaged = self._run(self._git_cmd("diff", "--stat")).stdout
        return f"{staged}\n{unstaged}".strip()

    def commit(self, message: str) -> None:
        """Stage all changes and commit with the given message."""
        self._run(self._git_cmd("add", "-A"))
        self._run(self._git_cmd("commit", "-m", message))
