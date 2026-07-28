"""Checkpoint system — shadow git snapshots for reversible agent actions.

Ports Cline's checkpoint mechanism: after every tool call, snapshot the
working tree into a hidden shadow git repo (``.myagent/checkpoints/``).
Each snapshot is tagged with the iteration number and a short hash, and
can be restored (files only, or files + conversation).

Reference: ``competitors/cline/sdk/packages/core/src/checkpoint/``.

Design:

* :class:`CheckpointManager` owns the shadow repo. It initializes a bare
  git repo at ``<project>/.myagent/checkpoints/`` on first use, then
  tracks the project working tree as a second worktree via
  ``git --work-tree=<project>``.
* Each :meth:`snapshot` call creates a commit on the shadow repo whose
  tree mirrors the project's current state. The commit hash is the
  checkpoint id.
* :meth:`restore` checks out a previous commit's tree back into the
  project directory. Optionally also rewinds the conversation to the
  iteration that created the checkpoint.
* The shadow repo is separate from the user's own git repo — it lives
  under ``.myagent/`` which is typically gitignored.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from myagent.exceptions import MyAgentError


class CheckpointError(MyAgentError):
    """Raised when a checkpoint operation fails."""


@dataclass(frozen=True)
class Checkpoint:
    """A single checkpoint (shadow git commit).

    Attributes:
        id: Short commit hash from the shadow repo.
        iteration: Agent-loop iteration when the checkpoint was created.
        tool_name: Name of the tool that triggered this checkpoint
            (empty for the initial checkpoint at iteration 0).
        timestamp: Unix epoch seconds.
        message: Human-readable description.
        metadata: Extra context (tool input, file count, etc.).
    """

    id: str
    iteration: int
    tool_name: str
    timestamp: float
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Configuration for :class:`CheckpointManager`.

    Attributes:
        enabled: If False, :meth:`snapshot` is a no-op.
        checkpoint_dir: Relative path under the project root for the
            shadow git repo. Defaults to ``.myagent/checkpoints``.
        max_checkpoints: Maximum number of checkpoints to retain.
            Older checkpoints are pruned via ``git gc``. 0 = unlimited.
        exclude_patterns: Glob patterns of files to exclude from
            snapshots (matched against the relative path). Defaults to
            common VCS/build artifacts.
    """

    enabled: bool = True
    checkpoint_dir: str = ".myagent/checkpoints"
    max_checkpoints: int = 100
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            ".git/",
            ".myagent/",
            "__pycache__/",
            "*.pyc",
            "node_modules/",
            ".venv/",
            "venv/",
            "*.egg-info/",
            "dist/",
            "build/",
        ]
    )


class CheckpointManager:
    """Manages shadow-git checkpoints for a project directory.

    The shadow repo is a separate git repository that tracks the
    *content* of the project directory without touching the user's own
    ``.git``. Snapshots are commits; restores check out a commit's tree
    back into the project directory.

    Example:

        >>> mgr = CheckpointManager("/path/to/project")
        >>> mgr.initialize()
        >>> cp = mgr.snapshot(iteration=0, tool_name="", message="initial")
        >>> # ... agent makes changes ...
        >>> cp2 = mgr.snapshot(iteration=1, tool_name="write_to_file",
        ...                     message="after write_to_file")
        >>> mgr.restore(cp.id)  # revert project to cp's state
    """

    def __init__(
        self,
        project_root: str | Path,
        config: CheckpointConfig | None = None,
    ) -> None:
        self._project_root = Path(project_root).resolve()
        self._config = config or CheckpointConfig()
        self._shadow_repo = self._project_root / self._config.checkpoint_dir
        self._git_env = {
            "GIT_AUTHOR_NAME": "myagent-checkpoint",
            "GIT_AUTHOR_EMAIL": "checkpoint@myagent.local",
            "GIT_COMMITTER_NAME": "myagent-checkpoint",
            "GIT_COMMITTER_EMAIL": "checkpoint@myagent.local",
        }
        self._initialized = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        """The project directory being snapshotted."""

        return self._project_root

    @property
    def shadow_repo_path(self) -> Path:
        """Path to the shadow git repository."""

        return self._shadow_repo

    @property
    def is_enabled(self) -> bool:
        """Whether checkpoints are enabled (per config)."""

        return self._config.enabled

    @property
    def is_initialized(self) -> bool:
        """Whether the shadow repo has been initialized."""

        return self._initialized

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize the shadow git repo if not already done.

        Creates ``.myagent/checkpoints/`` as a git repository. Safe to
        call multiple times. If checkpoints are disabled, this is a
        no-op.
        """

        if not self._config.enabled:
            return
        if self._initialized:
            return
        self._shadow_repo.mkdir(parents=True, exist_ok=True)
        # Init a fresh git repo in the shadow directory.
        self._run_git("init", "--quiet", cwd=self._shadow_repo)
        # Configure the shadow repo to allow committing as root even
        # though the working tree is elsewhere.
        self._run_git(
            "config", "user.name", "myagent-checkpoint", cwd=self._shadow_repo
        )
        self._run_git(
            "config", "user.email", "checkpoint@myagent.local", cwd=self._shadow_repo
        )
        # Write the exclude patterns to the shadow repo's
        # .git/info/exclude so they apply when adding files from the
        # project worktree. Using .git/info/exclude (instead of a
        # .gitignore in the worktree) keeps the exclude file inside the
        # shadow repo, not the user's project.
        exclude_file = self._shadow_repo / ".git" / "info" / "exclude"
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        exclude_file.write_text(
            "\n".join(self._config.exclude_patterns) + "\n", encoding="utf-8"
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(
        self,
        *,
        iteration: int,
        tool_name: str = "",
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint | None:
        """Create a checkpoint of the current project state.

        Returns the :class:`Checkpoint`, or ``None`` if checkpoints are
        disabled or the shadow repo isn't initialized.

        Raises :class:`CheckpointError` if the git operations fail.
        """

        if not self._config.enabled:
            return None
        with self._lock:
            if not self._initialized:
                self.initialize()
                if not self._initialized:
                    return None

            metadata = metadata or {}
            timestamp = time.time()
            commit_message = self._format_commit_message(
                iteration=iteration,
                tool_name=tool_name,
                message=message,
                timestamp=timestamp,
                metadata=metadata,
            )

            # Stage all files from the project directory into the shadow
            # repo's index, using --work-tree to point at the project.
            # We use `git add -A` (NOT --force) so the .gitignore in the
            # shadow repo excludes the patterns we don't want to track.
            # However, GIT_WORK_TREE pointing at the project means git add
            # will traverse the project dir; the .gitignore in the shadow
            # repo root applies to relative paths.
            with contextlib.suppress(CheckpointError):
                self._run_git(
                    "add",
                    "-A",
                    cwd=self._shadow_repo,
                    env_extra={"GIT_WORK_TREE": str(self._project_root)},
                )

            # Commit. --allow-empty ensures we always get a checkpoint even
            # if nothing changed (useful for marking iteration boundaries).
            try:
                self._run_git(
                    "commit",
                    "--quiet",
                    "--allow-empty",
                    "-m",
                    commit_message,
                    cwd=self._shadow_repo,
                    env_extra={"GIT_WORK_TREE": str(self._project_root)},
                )
            except CheckpointError as exc:
                # A commit can fail if there's nothing to commit AND
                # --allow-empty somehow doesn't apply. Fall back to reading
                # the current HEAD.
                if "nothing to commit" in str(exc).lower():
                    pass
                else:
                    raise

            # Get the commit hash.
            result = self._run_git(
                "rev-parse", "--short", "HEAD", cwd=self._shadow_repo
            )
            commit_id = result.stdout.strip()

            cp = Checkpoint(
                id=commit_id,
                iteration=iteration,
                tool_name=tool_name,
                timestamp=timestamp,
                message=message,
                metadata=metadata,
            )

            # Prune old checkpoints if needed.
            if self._config.max_checkpoints > 0:
                self._prune_old_checkpoints()

            return cp

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self, checkpoint_id: str) -> None:
        """Restore the project directory to the state at ``checkpoint_id``.

        This checks out the commit's tree into the project directory,
        overwriting current files. Deleted files are removed; new files
        since the checkpoint are also removed (to match the snapshot
        exactly). The user's own ``.git`` directory is preserved.

        Raises :class:`CheckpointError` if the checkpoint doesn't exist.
        """

        with self._lock:
            if not self._config.enabled:
                raise CheckpointError("Checkpoints are disabled")
            if not self._initialized:
                raise CheckpointError("Checkpoint manager not initialized")

            # Verify the checkpoint exists.
            verify = self._run_git(
                "cat-file", "-t", checkpoint_id, cwd=self._shadow_repo, check=False
            )
            if verify.returncode != 0 or verify.stdout.strip() != "commit":
                raise CheckpointError(
                    f"Unknown checkpoint id: {checkpoint_id}",
                    details={"id": checkpoint_id},
                )

            # Get the list of files tracked at the checkpoint commit.
            checkpoint_files = self._list_files_at_commit(checkpoint_id)

            # Strategy: for each file in the checkpoint, extract its content
            # from the commit and write it to the project directory. This
            # avoids index/worktree confusion that arises with
            # `git checkout <id> -- .` when GIT_WORK_TREE points elsewhere.
            protected = {".git", ".myagent"}

            # Step 1: Remove files in the project that are NOT in the
            # checkpoint (they were added after the snapshot). Check for
            # symlinks first so we never follow a symlink into an external
            # directory with shutil.rmtree.
            for item in self._project_root.iterdir():
                if item.name in protected:
                    continue
                rel = item.relative_to(self._project_root).as_posix()
                if rel not in checkpoint_files:
                    if item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)

            # Step 2: For each file in the checkpoint, extract its content
            # from the commit tree and write it to the project. We use the
            # bytes-returning variant of _run_git so binary files (images,
            # compiled artifacts, etc.) are restored verbatim without any
            # text round-trip that would corrupt them.
            for rel_path in checkpoint_files:
                # Extract the file content from the commit.
                result = self._run_git_bytes(
                    "show",
                    f"{checkpoint_id}:{rel_path}",
                    cwd=self._shadow_repo,
                    check=False,
                )
                if result.returncode != 0:
                    continue  # File may be binary or otherwise problematic.
                target = self._project_root / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(result.stdout)

            # Step 3: Remove any empty directories that were left behind
            # (except protected ones).
            for root, _dirs, _files in os.walk(self._project_root, topdown=False):
                root_path = Path(root)
                if root_path == self._project_root:
                    continue
                # Skip protected directories.
                try:
                    relative = root_path.relative_to(self._project_root)
                except ValueError:
                    continue
                relative_parts = relative.parts
                if relative_parts and relative_parts[0] in protected:
                    continue
                if not any(True for _ in root_path.iterdir()):
                    root_path.rmdir()

    # ------------------------------------------------------------------
    # List / get
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> list[Checkpoint]:
        """Return all checkpoints, newest first."""

        if not self._initialized:
            return []
        result = self._run_git(
            "log",
            "--pretty=format:%H%x1f%h%x1f%s",
            "--no-merges",
            cwd=self._shadow_repo,
            check=False,
        )
        if result.returncode != 0:
            return []
        checkpoints: list[Checkpoint] = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split("\x1f")
            if len(parts) < 3:
                continue
            short_hash, subject = parts[1], parts[2]
            cp = self._parse_commit_message(short_hash, subject)
            if cp is not None:
                checkpoints.append(cp)
        return checkpoints

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Return the checkpoint with the given id, or None."""

        for cp in self.list_checkpoints():
            if cp.id == checkpoint_id:
                return cp
        return None

    def latest(self) -> Checkpoint | None:
        """Return the most recent checkpoint, or None if there are none."""

        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(
        self, checkpoint_id_a: str | None = None, checkpoint_id_b: str | None = None
    ) -> str:
        """Return a unified diff between two checkpoints.

        If ``checkpoint_id_a`` is None, uses the parent of
        ``checkpoint_id_b``. If both are None, returns the diff of the
        latest checkpoint vs its parent.
        """

        if not self._initialized:
            return ""
        if checkpoint_id_a is None and checkpoint_id_b is None:
            commit_b = "HEAD"
            commit_a: str = "HEAD~1"
        elif checkpoint_id_a is None:
            commit_b = checkpoint_id_b or "HEAD"
            commit_a = f"{commit_b}~1"
        else:
            commit_a = checkpoint_id_a
            commit_b = checkpoint_id_b or "HEAD"
        result = self._run_git(
            "diff",
            "--no-color",
            commit_a,
            commit_b,
            cwd=self._shadow_repo,
            check=False,
        )
        return result.stdout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_git(
        self,
        *args: str,
        cwd: Path,
        check: bool = True,
        env_extra: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command in the shadow repo."""

        env = {**self._git_env}
        if env_extra:
            env.update(env_extra)
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=check,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ""
            if "nothing to commit" in stderr.lower():
                # Re-raise as a soft error the caller can handle.
                raise CheckpointError(
                    f"Nothing to commit: {stderr.strip()}",
                    details={"returncode": exc.returncode},
                ) from exc
            raise CheckpointError(
                f"Git command failed: git {' '.join(args)}",
                details={
                    "cmd": list(args),
                    "returncode": exc.returncode,
                    "stdout": exc.stdout,
                    "stderr": stderr,
                },
            ) from exc

    def _run_git_bytes(
        self,
        *args: str,
        cwd: Path,
        check: bool = True,
        env_extra: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a git command and return raw bytes output.

        Like :meth:`_run_git` but uses ``text=False`` so stdout is
        returned as raw bytes. Use this for commands whose output may
        contain binary data (e.g. ``git show <id>:<path>`` to fetch a
        file's content), so that binary files are not corrupted by a
        text round-trip.
        """

        env = {**self._git_env}
        if env_extra:
            env.update(env_extra)
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=check,
                capture_output=True,
                text=False,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            stderr_bytes = exc.stderr or b""
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            raise CheckpointError(
                f"Git command failed: git {' '.join(args)}",
                details={
                    "cmd": list(args),
                    "returncode": exc.returncode,
                    "stdout": exc.stdout,
                    "stderr": stderr,
                },
            ) from exc

    def _format_commit_message(
        self,
        *,
        iteration: int,
        tool_name: str,
        message: str,
        timestamp: float,
        metadata: dict[str, Any],
    ) -> str:
        """Format the commit message with structured metadata.

        The message is a JSON envelope so it can be parsed back by
        :meth:`_parse_commit_message`. We use a ``MYAGENT-CP:`` prefix
        to distinguish our commits from any user commits that might
        leak into the shadow repo.
        """

        envelope = {
            "iteration": iteration,
            "tool_name": tool_name,
            "message": message,
            "timestamp": timestamp,
            "metadata": metadata,
        }
        return f"MYAGENT-CP: {json.dumps(envelope, ensure_ascii=False)}"

    def _parse_commit_message(
        self, short_hash: str, subject: str
    ) -> Checkpoint | None:
        """Parse a commit subject back into a :class:`Checkpoint`."""

        prefix = "MYAGENT-CP: "
        if not subject.startswith(prefix):
            return None
        try:
            data = json.loads(subject[len(prefix):])
        except json.JSONDecodeError:
            return None
        return Checkpoint(
            id=short_hash,
            iteration=data.get("iteration", 0),
            tool_name=data.get("tool_name", ""),
            timestamp=data.get("timestamp", 0.0),
            message=data.get("message", ""),
            metadata=data.get("metadata", {}),
        )

    def _list_files_at_commit(self, commit_id: str) -> set[str]:
        """Return the set of file paths tracked at ``commit_id``."""

        result = self._run_git(
            "ls-tree",
            "-r",
            "--name-only",
            commit_id,
            cwd=self._shadow_repo,
            check=False,
        )
        if result.returncode != 0:
            return set()
        return {line for line in result.stdout.strip().splitlines() if line}

    def _prune_old_checkpoints(self) -> None:
        """If we exceed max_checkpoints, prune the oldest via gc."""

        result = self._run_git(
            "rev-list", "--count", "HEAD", cwd=self._shadow_repo, check=False
        )
        if result.returncode != 0:
            return
        try:
            count = int(result.stdout.strip())
        except ValueError:
            return
        if count <= self._config.max_checkpoints:
            return
        # Find the commit at the cutoff point and rewrite history to
        # drop everything before it. This is destructive but keeps the
        # shadow repo from growing unboundedly.
        cutoff = count - self._config.max_checkpoints

        # Capture the current branch name *before* switching to the
        # orphan branch, so we can delete it afterwards regardless of
        # what the repo's default branch is called (main, master, init,
        # trunk, develop, ...). Fall back to "main" if we're in detached
        # HEAD or the command fails.
        branch_result = self._run_git(
            "branch", "--show-current", cwd=self._shadow_repo, check=False
        )
        current_branch = (
            branch_result.stdout.strip()
            if branch_result.returncode == 0 and branch_result.stdout.strip()
            else "main"
        )

        result = self._run_git(
            "rev-list",
            "--reverse",
            "HEAD",
            cwd=self._shadow_repo,
        )
        commits = result.stdout.strip().splitlines()
        if cutoff < len(commits):
            new_root = commits[cutoff]
            # Create a new orphan branch starting at new_root.
            self._run_git(
                "checkout",
                "--orphan",
                "_myagent_temp",
                new_root,
                cwd=self._shadow_repo,
            )
            self._run_git(
                "commit",
                "--quiet",
                "-m",
                "MYAGENT-CP: pruned history (root)",
                cwd=self._shadow_repo,
            )
            # Drop the old branch (whatever it was actually called) so
            # the pre-cutoff commits become unreachable, then rename the
            # new orphan branch to "main".
            self._run_git(
                "branch",
                "-D",
                current_branch,
                cwd=self._shadow_repo,
                check=False,
            )
            self._run_git(
                "branch", "-m", "main", cwd=self._shadow_repo, check=False
            )
            # gc to actually reclaim space.
            self._run_git(
                "gc", "--quiet", "--prune=now", cwd=self._shadow_repo, check=False
            )


def diff_files(
    project_root: str | Path,
    shadow_repo: str | Path,
    commit_a: str,
    commit_b: str,
) -> list[str]:
    """Standalone helper: list files that differ between two commits."""

    result = subprocess.run(
        [
            "git",
            "-C",
            str(shadow_repo),
            "diff",
            "--name-only",
            commit_a,
            commit_b,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.strip().splitlines() if line]


__all__ = [
    "Checkpoint",
    "CheckpointConfig",
    "CheckpointError",
    "CheckpointManager",
    "diff_files",
]
