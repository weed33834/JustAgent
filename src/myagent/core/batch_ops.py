"""Batch operations across managed projects.

Run a command or workflow stage across multiple managed projects in
sequence (or parallel for read-only checks). Used by the
``myagent project batch-*`` commands.
"""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from myagent.core.project_store import ProjectStore
from myagent.exceptions import MyAgentError
from myagent.models.project import ManagedProject


class BatchOpsError(MyAgentError):
    """Raised when a batch operation is misconfigured or cannot proceed."""


class BatchOperation(str, Enum):  # noqa: UP042 - match existing codebase style
    """A kind of batch operation that can be run across projects."""

    STATUS = "status"
    CLEAN = "clean"
    VERIFY = "verify"
    COMMIT = "commit"
    SHIP = "ship"
    RUN = "run"


@dataclass(frozen=True)
class BatchResult:
    """The outcome of running a single operation in one project."""

    project_name: str
    project_path: str
    operation: BatchOperation
    success: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class BatchSummary:
    """Aggregate outcome of a batch operation across many projects."""

    operation: BatchOperation
    results: list[BatchResult]
    total: int
    succeeded: int
    failed: int
    elapsed_seconds: float


def _stream_to_str(value: str | bytes | None) -> str:
    """Normalize a subprocess output stream to a string."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


class BatchRunner:
    """Run operations across the projects tracked by a :class:`ProjectStore`.

    Read-only operations (such as ``git status``) may be parallelised across
    projects; operations with side effects (arbitrary commands, workflow
    stages) always run sequentially to avoid interleaving mutations.
    """

    def __init__(
        self,
        store: ProjectStore,
        parallel: bool = False,
        max_workers: int = 1,
    ) -> None:
        self.store = store
        self.parallel = parallel
        self.max_workers = max(max_workers, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_status(
        self, project_names: list[str] | None = None
    ) -> BatchSummary:
        """Run ``git status --porcelain`` across projects.

        Read-only, so when ``parallel`` is enabled the checks run
        concurrently. A clean project produces empty ``stdout``; a dirty
        project lists changed files in ``stdout``.
        """
        projects, missing = self._select_projects(project_names)
        operation = BatchOperation.STATUS
        results: list[BatchResult] = [
            self._missing_result(name, operation) for name in missing
        ]

        if self.parallel and len(projects) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_project = {
                    executor.submit(
                        self._exec, project, operation, ["git", "status", "--porcelain"]
                    ): project
                    for project in projects
                }
                for future in as_completed(future_to_project):
                    results.append(future.result())
        else:
            for project in projects:
                results.append(
                    self._exec(
                        project, operation, ["git", "status", "--porcelain"]
                    )
                )
        return self._summarize(operation, results)

    def run_command(
        self,
        project_names: list[str] | None,
        command: list[str],
    ) -> BatchSummary:
        """Run an arbitrary ``command`` in each project directory.

        Always sequential — arbitrary commands may have side effects.
        """
        projects, missing = self._select_projects(project_names)
        operation = BatchOperation.RUN
        results: list[BatchResult] = [
            self._missing_result(name, operation) for name in missing
        ]
        for project in projects:
            results.append(self._exec(project, operation, command))
        return self._summarize(operation, results)

    def run_pipeline(
        self,
        project_names: list[str] | None,
        stages: list[BatchOperation],
    ) -> BatchSummary:
        """Run workflow ``stages`` across projects by shelling out to ``myagent``.

        Each stage runs ``myagent <stage>`` in the project directory. Within
        a single project, if a stage fails the remaining stages for that
        project are skipped (workflow semantics), but other projects
        continue. Each ``(project, stage)`` pair produces its own
        :class:`BatchResult`.

        The summary's ``operation`` is set to the final stage (the
        workflow's culminating step).
        """
        projects, missing = self._select_projects(project_names)
        final_operation = stages[-1] if stages else BatchOperation.RUN
        results: list[BatchResult] = [
            self._missing_result(name, final_operation) for name in missing
        ]
        for project in projects:
            aborted = False
            for stage in stages:
                if aborted:
                    break
                command = ["myagent", stage.value]
                result = self._exec(project, stage, command)
                results.append(result)
                if not result.success:
                    aborted = True
        return self._summarize(final_operation, results)

    def format_summary(self, summary: BatchSummary) -> str:
        """Render a summary as a human-readable table.

        Columns: ``PROJECT``, ``STATUS`` (``OK``/``FAIL``), ``TIME``.
        """
        lines = [f"{'PROJECT':<24} {'STATUS':<8} {'TIME':>8}"]
        lines.append("-" * 44)
        for result in summary.results:
            status = "OK" if result.success else "FAIL"
            lines.append(
                f"{result.project_name:<24} {status:<8} "
                f"{result.elapsed_seconds:>7.2f}s"
            )
        lines.append("-" * 44)
        lines.append(
            f"Total: {summary.total}  Succeeded: {summary.succeeded}  "
            f"Failed: {summary.failed}  Time: {summary.elapsed_seconds:.2f}s"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_projects(
        self, project_names: list[str] | None
    ) -> tuple[list[ManagedProject], list[str]]:
        """Resolve ``project_names`` against the store.

        Returns ``(found_projects, missing_names)``. When ``project_names``
        is ``None`` all managed projects are returned.
        """
        if project_names is None:
            return self.store.list_all(), []
        found: list[ManagedProject] = []
        missing: list[str] = []
        for name in project_names:
            project = self.store.get(name)
            if project is None:
                missing.append(name)
            else:
                found.append(project)
        return found, missing

    def _missing_result(
        self, name: str, operation: BatchOperation
    ) -> BatchResult:
        """Build a failure result for a project that is not in the store."""
        return BatchResult(
            project_name=name,
            project_path="",
            operation=operation,
            success=False,
            error=f"project '{name}' not found in store",
        )

    def _exec(
        self,
        project: ManagedProject,
        operation: BatchOperation,
        command: list[str],
    ) -> BatchResult:
        """Run ``command`` in ``project``'s directory and capture the result."""
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=project.path,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - start
            return BatchResult(
                project_name=project.name,
                project_path=project.path,
                operation=operation,
                success=False,
                exit_code=-1,
                stdout=_stream_to_str(exc.stdout),
                stderr=_stream_to_str(exc.stderr),
                elapsed_seconds=elapsed,
                error=f"command timed out after {exc.timeout}s",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            elapsed = time.monotonic() - start
            return BatchResult(
                project_name=project.name,
                project_path=project.path,
                operation=operation,
                success=False,
                exit_code=-1,
                elapsed_seconds=elapsed,
                error=str(exc),
            )
        elapsed = time.monotonic() - start
        return BatchResult(
            project_name=project.name,
            project_path=project.path,
            operation=operation,
            success=completed.returncode == 0,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=elapsed,
        )

    def _summarize(
        self, operation: BatchOperation, results: list[BatchResult]
    ) -> BatchSummary:
        """Build a :class:`BatchSummary`, sorting results deterministically."""
        ordered = sorted(
            results, key=lambda r: (r.project_name, r.operation.value)
        )
        succeeded = sum(1 for r in ordered if r.success)
        failed = len(ordered) - succeeded
        total_elapsed = sum(r.elapsed_seconds for r in ordered)
        return BatchSummary(
            operation=operation,
            results=ordered,
            total=len(ordered),
            succeeded=succeeded,
            failed=failed,
            elapsed_seconds=total_elapsed,
        )
