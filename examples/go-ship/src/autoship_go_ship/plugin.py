"""AutoShip-CLI language pack for Go projects.

Hooks into the ``clean`` and ``verify`` lifecycle to report Go build
artifacts (``bin/``, ``*.test``, ``*.out``) and suggest the conventional
``go test ./...`` command.

This pack is non-destructive: it only observes and reports artifacts, leaving
removal to the user or to a custom extension of this skeleton. The set of
artifacts seen during ``pre_clean`` is stashed on the instance keyed by the
command ``trace_id`` so ``post_clean`` can report what actually disappeared.
"""

from __future__ import annotations

from pathlib import Path

from autoship_sdk import CommandContext, Plugin, hook

_TAG = "go-ship"


class GoShipPlugin(Plugin):
    """Reports Go build artifacts and suggests ``go test ./...``."""

    #: Artifacts this pack recognises. A trailing slash denotes a directory;
    #: any other value is treated as a glob relative to the project root.
    ARTIFACTS: tuple[str, ...] = ("bin/", "*.test", "*.out")

    #: Conventional Go test command.
    TEST_COMMAND = "go test ./..."

    #: Conventional Go linter invocation.
    LINT_COMMAND = "golangci-lint run"

    def __init__(self) -> None:
        super().__init__()
        self._found: dict[str, list[str]] = {}

    @hook
    def pre_clean(self, context: CommandContext) -> None:
        """Log Go build artifacts present before ``autoship clean`` runs.

        Records the detected artifacts keyed by ``context.trace_id`` so
        :meth:`post_clean` can report which ones were removed.
        """
        found = self._find_artifacts(context.project_root)
        self._found[context.trace_id] = found
        if not found:
            return
        print(f"[{_TAG}] build artifacts before clean: {', '.join(found)} ({len(found)})")

    @hook
    def post_clean(self, context: CommandContext) -> None:
        """Report which Go build artifacts were removed during this clean run."""
        found = self._found.pop(context.trace_id, [])
        if not found:
            return
        still_present = self._find_artifacts(context.project_root)
        removed = [item for item in found if item not in still_present]
        if removed:
            print(f"[{_TAG}] removed during clean: {', '.join(removed)} ({len(removed)})")
        else:
            print(
                f"[{_TAG}] {len(found)} artifact(s) still present after clean "
                "(built-in formatters do not remove build outputs)"
            )

    @hook
    def pre_verify(self, context: CommandContext) -> None:
        """Suggest the native Go test command before verification runs."""
        if not self._is_go_project(context.project_root):
            return
        test_cmd = self._detect_test_command(context.project_root)
        print(
            f"[{_TAG}] Go project detected. "
            f"Verify with: `{test_cmd}`. Lint with: `{self.LINT_COMMAND}`."
        )

    @staticmethod
    def _is_go_project(project_root: Path) -> bool:
        """Return True when ``go.mod`` is present at the project root."""
        return (project_root / "go.mod").is_file()

    @staticmethod
    def _detect_test_command(project_root: Path) -> str:
        """Return the conventional Go test command."""
        return GoShipPlugin.TEST_COMMAND

    def _find_artifacts(self, project_root: Path) -> list[str]:
        """Return the recognised Go artifacts currently present."""
        found: list[str] = []
        for pattern in self.ARTIFACTS:
            if pattern.endswith("/"):
                if (project_root / pattern[:-1]).is_dir():
                    found.append(pattern)
            elif any(project_root.glob(pattern)):
                found.append(pattern)
        return found


def register() -> GoShipPlugin:
    """Entry-point factory used by the ``autoship.plugins`` group."""
    return GoShipPlugin()
