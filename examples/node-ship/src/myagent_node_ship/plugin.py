"""MyAgent-CLI language pack for Node projects.

Hooks into the ``clean`` and ``verify`` lifecycle to report Node build
artifacts (``dist/``, ``build/``, ``*.tsbuildinfo``, ``coverage/``) and
suggest the conventional test command for the package manager in use
(``npm``/``pnpm``/``yarn``).

This pack is non-destructive: it only observes and reports artifacts, leaving
removal to the user or to a custom extension of this skeleton. The set of
artifacts seen during ``pre_clean`` is stashed on the instance keyed by the
command ``trace_id`` so ``post_clean`` can report what actually disappeared.
"""

from __future__ import annotations

from pathlib import Path

from myagent_sdk import CommandContext, Plugin, hook

_TAG = "node-ship"


class NodeShipPlugin(Plugin):
    """Reports Node build artifacts and suggests ``npm test``."""

    #: Artifacts this pack recognises. A trailing slash denotes a directory;
    #: any other value is treated as a glob relative to the project root.
    ARTIFACTS: tuple[str, ...] = ("dist/", "build/", "*.tsbuildinfo", "coverage/")

    #: Default test command (used when no lockfile is detected).
    TEST_COMMAND = "npm test"

    #: Conventional Node lint invocation.
    LINT_COMMAND = "npm run lint"

    def __init__(self) -> None:
        super().__init__()
        self._found: dict[str, list[str]] = {}

    @hook
    def pre_clean(self, context: CommandContext) -> None:
        """Log Node build artifacts present before ``myagent clean`` runs.

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
        """Report which Node build artifacts were removed during this clean run."""
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
        """Suggest the native Node test command before verification runs."""
        if not self._is_node_project(context.project_root):
            return
        test_cmd = self._detect_test_command(context.project_root)
        print(
            f"[{_TAG}] Node project detected. "
            f"Verify with: `{test_cmd}`. Lint with: `{self.LINT_COMMAND}`."
        )

    @staticmethod
    def _is_node_project(project_root: Path) -> bool:
        """Return True when ``package.json`` is present at the project root."""
        return (project_root / "package.json").is_file()

    @staticmethod
    def _detect_test_command(project_root: Path) -> str:
        """Pick the test command based on the lockfile present.

        Returns ``pnpm test`` when ``pnpm-lock.yaml`` exists, ``yarn test``
        when ``yarn.lock`` exists, and ``npm test`` otherwise.
        """
        if (project_root / "pnpm-lock.yaml").is_file():
            return "pnpm test"
        if (project_root / "yarn.lock").is_file():
            return "yarn test"
        return NodeShipPlugin.TEST_COMMAND

    def _find_artifacts(self, project_root: Path) -> list[str]:
        """Return the recognised Node artifacts currently present."""
        found: list[str] = []
        for pattern in self.ARTIFACTS:
            if pattern.endswith("/"):
                if (project_root / pattern[:-1]).is_dir():
                    found.append(pattern)
            elif any(project_root.glob(pattern)):
                found.append(pattern)
        return found


plugin = NodeShipPlugin()


def register() -> NodeShipPlugin:
    """Entry-point factory used by the ``myagent.plugins`` group."""
    return NodeShipPlugin()
