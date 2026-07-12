"""AutoShip-CLI language pack for Java projects.

Hooks into the ``clean`` and ``verify`` lifecycle to report Java build
artifacts (``target/classes``, ``target/test-classes``, loose ``*.class``
files) and suggest the conventional test command for the build tool in use
(``mvn test`` or ``gradle test``).

This pack is non-destructive: it only observes and reports artifacts, leaving
removal to the user or to a custom extension of this skeleton. The set of
artifacts seen during ``pre_clean`` is stashed on the instance keyed by the
command ``trace_id`` so ``post_clean`` can report what actually disappeared.
"""

from __future__ import annotations

from pathlib import Path

from autoship_sdk import CommandContext, Plugin, hook

_TAG = "java-ship"


class JavaShipPlugin(Plugin):
    """Reports Java build artifacts and suggests ``mvn test``."""

    #: Artifacts this pack recognises. A trailing slash denotes a directory;
    #: any other value is treated as a glob relative to the project root.
    #: ``target/classes`` and ``target/test-classes`` are scoped to avoid
    #: treating the whole ``target/`` tree (which also holds Maven metadata)
    #: as a single cleanable artifact.
    ARTIFACTS: tuple[str, ...] = ("target/classes/", "target/test-classes/", "*.class")

    #: Default test command (used for Maven projects).
    TEST_COMMAND = "mvn test"

    #: Conventional Java lint invocation (Maven checkstyle).
    LINT_COMMAND = "mvn checkstyle:check"

    def __init__(self) -> None:
        super().__init__()
        self._found: dict[str, list[str]] = {}

    @hook
    def pre_clean(self, context: CommandContext) -> None:
        """Log Java build artifacts present before ``autoship clean`` runs.

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
        """Report which Java build artifacts were removed during this clean run."""
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
        """Suggest the native Java test command before verification runs."""
        if not self._is_java_project(context.project_root):
            return
        test_cmd = self._detect_test_command(context.project_root)
        print(
            f"[{_TAG}] Java project detected. "
            f"Verify with: `{test_cmd}`. Lint with: `{self.LINT_COMMAND}`."
        )

    @staticmethod
    def _is_java_project(project_root: Path) -> bool:
        """Return True when a Maven or Gradle build file is present."""
        if (project_root / "pom.xml").is_file():
            return True
        return any(project_root.glob("build.gradle*"))

    @staticmethod
    def _detect_test_command(project_root: Path) -> str:
        """Pick the test command based on the build file present.

        Returns ``gradle test`` when a ``build.gradle`` or ``build.gradle.kts``
        file exists, and ``mvn test`` otherwise (Maven default).
        """
        if any(project_root.glob("build.gradle*")):
            return "gradle test"
        return JavaShipPlugin.TEST_COMMAND

    def _find_artifacts(self, project_root: Path) -> list[str]:
        """Return the recognised Java artifacts currently present."""
        found: list[str] = []
        for pattern in self.ARTIFACTS:
            if pattern.endswith("/"):
                if (project_root / pattern[:-1]).is_dir():
                    found.append(pattern)
            elif any(project_root.glob(pattern)):
                found.append(pattern)
        return found


plugin = JavaShipPlugin()


def register() -> JavaShipPlugin:
    """Entry-point factory used by the ``autoship.plugins`` group."""
    return JavaShipPlugin()
