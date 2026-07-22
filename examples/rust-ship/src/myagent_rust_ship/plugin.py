"""MyAgent-CLI language pack for Rust projects.

Hooks into the ``clean`` and ``verify`` lifecycle to report Rust build
artifacts and suggest the conventional ``cargo test`` command.

Artifact scope note
-------------------

This pack recognises ``target/debug`` and ``target/release`` -- the per-profile
build outputs -- but **never** matches ``target/`` wholesale. ``target/`` also
holds cargo's shared dependency cache and incremental compilation state; nuking
it all is what ``cargo clean`` is for, and is out of scope for a ``clean``-time
hook.

The pack is non-destructive: it only observes and reports. The set of artifacts
seen during ``pre_clean`` is stashed on the instance keyed by the command
``trace_id`` so ``post_clean`` can report what actually disappeared.
"""

from __future__ import annotations

from pathlib import Path

from myagent_sdk import CommandContext, Plugin, hook

_TAG = "rust-ship"


class RustShipPlugin(Plugin):
    """Reports Rust build artifacts and suggests ``cargo test``."""

    #: Per-profile build outputs only. ``target/`` itself is not matched, to
    #: avoid treating cargo's shared cache as a cleanable artifact.
    ARTIFACTS: tuple[str, ...] = ("target/debug/", "target/release/")

    #: Conventional Rust test command.
    TEST_COMMAND = "cargo test"

    #: Conventional Rust linter invocation.
    LINT_COMMAND = "cargo clippy"

    def __init__(self) -> None:
        super().__init__()
        self._found: dict[str, list[str]] = {}

    @hook
    def pre_clean(self, context: CommandContext) -> None:
        """Log Rust build artifacts present before ``myagent clean`` runs.

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
        """Report which Rust build artifacts were removed during this clean run."""
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
        """Suggest the native Rust test command before verification runs."""
        if not self._is_rust_project(context.project_root):
            return
        test_cmd = self._detect_test_command(context.project_root)
        print(
            f"[{_TAG}] Rust project detected. "
            f"Verify with: `{test_cmd}`. Lint with: `{self.LINT_COMMAND}`."
        )

    @staticmethod
    def _is_rust_project(project_root: Path) -> bool:
        """Return True when ``Cargo.toml`` is present at the project root."""
        return (project_root / "Cargo.toml").is_file()

    @staticmethod
    def _detect_test_command(project_root: Path) -> str:
        """Return the conventional Rust test command."""
        return RustShipPlugin.TEST_COMMAND

    def _find_artifacts(self, project_root: Path) -> list[str]:
        """Return the recognised Rust artifacts currently present.

        Only the per-profile output directories are checked; ``target/`` as a
        whole is skipped (see module docstring).
        """
        found: list[str] = []
        for pattern in self.ARTIFACTS:
            if pattern.endswith("/"):
                if (project_root / pattern[:-1]).is_dir():
                    found.append(pattern)
            elif any(project_root.glob(pattern)):
                found.append(pattern)
        return found


def register() -> RustShipPlugin:
    """Entry-point factory used by the ``myagent.plugins`` group."""
    return RustShipPlugin()
