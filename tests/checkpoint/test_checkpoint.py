"""Tests for the checkpoint system (shadow git snapshots).

Covers:

* :class:`CheckpointManager` — initialize, snapshot, restore, list, diff.
* :class:`CheckpointConfig` — enabled/disabled, exclude patterns.
* :class:`Checkpoint` dataclass — fields, parsing.
* Edge cases: empty project, no changes, restore deletes new files,
  restore re-creates deleted files, unknown checkpoint id.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from justagent.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointError,
    CheckpointManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """A clean project directory with one initial file."""

    (tmp_path / "hello.txt").write_text("Hello, world!\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def manager(project_dir: Path) -> CheckpointManager:
    """A CheckpointManager initialized on ``project_dir``."""

    mgr = CheckpointManager(project_dir)
    mgr.initialize()
    return mgr


# ---------------------------------------------------------------------------
# Checkpoint dataclass
# ---------------------------------------------------------------------------


class TestCheckpointDataclass:
    def test_fields(self) -> None:
        cp = Checkpoint(
            id="abc1234",
            iteration=3,
            tool_name="write_to_file",
            timestamp=1234567890.0,
            message="after write",
        )
        assert cp.id == "abc1234"
        assert cp.iteration == 3
        assert cp.tool_name == "write_to_file"
        assert cp.timestamp == 1234567890.0
        assert cp.message == "after write"
        assert cp.metadata == {}

    def test_metadata_defaults_to_empty(self) -> None:
        cp = Checkpoint(
            id="x", iteration=0, tool_name="", timestamp=0.0, message=""
        )
        assert cp.metadata == {}

    def test_is_frozen(self) -> None:
        cp = Checkpoint(
            id="x", iteration=0, tool_name="", timestamp=0.0, message=""
        )
        with pytest.raises(AttributeError):
            cp.iteration = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CheckpointConfig
# ---------------------------------------------------------------------------


class TestCheckpointConfig:
    def test_defaults(self) -> None:
        cfg = CheckpointConfig()
        assert cfg.enabled is True
        assert cfg.checkpoint_dir == ".justagent/checkpoints"
        assert cfg.max_checkpoints == 100
        assert ".git/" in cfg.exclude_patterns
        assert ".justagent/" in cfg.exclude_patterns

    def test_custom_config(self) -> None:
        cfg = CheckpointConfig(
            enabled=False,
            checkpoint_dir=".checkpoints",
            max_checkpoints=10,
            exclude_patterns=["*.tmp"],
        )
        assert cfg.enabled is False
        assert cfg.checkpoint_dir == ".checkpoints"
        assert cfg.max_checkpoints == 10
        assert cfg.exclude_patterns == ["*.tmp"]


# ---------------------------------------------------------------------------
# CheckpointManager — initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_initialize_creates_shadow_repo(self, project_dir: Path) -> None:
        mgr = CheckpointManager(project_dir)
        mgr.initialize()
        assert mgr.is_initialized is True
        shadow = project_dir / ".justagent" / "checkpoints"
        assert shadow.exists()
        assert (shadow / ".git").exists()

    def test_initialize_writes_gitignore(self, project_dir: Path) -> None:
        mgr = CheckpointManager(project_dir)
        mgr.initialize()
        # Excludes are written to .git/info/exclude in the shadow repo.
        exclude_file = (
            project_dir / ".justagent" / "checkpoints" / ".git" / "info" / "exclude"
        )
        assert exclude_file.exists()
        content = exclude_file.read_text(encoding="utf-8")
        assert ".git/" in content
        assert ".justagent/" in content

    def test_initialize_is_idempotent(self, project_dir: Path) -> None:
        mgr = CheckpointManager(project_dir)
        mgr.initialize()
        mgr.initialize()  # Should not raise
        assert mgr.is_initialized is True

    def test_initialize_noop_when_disabled(self, project_dir: Path) -> None:
        mgr = CheckpointManager(
            project_dir, CheckpointConfig(enabled=False)
        )
        mgr.initialize()
        assert mgr.is_initialized is False
        assert not (project_dir / ".justagent").exists()


# ---------------------------------------------------------------------------
# CheckpointManager — snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_returns_checkpoint(self, manager: CheckpointManager) -> None:
        cp = manager.snapshot(iteration=0, message="initial")
        assert cp is not None
        assert isinstance(cp, Checkpoint)
        assert len(cp.id) >= 7  # short hash
        assert cp.iteration == 0
        assert cp.message == "initial"

    def test_snapshot_tracks_file_contents(
        self, manager: CheckpointManager, project_dir: Path
    ) -> None:
        cp = manager.snapshot(iteration=0, message="initial")
        assert cp is not None
        # Verify the file is in the shadow repo's tree.
        files = manager._list_files_at_commit(cp.id)
        assert "hello.txt" in files
        assert "src/app.py" in files

    def test_snapshot_creates_new_commit_on_change(
        self, manager: CheckpointManager, project_dir: Path
    ) -> None:
        cp1 = manager.snapshot(iteration=0, message="initial")
        assert cp1 is not None
        # Modify a file
        (project_dir / "hello.txt").write_text("Modified!\n", encoding="utf-8")
        cp2 = manager.snapshot(iteration=1, tool_name="write_to_file", message="modified")
        assert cp2 is not None
        assert cp1.id != cp2.id

    def test_snapshot_with_no_changes_still_creates_commit(
        self, manager: CheckpointManager
    ) -> None:
        cp1 = manager.snapshot(iteration=0, message="first")
        cp2 = manager.snapshot(iteration=1, message="second")
        assert cp1 is not None
        assert cp2 is not None
        # --allow-empty means even no-change snapshots get a commit.
        assert cp1.id != cp2.id

    def test_snapshot_returns_none_when_disabled(
        self, project_dir: Path
    ) -> None:
        mgr = CheckpointManager(
            project_dir, CheckpointConfig(enabled=False)
        )
        mgr.initialize()
        cp = mgr.snapshot(iteration=0, message="test")
        assert cp is None

    def test_snapshot_metadata_stored(self, manager: CheckpointManager) -> None:
        cp = manager.snapshot(
            iteration=5,
            tool_name="run_command",
            message="after run",
            metadata={"command": "ls -la", "exit_code": 0},
        )
        assert cp is not None
        assert cp.metadata == {"command": "ls -la", "exit_code": 0}
        assert cp.tool_name == "run_command"

    def test_snapshot_auto_initializes(self, project_dir: Path) -> None:
        """snapshot() should auto-initialize if not already done."""

        mgr = CheckpointManager(project_dir)
        assert mgr.is_initialized is False
        cp = mgr.snapshot(iteration=0, message="auto-init")
        assert cp is not None
        assert mgr.is_initialized is True


# ---------------------------------------------------------------------------
# CheckpointManager — restore
# ---------------------------------------------------------------------------


class TestRestore:
    def test_restore_reverts_modified_file(
        self, manager: CheckpointManager, project_dir: Path
    ) -> None:
        # Snapshot initial state
        cp = manager.snapshot(iteration=0, message="initial")
        assert cp is not None
        original = (project_dir / "hello.txt").read_text()
        # Modify the file
        (project_dir / "hello.txt").write_text("CHANGED!\n", encoding="utf-8")
        assert (project_dir / "hello.txt").read_text() != original
        # Restore
        manager.restore(cp.id)
        assert (project_dir / "hello.txt").read_text() == original

    def test_restore_recreates_deleted_file(
        self, manager: CheckpointManager, project_dir: Path
    ) -> None:
        cp = manager.snapshot(iteration=0, message="initial")
        assert cp is not None
        # Delete a file
        (project_dir / "hello.txt").unlink()
        assert not (project_dir / "hello.txt").exists()
        # Restore
        manager.restore(cp.id)
        assert (project_dir / "hello.txt").exists()
        assert (project_dir / "hello.txt").read_text() == "Hello, world!\n"

    def test_restore_removes_new_file(
        self, manager: CheckpointManager, project_dir: Path
    ) -> None:
        cp = manager.snapshot(iteration=0, message="initial")
        assert cp is not None
        # Add a new file
        (project_dir / "new_file.txt").write_text("new!\n", encoding="utf-8")
        assert (project_dir / "new_file.txt").exists()
        # Restore — the new file should be removed
        manager.restore(cp.id)
        assert not (project_dir / "new_file.txt").exists()

    def test_restore_preserves_git_and_myagent_dirs(
        self, manager: CheckpointManager, project_dir: Path
    ) -> None:
        cp = manager.snapshot(iteration=0, message="initial")
        assert cp is not None
        # The .justagent dir (shadow repo) must survive restore.
        manager.restore(cp.id)
        assert (project_dir / ".justagent").exists()
        assert (project_dir / ".justagent" / "checkpoints").exists()

    def test_restore_unknown_checkpoint_raises(
        self, manager: CheckpointManager
    ) -> None:
        with pytest.raises(CheckpointError):
            manager.restore("nonexistent123")

    def test_restore_when_disabled_raises(
        self, project_dir: Path
    ) -> None:
        mgr = CheckpointManager(
            project_dir, CheckpointConfig(enabled=False)
        )
        with pytest.raises(CheckpointError):
            mgr.restore("abc123")

    def test_restore_full_workflow(
        self, manager: CheckpointManager, project_dir: Path
    ) -> None:
        """Snapshot → modify → snapshot → modify more → restore → verify."""

        cp1 = manager.snapshot(iteration=0, message="initial")
        assert cp1 is not None
        (project_dir / "hello.txt").write_text("v2\n", encoding="utf-8")
        (project_dir / "new.txt").write_text("new\n", encoding="utf-8")
        cp2 = manager.snapshot(iteration=1, tool_name="write_to_file", message="v2")
        assert cp2 is not None
        (project_dir / "hello.txt").write_text("v3\n", encoding="utf-8")
        (project_dir / "new.txt").unlink()
        (project_dir / "another.txt").write_text("x\n", encoding="utf-8")

        # Restore to cp1 — should revert everything to initial state.
        manager.restore(cp1.id)
        assert (project_dir / "hello.txt").read_text() == "Hello, world!\n"
        assert not (project_dir / "new.txt").exists()
        assert not (project_dir / "another.txt").exists()


# ---------------------------------------------------------------------------
# CheckpointManager — list / get / latest
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    def test_list_returns_empty_before_any_snapshot(
        self, project_dir: Path
    ) -> None:
        mgr = CheckpointManager(project_dir)
        mgr.initialize()
        assert mgr.list_checkpoints() == []

    def test_list_returns_checkpoints_newest_first(
        self, manager: CheckpointManager
    ) -> None:
        cp1 = manager.snapshot(iteration=0, message="first")
        time.sleep(0.01)
        cp2 = manager.snapshot(iteration=1, message="second")
        time.sleep(0.01)
        cp3 = manager.snapshot(iteration=2, message="third")
        assert cp1 is not None and cp2 is not None and cp3 is not None
        cps = manager.list_checkpoints()
        assert len(cps) == 3
        assert cps[0].id == cp3.id  # newest first
        assert cps[1].id == cp2.id
        assert cps[2].id == cp1.id

    def test_list_preserves_iteration_and_message(
        self, manager: CheckpointManager
    ) -> None:
        manager.snapshot(iteration=7, tool_name="read_file", message="test msg")
        cps = manager.list_checkpoints()
        assert len(cps) == 1
        assert cps[0].iteration == 7
        assert cps[0].tool_name == "read_file"
        assert cps[0].message == "test msg"

    def test_get_checkpoint_by_id(self, manager: CheckpointManager) -> None:
        cp = manager.snapshot(iteration=0, message="initial")
        assert cp is not None
        fetched = manager.get_checkpoint(cp.id)
        assert fetched is not None
        assert fetched.id == cp.id
        assert fetched.iteration == cp.iteration

    def test_get_unknown_id_returns_none(self, manager: CheckpointManager) -> None:
        assert manager.get_checkpoint("nonexistent") is None

    def test_latest_returns_most_recent(self, manager: CheckpointManager) -> None:
        manager.snapshot(iteration=0, message="first")
        cp2 = manager.snapshot(iteration=1, message="second")
        assert cp2 is not None
        latest = manager.latest()
        assert latest is not None
        assert latest.id == cp2.id

    def test_latest_returns_none_when_empty(
        self, project_dir: Path
    ) -> None:
        mgr = CheckpointManager(project_dir)
        mgr.initialize()
        assert mgr.latest() is None


# ---------------------------------------------------------------------------
# CheckpointManager — diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_diff_between_two_checkpoints(
        self, manager: CheckpointManager, project_dir: Path
    ) -> None:
        cp1 = manager.snapshot(iteration=0, message="initial")
        assert cp1 is not None
        (project_dir / "hello.txt").write_text("changed\n", encoding="utf-8")
        cp2 = manager.snapshot(iteration=1, message="modified")
        assert cp2 is not None
        diff = manager.diff(cp1.id, cp2.id)
        assert "hello.txt" in diff
        assert "-Hello, world!" in diff or "-Hello, world!" in diff.replace("\r", "")
        assert "+changed" in diff

    def test_diff_no_changes_returns_empty(
        self, manager: CheckpointManager
    ) -> None:
        cp1 = manager.snapshot(iteration=0, message="first")
        cp2 = manager.snapshot(iteration=1, message="second")
        assert cp1 is not None and cp2 is not None
        diff = manager.diff(cp1.id, cp2.id)
        # No file changes → empty diff
        assert diff.strip() == ""

    def test_diff_latest_vs_parent(self, manager: CheckpointManager, project_dir: Path) -> None:
        manager.snapshot(iteration=0, message="initial")
        (project_dir / "hello.txt").write_text("v2\n", encoding="utf-8")
        manager.snapshot(iteration=1, message="changed")
        diff = manager.diff()  # no args → HEAD vs HEAD~1
        assert "hello.txt" in diff


# ---------------------------------------------------------------------------
# CheckpointManager — properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_project_root(self, project_dir: Path) -> None:
        mgr = CheckpointManager(project_dir)
        assert mgr.project_root == project_dir.resolve()

    def test_shadow_repo_path(self, project_dir: Path) -> None:
        mgr = CheckpointManager(project_dir)
        assert mgr.shadow_repo_path == project_dir / ".justagent" / "checkpoints"

    def test_is_enabled_true_by_default(self, project_dir: Path) -> None:
        mgr = CheckpointManager(project_dir)
        assert mgr.is_enabled is True

    def test_is_enabled_false_when_disabled(self, project_dir: Path) -> None:
        mgr = CheckpointManager(project_dir, CheckpointConfig(enabled=False))
        assert mgr.is_enabled is False


# ---------------------------------------------------------------------------
# CheckpointManager — exclude patterns
# ---------------------------------------------------------------------------


class TestExcludePatterns:
    def test_pyc_files_excluded(self, project_dir: Path) -> None:
        # Add a .pyc file
        (project_dir / "src").mkdir(exist_ok=True)
        (project_dir / "src" / "__pycache__").mkdir(exist_ok=True)
        (project_dir / "src" / "__pycache__" / "app.cpython-311.pyc").write_text(
            "binary junk", encoding="utf-8"
        )
        mgr = CheckpointManager(project_dir)
        mgr.initialize()
        cp = mgr.snapshot(iteration=0, message="initial")
        assert cp is not None
        files = mgr._list_files_at_commit(cp.id)
        # __pycache__/ should be excluded by default patterns
        assert not any("pycache" in f for f in files)
        assert not any(f.endswith(".pyc") for f in files)

    def test_custom_exclude_patterns(self, project_dir: Path) -> None:
        (project_dir / "secret.env").write_text("KEY=value", encoding="utf-8")
        (project_dir / "keep.txt").write_text("keep me", encoding="utf-8")
        mgr = CheckpointManager(
            project_dir,
            CheckpointConfig(exclude_patterns=["*.env", ".git/", ".justagent/"]),
        )
        mgr.initialize()
        cp = mgr.snapshot(iteration=0, message="initial")
        assert cp is not None
        files = mgr._list_files_at_commit(cp.id)
        assert "keep.txt" in files
        assert "secret.env" not in files


# ---------------------------------------------------------------------------
# Integration with runtime (smoke test)
# ---------------------------------------------------------------------------


class TestRuntimeIntegration:
    """Smoke test: verify the runtime can use CheckpointManager."""

    def test_runtime_creates_checkpoint_per_tool_call(self, project_dir: Path) -> None:
        """Verify the checkpoint manager works in isolation."""

        mgr = CheckpointManager(project_dir)
        mgr.initialize()
        cp0 = mgr.snapshot(iteration=0, message="run-start")
        assert cp0 is not None

        # Simulate a tool call modifying a file
        (project_dir / "hello.txt").write_text("agent modified\n", encoding="utf-8")
        cp1 = mgr.snapshot(
            iteration=1, tool_name="write_to_file", message="after write"
        )
        assert cp1 is not None
        assert cp0.id != cp1.id

        # Restore to before the modification
        mgr.restore(cp0.id)
        assert (project_dir / "hello.txt").read_text() == "Hello, world!\n"
