"""Tests for :mod:`autoship.agent.change_tracker`."""

from __future__ import annotations

import time

import pytest

from autoship.agent.change_tracker import (
    ChangeTracker,
    FileChange,
    _compute_line_delta,
)

# ---------------------------------------------------------------------------
# _compute_line_delta
# ---------------------------------------------------------------------------


class TestComputeLineDelta:
    def test_identical_content(self) -> None:
        added, removed = _compute_line_delta("a\nb\n", "a\nb\n")
        assert added == 0
        assert removed == 0

    def test_pure_addition(self) -> None:
        added, removed = _compute_line_delta("a\n", "a\nb\nc\n")
        assert added == 2
        assert removed == 0

    def test_pure_removal(self) -> None:
        added, removed = _compute_line_delta("a\nb\nc\n", "a\n")
        assert added == 0
        assert removed == 2

    def test_replacement_counts_both(self) -> None:
        added, removed = _compute_line_delta("old\n", "new\n")
        assert added == 1
        assert removed == 1

    def test_empty_old(self) -> None:
        added, removed = _compute_line_delta("", "x\ny\n")
        assert added == 2
        assert removed == 0

    def test_empty_new(self) -> None:
        added, removed = _compute_line_delta("x\ny\n", "")
        assert added == 0
        assert removed == 2

    def test_no_trailing_newline(self) -> None:
        # splitlines() handles missing trailing newlines gracefully.
        added, removed = _compute_line_delta("a", "a\nb")
        assert added == 1
        assert removed == 0


# ---------------------------------------------------------------------------
# ChangeTracker.record_write
# ---------------------------------------------------------------------------


class TestRecordWrite:
    def test_record_write_new_file(self) -> None:
        tracker = ChangeTracker()
        tracker.record_write("a.txt", None, "hello\nworld\n", "write_to_file")
        changes = tracker.get_changes()
        assert len(changes) == 1
        ch = changes[0]
        assert ch.path == "a.txt"
        assert ch.action == "created"
        assert ch.tool == "write_to_file"
        assert ch.lines_added == 2
        assert ch.lines_removed == 0
        assert ch.timestamp > 0

    def test_record_write_existing_file(self) -> None:
        tracker = ChangeTracker()
        tracker.record_write(
            "a.txt", "old\n", "new\nextra\n", "write_to_file"
        )
        changes = tracker.get_changes()
        assert len(changes) == 1
        ch = changes[0]
        assert ch.action == "modified"
        assert ch.lines_added == 2
        assert ch.lines_removed == 1

    def test_record_write_empty_new_content(self) -> None:
        tracker = ChangeTracker()
        tracker.record_write("a.txt", "old\n", "", "write_to_file")
        ch = tracker.get_changes()[0]
        assert ch.action == "modified"
        assert ch.lines_added == 0
        assert ch.lines_removed == 1


# ---------------------------------------------------------------------------
# ChangeTracker.record_edit
# ---------------------------------------------------------------------------


class TestRecordEdit:
    def test_record_edit_marks_modified(self) -> None:
        tracker = ChangeTracker()
        tracker.record_edit("f.py", "a\nb\n", "a\nc\n", "replace_in_file")
        ch = tracker.get_changes()[0]
        assert ch.action == "modified"
        assert ch.lines_added == 1
        assert ch.lines_removed == 1
        assert ch.tool == "replace_in_file"

    def test_record_edit_with_no_changes(self) -> None:
        tracker = ChangeTracker()
        tracker.record_edit("f.py", "same\n", "same\n", "replace_in_file")
        ch = tracker.get_changes()[0]
        assert ch.action == "modified"
        assert ch.lines_added == 0
        assert ch.lines_removed == 0


# ---------------------------------------------------------------------------
# ChangeTracker.record_delete
# ---------------------------------------------------------------------------


class TestRecordDelete:
    def test_record_delete(self) -> None:
        tracker = ChangeTracker()
        tracker.record_delete("gone.txt", "apply_patch")
        ch = tracker.get_changes()[0]
        assert ch.action == "deleted"
        assert ch.lines_added == 0
        assert ch.lines_removed == 0
        assert ch.tool == "apply_patch"


# ---------------------------------------------------------------------------
# get_changes / get_changed_files
# ---------------------------------------------------------------------------


class TestGetChanges:
    def test_get_changes_returns_copy(self) -> None:
        tracker = ChangeTracker()
        tracker.record_write("a.txt", None, "x\n", "write_to_file")
        changes = tracker.get_changes()
        changes.clear()
        # Underlying list should be unaffected.
        assert len(tracker.get_changes()) == 1

    def test_get_changes_preserves_insertion_order(self) -> None:
        tracker = ChangeTracker()
        tracker.record_write("a.txt", None, "x\n", "write_to_file")
        tracker.record_edit("b.py", "old\n", "new\n", "replace_in_file")
        tracker.record_delete("c.txt", "apply_patch")
        changes = tracker.get_changes()
        assert [c.path for c in changes] == ["a.txt", "b.py", "c.txt"]
        assert [c.action for c in changes] == ["created", "modified", "deleted"]

    def test_get_changed_files_unique_in_first_seen_order(self) -> None:
        tracker = ChangeTracker()
        tracker.record_write("a.txt", None, "x\n", "write_to_file")
        tracker.record_write("b.py", None, "y\n", "write_to_file")
        tracker.record_edit("a.txt", "x\n", "z\n", "replace_in_file")
        files = tracker.get_changed_files()
        assert files == ["a.txt", "b.py"]

    def test_get_changed_files_empty(self) -> None:
        tracker = ChangeTracker()
        assert tracker.get_changed_files() == []


# ---------------------------------------------------------------------------
# get_summary / format_summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_get_summary_counts(self) -> None:
        tracker = ChangeTracker()
        tracker.record_write("a.txt", None, "x\n", "write_to_file")  # created
        tracker.record_write("b.py", "old\n", "new\n", "write_to_file")  # modified
        tracker.record_edit("b.py", "new\n", "newer\n", "replace_in_file")  # modified
        tracker.record_delete("c.txt", "apply_patch")  # deleted
        summary = tracker.get_summary()
        assert summary["total_files"] == 3  # unique paths
        assert summary["created"] == 1
        assert summary["modified"] == 2
        assert summary["deleted"] == 1
        # created a.txt (+1), modified b.py (+1 -1), modified b.py (+1 -1)
        assert summary["total_lines_added"] == 3
        assert summary["total_lines_removed"] == 2

    def test_get_summary_empty(self) -> None:
        tracker = ChangeTracker()
        summary = tracker.get_summary()
        assert summary == {
            "total_files": 0,
            "created": 0,
            "modified": 0,
            "deleted": 0,
            "total_lines_added": 0,
            "total_lines_removed": 0,
        }

    def test_format_summary_string(self) -> None:
        tracker = ChangeTracker()
        tracker.record_write("a.txt", None, "x\ny\n", "write_to_file")
        tracker.record_delete("b.txt", "apply_patch")
        s = tracker.format_summary()
        assert "2 file(s) changed" in s
        assert "1 created" in s
        assert "1 deleted" in s
        assert "+2" in s
        assert "-0" in s

    def test_format_summary_empty(self) -> None:
        tracker = ChangeTracker()
        s = tracker.format_summary()
        assert "0 file(s) changed" in s


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_changes(self) -> None:
        tracker = ChangeTracker()
        tracker.record_write("a.txt", None, "x\n", "write_to_file")
        tracker.record_delete("b.txt", "apply_patch")
        assert len(tracker.get_changes()) == 2
        tracker.clear()
        assert tracker.get_changes() == []
        assert tracker.get_changed_files() == []
        assert tracker.get_summary()["total_files"] == 0

    def test_clear_on_empty_tracker(self) -> None:
        tracker = ChangeTracker()
        tracker.clear()  # should not raise
        assert tracker.get_changes() == []


# ---------------------------------------------------------------------------
# Multiple changes to same file
# ---------------------------------------------------------------------------


class TestMultipleChangesSameFile:
    def test_multiple_changes_listed_separately(self) -> None:
        """Changes to the same file are NOT aggregated — each is a
        separate :class:`FileChange` entry (documented behaviour)."""

        tracker = ChangeTracker()
        tracker.record_write("f.py", None, "a\n", "write_to_file")
        tracker.record_edit("f.py", "a\n", "b\n", "replace_in_file")
        tracker.record_edit("f.py", "b\n", "c\n", "replace_in_file")
        changes = tracker.get_changes()
        assert len(changes) == 3
        assert all(c.path == "f.py" for c in changes)
        assert changes[0].action == "created"
        assert changes[1].action == "modified"
        assert changes[2].action == "modified"
        # But get_changed_files returns the path only once.
        assert tracker.get_changed_files() == ["f.py"]


# ---------------------------------------------------------------------------
# FileChange is frozen
# ---------------------------------------------------------------------------


class TestFileChangeFrozen:
    def test_file_change_is_frozen(self) -> None:
        tracker = ChangeTracker()
        tracker.record_write("a.txt", None, "x\n", "write_to_file")
        ch = tracker.get_changes()[0]
        # FrozenInstanceError is a subclass of AttributeError.
        with pytest.raises(AttributeError):
            ch.path = "other"  # type: ignore[misc]

    def test_file_change_fields(self) -> None:
        before = time.time()
        ch = FileChange(
            path="p",
            action="created",
            lines_added=1,
            lines_removed=0,
            tool="t",
            timestamp=before,
        )
        after = time.time()
        assert ch.path == "p"
        assert ch.action == "created"
        assert ch.lines_added == 1
        assert ch.lines_removed == 0
        assert ch.tool == "t"
        assert before <= ch.timestamp <= after
