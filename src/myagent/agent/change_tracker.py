"""Track file changes during an agent run for summary display.

The runtime records every file write/edit/patch and provides a
summary at the end: which files were created, modified, or deleted,
with line-count deltas.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FileChange:
    """A single recorded file change.

    ``action`` is one of ``"created"``, ``"modified"``, or ``"deleted"``.
    ``lines_added`` / ``lines_removed`` are computed by diffing the old
    and new content line-by-line. For deletions both are 0 (the file is
    gone). ``tool`` is the name of the tool that caused the change
    (e.g. ``"write_to_file"``, ``"replace_in_file"``, ``"apply_patch"``).
    """

    path: str
    action: str
    lines_added: int
    lines_removed: int
    tool: str
    timestamp: float


def _compute_line_delta(old: str, new: str) -> tuple[int, int]:
    """Return ``(lines_added, lines_removed)`` by diffing ``old`` → ``new``.

    Uses :class:`difflib.SequenceMatcher` on line lists so replacements
    are counted as both a removal and an addition.
    """

    old_lines = old.splitlines()
    new_lines = new.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    added = 0
    removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed


class ChangeTracker:
    """Track file changes during an agent run.

    Each call to :meth:`record_write`, :meth:`record_edit`, or
    :meth:`record_delete` appends a :class:`FileChange` entry. Multiple
    changes to the same file are listed separately (not aggregated) so
    the full history is preserved. Use :meth:`get_summary` or
    :meth:`format_summary` for an aggregated view.
    """

    def __init__(self) -> None:
        self._changes: list[FileChange] = []

    def record_write(
        self,
        path: str,
        old_content: str | None,
        new_content: str,
        tool: str,
    ) -> None:
        """Record a file write.

        If ``old_content`` is ``None`` the file is treated as newly
        created; otherwise it is a modification. Line deltas are
        computed by comparing old and new content.
        """

        if old_content is None:
            added, removed = _compute_line_delta("", new_content)
            action = "created"
        else:
            added, removed = _compute_line_delta(old_content, new_content)
            action = "modified"
        self._changes.append(
            FileChange(
                path=path,
                action=action,
                lines_added=added,
                lines_removed=removed,
                tool=tool,
                timestamp=time.time(),
            )
        )

    def record_edit(
        self,
        path: str,
        old_content: str,
        new_content: str,
        tool: str,
    ) -> None:
        """Record a file edit (same logic as :meth:`record_write` for an
        existing file)."""

        added, removed = _compute_line_delta(old_content, new_content)
        self._changes.append(
            FileChange(
                path=path,
                action="modified",
                lines_added=added,
                lines_removed=removed,
                tool=tool,
                timestamp=time.time(),
            )
        )

    def record_delete(self, path: str, tool: str) -> None:
        """Record a file deletion."""

        self._changes.append(
            FileChange(
                path=path,
                action="deleted",
                lines_added=0,
                lines_removed=0,
                tool=tool,
                timestamp=time.time(),
            )
        )

    def get_changes(self) -> list[FileChange]:
        """Return all recorded changes in insertion order."""

        return list(self._changes)

    def get_changed_files(self) -> list[str]:
        """Return unique file paths that were changed, in first-seen order."""

        seen: set[str] = set()
        result: list[str] = []
        for change in self._changes:
            if change.path not in seen:
                seen.add(change.path)
                result.append(change.path)
        return result

    def get_summary(self) -> dict[str, Any]:
        """Return an aggregated summary dict.

        Keys: ``total_files`` (unique), ``created``, ``modified``,
        ``deleted``, ``total_lines_added``, ``total_lines_removed``.
        """

        unique_paths = {change.path for change in self._changes}
        created = sum(1 for c in self._changes if c.action == "created")
        modified = sum(1 for c in self._changes if c.action == "modified")
        deleted = sum(1 for c in self._changes if c.action == "deleted")
        total_added = sum(c.lines_added for c in self._changes)
        total_removed = sum(c.lines_removed for c in self._changes)
        return {
            "total_files": len(unique_paths),
            "created": created,
            "modified": modified,
            "deleted": deleted,
            "total_lines_added": total_added,
            "total_lines_removed": total_removed,
        }

    def format_summary(self) -> str:
        """Return a human-readable one-line summary string."""

        summary = self.get_summary()
        return (
            f"{summary['total_files']} file(s) changed: "
            f"{summary['created']} created, "
            f"{summary['modified']} modified, "
            f"{summary['deleted']} deleted "
            f"(+{summary['total_lines_added']} "
            f"-{summary['total_lines_removed']} lines)"
        )

    def clear(self) -> None:
        """Clear all tracked changes."""

        self._changes.clear()


__all__ = ["ChangeTracker", "FileChange"]
