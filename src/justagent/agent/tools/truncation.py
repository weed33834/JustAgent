"""Output truncation service.

Ports OpenCode's ``Truncate`` service (``tool/truncate.ts``) to Python.

When a tool's output exceeds the configured size, the full text is
written to a temp file and the returned content becomes a head/tail
preview plus a hint that mentions the saved path. This keeps tool
results inside the LLM context window while preserving the full output
for the user to inspect.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("justagent.agent.tools.truncation")

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TruncationResult:
    """Outcome of :meth:`TruncationService.truncate`."""

    content: str
    """The (possibly truncated) content to send to the LLM."""

    truncated: bool
    """``True`` if the original content exceeded limits."""

    output_path: str | None = None
    """Path to the full output file, set when ``truncated`` is True."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50_000
DEFAULT_HEAD_LINES = 200
DEFAULT_TAIL_LINES = 200
DEFAULT_RETENTION_DAYS = 7


@dataclass
class TruncationService:
    """Truncate tool output that exceeds size limits.

    Defaults match OpenCode: 2000 lines or 50 KB. The full output is
    written to ``output_dir`` (a temp directory by default) and the
    returned content is a head + tail preview with a hint pointing to
    the saved file.

    Example:

    >>> service = TruncationService()
    >>> result = service.truncate("line\\n" * 5000)
    >>> result.truncated
    True
    >>> result.output_path is not None
    True
    """

    max_lines: int = DEFAULT_MAX_LINES
    max_bytes: int = DEFAULT_MAX_BYTES
    head_lines: int = DEFAULT_HEAD_LINES
    tail_lines: int = DEFAULT_TAIL_LINES
    output_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "justagent-tool-output"
    )

    def truncate(
        self,
        content: str,
        *,
        tool_id: str = "",
        call_id: str | None = None,
    ) -> TruncationResult:
        """Truncate ``content`` if it exceeds limits.

        ``tool_id`` and ``call_id`` are used to name the saved file
        for debugging. If neither is provided, a UUID is used.
        """

        if not content:
            return TruncationResult(content="", truncated=False)

        byte_len = len(content.encode("utf-8"))
        line_count = content.count("\n") + (0 if content.endswith("\n") else 1)

        if byte_len <= self.max_bytes and line_count <= self.max_lines:
            return TruncationResult(content=content, truncated=False)

        # Save full output to disk.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = call_id or uuid.uuid4().hex[:8]
        prefix = f"{tool_id}_" if tool_id else ""
        # Use mkstemp for race-free file creation.
        fd, path_str = tempfile.mkstemp(
            prefix=prefix,
            suffix=f"_{suffix}.txt",
            dir=str(self.output_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            # If we can't save the full output, just hard-truncate.
            logger.debug("Failed to save full tool output to disk", exc_info=True)
            return TruncationResult(
                content=self._head_tail(content),
                truncated=True,
                output_path=None,
            )

        return TruncationResult(
            content=self._head_tail(content, hint_path=path_str),
            truncated=True,
            output_path=path_str,
        )

    def _head_tail(self, content: str, *, hint_path: str | None = None) -> str:
        """Build a head + tail preview with an optional hint."""

        lines = content.splitlines(keepends=True)
        if len(lines) <= self.head_lines + self.tail_lines:
            # Content fits without truncation between head and tail.
            preview = "".join(lines)
        else:
            head = "".join(lines[: self.head_lines])
            tail = "".join(lines[-self.tail_lines :])
            omitted = len(lines) - self.head_lines - self.tail_lines
            preview = (
                head
                + f"\n... [{omitted} lines omitted] ...\n\n"
                + tail
            )

        if hint_path:
            preview += (
                f"\n[Output truncated. Full output saved to: {hint_path}]"
            )
        return preview

    def cleanup_old(self, *, max_age_days: int = DEFAULT_RETENTION_DAYS) -> int:
        """Remove files in ``output_dir`` older than ``max_age_days``.

        Returns the number of files removed. No-op if the directory
        doesn't exist.
        """

        if not self.output_dir.exists():
            return 0
        cutoff = time_now() - max_age_days * 86400
        removed = 0
        for entry in self.output_dir.iterdir():
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    entry.unlink()
                    removed += 1
            except OSError:
                continue
        return removed


def time_now() -> float:
    """Indirection for testing. Returns current Unix timestamp."""

    import time

    return time.time()


__all__ = [
    "DEFAULT_HEAD_LINES",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_TAIL_LINES",
    "TruncationResult",
    "TruncationService",
]
