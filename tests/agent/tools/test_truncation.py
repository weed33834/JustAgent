"""Tests for the TruncationService."""

from __future__ import annotations

import pytest

from autoship.agent.tools.truncation import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationService,
)


class TestTruncationService:
    def test_small_content_not_truncated(self) -> None:
        service = TruncationService()
        result = service.truncate("hello\n")
        assert not result.truncated
        assert result.content == "hello\n"
        assert result.output_path is None

    def test_empty_content_not_truncated(self) -> None:
        service = TruncationService()
        result = service.truncate("")
        assert not result.truncated
        assert result.content == ""

    def test_truncates_when_exceeds_max_lines(self) -> None:
        service = TruncationService(max_lines=10, head_lines=2, tail_lines=2)
        content = "\n".join(f"line {i}" for i in range(20)) + "\n"
        result = service.truncate(content, tool_id="test")
        assert result.truncated
        assert result.output_path is not None
        # Head + tail + omission notice + hint.
        assert "line 0" in result.content
        assert "line 1" in result.content
        assert "line 19" in result.content
        assert "line 18" in result.content
        assert "lines omitted" in result.content
        assert "Output truncated" in result.content

    def test_truncates_when_exceeds_max_bytes(self) -> None:
        service = TruncationService(max_bytes=100, head_lines=2, tail_lines=2)
        content = "x" * 200  # 200 bytes, single line
        result = service.truncate(content, tool_id="test")
        assert result.truncated
        assert result.output_path is not None

    def test_under_limit_not_truncated(self) -> None:
        service = TruncationService(max_lines=10, max_bytes=100)
        content = "\n".join(f"line {i}" for i in range(5))
        result = service.truncate(content)
        assert not result.truncated
        assert result.content == content

    def test_output_path_uses_tool_id_prefix(self) -> None:
        service = TruncationService(max_lines=5)
        content = "\n".join(f"line {i}" for i in range(20)) + "\n"
        result = service.truncate(content, tool_id="read_file", call_id="abc123")
        assert result.output_path is not None
        assert "read_file_" in result.output_path
        assert "abc123" in result.output_path

    def test_output_path_falls_back_to_uuid(self) -> None:
        service = TruncationService(max_lines=5)
        content = "\n".join(f"line {i}" for i in range(20)) + "\n"
        result = service.truncate(content)  # no tool_id, no call_id
        assert result.output_path is not None
        # Should still get a valid file path.

    def test_saved_file_contains_full_content(self) -> None:
        service = TruncationService(max_lines=5)
        content = "\n".join(f"line {i}" for i in range(20)) + "\n"
        result = service.truncate(content, tool_id="test")
        assert result.output_path is not None
        with open(result.output_path, encoding="utf-8") as f:
            saved = f.read()
        assert saved == content

    def test_head_tail_keeps_all_when_small_enough(self) -> None:
        # If head + tail >= total lines, just keep everything (no omission).
        service = TruncationService(
            max_lines=2, head_lines=10, tail_lines=10
        )
        content = "\n".join(f"line {i}" for i in range(5)) + "\n"
        result = service.truncate(content, tool_id="test")
        assert result.truncated
        # No "lines omitted" because head+tail > total.
        assert "lines omitted" not in result.content

    def test_custom_output_dir(self, tmp_path) -> None:
        service = TruncationService(
            max_lines=5, output_dir=tmp_path / "trunc"
        )
        content = "\n".join(f"line {i}" for i in range(20)) + "\n"
        result = service.truncate(content, tool_id="test")
        assert result.output_path is not None
        assert str(tmp_path / "trunc") in result.output_path

    def test_defaults_match_opencode(self) -> None:
        """Defaults should match OpenCode's Truncate service."""

        assert DEFAULT_MAX_LINES == 2000
        assert DEFAULT_MAX_BYTES == 50_000

    def test_cleanup_old_removes_aged_files(self, tmp_path) -> None:
        import os
        import time

        service = TruncationService(output_dir=tmp_path / "cleanup")
        service.output_dir.mkdir(parents=True, exist_ok=True)

        # Create an "old" file.
        old_file = service.output_dir / "old.txt"
        old_file.write_text("old content")
        old_time = time.time() - 8 * 86400  # 8 days ago
        os.utime(old_file, (old_time, old_time))

        # Create a "new" file.
        new_file = service.output_dir / "new.txt"
        new_file.write_text("new content")

        removed = service.cleanup_old(max_age_days=7)
        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_cleanup_old_no_dir_is_noop(self, tmp_path) -> None:
        service = TruncationService(output_dir=tmp_path / "nonexistent")
        # Should not raise.
        assert service.cleanup_old() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
