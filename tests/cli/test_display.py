"""Tests for :mod:`autoship.cli.display` (:class:`RichDisplay`)."""

from __future__ import annotations

import io

from rich.console import Console

from autoship.cli.display import RichDisplay


def _make_display(
    *,
    json_mode: bool = False,
    verbose: bool = False,
    force_terminal: bool = True,
) -> tuple[RichDisplay, io.StringIO]:
    """Build a :class:`RichDisplay` whose output is captured in a buffer.

    ``force_terminal=True`` makes Rich emit ANSI codes so we can assert
    on colour styles; set ``force_terminal=False`` for plain text.
    """

    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=force_terminal,
        width=120,
        record=True,
        highlight=False,
        soft_wrap=False,
    )
    display = RichDisplay(
        verbose=verbose,
        json_mode=json_mode,
        console=console,
    )
    return display, buf


def _output(buf: io.StringIO) -> str:
    """Read the captured output and reset the buffer."""

    val = buf.getvalue()
    buf.seek(0)
    buf.truncate(0)
    return val


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        display = RichDisplay()
        assert display.verbose is False
        assert display.json_mode is False

    def test_verbose_flag(self) -> None:
        display = RichDisplay(verbose=True)
        assert display.verbose is True

    def test_custom_console(self) -> None:
        display, _ = _make_display()
        # The custom console should be wired through.
        display.print_info("hello")
        # No assertion error means it works.

    def test_json_mode_flag(self) -> None:
        display = RichDisplay(json_mode=True)
        assert display.json_mode is True


# ---------------------------------------------------------------------------
# json_mode is no-op
# ---------------------------------------------------------------------------


class TestJsonModeNoop:
    def test_json_mode_print_assistant_message_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.print_assistant_message("hello")
        assert _output(buf) == ""

    def test_json_mode_print_tool_start_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.print_tool_start("read_file", {"path": "x"})
        assert _output(buf) == ""

    def test_json_mode_print_tool_result_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.print_tool_result("read_file", "out", False, 1.0)
        assert _output(buf) == ""

    def test_json_mode_print_warning_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.print_warning("careful")
        assert _output(buf) == ""

    def test_json_mode_print_error_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.print_error("boom")
        assert _output(buf) == ""

    def test_json_mode_print_info_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.print_info("note")
        assert _output(buf) == ""

    def test_json_mode_print_diff_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.print_diff("a\n", "b\n", "f.txt")
        assert _output(buf) == ""

    def test_json_mode_print_change_summary_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.print_change_summary(
            [{"path": "f", "action": "created", "lines_added": 1, "lines_removed": 0}]
        )
        assert _output(buf) == ""

    def test_json_mode_print_run_summary_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.print_run_summary(1, 10, 1.0, ["f"])
        assert _output(buf) == ""

    def test_json_mode_print_welcome_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.print_welcome("act", "gpt-4o", "/tmp")
        assert _output(buf) == ""

    def test_json_mode_start_spinner_no_output(self) -> None:
        display, buf = _make_display(json_mode=True)
        display.start_spinner("Thinking...")
        display.stop_spinner()
        assert _output(buf) == ""

    def test_json_mode_permission_prompt_returns_true(self) -> None:
        display, buf = _make_display(json_mode=True)
        assert display.print_permission_prompt("write_to_file", "edit f") is True
        assert _output(buf) == ""


# ---------------------------------------------------------------------------
# print_assistant_message
# ---------------------------------------------------------------------------


class TestPrintAssistantMessage:
    def test_prints_content(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_assistant_message("Hello, world!")
        out = _output(buf)
        assert "Hello, world!" in out

    def test_empty_content_no_output(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_assistant_message("")
        assert _output(buf) == ""

    def test_includes_assistant_title(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_assistant_message("hi")
        out = _output(buf)
        assert "Assistant" in out


# ---------------------------------------------------------------------------
# print_tool_start / print_tool_result
# ---------------------------------------------------------------------------


class TestPrintToolStart:
    def test_prints_tool_name(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_tool_start("read_file", {"path": "f.txt"})
        out = _output(buf)
        assert "read_file" in out

    def test_prints_input_preview(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_tool_start("read_file", {"path": "f.txt"})
        out = _output(buf)
        assert "f.txt" in out

    def test_empty_input_no_preview(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_tool_start("list_files", {})
        out = _output(buf)
        assert "list_files" in out


class TestPrintToolResult:
    def test_success_uses_checkmark(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_tool_result("read_file", "contents", False, 5.0)
        out = _output(buf)
        assert "read_file" in out
        assert "✓" in out

    def test_error_uses_x(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_tool_result("run_command", "not found", True, 1.0)
        out = _output(buf)
        assert "run_command" in out
        assert "✗" in out

    def test_truncates_long_output(self) -> None:
        display, buf = _make_display(force_terminal=False)
        long_output = "x" * 500
        display.print_tool_result("read_file", long_output, False, 1.0)
        out = _output(buf)
        assert "…" in out

    def test_includes_latency(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_tool_result("read_file", "ok", False, 42.0)
        out = _output(buf)
        assert "42" in out


# ---------------------------------------------------------------------------
# print_diff
# ---------------------------------------------------------------------------


class TestPrintDiff:
    def test_renders_added_line(self) -> None:
        display, buf = _make_display(force_terminal=True)
        display.print_diff("a\n", "a\nb\n", "f.txt")
        out = _output(buf)
        # The added line should appear with a "+" prefix.
        assert "+b" in out

    def test_renders_removed_line(self) -> None:
        display, buf = _make_display(force_terminal=True)
        display.print_diff("a\nb\n", "a\n", "f.txt")
        out = _output(buf)
        # The removed line should appear with a "-" prefix.
        assert "-b" in out

    def test_renders_hunk_header(self) -> None:
        display, buf = _make_display(force_terminal=True)
        display.print_diff("a\n", "b\n", "f.txt")
        out = _output(buf)
        # The @@ hunk header should be present.
        assert "@@" in out

    def test_renders_from_and_to_file_headers(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_diff("a\n", "b\n", "f.txt")
        out = _output(buf)
        assert "--- a/f.txt" in out
        assert "+++ b/f.txt" in out

    def test_no_changes_prints_no_change_message(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_diff("a\n", "a\n", "f.txt")
        out = _output(buf)
        assert "no changes" in out.lower()

    def test_includes_filename_in_panel(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_diff("a\n", "b\n", "src/f.py")
        out = _output(buf)
        assert "src/f.py" in out


# ---------------------------------------------------------------------------
# print_change_summary
# ---------------------------------------------------------------------------


class TestPrintChangeSummary:
    def test_prints_table_with_changes(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_change_summary(
            [
                {"path": "a.txt", "action": "created", "lines_added": 5, "lines_removed": 0},
                {"path": "b.py", "action": "modified", "lines_added": 2, "lines_removed": 1},
                {"path": "c.txt", "action": "deleted", "lines_added": 0, "lines_removed": 0},
            ]
        )
        out = _output(buf)
        assert "a.txt" in out
        assert "b.py" in out
        assert "c.txt" in out
        assert "File Changes" in out

    def test_empty_changes_no_output(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_change_summary([])
        assert _output(buf) == ""


# ---------------------------------------------------------------------------
# print_run_summary
# ---------------------------------------------------------------------------


class TestPrintRunSummary:
    def test_prints_summary_panel(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_run_summary(
            iterations=3,
            total_tokens=1500,
            elapsed_seconds=12.5,
            files_changed=["a.txt", "b.py"],
        )
        out = _output(buf)
        assert "3" in out
        assert "1500" in out
        assert "12.5" in out
        assert "a.txt" in out
        assert "b.py" in out

    def test_files_changed_truncated_at_10(self) -> None:
        display, buf = _make_display(force_terminal=False)
        files = [f"f{i}.txt" for i in range(15)]
        display.print_run_summary(1, 10, 1.0, files)
        out = _output(buf)
        assert "and 5 more" in out

    def test_no_files_changed(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_run_summary(1, 10, 1.0, [])
        out = _output(buf)
        assert "Files changed: 0" in out


# ---------------------------------------------------------------------------
# print_welcome
# ---------------------------------------------------------------------------


class TestPrintWelcome:
    def test_prints_welcome_panel(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_welcome("act", "gpt-4o", "/tmp/project")
        out = _output(buf)
        assert "act" in out
        assert "gpt-4o" in out
        assert "/tmp/project" in out

    def test_includes_help_hint(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_welcome("act", "gpt-4o", "/tmp")
        out = _output(buf)
        assert "/help" in out


# ---------------------------------------------------------------------------
# print_warning / print_error / print_info
# ---------------------------------------------------------------------------


class TestPrintWarningErrorInfo:
    def test_print_warning(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_warning("careful")
        out = _output(buf)
        assert "careful" in out

    def test_print_error(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_error("boom")
        out = _output(buf)
        assert "boom" in out

    def test_print_info(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.print_info("note")
        out = _output(buf)
        assert "note" in out


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------


class TestSpinner:
    def test_start_stop_spinner_in_non_terminal(self) -> None:
        # force_terminal=False simulates piped output — start_spinner
        # should fall back to a plain dim print.
        display, buf = _make_display(force_terminal=False)
        display.start_spinner("Thinking...")
        out_after_start = _output(buf)
        assert "Thinking..." in out_after_start
        # stop_spinner should not raise and should not add output.
        display.stop_spinner()
        assert _output(buf) == ""

    def test_start_spinner_in_terminal(self) -> None:
        display, buf = _make_display(force_terminal=True)
        display.start_spinner("Thinking...")
        # The Live display may not flush to the buffer until stopped,
        # but it should not raise.
        display.stop_spinner()
        # No assertion on content — Live with transient=True clears it.

    def test_stop_spinner_without_start_is_noop(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.stop_spinner()
        assert _output(buf) == ""

    def test_update_spinner_without_start_is_noop(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.update_spinner("new text")  # should not raise
        assert _output(buf) == ""

    def test_start_spinner_replaces_existing(self) -> None:
        display, buf = _make_display(force_terminal=False)
        display.start_spinner("First...")
        display.start_spinner("Second...")
        out = _output(buf)
        # Both messages may appear in non-terminal fallback mode.
        assert "First..." in out or "Second..." in out


# ---------------------------------------------------------------------------
# print_permission_prompt
# ---------------------------------------------------------------------------


class TestPrintPermissionPrompt:
    def test_prints_permission_request(self) -> None:
        display, buf = _make_display(force_terminal=False)
        # We can't easily test interactive Confirm.ask here without
        # mocking stdin, but we can at least verify the description is
        # printed before the prompt. Use a console that won't block.
        # Patch Confirm.ask to avoid blocking on stdin.
        import autoship.cli.display as display_module

        original = display_module.Confirm.ask
        display_module.Confirm.ask = lambda *a, **kw: True  # type: ignore[assignment]
        try:
            result = display.print_permission_prompt("write_to_file", "edit f.txt")
        finally:
            display_module.Confirm.ask = original  # type: ignore[assignment]
        out = _output(buf)
        assert "edit f.txt" in out
        assert result is True


# ---------------------------------------------------------------------------
# _format_preview helper
# ---------------------------------------------------------------------------


class TestFormatPreview:
    def test_empty_dict(self) -> None:
        assert RichDisplay._format_preview({}) == ""

    def test_simple_dict(self) -> None:
        result = RichDisplay._format_preview({"path": "f.txt"})
        assert "f.txt" in result

    def test_long_dict_truncated(self) -> None:
        big = {"x": "a" * 500}
        result = RichDisplay._format_preview(big)
        assert len(result) <= 120

    def test_non_serialisable_falls_back_to_str(self) -> None:
        class Weird:
            def __repr__(self) -> str:
                return "weird-repr"

        # json.dumps fails on non-serialisable objects, so the helper
        # falls back to str(dict) which uses repr() on values.
        result = RichDisplay._format_preview({"obj": Weird()})
        assert "weird-repr" in result
