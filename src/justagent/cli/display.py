"""Rich-based terminal display for the agent CLI.

Provides spinners during LLM calls, formatted tool execution panels,
unified diff rendering with syntax highlighting, and a change summary
table at the end of each run.

In ``json_mode`` every method is a no-op — the agent CLI emits NDJSON
events on stdout instead, and the Rich display must not interfere.
"""

from __future__ import annotations

import contextlib
import difflib
import json
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

# Module-level shared console so all callers (command modules, display
# helpers, etc.) render to the same Rich output stream by default.
_shared_console: Console | None = None


def get_console() -> Console:
    """Return a shared :class:`rich.console.Console` instance.

    A single module-level console is reused across calls so that Rich
    output (tables, panels, spinners) is consistent and does not fight
    over the terminal. Command modules that need Rich rendering without
    the full :class:`RichDisplay` lifecycle should use this helper
    rather than constructing their own ``Console()``.
    """

    global _shared_console
    if _shared_console is None:
        _shared_console = Console()
    return _shared_console


class RichDisplay:
    """Centralised Rich-based terminal output for the agent CLI.

    Construct with ``verbose=True`` for extra detail, or
    ``json_mode=True`` to silence all output (NDJSON mode).
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
        json_mode: bool = False,
        console: Console | None = None,
    ) -> None:
        self.verbose = verbose
        self.json_mode = json_mode
        self._console = console or Console()
        self._live: Live | None = None
        self._spinner: Spinner | None = None

    # -- spinner -----------------------------------------------------------

    def start_spinner(self, text: str) -> None:
        """Start a Rich spinner with ``text`` (e.g. ``"Thinking..."``)."""

        if self.json_mode:
            return
        if self._live is not None:
            self.stop_spinner()
        # In a non-interactive (piped) context Live animation is useless
        # and can confuse captured output — just print the text dimly.
        if not self._console.is_terminal:
            self._console.print(text, style="dim")
            return
        self._spinner = Spinner("dots", text=text)
        self._live = Live(
            self._spinner,
            console=self._console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()

    def stop_spinner(self) -> None:
        """Stop the current spinner if one is running."""

        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.stop()
            self._live = None
        self._spinner = None

    def update_spinner(self, text: str) -> None:
        """Update the spinner text. If no spinner is running, this is a no-op
        (use :meth:`start_spinner` first)."""

        if self.json_mode:
            return
        if self._spinner is not None:
            self._spinner.update(text=text)
        elif self._live is not None:
            self._live.update(Spinner("dots", text=text))

    # -- content panels ----------------------------------------------------

    def print_assistant_message(self, content: str) -> None:
        """Print the LLM's text response in a Rich Panel."""

        if self.json_mode or not content:
            return
        self._console.print(Panel(content, title="🤖 Assistant", border_style="blue"))

    def print_tool_start(self, tool_name: str, input_preview: dict[str, Any]) -> None:
        """Print a tool-start line with a spinner icon."""

        if self.json_mode:
            return
        preview_str = self._format_preview(input_preview)
        text = Text()
        text.append("▶ ", style="yellow")
        text.append(tool_name, style="bold yellow")
        if preview_str:
            text.append(f"  {preview_str}", style="dim")
        self._console.print(text)

    def print_tool_result(
        self,
        tool_name: str,
        output: str,
        is_error: bool,
        latency_ms: float,
    ) -> None:
        """Print a tool result with ✓/✗ and a truncated output preview."""

        if self.json_mode:
            return
        symbol = "✗" if is_error else "✓"
        style = "red" if is_error else "green"
        preview = output[:200].replace("\n", " ")
        if len(output) > 200:
            preview += "…"
        text = Text()
        text.append(f"  {symbol} ", style=style)
        text.append(tool_name, style=style)
        text.append(f" ({latency_ms:.0f}ms) ", style="dim")
        text.append(preview, style="dim")
        self._console.print(text)

    def print_warning(self, message: str) -> None:
        """Print a yellow warning line."""

        if self.json_mode:
            return
        self._console.print(f"⚠ {message}", style="yellow")

    def print_error(self, message: str) -> None:
        """Print a red error line."""

        if self.json_mode:
            return
        self._console.print(f"✗ {message}", style="red")

    def print_info(self, message: str) -> None:
        """Print a dim info line."""

        if self.json_mode:
            return
        self._console.print(message, style="dim")

    # -- diff --------------------------------------------------------------

    def print_diff(
        self,
        old_content: str,
        new_content: str,
        filename: str,
    ) -> None:
        """Render a unified diff with green/red syntax highlighting."""

        if self.json_mode:
            return
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                lineterm="",
            )
        )
        if not diff_lines:
            self._console.print(f"  (no changes in {filename})", style="dim")
            return
        text = Text()
        for line in diff_lines:
            line_stripped = line.rstrip("\n")
            if line.startswith("+++") or line.startswith("---"):
                text.append(line_stripped + "\n", style="dim")
            elif line.startswith("@@"):
                text.append(line_stripped + "\n", style="cyan")
            elif line.startswith("+"):
                text.append(line_stripped + "\n", style="green")
            elif line.startswith("-"):
                text.append(line_stripped + "\n", style="red")
            else:
                text.append(line_stripped + "\n")
        self._console.print(Panel(text, title=f"📝 {filename}", border_style="cyan"))

    # -- summaries ---------------------------------------------------------

    def print_change_summary(self, changes: list[dict[str, Any]]) -> None:
        """Print a table of file changes.

        Each dict should have keys: ``path``, ``action`` (created /
        modified / deleted), ``lines_added``, ``lines_removed``.
        """

        if self.json_mode or not changes:
            return
        table = Table(title="File Changes", border_style="cyan")
        table.add_column("File", style="white")
        table.add_column("Action", style="bold")
        table.add_column("+", style="green", justify="right")
        table.add_column("-", style="red", justify="right")
        for ch in changes:
            action = ch.get("action", "modified")
            action_style = {
                "created": "green",
                "modified": "yellow",
                "deleted": "red",
            }.get(action, "white")
            table.add_row(
                ch.get("path", "?"),
                Text(action, style=action_style),
                str(ch.get("lines_added", 0)),
                str(ch.get("lines_removed", 0)),
            )
        self._console.print(table)

    def print_run_summary(
        self,
        iterations: int,
        total_tokens: int,
        elapsed_seconds: float,
        files_changed: list[str],
    ) -> None:
        """Print a final summary panel."""

        if self.json_mode:
            return
        lines = [
            f"Iterations:  {iterations}",
            f"Tokens:      {total_tokens}",
            f"Elapsed:     {elapsed_seconds:.1f}s",
            f"Files changed: {len(files_changed)}",
        ]
        if files_changed:
            for f in files_changed[:10]:
                lines.append(f"  • {f}")
            if len(files_changed) > 10:
                lines.append(f"  … and {len(files_changed) - 10} more")
        self._console.print(
            Panel(
                "\n".join(lines),
                title="Run Summary",
                border_style="green",
            )
        )

    def print_welcome(self, mode: str, model: str, cwd: str) -> None:
        """Print a welcome banner panel."""

        if self.json_mode:
            return
        content = (
            f"Mode: {mode}  |  Model: {model}\nCWD:  {cwd}\nType /help for commands, /exit to quit"
        )
        self._console.print(
            Panel(
                content,
                title="JustAgent Agent (interactive)",
                border_style="cyan",
            )
        )

    def print_permission_prompt(self, tool: str, description: str) -> bool:
        """Print a permission request and return ``True``/``False``.

        Uses Rich's :class:`Confirm` prompt. In JSON mode (no TTY to
        ask), returns ``True`` (auto-approve — the caller should have
        already handled JSON/yolo mode before calling this).
        """

        if self.json_mode:
            return True
        self._console.print(f"\n📋 Permission needed: {description}", style="yellow")
        return Confirm.ask(f"  Approve {tool}?", default=True)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _format_preview(input_preview: dict[str, Any]) -> str:
        """Format a dict as a compact preview string."""

        if not input_preview:
            return ""
        try:
            raw = json.dumps(input_preview, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(input_preview)[:120]
        return raw[:120]


__all__ = ["RichDisplay", "get_console"]
