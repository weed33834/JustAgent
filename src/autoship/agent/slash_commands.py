"""Slash commands for the interactive agent loop.

When the user types input starting with ``/``, it is parsed as a slash
command instead of being sent to the LLM. Commands can return text to
display, trigger an action (clear history, switch mode, undo), or both.

Built-in commands:
    /help              Show available commands
    /mode [plan|act|yolo]   Switch agent mode (or show current)
    /clear             Clear conversation history
    /compact           Manually trigger context compaction
    /undo              Restore the last checkpoint
    /checkpoint [list|restore <id>]   Manage checkpoints
    /skills [list|load <name>]   Browse available skills
    /tools             List available tools
    /exit              Exit the agent loop
    /cost              Show token/cost usage so far
    /lint [file...]    Run the linter (ruff) on files or all dirty files
    /test [args...]    Run the test suite (pytest) with optional args
    /add <file>        Add a file to the agent's explicit context
    /drop <file>       Remove a file from the agent's explicit context
    /tokens            Show token usage breakdown
    /diff              Show uncommitted git changes
    /history           Show conversation history summary

Reference: Cline's ``registerCommands()`` and OpenCode's ``/`` command palette.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from autoship.exceptions import AutoShipError


class SlashCommandError(AutoShipError):
    """Raised when a slash command fails to parse or execute."""


class CommandAction(str, Enum):  # noqa: UP042
    """What the agent loop should do after a slash command runs."""

    #: Just display ``CommandResult.message`` to the user.
    DISPLAY = "display"
    #: Wipe the conversation history.
    CLEAR_HISTORY = "clear_history"
    #: Switch the agent mode (carries ``{"mode": "plan|act|yolo"}``).
    SWITCH_MODE = "switch_mode"
    #: Manually trigger context compaction.
    COMPACT = "compact"
    #: Restore the most recent checkpoint.
    UNDO = "undo"
    #: Restore a specific checkpoint (carries ``{"checkpoint_id": "..."}``).
    RESTORE_CHECKPOINT = "restore_checkpoint"
    #: Exit the interactive agent loop.
    EXIT = "exit"
    #: Do nothing — the command already handled everything itself.
    NOOP = "noop"


@dataclass(frozen=True)
class CommandResult:
    """Outcome of executing a slash command.

    Attributes:
        action: What the agent loop should do next.
        message: Text to show the user (may be empty for pure actions).
        data: Extra context, e.g. ``{"mode": "plan"}`` for SWITCH_MODE
            or ``{"checkpoint_id": "abc123"}`` for RESTORE_CHECKPOINT.
    """

    action: CommandAction
    message: str = ""
    data: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SlashCommand:
    """A single slash command definition.

    Attributes:
        name: Command name without the leading slash (e.g. ``"help"``).
            Matched case-insensitively.
        description: Short human-readable summary shown by ``/help``.
        handler: Callable invoked with ``(args, context)`` that returns
            a :class:`CommandResult`. The context dict may carry keys
            such as ``"current_mode"``, ``"available_skills"``,
            ``"checkpoints"``, ``"tools"``, ``"cost"`` and
            ``"registry"``.
    """

    name: str
    description: str
    handler: Callable[[list[str], dict[str, Any]], CommandResult]


class SlashCommandRegistry:
    """Registry of available slash commands.

    Commands are matched case-insensitively: ``/HELP`` works the same
    as ``/help``. Unknown commands never raise — :meth:`execute` returns
    a :class:`CommandResult` with ``action=DISPLAY`` and an error
    message instead.
    """

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        """Register a command. Overwrites an existing one with the same name."""

        self._commands[command.name.lower()] = command

    def unregister(self, name: str) -> bool:
        """Remove a command by name. Returns ``True`` if it was present."""

        return self._commands.pop(name.lower(), None) is not None

    def get(self, name: str) -> SlashCommand | None:
        """Return the command with the given name, or ``None``."""

        return self._commands.get(name.lower())

    def list_commands(self) -> list[SlashCommand]:
        """Return all commands sorted by name."""

        return sorted(self._commands.values(), key=lambda c: c.name)

    def parse(self, raw_input: str) -> tuple[str, list[str]] | None:
        """Parse ``/cmd arg1 arg2`` into ``(cmd, args)``.

        Returns ``None`` when the input is not a slash command (does
        not start with ``/``) or is just ``/`` with no command name.
        The command name is normalized to lowercase. Whitespace between
        args is collapsed via ``str.split``.
        """

        if not raw_input:
            return None
        stripped = raw_input.strip()
        if not stripped.startswith("/"):
            return None
        rest = stripped[1:].strip()
        if not rest:
            return None
        parts = rest.split()
        cmd = parts[0].lower()
        args = parts[1:]
        return cmd, args

    def execute(
        self,
        raw_input: str,
        context: dict[str, Any] | None = None,
    ) -> CommandResult | None:
        """Parse and execute a slash command.

        Returns ``None`` when the input is not a slash command, so the
        caller can forward it to the LLM. Unknown commands return a
        :class:`CommandResult` with ``action=DISPLAY`` and an error
        message — this method never raises for unknown commands.
        """

        parsed = self.parse(raw_input)
        if parsed is None:
            return None
        name, args = parsed
        command = self.get(name)
        if command is None:
            return CommandResult(
                action=CommandAction.DISPLAY,
                message=(
                    f"Unknown command: /{name}. "
                    "Type /help for available commands."
                ),
            )
        return command.handler(args, context or {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_attr(obj: Any, name: str, default: Any = "") -> Any:
    """Return ``obj.name`` for objects or ``obj[name]`` for dicts.

    Falls back to ``default`` when neither is available. Used by
    handlers so callers can pass either plain dicts or small dataclasses
    (e.g. ``Checkpoint``, ``Tool``) as context entries.
    """

    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


#: Valid agent modes for the ``/mode`` command.
_VALID_MODES: tuple[str, ...] = ("plan", "act", "yolo")

#: Built-in command metadata used by ``/help`` when no registry is
#: available in the context dict.
_BUILTIN_COMMANDS: tuple[tuple[str, str], ...] = (
    ("help", "Show available commands"),
    ("mode", "Switch agent mode (or show current): /mode [plan|act|yolo]"),
    ("clear", "Clear conversation history"),
    ("compact", "Manually trigger context compaction"),
    ("undo", "Restore the last checkpoint"),
    ("checkpoint", "Manage checkpoints: /checkpoint [list|restore <id>]"),
    ("skills", "Browse available skills: /skills [list|load <name>]"),
    ("tools", "List available tools"),
    ("exit", "Exit the agent loop"),
    ("cost", "Show token/cost usage so far"),
    ("lint", "Run the linter (ruff): /lint [file...]"),
    ("test", "Run the test suite (pytest): /test [args...]"),
    ("add", "Add a file to explicit context: /add <file>"),
    ("drop", "Remove a file from explicit context: /drop <file>"),
    ("tokens", "Show token usage breakdown"),
    ("diff", "Show uncommitted git changes"),
    ("history", "Show conversation history summary"),
)


# ---------------------------------------------------------------------------
# Built-in command handlers
# ---------------------------------------------------------------------------


def _handle_help(args: list[str], context: dict[str, Any]) -> CommandResult:
    """List all available commands."""

    registry = context.get("registry")
    if isinstance(registry, SlashCommandRegistry):
        commands = registry.list_commands()
        lines = ["Available commands:"]
        for cmd in commands:
            lines.append(f"  /{cmd.name} - {cmd.description}")
        return CommandResult(action=CommandAction.DISPLAY, message="\n".join(lines))
    lines = ["Available commands:"]
    for name, desc in _BUILTIN_COMMANDS:
        lines.append(f"  /{name} - {desc}")
    return CommandResult(action=CommandAction.DISPLAY, message="\n".join(lines))


def _handle_mode(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Switch the agent mode, or show the current one when called with no args."""

    if not args:
        current = str(context.get("current_mode", "act"))
        return CommandResult(
            action=CommandAction.DISPLAY,
            message=f"Current mode: {current}",
        )
    mode = args[0].lower()
    if mode not in _VALID_MODES:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message=(
                f"Invalid mode: {mode}. "
                f"Valid modes: {', '.join(_VALID_MODES)}"
            ),
        )
    return CommandResult(
        action=CommandAction.SWITCH_MODE,
        message=f"Switched to {mode} mode.",
        data={"mode": mode},
    )


def _handle_clear(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Clear the conversation history."""

    return CommandResult(
        action=CommandAction.CLEAR_HISTORY,
        message="Conversation history cleared.",
    )


def _handle_compact(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Manually trigger context compaction."""

    return CommandResult(
        action=CommandAction.COMPACT,
        message="Context compaction triggered.",
    )


def _handle_undo(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Restore the most recent checkpoint."""

    return CommandResult(
        action=CommandAction.UNDO,
        message="Restoring last checkpoint...",
    )


def _handle_checkpoint(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Manage checkpoints: ``list`` or ``restore <id>``."""

    if not args:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message=(
                "Usage: /checkpoint [list|restore <id>]\n"
                "  list           List available checkpoints\n"
                "  restore <id>   Restore the checkpoint with the given id"
            ),
        )
    sub = args[0].lower()
    if sub == "list":
        checkpoints = context.get("checkpoints")
        if not checkpoints:
            return CommandResult(
                action=CommandAction.DISPLAY,
                message="No checkpoints available.",
            )
        lines = ["Checkpoints:"]
        for cp in checkpoints:
            cp_id = str(_get_attr(cp, "id", str(cp)))
            cp_msg = str(_get_attr(cp, "message", ""))
            lines.append(f"  {cp_id} - {cp_msg}".rstrip())
        return CommandResult(action=CommandAction.DISPLAY, message="\n".join(lines))
    if sub == "restore":
        if len(args) < 2:
            return CommandResult(
                action=CommandAction.DISPLAY,
                message="Usage: /checkpoint restore <id>",
            )
        cp_id = args[1]
        return CommandResult(
            action=CommandAction.RESTORE_CHECKPOINT,
            message=f"Restoring checkpoint {cp_id}...",
            data={"checkpoint_id": cp_id},
        )
    return CommandResult(
        action=CommandAction.DISPLAY,
        message=(
            f"Unknown subcommand: {sub}. "
            "Usage: /checkpoint [list|restore <id>]"
        ),
    )


def _handle_skills(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Browse available skills: ``list`` or ``load <name>``."""

    if not args:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message=(
                "Usage: /skills [list|load <name>]\n"
                "  list          List available skills\n"
                "  load <name>   Load and display a skill"
            ),
        )
    sub = args[0].lower()
    if sub == "list":
        skills = context.get("available_skills")
        if not skills:
            return CommandResult(
                action=CommandAction.DISPLAY,
                message="No skills available.",
            )
        lines = ["Available skills:"]
        for skill in skills:
            name = str(_get_attr(skill, "name", str(skill)))
            desc = str(_get_attr(skill, "description", ""))
            lines.append(f"  {name} - {desc}".rstrip())
        return CommandResult(action=CommandAction.DISPLAY, message="\n".join(lines))
    if sub == "load":
        if len(args) < 2:
            return CommandResult(
                action=CommandAction.DISPLAY,
                message="Usage: /skills load <name>",
            )
        skill_name = args[1]
        skills = context.get("available_skills") or []
        content = ""
        found = False
        for skill in skills:
            if str(_get_attr(skill, "name", "")) == skill_name:
                content = str(
                    _get_attr(skill, "content", _get_attr(skill, "description", ""))
                )
                found = True
                break
        if not found:
            return CommandResult(
                action=CommandAction.DISPLAY,
                message=f"Skill not found: {skill_name}",
            )
        return CommandResult(
            action=CommandAction.DISPLAY,
            message=content or f"Loaded skill: {skill_name}",
        )
    return CommandResult(
        action=CommandAction.DISPLAY,
        message=(
            f"Unknown subcommand: {sub}. "
            "Usage: /skills [list|load <name>]"
        ),
    )


def _handle_tools(args: list[str], context: dict[str, Any]) -> CommandResult:
    """List available tools from the context."""

    tools = context.get("tools")
    if not tools:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message="No tools available.",
        )
    lines = ["Available tools:"]
    for tool in tools:
        name = str(_get_attr(tool, "name", _get_attr(tool, "id", str(tool))))
        desc = str(_get_attr(tool, "description", ""))
        lines.append(f"  {name} - {desc}".rstrip())
    return CommandResult(action=CommandAction.DISPLAY, message="\n".join(lines))


def _handle_exit(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Exit the interactive agent loop."""

    return CommandResult(
        action=CommandAction.EXIT,
        message="Goodbye!",
    )


def _handle_cost(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Show token/cost usage so far from the context."""

    cost = context.get("cost")
    if cost is None:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message="No cost information available.",
        )
    if isinstance(cost, dict):
        lines = ["Cost usage so far:"]
        for key, value in cost.items():
            lines.append(f"  {key}: {value}")
        return CommandResult(action=CommandAction.DISPLAY, message="\n".join(lines))
    return CommandResult(
        action=CommandAction.DISPLAY,
        message=f"Cost usage so far: {cost}",
    )


#: Token usage fields shown by ``/tokens``, in display order.
_TOKEN_FIELDS: tuple[tuple[str, str], ...] = (
    ("prompt_tokens", "Prompt"),
    ("completion_tokens", "Completion"),
    ("total_tokens", "Total"),
    ("system_tokens", "System"),
    ("history_tokens", "History"),
    ("file_tokens", "Files"),
    ("tool_tokens", "Tools"),
    ("max_tokens", "Max"),
)


def _handle_lint(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Run the linter (ruff by default) on the given files or all dirty files.

    When no files are given, ruff checks the current working directory.
    """

    cwd = context.get("cwd", ".")
    try:
        result = subprocess.run(
            ["ruff", "check", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message=(
                "ruff is not installed or not on PATH. "
                "Install it with `pip install ruff`."
            ),
        )
    output = (result.stdout or "") + (result.stderr or "")
    return CommandResult(
        action=CommandAction.DISPLAY,
        message=output.strip() or "No lint issues found.",
    )


def _handle_test(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Run the test suite (pytest) with optional args."""

    cwd = context.get("cwd", ".")
    try:
        result = subprocess.run(
            ["pytest", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message=(
                "pytest is not installed or not on PATH. "
                "Install it with `pip install pytest`."
            ),
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message="Test run timed out after 300 seconds.",
        )
    output = (result.stdout or "") + (result.stderr or "")
    return CommandResult(
        action=CommandAction.DISPLAY,
        message=output.strip() or "Tests passed (no output).",
    )


def _handle_add(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Add a file to the agent's explicit context.

    The file path is stored in ``context["explicit_files"]`` so the agent
    loop always includes its full content in the LLM context.
    """

    if not args:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message="Usage: /add <file>",
        )
    file_path = args[0]
    cwd = context.get("cwd", ".")
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = Path(cwd) / file_path
    if not candidate.exists():
        return CommandResult(
            action=CommandAction.DISPLAY,
            message=f"File not found: {file_path}",
        )
    explicit_files = context.setdefault("explicit_files", [])
    if file_path not in explicit_files:
        explicit_files.append(file_path)
    return CommandResult(
        action=CommandAction.DISPLAY,
        message=f"Added {file_path} to context",
    )


def _handle_drop(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Remove a file from the agent's explicit context."""

    if not args:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message="Usage: /drop <file>",
        )
    file_path = args[0]
    explicit_files = context.get("explicit_files", [])
    if file_path in explicit_files:
        explicit_files.remove(file_path)
        return CommandResult(
            action=CommandAction.DISPLAY,
            message=f"Removed {file_path} from context",
        )
    return CommandResult(
        action=CommandAction.DISPLAY,
        message=f"{file_path} is not in the explicit context.",
    )


def _handle_tokens(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Show a token usage breakdown from the context."""

    usage = context.get("token_usage", {})
    if not usage:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message="No token usage information available.",
        )
    lines = ["Token usage:"]
    for key, label in _TOKEN_FIELDS:
        if key in usage:
            lines.append(f"  {label:<10}: {usage[key]}")
    return CommandResult(action=CommandAction.DISPLAY, message="\n".join(lines))


def _handle_diff(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Show uncommitted git changes in the project."""

    cwd = context.get("cwd", ".")
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message="git is not installed or not on PATH.",
        )
    output = (result.stdout or "") + (result.stderr or "")
    if not output.strip():
        return CommandResult(
            action=CommandAction.DISPLAY,
            message="No uncommitted changes.",
        )
    return CommandResult(
        action=CommandAction.DISPLAY,
        message=output.strip(),
    )


def _handle_history(args: list[str], context: dict[str, Any]) -> CommandResult:
    """Show a conversation history summary from the context."""

    messages = context.get("messages", [])
    if not messages:
        return CommandResult(
            action=CommandAction.DISPLAY,
            message="No conversation history.",
        )
    lines = ["Conversation history:"]
    for index, msg in enumerate(messages, 1):
        role = str(_get_attr(msg, "role", "?"))
        # Support both dict-style (``content_preview``) and Message
        # dataclass (``content``) inputs — the REPL passes live Message
        # objects, while tests may pass dicts with a preview field.
        preview = str(
            _get_attr(msg, "content_preview", "")
            or _get_attr(msg, "content", "")
        )
        if len(preview) > 80:
            preview = preview[:77] + "..."
        # Collapse newlines for single-line display.
        preview = " ".join(preview.splitlines()).strip()
        lines.append(f"  {index}. [{role}] {preview}".rstrip())
    return CommandResult(action=CommandAction.DISPLAY, message="\n".join(lines))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_registry() -> SlashCommandRegistry:
    """Create a registry pre-populated with all built-in slash commands."""

    registry = SlashCommandRegistry()
    registry.register(
        SlashCommand(
            name="help",
            description="Show available commands",
            handler=_handle_help,
        )
    )
    registry.register(
        SlashCommand(
            name="mode",
            description="Switch agent mode (or show current): /mode [plan|act|yolo]",
            handler=_handle_mode,
        )
    )
    registry.register(
        SlashCommand(
            name="clear",
            description="Clear conversation history",
            handler=_handle_clear,
        )
    )
    registry.register(
        SlashCommand(
            name="compact",
            description="Manually trigger context compaction",
            handler=_handle_compact,
        )
    )
    registry.register(
        SlashCommand(
            name="undo",
            description="Restore the last checkpoint",
            handler=_handle_undo,
        )
    )
    registry.register(
        SlashCommand(
            name="checkpoint",
            description="Manage checkpoints: /checkpoint [list|restore <id>]",
            handler=_handle_checkpoint,
        )
    )
    registry.register(
        SlashCommand(
            name="skills",
            description="Browse available skills: /skills [list|load <name>]",
            handler=_handle_skills,
        )
    )
    registry.register(
        SlashCommand(
            name="tools",
            description="List available tools",
            handler=_handle_tools,
        )
    )
    registry.register(
        SlashCommand(
            name="exit",
            description="Exit the agent loop",
            handler=_handle_exit,
        )
    )
    registry.register(
        SlashCommand(
            name="cost",
            description="Show token/cost usage so far",
            handler=_handle_cost,
        )
    )
    registry.register(
        SlashCommand(
            name="lint",
            description="Run the linter (ruff): /lint [file...]",
            handler=_handle_lint,
        )
    )
    registry.register(
        SlashCommand(
            name="test",
            description="Run the test suite (pytest): /test [args...]",
            handler=_handle_test,
        )
    )
    registry.register(
        SlashCommand(
            name="add",
            description="Add a file to explicit context: /add <file>",
            handler=_handle_add,
        )
    )
    registry.register(
        SlashCommand(
            name="drop",
            description="Remove a file from explicit context: /drop <file>",
            handler=_handle_drop,
        )
    )
    registry.register(
        SlashCommand(
            name="tokens",
            description="Show token usage breakdown",
            handler=_handle_tokens,
        )
    )
    registry.register(
        SlashCommand(
            name="diff",
            description="Show uncommitted git changes",
            handler=_handle_diff,
        )
    )
    registry.register(
        SlashCommand(
            name="history",
            description="Show conversation history summary",
            handler=_handle_history,
        )
    )
    return registry


__all__ = [
    "CommandAction",
    "CommandResult",
    "SlashCommand",
    "SlashCommandError",
    "SlashCommandRegistry",
    "create_default_registry",
]
