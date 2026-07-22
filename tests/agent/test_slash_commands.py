"""Tests for ``autoship.agent.slash_commands``.

Covers:

* :class:`CommandAction` enum values.
* :class:`CommandResult` — construction, frozen, defaults.
* :class:`SlashCommand` — construction, frozen.
* :class:`SlashCommandRegistry.parse` — various input shapes,
  case-insensitivity, whitespace handling.
* :class:`SlashCommandRegistry` — register / unregister / get /
  list_commands sorted.
* :meth:`SlashCommandRegistry.execute` — known / unknown / non-slash
  input, ``None`` context.
* Each built-in command: help, mode, clear, compact, undo, checkpoint,
  skills, tools, exit, cost.
* Edge cases — empty context, ``None`` context, args with quotes.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoship.agent.slash_commands import (
    CommandAction,
    CommandResult,
    SlashCommand,
    SlashCommandRegistry,
    create_default_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler() -> Any:
    """Return a no-op handler suitable for a SlashCommand."""

    def handler(args: list[str], ctx: dict[str, Any]) -> CommandResult:
        return CommandResult(action=CommandAction.NOOP)

    return handler


def _make_command(name: str) -> SlashCommand:
    return SlashCommand(
        name=name,
        description=f"cmd {name}",
        handler=_make_handler(),
    )


# ---------------------------------------------------------------------------
# CommandAction
# ---------------------------------------------------------------------------


class TestCommandAction:
    def test_display_value(self) -> None:
        assert CommandAction.DISPLAY.value == "display"

    def test_clear_history_value(self) -> None:
        assert CommandAction.CLEAR_HISTORY.value == "clear_history"

    def test_switch_mode_value(self) -> None:
        assert CommandAction.SWITCH_MODE.value == "switch_mode"

    def test_compact_value(self) -> None:
        assert CommandAction.COMPACT.value == "compact"

    def test_undo_value(self) -> None:
        assert CommandAction.UNDO.value == "undo"

    def test_restore_checkpoint_value(self) -> None:
        assert CommandAction.RESTORE_CHECKPOINT.value == "restore_checkpoint"

    def test_exit_value(self) -> None:
        assert CommandAction.EXIT.value == "exit"

    def test_noop_value(self) -> None:
        assert CommandAction.NOOP.value == "noop"

    def test_is_str_subclass(self) -> None:
        assert isinstance(CommandAction.DISPLAY, str)

    def test_from_string(self) -> None:
        assert CommandAction("display") is CommandAction.DISPLAY
        assert CommandAction("exit") is CommandAction.EXIT


# ---------------------------------------------------------------------------
# CommandResult
# ---------------------------------------------------------------------------


class TestCommandResult:
    def test_defaults(self) -> None:
        result = CommandResult(action=CommandAction.DISPLAY)
        assert result.action is CommandAction.DISPLAY
        assert result.message == ""
        assert result.data == {}

    def test_with_message(self) -> None:
        result = CommandResult(action=CommandAction.DISPLAY, message="hi")
        assert result.message == "hi"

    def test_with_data(self) -> None:
        result = CommandResult(
            action=CommandAction.SWITCH_MODE,
            data={"mode": "plan"},
        )
        assert result.data == {"mode": "plan"}

    def test_is_frozen(self) -> None:
        result = CommandResult(action=CommandAction.DISPLAY)
        with pytest.raises(AttributeError):
            result.action = CommandAction.EXIT  # type: ignore[misc]

    def test_default_data_instances_are_independent(self) -> None:
        r1 = CommandResult(action=CommandAction.DISPLAY)
        r2 = CommandResult(action=CommandAction.DISPLAY)
        r1.data["x"] = "y"
        assert "x" not in r2.data


# ---------------------------------------------------------------------------
# SlashCommand
# ---------------------------------------------------------------------------


class TestSlashCommand:
    def test_construction(self) -> None:
        handler = _make_handler()
        cmd = SlashCommand(
            name="test",
            description="a test command",
            handler=handler,
        )
        assert cmd.name == "test"
        assert cmd.description == "a test command"
        assert cmd.handler is handler

    def test_is_frozen(self) -> None:
        cmd = SlashCommand(
            name="test",
            description="d",
            handler=_make_handler(),
        )
        with pytest.raises(AttributeError):
            cmd.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def setup_method(self) -> None:
        self.registry = SlashCommandRegistry()

    def test_no_slash_returns_none(self) -> None:
        assert self.registry.parse("hello") is None

    def test_empty_string_returns_none(self) -> None:
        assert self.registry.parse("") is None

    def test_just_slash_returns_none(self) -> None:
        assert self.registry.parse("/") is None

    def test_only_whitespace_returns_none(self) -> None:
        assert self.registry.parse("   ") is None

    def test_slash_with_trailing_whitespace(self) -> None:
        assert self.registry.parse("/  ") is None

    def test_command_no_args(self) -> None:
        assert self.registry.parse("/help") == ("help", [])

    def test_command_with_one_arg(self) -> None:
        assert self.registry.parse("/mode plan") == ("mode", ["plan"])

    def test_command_with_two_args(self) -> None:
        assert self.registry.parse("/checkpoint restore abc123") == (
            "checkpoint",
            ["restore", "abc123"],
        )

    def test_extra_spaces_collapsed(self) -> None:
        assert self.registry.parse("/cmd  arg1  arg2") == (
            "cmd",
            ["arg1", "arg2"],
        )

    def test_leading_trailing_whitespace_stripped(self) -> None:
        assert self.registry.parse("   /help   ") == ("help", [])

    def test_command_case_insensitive(self) -> None:
        assert self.registry.parse("/HELP") == ("help", [])

    def test_mixed_case_command(self) -> None:
        assert self.registry.parse("/Help") == ("help", [])

    def test_unknown_command_parses(self) -> None:
        # parse only splits; unknown commands are handled by execute.
        assert self.registry.parse("/unknown") == ("unknown", [])

    def test_space_after_slash(self) -> None:
        assert self.registry.parse("/ cmd") == ("cmd", [])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def setup_method(self) -> None:
        self.registry = SlashCommandRegistry()

    def test_register_and_get(self) -> None:
        cmd = _make_command("foo")
        self.registry.register(cmd)
        assert self.registry.get("foo") is cmd

    def test_get_unknown_returns_none(self) -> None:
        assert self.registry.get("nope") is None

    def test_get_is_case_insensitive(self) -> None:
        cmd = _make_command("Foo")
        self.registry.register(cmd)
        assert self.registry.get("foo") is cmd
        assert self.registry.get("FOO") is cmd

    def test_unregister_existing(self) -> None:
        cmd = _make_command("foo")
        self.registry.register(cmd)
        assert self.registry.unregister("foo") is True
        assert self.registry.get("foo") is None

    def test_unregister_missing(self) -> None:
        assert self.registry.unregister("nope") is False

    def test_unregister_case_insensitive(self) -> None:
        cmd = _make_command("Foo")
        self.registry.register(cmd)
        assert self.registry.unregister("FOO") is True

    def test_list_commands_empty(self) -> None:
        assert self.registry.list_commands() == []

    def test_list_commands_sorted_by_name(self) -> None:
        for name in ("zeta", "alpha", "mid"):
            self.registry.register(_make_command(name))
        names = [c.name for c in self.registry.list_commands()]
        assert names == ["alpha", "mid", "zeta"]

    def test_register_overwrites_same_name(self) -> None:
        cmd1 = _make_command("foo")
        cmd2 = _make_command("foo")
        self.registry.register(cmd1)
        self.registry.register(cmd2)
        assert self.registry.get("foo") is cmd2
        assert len(self.registry.list_commands()) == 1


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


class TestExecute:
    def setup_method(self) -> None:
        self.registry = create_default_registry()

    def test_non_slash_returns_none(self) -> None:
        assert self.registry.execute("hello world") is None

    def test_empty_input_returns_none(self) -> None:
        assert self.registry.execute("") is None

    def test_known_command(self) -> None:
        result = self.registry.execute("/clear")
        assert result is not None
        assert result.action is CommandAction.CLEAR_HISTORY

    def test_unknown_command_returns_display(self) -> None:
        result = self.registry.execute("/unknown")
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "Unknown command" in result.message

    def test_unknown_command_does_not_raise(self) -> None:
        result = self.registry.execute("/nonexistent arg1 arg2")
        assert result is not None
        assert result.action is CommandAction.DISPLAY

    def test_case_insensitive_command(self) -> None:
        result = self.registry.execute("/CLEAR")
        assert result is not None
        assert result.action is CommandAction.CLEAR_HISTORY

    def test_none_context_handled(self) -> None:
        result = self.registry.execute("/clear", context=None)
        assert result is not None
        assert result.action is CommandAction.CLEAR_HISTORY

    def test_execute_passes_args(self) -> None:
        result = self.registry.execute("/mode plan", context={})
        assert result is not None
        assert result.action is CommandAction.SWITCH_MODE
        assert result.data == {"mode": "plan"}


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------


class TestHelpCommand:
    def test_lists_all_builtin_commands(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/help", context={"registry": registry})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        for name in (
            "help",
            "mode",
            "clear",
            "compact",
            "undo",
            "checkpoint",
            "skills",
            "tools",
            "exit",
            "cost",
        ):
            assert f"/{name}" in result.message

    def test_falls_back_without_registry_in_context(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/help", context={})
        assert result is not None
        assert "Available commands:" in result.message
        assert "/help" in result.message

    def test_falls_back_with_none_context(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/help", context=None)
        assert result is not None
        assert "Available commands:" in result.message


# ---------------------------------------------------------------------------
# /mode
# ---------------------------------------------------------------------------


class TestModeCommand:
    def setup_method(self) -> None:
        self.registry = create_default_registry()

    def test_switch_to_plan(self) -> None:
        result = self.registry.execute("/mode plan", context={})
        assert result is not None
        assert result.action is CommandAction.SWITCH_MODE
        assert result.data == {"mode": "plan"}

    def test_switch_to_act(self) -> None:
        result = self.registry.execute("/mode act", context={})
        assert result is not None
        assert result.action is CommandAction.SWITCH_MODE
        assert result.data == {"mode": "act"}

    def test_switch_to_yolo(self) -> None:
        result = self.registry.execute("/mode yolo", context={})
        assert result is not None
        assert result.action is CommandAction.SWITCH_MODE
        assert result.data == {"mode": "yolo"}

    def test_no_arg_shows_current(self) -> None:
        result = self.registry.execute(
            "/mode", context={"current_mode": "plan"}
        )
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "plan" in result.message

    def test_no_arg_default_mode(self) -> None:
        result = self.registry.execute("/mode", context={})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "act" in result.message

    def test_invalid_mode(self) -> None:
        result = self.registry.execute("/mode invalid", context={})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "Invalid mode" in result.message

    def test_mode_arg_case_insensitive(self) -> None:
        result = self.registry.execute("/mode PLAN", context={})
        assert result is not None
        assert result.action is CommandAction.SWITCH_MODE
        assert result.data == {"mode": "plan"}


# ---------------------------------------------------------------------------
# /clear
# ---------------------------------------------------------------------------


class TestClearCommand:
    def test_returns_clear_history(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/clear", context={})
        assert result is not None
        assert result.action is CommandAction.CLEAR_HISTORY

    def test_message_present(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/clear", context={})
        assert result is not None
        assert result.message != ""


# ---------------------------------------------------------------------------
# /compact
# ---------------------------------------------------------------------------


class TestCompactCommand:
    def test_returns_compact(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/compact", context={})
        assert result is not None
        assert result.action is CommandAction.COMPACT


# ---------------------------------------------------------------------------
# /undo
# ---------------------------------------------------------------------------


class TestUndoCommand:
    def test_returns_undo(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/undo", context={})
        assert result is not None
        assert result.action is CommandAction.UNDO


# ---------------------------------------------------------------------------
# /checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointCommand:
    def setup_method(self) -> None:
        self.registry = create_default_registry()

    def test_no_args_shows_usage(self) -> None:
        result = self.registry.execute("/checkpoint", context={})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "Usage" in result.message

    def test_list_displays_checkpoints(self) -> None:
        cps = [
            {"id": "abc123", "message": "first"},
            {"id": "def456", "message": "second"},
        ]
        result = self.registry.execute(
            "/checkpoint list", context={"checkpoints": cps}
        )
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "abc123" in result.message
        assert "def456" in result.message

    def test_list_empty(self) -> None:
        result = self.registry.execute(
            "/checkpoint list", context={"checkpoints": []}
        )
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "No checkpoints" in result.message

    def test_list_missing_context(self) -> None:
        result = self.registry.execute("/checkpoint list", context={})
        assert result is not None
        assert "No checkpoints" in result.message

    def test_restore_returns_restore_action(self) -> None:
        result = self.registry.execute(
            "/checkpoint restore abc123", context={}
        )
        assert result is not None
        assert result.action is CommandAction.RESTORE_CHECKPOINT
        assert result.data == {"checkpoint_id": "abc123"}

    def test_restore_without_id(self) -> None:
        result = self.registry.execute("/checkpoint restore", context={})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "Usage" in result.message

    def test_unknown_subcommand(self) -> None:
        result = self.registry.execute("/checkpoint frobnicate", context={})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "Unknown subcommand" in result.message


# ---------------------------------------------------------------------------
# /skills
# ---------------------------------------------------------------------------


class TestSkillsCommand:
    def setup_method(self) -> None:
        self.registry = create_default_registry()

    def test_no_args_shows_usage(self) -> None:
        result = self.registry.execute("/skills", context={})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "Usage" in result.message

    def test_list_displays_skills(self) -> None:
        skills = [
            {"name": "git-sop", "description": "Git SOP"},
            {"name": "registry", "description": "Tool registry"},
        ]
        result = self.registry.execute(
            "/skills list", context={"available_skills": skills}
        )
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "git-sop" in result.message
        assert "registry" in result.message

    def test_list_empty(self) -> None:
        result = self.registry.execute(
            "/skills list", context={"available_skills": []}
        )
        assert result is not None
        assert "No skills" in result.message

    def test_list_missing_context(self) -> None:
        result = self.registry.execute("/skills list", context={})
        assert result is not None
        assert "No skills" in result.message

    def test_load_existing_skill(self) -> None:
        skills = [
            {"name": "git-sop", "content": "## Git SOP\n- commit often"},
        ]
        result = self.registry.execute(
            "/skills load git-sop", context={"available_skills": skills}
        )
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "## Git SOP" in result.message

    def test_load_missing_skill(self) -> None:
        result = self.registry.execute(
            "/skills load nonexistent", context={"available_skills": []}
        )
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "not found" in result.message.lower()

    def test_load_without_name(self) -> None:
        result = self.registry.execute("/skills load", context={})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "Usage" in result.message


# ---------------------------------------------------------------------------
# /tools
# ---------------------------------------------------------------------------


class TestToolsCommand:
    def test_lists_tools_from_context(self) -> None:
        registry = create_default_registry()
        tools = [
            {"name": "read_file", "description": "Read a file"},
            {"name": "write_file", "description": "Write a file"},
        ]
        result = registry.execute("/tools", context={"tools": tools})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "read_file" in result.message
        assert "write_file" in result.message

    def test_no_tools_message(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/tools", context={})
        assert result is not None
        assert "No tools" in result.message

    def test_empty_tools_list(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/tools", context={"tools": []})
        assert result is not None
        assert "No tools" in result.message


# ---------------------------------------------------------------------------
# /exit
# ---------------------------------------------------------------------------


class TestExitCommand:
    def test_returns_exit(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/exit", context={})
        assert result is not None
        assert result.action is CommandAction.EXIT


# ---------------------------------------------------------------------------
# /cost
# ---------------------------------------------------------------------------


class TestCostCommand:
    def test_shows_cost_from_context(self) -> None:
        registry = create_default_registry()
        cost = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_cost": 0.0123,
        }
        result = registry.execute("/cost", context={"cost": cost})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "1000" in result.message
        assert "0.0123" in result.message

    def test_no_cost_in_context(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/cost", context={})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "No cost" in result.message

    def test_cost_as_scalar(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/cost", context={"cost": 0.05})
        assert result is not None
        assert result.action is CommandAction.DISPLAY
        assert "0.05" in result.message


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_context_dict(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/help", context={})
        assert result is not None
        assert result.action is CommandAction.DISPLAY

    def test_none_context(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/mode", context=None)
        assert result is not None
        assert result.action is CommandAction.DISPLAY

    def test_args_with_quotes_basic_split(self) -> None:
        # Basic handling: quotes are preserved as part of args (no
        # shell-like quote stripping). Splitting on whitespace only.
        registry = SlashCommandRegistry()
        parsed = registry.parse('/cmd "quoted arg"')
        assert parsed == ("cmd", ['"quoted', 'arg"'])

    def test_args_with_quotes_execute(self) -> None:
        # The quoted text is split across args; /skills load falls
        # back to a "not found" because the skill name doesn't match.
        registry = create_default_registry()
        result = registry.execute(
            '/skills load "my skill"', context={"available_skills": []}
        )
        assert result is not None
        assert result.action is CommandAction.DISPLAY

    def test_command_with_special_chars_in_args(self) -> None:
        registry = SlashCommandRegistry()
        assert registry.parse("/cmd arg/with/slashes") == (
            "cmd",
            ["arg/with/slashes"],
        )

    def test_default_registry_has_all_commands(self) -> None:
        registry = create_default_registry()
        names = {c.name for c in registry.list_commands()}
        assert names == {
            "help",
            "mode",
            "clear",
            "compact",
            "undo",
            "checkpoint",
            "skills",
            "tools",
            "exit",
            "cost",
        }

    def test_multiple_spaces_in_command_args(self) -> None:
        registry = create_default_registry()
        result = registry.execute(
            "/checkpoint    restore    abc123", context={}
        )
        assert result is not None
        assert result.action is CommandAction.RESTORE_CHECKPOINT
        assert result.data == {"checkpoint_id": "abc123"}

    def test_uppercase_command_with_args(self) -> None:
        registry = create_default_registry()
        result = registry.execute("/CHECKPOINT RESTORE abc", context={})
        assert result is not None
        assert result.action is CommandAction.RESTORE_CHECKPOINT
        assert result.data == {"checkpoint_id": "abc"}
