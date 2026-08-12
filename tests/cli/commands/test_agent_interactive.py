"""Tests for the interactive REPL mode of the ``justagent agent`` command.

Covers:

* The ``--interactive`` / ``-i`` flag exists and is recognised.
* The REPL loop reads from stdin (mocked ``input()``).
* Slash commands (``/exit``, ``/clear``, ``/mode``, ``/help``) are processed.
* Regular input calls ``runtime.continue_run()``.
* ``Ctrl+C`` / ``Ctrl+D`` (EOFError / KeyboardInterrupt) exits gracefully.
* The welcome banner is shown in pretty mode (and skipped in JSON mode).
* The ``_print_result`` helper formats output correctly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from justagent.agent.plan_act import AgentMode
from justagent.agent.runtime import RunResult
from justagent.agent.slash_commands import create_default_registry
from justagent.cli.commands import agent as agent_module
from justagent.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_legacy_llm_config(root: Path, *, api_key: str = "fake-key") -> None:
    """Write a ``.justagent.toml`` with a legacy ``[llm]`` section."""

    config_file = root / ".justagent.toml"
    config_file.write_text(
        f'[llm]\napi_key = "{api_key}"\nmodel = "gpt-4o-mini"\n',
        encoding="utf-8",
    )


def _make_run_result(
    *,
    status: str = "completed",
    final_content: str = "Done!",
    iterations: int = 1,
    error: str = "",
    stop_reason: str = "",
) -> RunResult:
    """Build a real RunResult (not a mock) for predictable attribute access."""

    return RunResult(
        status=status,  # type: ignore[arg-type]
        final_content=final_content,
        iterations=iterations,
        messages=[],
        total_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        error=error,
        stop_reason=stop_reason,
    )


def _async_wrap(value: Any) -> Any:
    """Wrap a plain value in an awaitable that ``asyncio.run`` can drive."""

    async def _coro() -> Any:
        return value

    return _coro()


def _make_mock_runtime(
    *,
    run_result: RunResult | None = None,
    continue_result: RunResult | None = None,
) -> MagicMock:
    """Build a mock ``AgentRuntime`` suitable for interactive-mode tests.

    ``run`` and ``continue_run`` are set up as ``MagicMock``s whose
    ``side_effect`` produces a **fresh** awaitable on each call (a
    coroutine cannot be awaited more than once).
    """

    _run_result = run_result or _make_run_result()
    _continue_result = continue_result or _make_run_result()

    mock = MagicMock()
    mock.run = MagicMock(side_effect=lambda *a, **kw: _async_wrap(_run_result))
    mock.continue_run = MagicMock(
        side_effect=lambda *a, **kw: _async_wrap(_continue_result)
    )
    mock.reset = MagicMock()
    mock.abort = MagicMock()
    mock.switch_mode = MagicMock()
    mock._abort = MagicMock()
    mock._abort.clear = MagicMock()
    mock.mode = MagicMock()
    mock.mode.value = "act"
    mock._total_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return mock


# ---------------------------------------------------------------------------
# Flag existence
# ---------------------------------------------------------------------------


class TestInteractiveFlag:
    def test_help_lists_interactive_flag(self) -> None:
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0
        out = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--interactive" in out
        assert "-i" in out

    def test_interactive_flag_short_form_accepted(self, tmp_path: Path, monkeypatch) -> None:
        """``-i`` should be accepted and enter the REPL (which immediately
        exits via EOF)."""

        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = _make_mock_runtime()
        with (
            patch(
                "justagent.cli.commands.agent.AgentRuntime", return_value=mock_runtime
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            result = runner.invoke(app, ["agent", "-i"])

        assert result.exit_code == 0

    def test_non_interactive_without_prompt_still_errors(self, tmp_path: Path, monkeypatch) -> None:
        """Without ``-i``, a missing prompt should still error."""

        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["agent"])
        assert result.exit_code != 0
        assert "Prompt is required" in result.output or "prompt" in result.output.lower()


# ---------------------------------------------------------------------------
# Welcome banner
# ---------------------------------------------------------------------------


class TestWelcomeBanner:
    def test_banner_shown_in_pretty_mode(self, capsys: pytest.CaptureFixture) -> None:
        agent_module._print_welcome_banner(
            mode="act", model="gpt-4o", cwd="/tmp", json_mode=False
        )
        captured = capsys.readouterr()
        assert "JustAgent Agent" in captured.out
        assert "interactive" in captured.out
        assert "act" in captured.out
        assert "gpt-4o" in captured.out
        assert "/tmp" in captured.out
        assert "/help" in captured.out
        assert "/exit" in captured.out

    def test_banner_skipped_in_json_mode(self, capsys: pytest.CaptureFixture) -> None:
        agent_module._print_welcome_banner(
            mode="act", model="gpt-4o", cwd="/tmp", json_mode=True
        )
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# _print_result
# ---------------------------------------------------------------------------


class TestPrintResult:
    def test_pretty_mode_prints_turn_summary(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        result = _make_run_result(iterations=2, final_content="hello")
        agent_module._print_result(result, json_mode=False, turn=3)
        captured = capsys.readouterr()
        assert "turn 3" in captured.out
        assert "2 iteration" in captured.out
        assert "15 tokens" in captured.out

    def test_json_mode_emits_result_envelope(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        result = _make_run_result(final_content="done")
        agent_module._print_result(result, json_mode=True, turn=1)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["type"] == "result"
        assert parsed["turn"] == 1
        assert parsed["status"] == "completed"
        assert parsed["final_content"] == "done"

    def test_pretty_mode_shows_error_status(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        result = _make_run_result(
            status="failed", error="boom", final_content=""
        )
        agent_module._print_result(result, json_mode=False, turn=1)
        captured = capsys.readouterr()
        assert "failed" in captured.err or "failed" in captured.out


# ---------------------------------------------------------------------------
# _run_interactive — direct async tests
# ---------------------------------------------------------------------------


class TestRunInteractiveDirect:
    """Tests that drive ``_run_interactive`` directly as an async function,
    mocking ``input()`` and the runtime. This gives fine-grained control
    over the REPL loop without going through the Typer CLI layer."""

    @pytest.mark.asyncio
    async def test_exit_command_breaks_loop(self) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        with patch("builtins.input", side_effect=["/exit"]):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        # /exit should break before any continue_run call.
        runtime.continue_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_eof_exits_gracefully(self) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        with patch("builtins.input", side_effect=EOFError):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        runtime.continue_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_at_prompt_exits(self) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        runtime.continue_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_regular_input_calls_continue_run(self) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        inputs = ["hello world", "/exit"]
        with patch("builtins.input", side_effect=inputs):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        runtime.continue_run.assert_called_once_with("hello world")

    @pytest.mark.asyncio
    async def test_initial_prompt_calls_run_not_continue(self) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        with patch("builtins.input", side_effect=["/exit"]):
            await agent_module._run_interactive(
                runtime, "initial task", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        runtime.run.assert_called_once_with("initial task")
        runtime.continue_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate(self) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        inputs = ["turn one", "turn two", "/exit"]
        with patch("builtins.input", side_effect=inputs):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        assert runtime.continue_run.call_count == 2
        runtime.continue_run.assert_any_call("turn one")
        runtime.continue_run.assert_any_call("turn two")

    @pytest.mark.asyncio
    async def test_clear_command_calls_reset(self) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        with patch("builtins.input", side_effect=["/clear", "/exit"]):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        runtime.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_mode_command_switches_mode(self) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        with patch("builtins.input", side_effect=["/mode plan", "/exit"]):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        runtime.switch_mode.assert_called_once_with(AgentMode.PLAN)

    @pytest.mark.asyncio
    async def test_help_command_displays_output(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        with patch("builtins.input", side_effect=["/help", "/exit"]):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        captured = capsys.readouterr()
        assert "Available commands" in captured.out

    @pytest.mark.asyncio
    async def test_empty_lines_are_skipped(self) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        inputs = ["", "  ", "real input", "/exit"]
        with patch("builtins.input", side_effect=inputs):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        runtime.continue_run.assert_called_once_with("real input")

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_during_turn_aborts(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """When a turn is interrupted, the runtime should be aborted and
        the REPL should continue accepting input."""

        runtime = _make_mock_runtime()
        # First continue_run raises KeyboardInterrupt, second succeeds.
        runtime.continue_run = MagicMock(
            side_effect=[KeyboardInterrupt, _async_wrap(_make_run_result())]
        )
        registry = create_default_registry()
        inputs = ["first", "second", "/exit"]
        with patch("builtins.input", side_effect=inputs):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        runtime.abort.assert_called_once()
        captured = capsys.readouterr()
        assert "aborted" in captured.err or "aborted" in captured.out

    @pytest.mark.asyncio
    async def test_json_mode_reads_until_empty_line(self) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        # In JSON mode, an empty line signals end of input.
        inputs = ["do something", ""]
        with patch("builtins.input", side_effect=inputs):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=True, verbose=False,
                model_name="test", cwd="/tmp",
            )
        runtime.continue_run.assert_called_once_with("do something")

    @pytest.mark.asyncio
    async def test_unknown_slash_command_shows_message(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        runtime = _make_mock_runtime()
        registry = create_default_registry()
        with patch("builtins.input", side_effect=["/nonexistent", "/exit"]):
            await agent_module._run_interactive(
                runtime, "", registry,
                json_mode=False, verbose=False,
                model_name="test", cwd="/tmp",
            )
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out
        runtime.continue_run.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end CLI tests (mocked runtime + mocked stdin)
# ---------------------------------------------------------------------------


class TestAgentInteractiveCli:
    def test_interactive_with_exit(self, tmp_path: Path, monkeypatch) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = _make_mock_runtime()
        with (
            patch(
                "justagent.cli.commands.agent.AgentRuntime", return_value=mock_runtime
            ),
            patch("builtins.input", side_effect=["/exit"]),
        ):
            result = runner.invoke(app, ["agent", "-i"])

        assert result.exit_code == 0
        assert "Goodbye" in result.output

    def test_interactive_with_eof(self, tmp_path: Path, monkeypatch) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = _make_mock_runtime()
        with (
            patch(
                "justagent.cli.commands.agent.AgentRuntime", return_value=mock_runtime
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            result = runner.invoke(app, ["agent", "--interactive"])

        assert result.exit_code == 0
        assert "Goodbye" in result.output

    def test_interactive_banner_shown(self, tmp_path: Path, monkeypatch) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = _make_mock_runtime()
        with (
            patch(
                "justagent.cli.commands.agent.AgentRuntime", return_value=mock_runtime
            ),
            patch("builtins.input", side_effect=["/exit"]),
        ):
            result = runner.invoke(app, ["agent", "-i"])

        assert "JustAgent Agent" in result.output
        assert "interactive" in result.output

    def test_interactive_with_initial_prompt_then_exit(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = _make_mock_runtime()
        with (
            patch(
                "justagent.cli.commands.agent.AgentRuntime", return_value=mock_runtime
            ),
            patch("builtins.input", side_effect=["/exit"]),
        ):
            result = runner.invoke(app, ["agent", "-i", "initial task"])

        assert result.exit_code == 0
        mock_runtime.run.assert_called_once_with("initial task")

    def test_interactive_regular_input_uses_continue_run(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = _make_mock_runtime()
        with (
            patch(
                "justagent.cli.commands.agent.AgentRuntime", return_value=mock_runtime
            ),
            patch("builtins.input", side_effect=["hello there", "/exit"]),
        ):
            result = runner.invoke(app, ["agent", "-i"])

        assert result.exit_code == 0
        mock_runtime.continue_run.assert_called_once_with("hello there")

    def test_interactive_clear_command(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = _make_mock_runtime()
        with (
            patch(
                "justagent.cli.commands.agent.AgentRuntime", return_value=mock_runtime
            ),
            patch("builtins.input", side_effect=["/clear", "/exit"]),
        ):
            result = runner.invoke(app, ["agent", "-i"])

        assert result.exit_code == 0
        mock_runtime.reset.assert_called_once()
        assert "cleared" in result.output.lower()

    def test_interactive_mode_switch(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = _make_mock_runtime()
        with (
            patch(
                "justagent.cli.commands.agent.AgentRuntime", return_value=mock_runtime
            ),
            patch("builtins.input", side_effect=["/mode plan", "/exit"]),
        ):
            result = runner.invoke(app, ["agent", "-i"])

        assert result.exit_code == 0
        mock_runtime.switch_mode.assert_called_once_with(AgentMode.PLAN)
        assert "plan" in result.output.lower()

    def test_interactive_long_flag_works(self, tmp_path: Path, monkeypatch) -> None:
        """Both ``--interactive`` and ``-i`` should work."""

        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = _make_mock_runtime()
        with (
            patch(
                "justagent.cli.commands.agent.AgentRuntime", return_value=mock_runtime
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            result = runner.invoke(app, ["agent", "--interactive"])

        assert result.exit_code == 0
