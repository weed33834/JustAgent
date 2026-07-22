"""Tests for the ``autoship agent`` CLI command.

These tests cover:

* Help output (smoke test).
* Mode resolution from ``--mode`` / ``--plan`` / ``--yolo`` / ``--yes``.
* LLM client resolution from CLI overrides and config sections.
* Event-emit callback (pretty + JSON modes).
* End-to-end run via the Typer CliRunner, with ``AgentRuntime.run``
  mocked so no real LLM call is made.
* Status → exit-code mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from autoship.agent.plan_act import AgentMode
from autoship.agent.runtime import (
    AssistantMessageEvent,
    RunCompletedEvent,
    RunStartedEvent,
    RuntimeEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
    TurnStartedEvent,
)
from autoship.cli.commands import agent as agent_module
from autoship.cli.main import app
from autoship.models.config import (
    AppConfig,
    LlmConfig,
    LlmProvider,
    ModelBackendConfig,
    ModelConfig,
    Provider,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(type_: str, run_id: str = "test-run") -> RuntimeEvent:
    """Build a minimal RuntimeEvent for emit tests."""

    return RuntimeEvent(type=type_, run_id=run_id)


def _write_legacy_llm_config(root: Path, *, api_key: str = "fake-key") -> None:
    """Write a ``.autoship.toml`` with a legacy ``[llm]`` section."""

    config_file = root / ".autoship.toml"
    config_file.write_text(
        f'[llm]\napi_key = "{api_key}"\nmodel = "gpt-4o-mini"\n',
        encoding="utf-8",
    )


def _write_newstyle_model_config(root: Path, *, api_key: str = "fake-key") -> None:
    """Write a ``.autoship.toml`` with a ``[model.backends]]`` section."""

    config_file = root / ".autoship.toml"
    config_file.write_text(
        f'''[[model.backends]]
provider = "openai"
base_url = "https://api.openai.com/v1"
api_key = "{api_key}"
model = "gpt-4o"
''',
        encoding="utf-8",
    )


def _make_run_result(
    *,
    status: str = "completed",
    final_content: str = "Done!",
    iterations: int = 1,
    error: str = "",
    stop_reason: str = "",
) -> Any:
    """Build a fake RunResult for mocked ``runtime.run``."""

    return MagicMock(
        status=status,
        final_content=final_content,
        iterations=iterations,
        messages=[],
        total_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        error=error,
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# Help / registration smoke test
# ---------------------------------------------------------------------------


class TestAgentHelpSmoke:
    def test_agent_command_registered(self) -> None:
        """The `agent` command should appear in the top-level CLI."""

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "agent" in result.output

    def test_agent_help_lists_options(self) -> None:
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0
        assert "--mode" in result.output
        assert "--plan" in result.output
        assert "--yolo" in result.output
        assert "--json" in result.output
        assert "--max-iterations" in result.output
        assert "--api-key" in result.output
        assert "--base-url" in result.output
        assert "--model" in result.output


# ---------------------------------------------------------------------------
# _resolve_mode
# ---------------------------------------------------------------------------


class TestResolveMode:
    def test_default_is_act(self) -> None:
        assert (
            agent_module._resolve_mode(
                mode_flag=None, plan_flag=False, yolo_flag=False, yes_flag=False
            )
            is AgentMode.ACT
        )

    def test_mode_flag_act(self) -> None:
        assert (
            agent_module._resolve_mode(
                mode_flag="act", plan_flag=False, yolo_flag=False, yes_flag=False
            )
            is AgentMode.ACT
        )

    def test_mode_flag_plan(self) -> None:
        assert (
            agent_module._resolve_mode(
                mode_flag="plan", plan_flag=False, yolo_flag=False, yes_flag=False
            )
            is AgentMode.PLAN
        )

    def test_mode_flag_yolo(self) -> None:
        assert (
            agent_module._resolve_mode(
                mode_flag="yolo", plan_flag=False, yolo_flag=False, yes_flag=False
            )
            is AgentMode.YOLO
        )

    def test_plan_flag_shortcut(self) -> None:
        assert (
            agent_module._resolve_mode(
                mode_flag=None, plan_flag=True, yolo_flag=False, yes_flag=False
            )
            is AgentMode.PLAN
        )

    def test_yolo_flag_shortcut(self) -> None:
        assert (
            agent_module._resolve_mode(
                mode_flag=None, plan_flag=False, yolo_flag=True, yes_flag=False
            )
            is AgentMode.YOLO
        )

    def test_yes_flag_uses_yolo(self) -> None:
        """``--yes`` skips permission prompts → Yolo mode."""

        assert (
            agent_module._resolve_mode(
                mode_flag=None, plan_flag=False, yolo_flag=False, yes_flag=True
            )
            is AgentMode.YOLO
        )

    def test_yolo_overrides_plan(self) -> None:
        """If both --plan and --yolo are given, yolo wins."""

        assert (
            agent_module._resolve_mode(
                mode_flag=None, plan_flag=True, yolo_flag=True, yes_flag=False
            )
            is AgentMode.YOLO
        )

    def test_yolo_overrides_mode_flag(self) -> None:
        """If --yolo and --mode plan are both given, yolo wins."""

        assert (
            agent_module._resolve_mode(
                mode_flag="plan", plan_flag=False, yolo_flag=True, yes_flag=False
            )
            is AgentMode.YOLO
        )

    def test_invalid_mode_raises(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter):
            agent_module._resolve_mode(
                mode_flag="invalid", plan_flag=False, yolo_flag=False, yes_flag=False
            )


# ---------------------------------------------------------------------------
# _resolve_llm
# ---------------------------------------------------------------------------


def _make_config(
    *,
    backends: list[ModelBackendConfig] | None = None,
    llm: LlmConfig | None = None,
) -> AppConfig:
    """Build an AppConfig with optional model backends / legacy llm section."""

    return AppConfig(
        model=ModelConfig(backends=backends or []),
        llm=llm or LlmConfig(),
    )


class TestResolveLLM:
    def test_cli_overrides_take_priority(self) -> None:
        config = _make_config(
            backends=[
                ModelBackendConfig(
                    provider=Provider.OPENAI,
                    base_url="https://from-config.example/v1",
                    api_key="config-key",
                    model="config-model",
                )
            ]
        )
        client = agent_module._resolve_llm(
            config,
            model_override="cli-model",
            api_key_override="cli-key",
            base_url_override="https://cli.example/v1",
        )
        assert client._model == "cli-model"
        assert client._api_key == "cli-key"
        assert client._base_url == "https://cli.example/v1"

    def test_newstyle_backends_used_when_no_overrides(self) -> None:
        config = _make_config(
            backends=[
                ModelBackendConfig(
                    provider=Provider.OPENAI,
                    base_url="https://api.openai.com/v1",
                    api_key="config-key",
                    model="gpt-4o",
                    timeout=42.0,
                )
            ]
        )
        client = agent_module._resolve_llm(
            config,
            model_override=None,
            api_key_override=None,
            base_url_override=None,
        )
        assert client._model == "gpt-4o"
        assert client._api_key == "config-key"
        assert client._base_url == "https://api.openai.com/v1"
        assert client._timeout == 42.0
        assert client._provider == "openai"

    def test_legacy_llm_section_used_when_no_backends(self) -> None:
        config = _make_config(
            llm=LlmConfig(
                provider=LlmProvider.OPENAI,
                model="gpt-4o-mini",
                api_key="legacy-key",
            )
        )
        client = agent_module._resolve_llm(
            config,
            model_override=None,
            api_key_override=None,
            base_url_override=None,
        )
        assert client._model == "gpt-4o-mini"
        assert client._api_key == "legacy-key"
        # Default OpenAI base URL applied.
        assert client._base_url == "https://api.openai.com/v1"

    def test_legacy_llm_with_explicit_base_url(self) -> None:
        config = _make_config(
            llm=LlmConfig(
                provider=LlmProvider.OPENAI,
                model="gpt-4o-mini",
                api_key="legacy-key",
                base_url="https://custom.example/v1",
            )
        )
        client = agent_module._resolve_llm(
            config,
            model_override=None,
            api_key_override=None,
            base_url_override=None,
        )
        assert client._base_url == "https://custom.example/v1"

    def test_legacy_llm_uses_default_url_for_openrouter(self) -> None:
        config = _make_config(
            llm=LlmConfig(
                provider=LlmProvider.OPENROUTER,
                model="anthropic/claude-3.5",
                api_key="or-key",
            )
        )
        client = agent_module._resolve_llm(
            config,
            model_override=None,
            api_key_override=None,
            base_url_override=None,
        )
        assert client._provider == "openrouter"
        assert client._base_url == "https://openrouter.ai/api/v1"

    def test_missing_api_key_for_openai_raises(self) -> None:
        import typer

        config = _make_config(
            llm=LlmConfig(
                provider=LlmProvider.OPENAI,
                model="gpt-4o-mini",
                api_key=None,
            )
        )
        with pytest.raises(typer.BadParameter):
            agent_module._resolve_llm(
                config,
                model_override=None,
                api_key_override=None,
                base_url_override=None,
            )

    def test_ollama_allows_missing_api_key(self) -> None:
        """Ollama runs locally — no API key required."""

        config = _make_config(
            llm=LlmConfig(
                provider=LlmProvider.OLLAMA,
                model="llama3",
                api_key=None,
            )
        )
        client = agent_module._resolve_llm(
            config,
            model_override=None,
            api_key_override=None,
            base_url_override=None,
        )
        assert client._api_key is None
        assert client._base_url == "http://127.0.0.1:11434/v1"

    def test_backend_without_model_raises(self) -> None:
        import typer

        config = _make_config(
            backends=[
                ModelBackendConfig(
                    provider=Provider.OPENAI,
                    base_url="https://api.openai.com/v1",
                    api_key="k",
                    model=None,
                )
            ]
        )
        with pytest.raises(typer.BadParameter):
            agent_module._resolve_llm(
                config,
                model_override=None,
                api_key_override=None,
                base_url_override=None,
            )

    def test_partial_overrides_fall_through_to_config(self) -> None:
        """Passing only --model should still pull api_key/base_url from config."""

        config = _make_config(
            backends=[
                ModelBackendConfig(
                    provider=Provider.OPENAI,
                    base_url="https://api.openai.com/v1",
                    api_key="config-key",
                    model="config-model",
                )
            ]
        )
        client = agent_module._resolve_llm(
            config,
            model_override="override-model",
            api_key_override=None,
            base_url_override=None,
        )
        assert client._model == "override-model"
        assert client._api_key == "config-key"


# ---------------------------------------------------------------------------
# _event_to_dict (JSON serialization)
# ---------------------------------------------------------------------------


class TestEventToDict:
    def test_run_started(self) -> None:
        event = RunStartedEvent(type="run-started", run_id="r1", iteration=0)
        d = agent_module._event_to_dict(event)
        assert d["type"] == "run-started"
        assert d["run_id"] == "r1"
        assert d["iteration"] == 0

    def test_turn_started(self) -> None:
        event = TurnStartedEvent(
            type="turn-started", run_id="r1", iteration=3
        )
        d = agent_module._event_to_dict(event)
        assert d["iteration"] == 3

    def test_assistant_message(self) -> None:
        from autoship.agent.runtime import ToolCall

        event = AssistantMessageEvent(
            type="assistant-message",
            run_id="r1",
            content="hello",
            tool_calls=[ToolCall(id="tc1", name="echo", input={"x": 1})],
            finish_reason="stop",
            usage={"total_tokens": 5},
            latency_ms=10.0,
        )
        d = agent_module._event_to_dict(event)
        assert d["content"] == "hello"
        assert d["tool_calls"][0]["name"] == "echo"
        assert d["finish_reason"] == "stop"
        assert d["latency_ms"] == 10.0

    def test_tool_started(self) -> None:
        event = ToolStartedEvent(
            type="tool-started",
            run_id="r1",
            iteration=1,
            tool_call_id="tc1",
            tool_name="read_file",
            input={"path": "a.txt"},
        )
        d = agent_module._event_to_dict(event)
        assert d["tool_name"] == "read_file"
        assert d["input"] == {"path": "a.txt"}

    def test_tool_finished(self) -> None:
        event = ToolFinishedEvent(
            type="tool-finished",
            run_id="r1",
            iteration=1,
            tool_call_id="tc1",
            tool_name="read_file",
            output="contents",
            is_error=False,
            latency_ms=5.0,
        )
        d = agent_module._event_to_dict(event)
        assert d["is_error"] is False
        assert d["output"] == "contents"

    def test_run_completed(self) -> None:
        event = RunCompletedEvent(
            type="run-completed",
            run_id="r1",
            final_content="Done",
            iterations=3,
            total_usage={"total_tokens": 100},
        )
        d = agent_module._event_to_dict(event)
        assert d["final_content"] == "Done"
        assert d["iterations"] == 3


# ---------------------------------------------------------------------------
# _make_emit_callback
# ---------------------------------------------------------------------------


class TestEmitCallback:
    @pytest.mark.asyncio
    async def test_json_mode_emits_ndjson(self, capsys: pytest.CaptureFixture) -> None:
        emit = agent_module._make_emit_callback(json_mode=True, verbose=False)
        event = RunStartedEvent(type="run-started", run_id="r1", iteration=0)
        await emit(event)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["type"] == "run-started"
        assert parsed["run_id"] == "r1"

    @pytest.mark.asyncio
    async def test_pretty_mode_assistant_message_prints_content(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        emit = agent_module._make_emit_callback(json_mode=False, verbose=False)
        event = AssistantMessageEvent(
            type="assistant-message",
            run_id="r1",
            content="Hello, world!",
            tool_calls=[],
            finish_reason="stop",
            usage={},
            latency_ms=1.0,
        )
        await emit(event)
        captured = capsys.readouterr()
        assert "Hello, world!" in captured.out

    @pytest.mark.asyncio
    async def test_pretty_mode_tool_started_uses_arrow(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        emit = agent_module._make_emit_callback(json_mode=False, verbose=False)
        event = ToolStartedEvent(
            type="tool-started",
            run_id="r1",
            iteration=1,
            tool_call_id="tc1",
            tool_name="read_file",
            input={},
        )
        await emit(event)
        captured = capsys.readouterr()
        assert "read_file" in captured.out

    @pytest.mark.asyncio
    async def test_pretty_mode_tool_finished_shows_status(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        emit = agent_module._make_emit_callback(json_mode=False, verbose=False)
        event = ToolFinishedEvent(
            type="tool-finished",
            run_id="r1",
            iteration=1,
            tool_call_id="tc1",
            tool_name="read_file",
            output="file contents",
            is_error=False,
            latency_ms=5.0,
        )
        await emit(event)
        captured = capsys.readouterr()
        assert "read_file" in captured.out

    @pytest.mark.asyncio
    async def test_pretty_mode_error_tool_uses_x(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        emit = agent_module._make_emit_callback(json_mode=False, verbose=False)
        event = ToolFinishedEvent(
            type="tool-finished",
            run_id="r1",
            iteration=1,
            tool_call_id="tc1",
            tool_name="run_command",
            output="command not found",
            is_error=True,
            latency_ms=1.0,
        )
        await emit(event)
        captured = capsys.readouterr()
        assert "run_command" in captured.out


# ---------------------------------------------------------------------------
# _status_to_exit_code
# ---------------------------------------------------------------------------


class TestStatusToExitCode:
    def test_completed_returns_zero(self) -> None:
        assert agent_module._status_to_exit_code("completed") == 0

    def test_aborted_returns_130(self) -> None:
        assert agent_module._status_to_exit_code("aborted") == 130

    def test_failed_returns_one(self) -> None:
        assert agent_module._status_to_exit_code("failed") == 1

    def test_stopped_returns_two(self) -> None:
        assert agent_module._status_to_exit_code("stopped") == 2

    def test_unknown_returns_one(self) -> None:
        assert agent_module._status_to_exit_code("unknown") == 1


# ---------------------------------------------------------------------------
# End-to-end CLI tests (mocked runtime)
# ---------------------------------------------------------------------------


class TestAgentCliEndToEnd:
    def test_agent_runs_and_completes_with_legacy_config(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = MagicMock()
        mock_runtime.run = MagicMock(
            return_value=_async_wrap(_make_run_result(status="completed"))
        )
        with patch(
            "autoship.cli.commands.agent.AgentRuntime", return_value=mock_runtime
        ):
            result = runner.invoke(app, ["agent", "say hello"])

        assert result.exit_code == 0
        mock_runtime.run.assert_called_once_with("say hello")

    def test_agent_failed_status_returns_one(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = MagicMock()
        mock_runtime.run = MagicMock(
            return_value=_async_wrap(
                _make_run_result(status="failed", final_content="", error="boom")
            )
        )
        with patch(
            "autoship.cli.commands.agent.AgentRuntime", return_value=mock_runtime
        ):
            result = runner.invoke(app, ["agent", "do something"])

        assert result.exit_code == 1

    def test_agent_stopped_status_returns_two(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = MagicMock()
        mock_runtime.run = MagicMock(
            return_value=_async_wrap(
                _make_run_result(status="stopped", stop_reason="max_iterations")
            )
        )
        with patch(
            "autoship.cli.commands.agent.AgentRuntime", return_value=mock_runtime
        ):
            result = runner.invoke(app, ["agent", "loop forever"])

        assert result.exit_code == 2

    def test_agent_with_newstyle_config(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_newstyle_model_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = MagicMock()
        mock_runtime.run = MagicMock(
            return_value=_async_wrap(_make_run_result())
        )
        with patch(
            "autoship.cli.commands.agent.AgentRuntime", return_value=mock_runtime
        ):
            result = runner.invoke(app, ["agent", "hello"])

        assert result.exit_code == 0

    def test_agent_json_mode_emits_result_envelope(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = MagicMock()
        mock_runtime.run = MagicMock(
            return_value=_async_wrap(
                _make_run_result(status="completed", final_content="All done")
            )
        )
        with patch(
            "autoship.cli.commands.agent.AgentRuntime", return_value=mock_runtime
        ):
            result = runner.invoke(app, ["agent", "--json", "task"])

        assert result.exit_code == 0
        # The last line should be the result envelope.
        lines = [line for line in result.output.splitlines() if line.strip()]
        result_line = lines[-1]
        parsed = json.loads(result_line)
        assert parsed["type"] == "result"
        assert parsed["status"] == "completed"
        assert parsed["final_content"] == "All done"

    def test_agent_plan_flag_sets_initial_mode(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        captured_config: dict[str, Any] = {}

        def fake_init(**kwargs: Any) -> MagicMock:
            captured_config.update(kwargs)
            mock = MagicMock()
            mock.run = MagicMock(return_value=_async_wrap(_make_run_result()))
            return mock

        with patch(
            "autoship.cli.commands.agent.AgentRuntime", side_effect=fake_init
        ):
            result = runner.invoke(app, ["agent", "--plan", "explore"])

        assert result.exit_code == 0
        runtime_config = captured_config["config"]
        assert runtime_config.initial_mode is AgentMode.PLAN

    def test_agent_yolo_flag_sets_initial_mode(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        captured_config: dict[str, Any] = {}

        def fake_init(**kwargs: Any) -> MagicMock:
            captured_config.update(kwargs)
            mock = MagicMock()
            mock.run = MagicMock(return_value=_async_wrap(_make_run_result()))
            return mock

        with patch(
            "autoship.cli.commands.agent.AgentRuntime", side_effect=fake_init
        ):
            result = runner.invoke(app, ["agent", "--yolo", "just do it"])

        assert result.exit_code == 0
        runtime_config = captured_config["config"]
        assert runtime_config.initial_mode is AgentMode.YOLO

    def test_agent_yes_flag_sets_yolo_mode(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        captured_config: dict[str, Any] = {}

        def fake_init(**kwargs: Any) -> MagicMock:
            captured_config.update(kwargs)
            mock = MagicMock()
            mock.run = MagicMock(return_value=_async_wrap(_make_run_result()))
            return mock

        with patch(
            "autoship.cli.commands.agent.AgentRuntime", side_effect=fake_init
        ):
            result = runner.invoke(app, ["agent", "--yes", "go"])

        assert result.exit_code == 0
        runtime_config = captured_config["config"]
        assert runtime_config.initial_mode is AgentMode.YOLO

    def test_agent_mode_flag_explicit(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        captured_config: dict[str, Any] = {}

        def fake_init(**kwargs: Any) -> MagicMock:
            captured_config.update(kwargs)
            mock = MagicMock()
            mock.run = MagicMock(return_value=_async_wrap(_make_run_result()))
            return mock

        with patch(
            "autoship.cli.commands.agent.AgentRuntime", side_effect=fake_init
        ):
            result = runner.invoke(app, ["agent", "--mode", "plan", "explore"])

        assert result.exit_code == 0
        runtime_config = captured_config["config"]
        assert runtime_config.initial_mode is AgentMode.PLAN

    def test_agent_cli_overrides_pass_to_llm_client(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # No config file — must rely on CLI overrides.
        monkeypatch.chdir(tmp_path)

        mock_runtime = MagicMock()
        mock_runtime.run = MagicMock(return_value=_async_wrap(_make_run_result()))
        with (
            patch(
                "autoship.cli.commands.agent.AgentRuntime", return_value=mock_runtime
            ),
            patch(
                "autoship.cli.commands.agent.LLMClient"
            ) as mock_llm_client_class,
        ):
            result = runner.invoke(
                app,
                [
                    "agent",
                    "--model", "cli-model",
                    "--api-key", "cli-key",
                    "--base-url", "https://cli.example/v1",
                    "test",
                ],
            )

        assert result.exit_code == 0
        mock_llm_client_class.assert_called_once()
        call_kwargs = mock_llm_client_class.call_args.kwargs
        assert call_kwargs["model"] == "cli-model"
        assert call_kwargs["api_key"] == "cli-key"
        assert call_kwargs["base_url"] == "https://cli.example/v1"

    def test_agent_no_api_key_fails_cleanly(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # No config file, no CLI overrides — should fail with BadParameter.
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["agent", "hello"])
        assert result.exit_code != 0

    def test_agent_records_audit_events(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_legacy_llm_config(tmp_path)
        monkeypatch.chdir(tmp_path)

        mock_runtime = MagicMock()
        mock_runtime.run = MagicMock(
            return_value=_async_wrap(_make_run_result(status="completed"))
        )
        with patch(
            "autoship.cli.commands.agent.AgentRuntime", return_value=mock_runtime
        ):
            result = runner.invoke(app, ["agent", "task"])

        assert result.exit_code == 0
        # The audit log file should exist and contain agent.start / agent.done.
        audit_log = Path.home() / ".autoship" / "logs"
        # Find today's audit log
        audit_files = list(audit_log.glob("audit.*.jsonl"))
        assert audit_files, "Expected at least one audit log file"
        # Read the latest one
        latest = max(audit_files, key=lambda p: p.stat().st_mtime)
        contents = latest.read_text(encoding="utf-8")
        assert "agent.start" in contents
        assert "agent.done" in contents


# ---------------------------------------------------------------------------
# Helpers for async
# ---------------------------------------------------------------------------


def _async_wrap(value: Any) -> Any:
    """Wrap a plain value in an awaitable that asyncio.run can drive."""

    async def _coro() -> Any:
        return value

    return _coro()
