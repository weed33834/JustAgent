"""The ``autoship agent`` command — run the local-first AI agent loop.

Wraps :class:`autoship.agent.runtime.AgentRuntime` in a Typer command so
users can drive the agent from the CLI:

* ``autoship agent "list files in this dir"`` — default Act mode, tools
  available, prompts before destructive ops (or ``--yes`` to skip).
* ``autoship agent --plan "explore the codebase and propose a refactor"``
  — Plan mode: edit tools hidden, read-only investigation only.
* ``autoship agent --yolo "fix the failing tests"`` — Yolo mode: all
  tools, no permission prompts.
* ``autoship agent --json "..."`` — emit NDJSON events on stdout for
  headless / programmatic consumption (mirrors Cline's ``--json`` and
  OpenCode's ``session event stream``).

LLM credentials are resolved in this priority order:

1. ``--model`` / ``--api-key`` / ``--base-url`` CLI flags (highest).
2. ``config.model.backends[*]`` (preferred new-style config).
3. ``config.llm`` (legacy single-backend section, also bridged by
   :func:`autoship.cli.commands.fix._model_router`).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import typer

from autoship.agent.plan_act import AgentMode
from autoship.agent.runtime import (
    AgentRuntime,
    AgentRuntimeConfig,
    AssistantMessageEvent,
    LLMClient,
    LoopWarningEvent,
    MistakeLimitHitEvent,
    RunAbortedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    RuntimeEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
    TurnStartedEvent,
)
from autoship.agent.tools.builtin import make_default_tools
from autoship.exceptions import AutoShipError
from autoship.models.config import AppConfig, LlmProvider, Provider

app = typer.Typer()


#: Default system prompt — mirrors Cline's "You are Roo, a highly skilled
#: software engineer..." preamble, trimmed for autoship's scope.
DEFAULT_SYSTEM_PROMPT = (
    "You are autoship's built-in agent — a highly skilled software "
    "engineer with deep knowledge of the local codebase. You work "
    "iteratively: read files and search to build context, then make "
    "targeted edits and verify your work with shell commands. Prefer "
    "small, surgical changes over large rewrites. When you're done, "
    "respond with a short summary (no tool calls)."
)

#: Map legacy ``[llm].provider`` → ``Provider`` enum (mirrors fix.py).
_LLM_PROVIDER_TO_BACKEND: dict[LlmProvider, Provider] = {
    LlmProvider.OPENAI: Provider.OPENAI,
    LlmProvider.OPENROUTER: Provider.OPENROUTER,
    LlmProvider.OLLAMA: Provider.OLLAMA,
}

#: Default base URLs for providers that don't specify one in config.
_DEFAULT_BASE_URLS: dict[Provider, str] = {
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.OPENROUTER: "https://openrouter.ai/api/v1",
    Provider.OLLAMA: "http://127.0.0.1:11434/v1",
}


def register(parent: typer.Typer) -> None:
    parent.command(name="agent", help="运行本地 AI 智能体循环。")(agent)


def _resolve_mode(
    *,
    mode_flag: str | None,
    plan_flag: bool,
    yolo_flag: bool,
    yes_flag: bool,
) -> AgentMode:
    """Resolve the initial :class:`AgentMode` from CLI flags.

    Precedence: ``--yolo`` > ``--plan`` > ``--mode`` > ``--yes`` (which
    upgrades Act → Yolo to skip prompts). Explicit ``--mode`` wins over
    ``--plan`` / ``--yolo`` only when they don't conflict — if both are
    given, the shorthand flags win so users can override config-level
    ``--mode`` defaults easily.
    """

    if yolo_flag or yes_flag:
        return AgentMode.YOLO
    if plan_flag:
        return AgentMode.PLAN
    if mode_flag:
        try:
            return AgentMode(mode_flag)
        except ValueError as exc:
            raise typer.BadParameter(
                f"Invalid --mode value: {mode_flag!r} (expected act/plan/yolo)"
            ) from exc
    return AgentMode.ACT


def _resolve_llm(
    config: AppConfig,
    *,
    model_override: str | None,
    api_key_override: str | None,
    base_url_override: str | None,
) -> LLMClient:
    """Build an :class:`LLMClient` from CLI overrides or config.

    Priority: CLI flag > ``config.model.backends[0]`` > ``config.llm``.
    Raises :class:`typer.BadParameter` if no usable backend is found.
    """

    # 1. CLI overrides — if the user gave all three, use them directly.
    if model_override and api_key_override and base_url_override:
        return LLMClient(
            model=model_override,
            api_key=api_key_override,
            base_url=base_url_override,
        )

    # 2. New-style config.model.backends.
    if config.model.backends:
        backend = config.model.backends[0]
        model = model_override or backend.model or ""
        api_key = api_key_override or backend.api_key
        base_url = base_url_override or str(backend.base_url)
        if not model:
            raise typer.BadParameter(
                "Model backend has no `model` field — pass --model or fix .autoship.toml"
            )
        return LLMClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_version=backend.api_version,
            timeout=backend.timeout,
            provider=backend.provider.value,
        )

    # 3. Legacy config.llm (single backend).
    llm = config.llm
    provider = _LLM_PROVIDER_TO_BACKEND.get(llm.provider, Provider.OPENAI)
    model = model_override or llm.model
    api_key = api_key_override or llm.api_key
    base_url = base_url_override or str(llm.base_url or _DEFAULT_BASE_URLS[provider])
    if not api_key and provider is not Provider.OLLAMA:
        raise typer.BadParameter(
            "No API key configured — pass --api-key or set [llm].api_key in .autoship.toml"
        )
    return LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=llm.api_version,
        timeout=llm.timeout,
        provider=provider.value,
    )


def _make_emit_callback(*, json_mode: bool, verbose: bool) -> Any:
    """Build an event-emitter callback for the runtime.

    In ``--json`` mode, every event is serialized as one NDJSON line on
    stdout (headless-friendly). In pretty mode, only "interesting"
    events are surfaced — assistant messages, tool starts/finishes,
    warnings, and the final outcome.
    """

    async def emit(event: RuntimeEvent) -> None:
        if json_mode:
            typer.echo(json.dumps(_event_to_dict(event), ensure_ascii=False))
            return

        if isinstance(event, RunStartedEvent):
            if verbose:
                typer.secho(
                    f"[run {event.run_id}] started", fg=typer.colors.CYAN, dim=True
                )
        elif isinstance(event, TurnStartedEvent):
            if verbose:
                typer.secho(
                    f"  ── turn {event.iteration} ──", fg=typer.colors.BLUE, dim=True
                )
        elif isinstance(event, AssistantMessageEvent):
            if event.content:
                typer.echo(event.content)
            if event.tool_calls and verbose:
                for tc in event.tool_calls:
                    typer.secho(
                        f"  → calling {tc.name}({tc.input})", fg=typer.colors.MAGENTA
                    )
        elif isinstance(event, ToolStartedEvent):
            typer.secho(
                f"  ▶ {event.tool_name}", fg=typer.colors.YELLOW, dim=True
            )
        elif isinstance(event, ToolFinishedEvent):
            symbol = "✗" if event.is_error else "✓"
            color = typer.colors.RED if event.is_error else typer.colors.GREEN
            preview = event.output[:200].replace("\n", " ")
            if len(event.output) > 200:
                preview += "…"
            typer.secho(
                f"  {symbol} {event.tool_name} ({event.latency_ms:.0f}ms) {preview}",
                fg=color,
                dim=True,
            )
        elif isinstance(event, LoopWarningEvent):
            typer.secho(
                f"  ⚠ loop warning: {event.message}",
                fg=typer.colors.YELLOW,
                err=True,
            )
        elif isinstance(event, MistakeLimitHitEvent):
            typer.secho(
                f"  ⚠ mistake limit hit ({event.consecutive_mistakes}/"
                f"{event.max_consecutive_mistakes}): {event.reason}",
                fg=typer.colors.YELLOW,
                err=True,
            )
        elif isinstance(event, RunCompletedEvent):
            typer.secho(
                f"[done] {event.iterations} iteration(s), "
                f"{event.total_usage.get('total_tokens', 0)} tokens",
                fg=typer.colors.GREEN,
                dim=True,
            )
        elif isinstance(event, RunAbortedEvent):
            typer.secho(
                f"[aborted] {event.reason}", fg=typer.colors.YELLOW, err=True
            )
        elif isinstance(event, RunFailedEvent):
            typer.secho(
                f"[failed] {event.error}", fg=typer.colors.RED, err=True
            )

    return emit


def _event_to_dict(event: RuntimeEvent) -> dict[str, Any]:
    """Serialize a :class:`RuntimeEvent` to a JSON-friendly dict."""

    data: dict[str, Any] = {
        "type": event.type,
        "run_id": event.run_id,
        "timestamp": event.timestamp,
    }
    if isinstance(event, (RunStartedEvent, TurnStartedEvent)):
        data["iteration"] = event.iteration
    elif isinstance(event, AssistantMessageEvent):
        data["content"] = event.content
        data["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "input": tc.input}
            for tc in event.tool_calls
        ]
        data["finish_reason"] = event.finish_reason
        data["usage"] = event.usage
        data["latency_ms"] = event.latency_ms
    elif isinstance(event, ToolStartedEvent):
        data["iteration"] = event.iteration
        data["tool_call_id"] = event.tool_call_id
        data["tool_name"] = event.tool_name
        data["input"] = event.input
    elif isinstance(event, ToolFinishedEvent):
        data["iteration"] = event.iteration
        data["tool_call_id"] = event.tool_call_id
        data["tool_name"] = event.tool_name
        data["output"] = event.output
        data["is_error"] = event.is_error
        data["latency_ms"] = event.latency_ms
    elif isinstance(event, LoopWarningEvent):
        data["iteration"] = event.iteration
        data["tool_name"] = event.tool_name
        data["consecutive_count"] = event.consecutive_count
        data["message"] = event.message
    elif isinstance(event, MistakeLimitHitEvent):
        data["iteration"] = event.iteration
        data["consecutive_mistakes"] = event.consecutive_mistakes
        data["max_consecutive_mistakes"] = event.max_consecutive_mistakes
        data["reason"] = event.reason
    elif isinstance(event, RunCompletedEvent):
        data["final_content"] = event.final_content
        data["iterations"] = event.iterations
        data["total_usage"] = event.total_usage
    elif isinstance(event, RunAbortedEvent):
        data["reason"] = event.reason
    elif isinstance(event, RunFailedEvent):
        data["error"] = event.error
        data["iterations"] = event.iterations
    return data


def _status_to_exit_code(status: str) -> int:
    """Map a :class:`RunResult.status` to a process exit code."""

    return {
        "completed": 0,
        "aborted": 130,  # 130 = SIGINT-like
        "failed": 1,
        "stopped": 2,
    }.get(status, 1)


@app.command(name="agent")
def agent(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="用户的指令（用引号包起来）"),
    mode: str | None = typer.Option(
        None, "--mode", help="初始模式：act / plan / yolo（默认 act）"
    ),
    plan: bool = typer.Option(False, "--plan", help="Plan 模式快捷方式（等价 --mode plan）"),
    yolo: bool = typer.Option(
        False, "--yolo", help="Yolo 模式快捷方式（等价 --mode yolo，禁用权限提示）"
    ),
    max_iterations: int = typer.Option(
        50, "--max-iterations", help="最大迭代次数（默认 50）"
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="LLM 响应最大 token 数"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", help="LLM 采样温度（默认 0.7）"
    ),
    system_prompt: str | None = typer.Option(
        None, "--system-prompt", help="覆盖默认系统提示词"
    ),
    model: str | None = typer.Option(None, "--model", help="覆盖配置中的模型名"),
    api_key: str | None = typer.Option(
        None, "--api-key", help="覆盖配置中的 API key"
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="覆盖配置中的 base URL"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="以 NDJSON 流式输出事件（适合程序化消费）"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过权限提示（启用 Yolo 模式）"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
) -> None:
    """运行本地 AI 智能体循环。"""

    # Defensive: typer may pass OptionInfo when called directly in tests.
    if not isinstance(prompt, str) or not prompt.strip():
        raise typer.BadParameter("Prompt is required")
    mode = mode if isinstance(mode, str) else None
    model = model if isinstance(model, str) else None
    api_key = api_key if isinstance(api_key, str) else None
    base_url = base_url if isinstance(base_url, str) else None
    system_prompt = system_prompt if isinstance(system_prompt, str) else None
    max_tokens = max_tokens if isinstance(max_tokens, int) else None
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        max_iterations = 50
    if not isinstance(temperature, (int, float)) or temperature < 0:
        temperature = 0.7

    config: AppConfig = ctx.obj["config"]
    global_yes = ctx.obj.get("yes", False) or yes
    global_verbose = ctx.obj.get("verbose", False) or verbose

    initial_mode = _resolve_mode(
        mode_flag=mode,
        plan_flag=plan,
        yolo_flag=yolo,
        yes_flag=global_yes,
    )

    # Build the LLM client from CLI flags or config.
    client = _resolve_llm(
        config,
        model_override=model,
        api_key_override=api_key,
        base_url_override=base_url,
    )

    # Build the tool registry and runtime.
    tools = make_default_tools()
    runtime_config = AgentRuntimeConfig(
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        max_iterations=max_iterations,
        temperature=temperature,
        max_tokens=max_tokens,
        initial_mode=initial_mode,
    )
    emit = _make_emit_callback(json_mode=json_output, verbose=global_verbose)
    runtime = AgentRuntime(
        client=client,
        tools=tools,
        config=runtime_config,
        cwd=str(config.project_root),
        emit=emit,
    )

    # Audit log start (best-effort).
    audit = ctx.obj.get("audit_logger")
    if audit is not None:
        audit.record(
            "agent.start",
            {
                "prompt": prompt[:500],  # truncate for log safety
                "mode": initial_mode.value,
                "max_iterations": max_iterations,
                "model": model or "(from config)",
            },
        )

    # Run the agent loop.
    try:
        result = asyncio.run(runtime.run(prompt))
    except KeyboardInterrupt:
        if audit is not None:
            audit.record("agent.interrupted", {})
        typer.secho("\n[interrupted]", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=130) from None
    except AutoShipError as exc:
        if audit is not None:
            audit.record("agent.error", {"error": str(exc)})
        typer.secho(f"[error] {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    # In pretty mode, surface the final content if the runtime didn't
    # already print it via the assistant-message event.
    if not json_output and result.final_content:
        # Avoid double-printing: the last AssistantMessageEvent already
        # printed the content. Only print if there were no iterations
        # (defensive — should never happen) or as a separator.
        pass

    # Audit log completion.
    if audit is not None:
        audit.record(
            "agent.done",
            {
                "status": result.status,
                "iterations": result.iterations,
                "total_tokens": result.total_usage.get("total_tokens", 0),
                "stop_reason": result.stop_reason,
                "error": result.error,
            },
        )

    # In JSON mode, emit a final ``result`` envelope so consumers can
    # detect the run outcome without parsing every event.
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "type": "result",
                    "status": result.status,
                    "final_content": result.final_content,
                    "iterations": result.iterations,
                    "total_usage": result.total_usage,
                    "error": result.error,
                    "stop_reason": result.stop_reason,
                },
                ensure_ascii=False,
            )
        )
    else:
        if result.status != "completed":
            typer.secho(
                f"\n[{result.status}] {result.error or result.stop_reason}",
                fg=typer.colors.YELLOW,
                err=True,
            )

    raise typer.Exit(code=_status_to_exit_code(result.status))


__all__ = ["DEFAULT_SYSTEM_PROMPT", "agent", "app", "register"]
