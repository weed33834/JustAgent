"""The ``myagent agent`` command — run the local-first AI agent loop.

Wraps :class:`myagent.agent.runtime.AgentRuntime` in a Typer command so
users can drive the agent from the CLI:

* ``myagent agent "list files in this dir"`` — default Act mode, tools
  available, prompts before destructive ops (or ``--yes`` to skip).
* ``myagent agent --plan "explore the codebase and propose a refactor"``
  — Plan mode: edit tools hidden, read-only investigation only.
* ``myagent agent --yolo "fix the failing tests"`` — Yolo mode: all
  tools, no permission prompts.
* ``myagent agent --json "..."`` — emit NDJSON events on stdout for
  headless / programmatic consumption (mirrors Cline's ``--json`` and
  OpenCode's ``session event stream``).

LLM credentials are resolved in this priority order:

1. ``--model`` / ``--api-key`` / ``--base-url`` CLI flags (highest).
2. ``config.model.backends[*]`` (preferred new-style config).
3. ``config.llm`` (legacy single-backend section, also bridged by
   :func:`myagent.cli.commands.fix._model_router`).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import typer

from myagent.agent.change_tracker import ChangeTracker
from myagent.agent.plan_act import AgentMode
from myagent.agent.runtime import (
    AgentRuntime,
    AgentRuntimeConfig,
    AssistantMessageEvent,
    LLMClient,
    LoopWarningEvent,
    MistakeLimitHitEvent,
    RunAbortedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunResult,
    RunStartedEvent,
    RuntimeEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
    TurnStartedEvent,
)
from myagent.agent.session import Session, SessionError, get_session_store
from myagent.agent.slash_commands import CommandAction, create_default_registry
from myagent.agent.tools.builtin import make_default_tools
from myagent.cli.display import RichDisplay
from myagent.exceptions import MyAgentError
from myagent.models.config import AppConfig, LlmProvider, Provider
from myagent.permissions import (
    PermissionAction,
    PermissionEngine,
    PermissionScope,
    create_act_mode_engine,
    create_plan_mode_engine,
    create_yolo_mode_engine,
)

app = typer.Typer()


#: Default system prompt — mirrors Cline's "You are Roo, a highly skilled
#: software engineer..." preamble, trimmed for myagent's scope.
DEFAULT_SYSTEM_PROMPT = (
    "You are myagent's built-in agent — a highly skilled software "
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
                "Model backend has no `model` field — pass --model or fix .myagent.toml"
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
            "No API key configured — pass --api-key or set [llm].api_key in .myagent.toml"
        )
    return LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=llm.api_version,
        timeout=llm.timeout,
        provider=provider.value,
    )


def _make_emit_callback(
    *,
    json_mode: bool,
    verbose: bool,
    display: RichDisplay | None = None,
    change_tracker: ChangeTracker | None = None,
) -> Any:
    """Build an event-emitter callback for the runtime.

    In ``--json`` mode, every event is serialized as one NDJSON line on
    stdout (headless-friendly). In pretty mode, the :class:`RichDisplay`
    instance renders spinners, panels, and summary tables.

    Parameters
    ----------
    display
        Optional pre-constructed :class:`RichDisplay`. If ``None``, a
        default one is created from ``json_mode`` / ``verbose``.
    change_tracker
        Optional :class:`ChangeTracker` for the run-summary. If ``None``,
        no files-changed info is shown.
    """

    _display = display or RichDisplay(verbose=verbose, json_mode=json_mode)
    _change_tracker = change_tracker
    _run_start: list[float] = [0.0]

    async def emit(event: RuntimeEvent) -> None:
        if json_mode:
            typer.echo(json.dumps(_event_to_dict(event), ensure_ascii=False))
            return

        if isinstance(event, RunStartedEvent):
            _run_start[0] = event.timestamp
            if verbose:
                _display.print_info(f"[run {event.run_id}] started")
        elif isinstance(event, TurnStartedEvent):
            if verbose:
                _display.print_info(f"── turn {event.iteration} ──")
            _display.start_spinner("Thinking...")
        elif isinstance(event, AssistantMessageEvent):
            _display.stop_spinner()
            if event.content:
                _display.print_assistant_message(event.content)
            if event.tool_calls and verbose:
                for tc in event.tool_calls:
                    _display.print_info(
                        f"  → calling {tc.name}({tc.input})"
                    )
        elif isinstance(event, ToolStartedEvent):
            _display.stop_spinner()
            _display.print_tool_start(event.tool_name, event.input)
        elif isinstance(event, ToolFinishedEvent):
            _display.print_tool_result(
                event.tool_name,
                event.output,
                event.is_error,
                event.latency_ms,
            )
        elif isinstance(event, LoopWarningEvent):
            _display.print_warning(f"loop warning: {event.message}")
        elif isinstance(event, MistakeLimitHitEvent):
            _display.print_warning(
                f"mistake limit hit ({event.consecutive_mistakes}/"
                f"{event.max_consecutive_mistakes}): {event.reason}"
            )
        elif isinstance(event, RunCompletedEvent):
            _display.stop_spinner()
            files_changed = (
                _change_tracker.get_changed_files()
                if _change_tracker is not None
                else []
            )
            elapsed = (
                event.timestamp - _run_start[0]
                if _run_start[0] > 0
                else 0.0
            )
            _display.print_run_summary(
                iterations=event.iterations,
                total_tokens=int(event.total_usage.get("total_tokens", 0)),
                elapsed_seconds=elapsed,
                files_changed=files_changed,
            )
            if _change_tracker is not None and _change_tracker.get_changes():
                _display.print_change_summary(
                    [
                        {
                            "path": c.path,
                            "action": c.action,
                            "lines_added": c.lines_added,
                            "lines_removed": c.lines_removed,
                        }
                        for c in _change_tracker.get_changes()
                    ]
                )
        elif isinstance(event, RunAbortedEvent):
            _display.stop_spinner()
            _display.print_warning(f"[aborted] {event.reason}")
        elif isinstance(event, RunFailedEvent):
            _display.stop_spinner()
            _display.print_error(f"[failed] {event.error}")

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


# ---------------------------------------------------------------------------
# Interactive REPL helpers
# ---------------------------------------------------------------------------


def _print_welcome_banner(
    *, mode: str, model: str, cwd: str, json_mode: bool
) -> None:
    """Print the interactive-mode welcome banner.

    Skipped in JSON mode (headless consumers don't need a banner).
    """

    if json_mode:
        return
    width = 50
    typer.secho(
        "╭" + "─" * (width - 2) + "╮", fg=typer.colors.CYAN, bold=True
    )
    for line in (
        " MyAgent Agent (interactive)",
        f" Mode: {mode} | Model: {model} | CWD: {cwd}",
        " Type /help for commands, /exit to quit",
    ):
        pad = width - 2 - _display_width(line)
        if pad < 0:
            pad = 0
        typer.secho(
            "│" + line + " " * pad + "│", fg=typer.colors.CYAN, bold=True
        )
    typer.secho(
        "╰" + "─" * (width - 2) + "╯", fg=typer.colors.CYAN, bold=True
    )


def _display_width(text: str) -> int:
    """Approximate display width (CJK chars count as 2)."""

    return sum(2 if ord(c) > 0x2E7F else 1 for c in text)


def _print_result(result: RunResult, *, json_mode: bool, turn: int) -> None:
    """Print a run result and a brief turn summary.

    In JSON mode, emits a ``result`` envelope as NDJSON. In pretty mode,
    prints a dim ``[turn N done, M tokens]`` line (the assistant content
    was already printed by the emit callback).
    """

    if json_mode:
        typer.echo(
            json.dumps(
                {
                    "type": "result",
                    "turn": turn,
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
        return

    tokens = result.total_usage.get("total_tokens", 0)
    typer.secho(
        f"[turn {turn} done] {result.iterations} iteration(s), "
        f"{tokens} tokens",
        fg=typer.colors.GREEN,
        dim=True,
    )
    if result.status != "completed":
        typer.secho(
            f"  [{result.status}] {result.error or result.stop_reason}",
            fg=typer.colors.YELLOW,
            err=True,
        )


async def _run_interactive(
    runtime: AgentRuntime,
    initial_prompt: str,
    slash_registry: Any,
    *,
    json_mode: bool,
    verbose: bool,
    model_name: str,
    cwd: str,
) -> None:
    """Run the interactive REPL loop.

    Reads input line-by-line from stdin. Lines starting with ``/`` are
    treated as slash commands; everything else is forwarded to
    :meth:`AgentRuntime.continue_run`.

    Ctrl+C / Ctrl+D handling:
        * During a running turn → abort the turn, return to the prompt.
        * At the prompt → exit the REPL cleanly.
    """

    _print_welcome_banner(
        mode=runtime.mode.value,
        model=model_name,
        cwd=cwd,
        json_mode=json_mode,
    )

    turn = 0

    # Run the initial prompt (if any) as a fresh ``run`` to seed the
    # system prompt and conversation history.
    if initial_prompt.strip():
        turn += 1
        try:
            result = await runtime.run(initial_prompt)
            _print_result(result, json_mode=json_mode, turn=turn)
        except KeyboardInterrupt:
            typer.secho(
                "\n[turn aborted]", fg=typer.colors.YELLOW, err=True
            )
            runtime.abort()
            runtime._abort.clear()  # reset for next turn

    # REPL loop.
    while True:
        try:
            if json_mode:
                line = input()
                if not line:
                    break
            else:
                typer.secho(
                    "\nmyagent> ", fg=typer.colors.CYAN, bold=True, nl=False
                )
                line = input()
        except (EOFError, KeyboardInterrupt):
            typer.secho("\nGoodbye!", fg=typer.colors.YELLOW)
            break

        line = line.strip()
        if not line:
            continue

        # --- Slash commands ---
        cmd_result = slash_registry.execute(
            line,
            {
                "current_mode": runtime.mode.value,
                "runtime": runtime,
                "registry": slash_registry,
                # Pass live state so commands like /tokens, /history,
                # /diff, /lint, /test can report on the current session.
                "messages": runtime.messages,
                "token_usage": runtime.total_usage,
                "cwd": cwd,
                "tools": list(runtime.tools),
            },
        )
        if cmd_result is not None:
            if cmd_result.action == CommandAction.EXIT:
                typer.secho("Goodbye!", fg=typer.colors.YELLOW)
                break
            elif cmd_result.action == CommandAction.CLEAR_HISTORY:
                runtime.reset()
                typer.secho(
                    "Conversation cleared.", fg=typer.colors.GREEN
                )
                continue
            elif cmd_result.action == CommandAction.SWITCH_MODE:
                new_mode = cmd_result.data.get("mode", "act")
                runtime.switch_mode(AgentMode(new_mode))
                typer.secho(
                    f"Switched to {new_mode} mode.",
                    fg=typer.colors.GREEN,
                )
                continue
            elif cmd_result.action == CommandAction.COMPACT:
                typer.secho(
                    "Compacting context...", fg=typer.colors.YELLOW
                )
                continue
            elif cmd_result.action == CommandAction.UNDO:
                typer.secho(
                    "Undoing last change...", fg=typer.colors.YELLOW
                )
                continue
            elif cmd_result.message:
                typer.echo(cmd_result.message)
                continue
            else:
                continue

        # --- Regular user input — continue the conversation ---
        turn += 1
        try:
            result = await runtime.continue_run(line)
            _print_result(result, json_mode=json_mode, turn=turn)
        except KeyboardInterrupt:
            typer.secho(
                "\n[turn aborted]", fg=typer.colors.YELLOW, err=True
            )
            runtime.abort()
            runtime._abort.clear()  # reset for next turn
        except MyAgentError as exc:
            typer.secho(
                f"  [error] {exc}", fg=typer.colors.RED, err=True
            )


@app.command(name="agent")
def agent(
    ctx: typer.Context,
    prompt: str = typer.Argument("", help="用户的指令（用引号包起来；交互模式下可省略）"),
    mode: str | None = typer.Option(
        None, "--mode", help="初始模式：act / plan / yolo（默认 act）"
    ),
    plan: bool = typer.Option(False, "--plan", help="Plan 模式快捷方式（等价 --mode plan）"),
    yolo: bool = typer.Option(
        False, "--yolo", help="Yolo 模式快捷方式（等价 --mode yolo，禁用权限提示）"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="交互式多轮聊天模式（REPL）"
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
    resume: str | None = typer.Option(
        None, "--resume", help="恢复指定会话（加载历史后进入交互模式）"
    ),
) -> None:
    """运行本地 AI 智能体循环。

    一次性模式：``myagent agent "task"``
    交互模式：``myagent agent -i`` 或 ``myagent agent -i "initial task"``
    恢复模式：``myagent agent -i --resume <SESSION_ID>``
    """

    # Defensive: typer may pass OptionInfo when called directly in tests.
    prompt = prompt if isinstance(prompt, str) else ""
    mode = mode if isinstance(mode, str) else None
    model = model if isinstance(model, str) else None
    api_key = api_key if isinstance(api_key, str) else None
    base_url = base_url if isinstance(base_url, str) else None
    system_prompt = system_prompt if isinstance(system_prompt, str) else None
    resume = resume if isinstance(resume, str) else None
    max_tokens = max_tokens if isinstance(max_tokens, int) else None
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        max_iterations = 50
    if not isinstance(temperature, (int, float)) or temperature < 0:
        temperature = 0.7
    interactive = bool(interactive)

    # ``--resume`` implies interactive mode (the restored history is
    # continued via ``continue_run`` in the REPL).
    if resume:
        interactive = True

    # In non-interactive mode, a prompt is required.
    if not interactive and not prompt.strip():
        raise typer.BadParameter(
            "Prompt is required (or use --interactive / -i for REPL mode)"
        )

    config: AppConfig = ctx.obj["config"]
    global_yes = ctx.obj.get("yes", False) or yes
    global_verbose = ctx.obj.get("verbose", False) or verbose

    initial_mode = _resolve_mode(
        mode_flag=mode,
        plan_flag=plan,
        yolo_flag=yolo,
        yes_flag=global_yes,
    )

    # Session persistence: load an existing session to resume, or create a
    # fresh one so the conversation can be saved and resumed later.
    session_store = get_session_store()
    resume_session: Session | None = None
    interactive_seed_prompt = prompt
    if resume:
        try:
            resume_session = session_store.load(resume)
        except SessionError as exc:
            raise typer.BadParameter(str(exc)) from exc
        # Prefer the resumed session's model/mode unless overridden on the CLI.
        if not model:
            model = resume_session.metadata.model
        initial_mode = AgentMode(resume_session.metadata.mode)
        # Don't re-seed the conversation — the REPL uses ``continue_run``
        # on the restored history, so the initial prompt is skipped.
        interactive_seed_prompt = ""

    # Build the LLM client from CLI flags or config.
    client = _resolve_llm(
        config,
        model_override=model,
        api_key_override=api_key,
        base_url_override=base_url,
    )

    # Build the tool registry and runtime.
    tools = make_default_tools()
    # Create a shared change tracker so both the runtime and the
    # display can access the same instance for the run summary.
    change_tracker = ChangeTracker()
    display = RichDisplay(verbose=global_verbose, json_mode=json_output)
    runtime_config = AgentRuntimeConfig(
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        max_iterations=max_iterations,
        temperature=temperature,
        max_tokens=max_tokens,
        initial_mode=initial_mode,
        change_tracker=change_tracker,
    )
    emit = _make_emit_callback(
        json_mode=json_output,
        verbose=global_verbose,
        display=display,
        change_tracker=change_tracker,
    )

    # Wire up the permission engine based on the resolved mode. The
    # engine decides whether each destructive tool call is allowed,
    # denied, or must prompt the user. The ``_ask`` callback below is
    # passed to the runtime so tools can consult the engine via
    # ``ctx.request_permission()``.
    if initial_mode is AgentMode.YOLO:
        engine: PermissionEngine = create_yolo_mode_engine()
    elif initial_mode is AgentMode.PLAN:
        engine = create_plan_mode_engine()
    else:
        engine = create_act_mode_engine()

    async def _ask(request: dict[str, Any]) -> bool:
        """Permission callback wired to :class:`PermissionEngine`."""

        tool = request.get("tool", "")
        tool_input: dict[str, Any] = {}
        if "path" in request:
            tool_input["path"] = request["path"]
        elif "command" in request:
            tool_input["command"] = request["command"]

        decision = engine.check(tool, tool_input)
        if decision.action == PermissionAction.ALLOW:
            return True
        if decision.action == PermissionAction.DENY:
            return False

        # ASK — prompt the user. In non-interactive modes (JSON output
        # or --yes/--yolo) we auto-approve since there's no one to ask.
        if json_output or global_yes:
            return True

        description = request.get(
            "description",
            f"{tool} on {request.get('path', request.get('command', '?'))}",
        )
        approved = display.print_permission_prompt(tool, description)
        if approved:
            remember = typer.confirm(
                "  Remember for this session?", default=False
            )
            if remember:
                engine.remember(
                    tool,
                    tool_input,
                    PermissionAction.ALLOW,
                    PermissionScope.ALWAYS,
                )
        else:
            remember = typer.confirm(
                "  Remember denial for this session?", default=False
            )
            if remember:
                engine.remember(
                    tool,
                    tool_input,
                    PermissionAction.DENY,
                    PermissionScope.ALWAYS,
                )
        return approved

    # Build (or reuse) the conversation session so it can be saved and
    # later resumed via ``myagent session resume <ID>``.
    if resume_session is not None:
        session = resume_session
    else:
        session = session_store.create_session(
            mode=initial_mode.value,
            model=client.model,
            cwd=str(config.project_root),
            initial_prompt=prompt,
        )

    runtime = AgentRuntime(
        client=client,
        tools=tools,
        config=runtime_config,
        cwd=str(config.project_root),
        emit=emit,
        ask=_ask,
        session=session,
        session_store=session_store,
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
                "interactive": interactive,
            },
        )

    # Interactive REPL mode — multi-turn conversation.
    if interactive:
        slash_registry = create_default_registry()
        model_label = model or "(from config)"
        try:
            asyncio.run(
                _run_interactive(
                    runtime,
                    interactive_seed_prompt,
                    slash_registry,
                    json_mode=json_output,
                    verbose=global_verbose,
                    model_name=model_label,
                    cwd=str(config.project_root),
                )
            )
        except KeyboardInterrupt:
            if audit is not None:
                audit.record("agent.interrupted", {})
            typer.secho("\nGoodbye!", fg=typer.colors.YELLOW)
            raise typer.Exit(code=130) from None
        except MyAgentError as exc:
            if audit is not None:
                audit.record("agent.error", {"error": str(exc)})
            typer.secho(f"[error] {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        if audit is not None:
            audit.record(
                "agent.done",
                {
                    "status": "completed",
                    "interactive": True,
                    "total_tokens": runtime._total_usage.get(
                        "total_tokens", 0
                    ),
                },
            )
        raise typer.Exit(code=0)

    # One-shot mode — run the agent loop once and exit.
    try:
        result = asyncio.run(runtime.run(prompt))
    except KeyboardInterrupt:
        if audit is not None:
            audit.record("agent.interrupted", {})
        typer.secho("\n[interrupted]", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=130) from None
    except MyAgentError as exc:
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
