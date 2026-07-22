"""Agent runtime — the iterative loop that drives the LLM ↔ tool cycle.

Ports the core loop from Cline's ``AgentRuntime.execute()``
(``competitors/cline/sdk/packages/agents/src/agent-runtime.ts``) and
OpenCode's ``runLoop``
(``competitors/opencode/packages/opencode/src/session/prompt.ts``).

Design:

* :class:`AgentRuntime` owns the conversation state, tool registry, and
  the LLM client. :meth:`AgentRuntime.run` is the main entry point —
  it loops: send messages → get assistant response → if tool calls,
  execute them and feed results back → repeat until the assistant
  responds without tool calls (or hits the iteration cap).
* :class:`LLMClient` is a thin wrapper around LiteLLM that supports
  tool calling. It's decoupled from the existing
  :class:`autoship.adapters.model_gateway.ModelGateway` because that
  abstraction doesn't expose tool-calling fields.
* Messages use a simple dataclass model (:class:`Message`,
  :class:`ToolCall`, :class:`ToolResultPart`) that mirrors the
  OpenAI/Cline tool-calling message format.
* Abort is via :class:`asyncio.Event` (matching
  :class:`autoship.agent.tools.base.ToolContext.abort`).
* Events (:class:`RuntimeEvent` subclasses) are emitted at key points
  (turn started, assistant message, tool started/finished, run
  completed/aborted/failed) — mirroring Cline's ``emit()`` and
  OpenCode's ``SessionEvent``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from autoship.agent.change_tracker import ChangeTracker
from autoship.agent.loop_detection import (
    LoopDetectionCall,
    LoopDetectionConfig,
    LoopDetectionTracker,
)
from autoship.agent.mistake_tracker import (
    MistakeOutcome,
    MistakeReason,
    MistakeTracker,
    MistakeTrackerOptions,
    RecordMistakeInput,
)
from autoship.agent.plan_act import (
    AgentMode,
    ModeConfig,
    build_system_prompt,
    filter_tools_for_mode,
    format_user_message,
)
from autoship.agent.session import (
    Session,
    SessionError,
    SessionMetadata,
    SessionStatus,
    SessionStore,
    deserialize_message,
    serialize_message,
)
from autoship.agent.tools.base import (
    InvalidArgumentsError,
    Tool,
    ToolAbortedError,
    ToolContext,
    ToolError,
    ToolResult,
    ToolTimeoutError,
)
from autoship.agent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# No-op callbacks (named so mypy can infer their types — lambdas can't)
# ---------------------------------------------------------------------------


def _noop_emit(event: dict[str, Any]) -> None:
    """No-op event sink for MistakeTrackerOptions."""


def _noop_log(level: str, msg: str, meta: dict[str, Any] | None = None) -> None:
    """No-op leveled-log sink for MistakeTrackerOptions."""


def _noop_recovery_notice(msg: str, reason: MistakeReason) -> None:
    """No-op recovery-notice sink for MistakeTrackerOptions."""


def _noop_get_id() -> str:
    """No-op ID getter for MistakeTrackerOptions."""

    return ""


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM.

    Mirrors OpenAI's ``tool_calls`` schema and Cline's
    ``AgentToolCallPart``.
    """

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultPart:
    """The result of a tool call, to be sent back to the LLM.

    Mirrors OpenAI's ``role: "tool"`` message and Cline's
    ``AgentToolResultPart``.
    """

    tool_call_id: str
    name: str
    output: str
    is_error: bool = False


@dataclass
class Message:
    """A conversation message.

    ``role`` is one of ``"system"``, ``"user"``, ``"assistant"``, or
    ``"tool"``. ``content`` is the text body (may be empty for
    assistant messages that only contain tool calls).
    ``tool_calls`` is set on assistant messages that request tools.
    ``tool_result`` is set on ``role="tool"`` messages.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: ToolResultPart | None = None
    #: Name of the tool this ``role="tool"`` message answers (optional).
    name: str | None = None
    #: Arbitrary metadata (e.g. ``{"iteration": 3}``).
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to an OpenAI-compatible message dict."""

        if self.role == "tool":
            assert self.tool_result is not None
            return {
                "role": "tool",
                "tool_call_id": self.tool_result.tool_call_id,
                "content": self.tool_result.output,
            }
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


# ---------------------------------------------------------------------------
# LLM client (LiteLLM-backed)
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """The assistant's response from a single LLM call."""

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0


@dataclass
class LLMRequest:
    """A request to the LLM."""

    messages: list[Message]
    tools: list[Tool]
    temperature: float = 0.7
    max_tokens: int | None = None


class LLMClient:
    """Thin LiteLLM wrapper supporting tool calls.

    Decoupled from :class:`autoship.adapters.model_gateway.ModelGateway`
    because that abstraction doesn't expose tool-calling fields. Uses
    LiteLLM's ``acompletion`` (async) so the runtime can be aborted
    mid-call via :class:`asyncio.wait`.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        timeout: float | None = None,
        provider: str = "openai",
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._api_version = api_version
        self._timeout = timeout
        self._provider = provider

    @property
    def model(self) -> str:
        """The model identifier this client is configured to use."""

        return self._model

    async def complete(
        self,
        request: LLMRequest,
        *,
        abort: asyncio.Event | None = None,
    ) -> LLMResponse:
        """Send ``request`` and return the assistant's response.

        If ``abort`` is set and fires before the response arrives,
        :class:`asyncio.CancelledError` is raised.
        """

        import litellm

        litellm.drop_params = True
        litellm.suppress_debug_info = True

        messages = [m.to_dict() for m in request.messages]
        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": tool.id,
                    "description": tool.description,
                    "parameters": tool.json_schema(),
                },
            }
            for tool in request.tools
        ]

        model_str = (
            f"{self._provider}/{self._model}"
            if self._provider and "/" not in self._model
            else self._model
        )

        kwargs: dict[str, Any] = {
            "model": model_str,
            "messages": messages,
            "temperature": request.temperature,
            "api_base": self._base_url,
            "api_key": self._api_key,
            "api_version": self._api_version,
            "timeout": self._timeout,
            "tools": tools_schema if tools_schema else None,
            "tool_choice": "auto" if tools_schema else None,
        }
        # Strip None values — LiteLLM prefers absent over None.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        start = time.time()
        coro = litellm.acompletion(**kwargs)

        if abort is not None:
            done, pending = await asyncio.wait(
                {asyncio.ensure_future(coro), asyncio.ensure_future(abort.wait())},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if abort.is_set():
                raise asyncio.CancelledError("LLM call aborted by user")
            response = await next(iter(done))
        else:
            response = await coro

        latency_ms = (time.time() - start) * 1000
        return self._parse_response(response, latency_ms)

    @staticmethod
    def _parse_response(response: Any, latency_ms: float) -> LLMResponse:
        """Convert a LiteLLM response into :class:`LLMResponse`."""

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(msg, "tool_calls", None) or []
        for raw in raw_tool_calls:
            try:
                args = json.loads(raw.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": raw.function.arguments}
            tool_calls.append(
                ToolCall(
                    id=raw.id,
                    name=raw.function.name,
                    input=args,
                )
            )

        usage: dict[str, Any] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            model=response.model or "",
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Runtime events
# ---------------------------------------------------------------------------


@dataclass
class RuntimeEvent:
    """Base class for runtime events.

    Mirrors Cline's ``AgentEvent`` union and OpenCode's
    ``SessionEvent``. Subscribers receive these via the ``emit``
    callback.
    """

    type: str
    run_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class RunStartedEvent(RuntimeEvent):
    iteration: int = 0


@dataclass(kw_only=True)
class TurnStartedEvent(RuntimeEvent):
    iteration: int


@dataclass
class AssistantMessageEvent(RuntimeEvent):
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(kw_only=True)
class ToolStartedEvent(RuntimeEvent):
    iteration: int
    tool_call_id: str
    tool_name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ToolFinishedEvent(RuntimeEvent):
    iteration: int
    tool_call_id: str
    tool_name: str
    output: str = ""
    is_error: bool = False
    latency_ms: float = 0.0


@dataclass(kw_only=True)
class LoopWarningEvent(RuntimeEvent):
    iteration: int
    tool_name: str
    consecutive_count: int
    message: str = ""


@dataclass
class RunCompletedEvent(RuntimeEvent):
    final_content: str = ""
    iterations: int = 0
    total_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunAbortedEvent(RuntimeEvent):
    reason: str = ""


@dataclass
class RunFailedEvent(RuntimeEvent):
    error: str = ""
    iterations: int = 0


@dataclass(kw_only=True)
class MistakeLimitHitEvent(RuntimeEvent):
    iteration: int
    consecutive_mistakes: int
    max_consecutive_mistakes: int
    reason: str = ""


EventEmitter = Callable[[RuntimeEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------


@dataclass
class AgentRuntimeConfig:
    """Configuration for :class:`AgentRuntime`."""

    #: The system prompt (role=system message prepended to every request).
    system_prompt: str = ""
    #: Maximum number of iterations before the run is aborted.
    max_iterations: int = 50
    #: Temperature for LLM calls.
    temperature: float = 0.7
    #: Max tokens for LLM responses.
    max_tokens: int | None = None
    #: Max consecutive mistakes before the run stops.
    max_consecutive_mistakes: int = 3
    #: Loop detection soft threshold (warning).
    loop_soft_threshold: int = 3
    #: Loop detection hard threshold (stop).
    loop_hard_threshold: int = 5
    #: Whether to allow parallel tool execution. Defaults to False
    #: (sequential) for safety and determinism — mirrors Cline's
    #: default ``toolExecution: "sequential"``.
    parallel_tool_execution: bool = False
    #: Whether the runtime should auto-abort when the hard loop
    #: threshold is reached.
    abort_on_hard_loop: bool = True
    #: Initial agent mode (act/plan/yolo). The user can switch modes
    #: at runtime via :meth:`AgentRuntime.switch_mode`.
    initial_mode: AgentMode = AgentMode.ACT
    #: Optional change tracker. If ``None``, a new :class:`ChangeTracker`
    #: is created automatically. Set to ``None`` explicitly to disable
    #: tracking (the runtime still creates one — pass a pre-populated
    #: instance only if you need to share state across runs).
    change_tracker: ChangeTracker | None = None


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Outcome of a :meth:`AgentRuntime.run` call."""

    status: Literal["completed", "aborted", "failed", "stopped"]
    final_content: str = ""
    iterations: int = 0
    messages: list[Message] = field(default_factory=list)
    total_usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    stop_reason: str = ""


# ---------------------------------------------------------------------------
# AgentRuntime
# ---------------------------------------------------------------------------


class AgentRuntime:
    """The iterative agent loop.

    Usage::

        runtime = AgentRuntime(
            client=LLMClient(model="gpt-4o", api_key="..."),
            tools=make_default_tools(),
            config=AgentRuntimeConfig(system_prompt="You are..."),
        )
        result = await runtime.run("Hello, what files are in this dir?")
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        tools: list[Tool] | ToolRegistry,
        config: AgentRuntimeConfig | None = None,
        cwd: str = ".",
        emit: EventEmitter | None = None,
        ask: Callable[[dict[str, Any]], Awaitable[bool]] | None = None,
        session: Session | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self._client = client
        if isinstance(tools, ToolRegistry):
            self._registry = tools
        else:
            self._registry = ToolRegistry()
            for tool in tools:
                self._registry.register(tool)
        self._config = config or AgentRuntimeConfig()
        self._cwd = cwd
        self._emit = emit
        self._ask = ask
        self._mode_config = ModeConfig(mode=self._config.initial_mode)

        self._messages: list[Message] = []
        self._abort = asyncio.Event()
        self._iteration = 0
        self._run_id = ""
        self._total_usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._loop_tracker = LoopDetectionTracker(
            config=LoopDetectionConfig(
                soft_threshold=self._config.loop_soft_threshold,
                hard_threshold=self._config.loop_hard_threshold,
            )
        )
        self._mistake_tracker: MistakeTracker | None = None
        self._change_tracker: ChangeTracker = (
            self._config.change_tracker
            if self._config.change_tracker is not None
            else ChangeTracker()
        )
        self._session: Session | None = session
        self._session_store: SessionStore | None = session_store
        if session is not None:
            # Restore conversation history from the persisted session so
            # ``continue_run`` picks up where the previous process left off.
            self._messages = [deserialize_message(m) for m in session.messages]
            # Defensive: a freshly-created session has ``usage={}``; fill in
            # the canonical keys so ``_accumulate_usage`` never hits a
            # ``KeyError`` on the first turn of an interactive REPL.
            self._total_usage = {
                "prompt_tokens": int(session.usage.get("prompt_tokens", 0)),
                "completion_tokens": int(
                    session.usage.get("completion_tokens", 0)
                ),
                "total_tokens": int(session.usage.get("total_tokens", 0)),
            }
            if not self._run_id:
                self._run_id = uuid.uuid4().hex[:16]

    # -- public API -------------------------------------------------------

    def abort(self) -> None:
        """Signal the current run to abort at the next check point."""

        self._abort.set()

    def switch_mode(self, new_mode: AgentMode) -> None:
        """Switch the agent's execution mode (act/plan/yolo).

        Records the switch in the tracker so a ``<mode_notice>`` is
        emitted on the next user message. Safe to call mid-run.
        """

        self._mode_config.switch_to(new_mode)

    @property
    def mode(self) -> AgentMode:
        """The current execution mode."""

        return self._mode_config.mode

    @property
    def messages(self) -> list[Message]:
        """Return a snapshot of the conversation history."""

        return list(self._messages)

    @property
    def total_usage(self) -> dict[str, Any]:
        """Return a snapshot of the accumulated token usage."""

        return dict(self._total_usage)

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def change_tracker(self) -> ChangeTracker:
        """The change tracker recording file modifications during runs."""

        return self._change_tracker

    @property
    def tools(self) -> list[Tool]:
        """Return all registered tools."""

        return list(self._registry.all())

    @property
    def session(self) -> Session | None:
        """The active session, if any (set when persistence is enabled)."""

        return self._session

    def get_session_metadata(self) -> SessionMetadata:
        """Build a :class:`SessionMetadata` snapshot from the current state.

        Requires an active session (``session=`` passed to the runtime).
        Raises :class:`SessionError` if no session is attached.
        """

        if self._session is None:
            raise SessionError("No active session")
        base = self._session.metadata
        return SessionMetadata(
            id=base.id,
            created_at=base.created_at,
            updated_at=time.time(),
            status=base.status,
            mode=self._mode_config.mode.value,
            model=base.model,
            cwd=self._cwd,
            prompt_preview=base.prompt_preview,
            iterations=self._iteration,
            total_tokens=int(self._total_usage.get("total_tokens", 0)),
            message_count=len(self._messages),
            files_changed=self._change_tracker.get_changed_files(),
        )

    def save_session(self) -> None:
        """Persist the current conversation state to the session store.

        No-op when no session or store is attached. Updates ``self._session``
        in place with the latest messages / usage / metadata so subsequent
        saves are idempotent.
        """

        if self._session is None or self._session_store is None:
            return
        metadata = self.get_session_metadata()
        self._session = Session(
            metadata=metadata,
            messages=[serialize_message(m) for m in self._messages],
            usage=dict(self._total_usage),
        )
        self._session_store.save(self._session)

    def _safe_save_session(self) -> None:
        """Best-effort session persistence — never aborts the run."""

        if self._session is None or self._session_store is None:
            return
        with contextlib.suppress(Exception):
            self.save_session()

    async def run(self, user_input: str) -> RunResult:
        """Run the agent loop with ``user_input`` as the initial prompt.

        Returns when the assistant responds without tool calls, the
        iteration cap is hit, or the run is aborted/fails.

        This is a **fresh** run: it resets the conversation history,
        usage counters, run id, and iteration counter. For multi-turn
        conversations that preserve history, use :meth:`continue_run`.
        """

        self._abort.clear()
        self._iteration = 0
        self._run_id = uuid.uuid4().hex[:16]
        self._messages = []
        self._total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._mistake_tracker = self._make_mistake_tracker()

        # Seed the conversation. Build the system prompt with mode-specific
        # instructions appended, and wrap the user message with the mode tag.
        system_prompt = build_system_prompt(
            self._config.system_prompt,
            self._mode_config.mode,
        )
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

        self._append_user_message(user_input)

        await self._emit_event(
            RunStartedEvent(type="run-started", run_id=self._run_id, iteration=0)
        )

        result = await self._run_loop()
        await self._emit_terminal_event(result)
        self._safe_save_session()
        return result

    async def continue_run(self, user_input: str) -> RunResult:
        """Continue the conversation without resetting history.

        Unlike :meth:`run`, this keeps the existing ``_messages``,
        ``_total_usage``, and ``_run_id`` so the LLM sees the full
        conversation context. Use this for multi-turn REPL sessions.

        The iteration counter is reset to 0 for this turn (each turn
        gets its own iteration budget), but the conversation history
        and accumulated token usage are preserved.
        """

        self._abort.clear()
        self._iteration = 0
        # Keep _run_id, _messages, _total_usage — the key difference
        # from run(). Give the new turn a fresh mistake budget.
        self._mistake_tracker = self._make_mistake_tracker()

        # If the conversation has no system prompt yet (e.g. the user
        # started the REPL without an initial prompt and ``run`` was
        # never called), seed one now so the LLM has the mode context.
        if not self._messages or self._messages[0].role != "system":
            system_prompt = build_system_prompt(
                self._config.system_prompt,
                self._mode_config.mode,
            )
            if system_prompt:
                self._messages.insert(
                    0, Message(role="system", content=system_prompt)
                )
            if not self._run_id:
                self._run_id = uuid.uuid4().hex[:16]

        self._append_user_message(user_input)

        await self._emit_event(
            RunStartedEvent(type="run-started", run_id=self._run_id, iteration=0)
        )

        result = await self._run_loop()
        await self._emit_terminal_event(result)
        self._safe_save_session()
        return result

    def reset(self) -> None:
        """Clear conversation history and usage stats for a fresh start.

        Does not change the run id or current mode — those persist for
        the lifetime of the runtime instance. Useful in interactive
        mode when the user issues ``/clear``.
        """

        self._messages = []
        self._iteration = 0
        self._total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._abort.clear()
        self._mistake_tracker = None

    async def _run_loop(self) -> RunResult:
        """The main LLM ↔ tool loop.

        Assumes ``self._messages`` is already seeded (system + user)
        and that ``self._run_id`` / ``self._mistake_tracker`` are set.
        Shared by :meth:`run` and :meth:`continue_run`.
        """

        try:
            while self._iteration < self._config.max_iterations:
                self._check_aborted()
                self._iteration += 1

                await self._emit_event(
                    TurnStartedEvent(
                        type="turn-started",
                        run_id=self._run_id,
                        iteration=self._iteration,
                    )
                )

                # --- 1. Call the LLM ---
                try:
                    response = await self._call_llm()
                except asyncio.CancelledError:
                    return self._aborted_result("LLM call cancelled")
                except Exception as exc:
                    # Record the mistake; if the tracker says stop, end the
                    # run. Otherwise continue the loop and retry the LLM
                    # call (the mistake tracker may have injected a
                    # recovery notice via its callbacks).
                    outcome = await self._handle_mistake(
                        MistakeReason.API_ERROR, str(exc)
                    )
                    if outcome.status in {"stopped", "failed"}:
                        return outcome
                    continue

                # --- 2. Append the assistant message ---
                assistant_msg = Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                    metadata={"iteration": self._iteration},
                )
                self._messages.append(assistant_msg)
                self._accumulate_usage(response.usage)

                await self._emit_event(
                    AssistantMessageEvent(
                        type="assistant-message",
                        run_id=self._run_id,
                        content=response.content,
                        tool_calls=response.tool_calls,
                        finish_reason=response.finish_reason,
                        usage=response.usage,
                        latency_ms=response.latency_ms,
                    )
                )

                # --- 3. If no tool calls, the run is complete ---
                if not response.tool_calls:
                    return self._completed_result(response.content)

                # --- 4. Execute the tool calls ---
                tool_results = await self._execute_tool_calls(
                    response.tool_calls
                )

                # --- 5. Check for terminal tool (completes_run) ---
                terminal = any(
                    self._is_terminal_tool(tc.name) for tc in response.tool_calls
                )
                if terminal:
                    final = (
                        tool_results[0].output
                        if tool_results
                        else response.content
                    )
                    return self._completed_result(final)

            # Iteration cap hit.
            return RunResult(
                status="stopped",
                final_content="",
                iterations=self._iteration,
                messages=list(self._messages),
                total_usage=dict(self._total_usage),
                error=f"Reached max iterations ({self._config.max_iterations})",
                stop_reason="max_iterations",
            )

        except ToolAbortedError:
            return self._aborted_result("Tool execution aborted")
        except asyncio.CancelledError:
            return self._aborted_result("Run cancelled")
        except Exception as exc:
            return RunResult(
                status="failed",
                error=str(exc),
                iterations=self._iteration,
                messages=list(self._messages),
                total_usage=dict(self._total_usage),
                stop_reason="exception",
            )

    # -- internal: LLM call ----------------------------------------------

    async def _call_llm(self) -> LLMResponse:
        """Send the conversation to the LLM and return the response."""

        # Filter tools by the current mode (Plan mode hides edit tools).
        available_tools = filter_tools_for_mode(
            list(self._registry.all()), self._mode_config.mode
        )
        request = LLMRequest(
            messages=list(self._messages),
            tools=available_tools,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return await self._client.complete(request, abort=self._abort)

    # -- internal: tool execution ----------------------------------------

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolResultPart]:
        """Execute each tool call and append the results to history."""

        if self._config.parallel_tool_execution:
            results: list[ToolResultPart] = list(
                await asyncio.gather(
                    *(self._execute_one(tc) for tc in tool_calls)
                )
            )
        else:
            results = []
            for tc in tool_calls:
                self._check_aborted()
                result = await self._execute_one(tc)
                results.append(result)

        # Append all tool results as messages.
        for result in results:
            self._messages.append(
                Message(role="tool", tool_result=result, name=result.name)
            )

        return results

    async def _execute_one(self, call: ToolCall) -> ToolResultPart:
        """Execute a single tool call, with loop detection + mistake tracking."""

        # --- Loop detection ---
        verdict = self._loop_tracker.inspect(
            LoopDetectionCall(name=call.name, input=call.input)
        )
        if verdict.kind == "soft":
            await self._emit_event(
                LoopWarningEvent(
                    type="loop-warning",
                    run_id=self._run_id,
                    iteration=self._iteration,
                    tool_name=call.name,
                    consecutive_count=self._loop_tracker_count(),
                    message=verdict.message or "",
                )
            )
        elif verdict.kind == "hard" and self._config.abort_on_hard_loop:
            return ToolResultPart(
                tool_call_id=call.id,
                name=call.name,
                output=(
                    f"Tool call refused: detected a loop of "
                    f"{self._loop_tracker_count()} consecutive identical "
                    f"calls to '{call.name}'. Try a different approach."
                ),
                is_error=True,
            )

        # --- Look up the tool ---
        tool = self._registry.get(call.name)
        if tool is None:
            outcome = await self._handle_mistake(
                MistakeReason.INVALID_TOOL_CALL,
                f"Unknown tool: {call.name}",
            )
            # _handle_mistake may have returned a stopped/failed result,
            # but we still need to produce a ToolResultPart for the history.
            if outcome.status in {"stopped", "failed"}:
                raise ToolAbortedError(outcome.error or outcome.stop_reason)
            return ToolResultPart(
                tool_call_id=call.id,
                name=call.name,
                output=f"Error: unknown tool '{call.name}'. Available: "
                f"{', '.join(self._registry.ids())}",
                is_error=True,
            )

        # --- Build the context ---
        ctx = ToolContext(
            tool_call_id=call.id,
            iteration=self._iteration,
            cwd=self._cwd,
            abort=self._abort,
            ask=self._ask,
        )

        await self._emit_event(
            ToolStartedEvent(
                type="tool-started",
                run_id=self._run_id,
                iteration=self._iteration,
                tool_call_id=call.id,
                tool_name=call.name,
                input=call.input,
            )
        )

        start = time.time()
        try:
            result = await tool.invoke(call.input, ctx)
        except InvalidArgumentsError as exc:
            # Treat invalid args as a mistake (the LLM should fix it).
            outcome = await self._handle_mistake(
                MistakeReason.INVALID_TOOL_CALL, str(exc)
            )
            if outcome.status in {"stopped", "failed"}:
                raise ToolAbortedError(
                    outcome.error or outcome.stop_reason
                ) from exc
            result = ToolResult.failure(str(exc))
        except ToolTimeoutError as exc:
            result = ToolResult.failure(f"Tool timed out: {exc}")
        except ToolAbortedError:
            raise
        except ToolError as exc:
            outcome = await self._handle_mistake(
                MistakeReason.TOOL_EXECUTION_FAILED, str(exc)
            )
            if outcome.status in {"stopped", "failed"}:
                raise ToolAbortedError(
                    outcome.error or outcome.stop_reason
                ) from exc
            result = ToolResult.failure(str(exc))
        except Exception as exc:  # noqa: BLE001
            outcome = await self._handle_mistake(
                MistakeReason.TOOL_EXECUTION_FAILED, str(exc)
            )
            if outcome.status in {"stopped", "failed"}:
                raise ToolAbortedError(
                    outcome.error or outcome.stop_reason
                ) from exc
            result = ToolResult.failure(f"Tool crashed: {exc}")
        latency_ms = (time.time() - start) * 1000

        output = result.output if result.output else (
            result.error or "(no output)"
        )
        is_error = result.is_error

        # Record file changes for write/edit/patch tools (best-effort).
        if not is_error:
            self._record_tool_changes(call.name, result)

        await self._emit_event(
            ToolFinishedEvent(
                type="tool-finished",
                run_id=self._run_id,
                iteration=self._iteration,
                tool_call_id=call.id,
                tool_name=call.name,
                output=output,
                is_error=is_error,
                latency_ms=latency_ms,
            )
        )

        return ToolResultPart(
            tool_call_id=call.id,
            name=call.name,
            output=output,
            is_error=is_error,
        )

    # -- internal: mistake tracking --------------------------------------

    async def _handle_mistake(
        self, reason: MistakeReason, details: str
    ) -> RunResult:
        """Record a mistake and decide whether to continue or stop."""

        assert self._mistake_tracker is not None
        outcome: MistakeOutcome = self._mistake_tracker.record(
            RecordMistakeInput(
                iteration=self._iteration,
                reason=reason,
                details=details,
            )
        )
        if outcome.action == "stop":
            stop_msg = outcome.message
            await self._emit_event(
                MistakeLimitHitEvent(
                    type="mistake-limit-hit",
                    run_id=self._run_id,
                    iteration=self._iteration,
                    consecutive_mistakes=self._mistake_tracker.value,
                    max_consecutive_mistakes=self._config.max_consecutive_mistakes,
                    reason=stop_msg,
                )
            )
            return RunResult(
                status="stopped",
                error=stop_msg,
                stop_reason=f"mistake_limit:{reason.value}",
                iterations=self._iteration,
                messages=list(self._messages),
                total_usage=dict(self._total_usage),
            )
        # Continue: return a no-op result (the caller keeps looping).
        return RunResult(
            status="completed",
            iterations=self._iteration,
            messages=list(self._messages),
        )

    # -- internal: helpers -----------------------------------------------

    def _record_tool_changes(self, tool_name: str, result: ToolResult) -> None:
        """Feed change metadata from a tool result into the change tracker.

        Tools that modify files (``write_to_file``, ``replace_in_file``,
        ``apply_patch``) include a ``changes`` list in their result
        metadata. Each entry is a dict with ``path``, ``old_content``,
        ``new_content``, and optionally ``action``.
        """

        changes_meta = result.metadata.get("changes")
        if not changes_meta:
            # write_to_file uses flat ``path`` / ``old_content`` /
            # ``new_content`` metadata (single file).
            path = result.metadata.get("path")
            if path and tool_name == "write_to_file":
                old = result.metadata.get("old_content")
                new = result.metadata.get("new_content", "")
                self._change_tracker.record_write(
                    str(path), old, str(new), tool_name
                )
            return

        for ch in changes_meta:
            path = ch.get("path", "")
            if not path:
                continue
            action = ch.get("action", "")
            old_content = ch.get("old_content")
            new_content = ch.get("new_content", "")
            if action == "deleted":
                self._change_tracker.record_delete(str(path), tool_name)
            elif action == "created" or old_content is None:
                self._change_tracker.record_write(
                    str(path), None, str(new_content), tool_name
                )
            else:
                self._change_tracker.record_edit(
                    str(path), str(old_content), str(new_content), tool_name
                )

    def _make_mistake_tracker(self) -> MistakeTracker:
        """Build a fresh :class:`MistakeTracker` bound to the current run id."""

        return MistakeTracker(
            MistakeTrackerOptions(
                max_consecutive_mistakes=self._config.max_consecutive_mistakes,
                emit=_noop_emit,
                log=_noop_log,
                agent_id="autoship-agent",
                get_conversation_id=lambda: self._run_id,
                get_active_run_id=lambda: self._run_id,
                append_recovery_notice=_noop_recovery_notice,
            )
        )

    def _append_user_message(self, user_input: str) -> None:
        """Wrap ``user_input`` with the mode tag and append to history.

        Consumes any pending mode-switch notice and prepends it so the
        LLM sees when the user toggled modes.
        """

        switch_notice = self._mode_config.consume_switch_notice()
        user_body = (
            switch_notice + "\n" + format_user_message(
                user_input, self._mode_config.mode
            )
            if switch_notice
            else format_user_message(user_input, self._mode_config.mode)
        )
        self._messages.append(Message(role="user", content=user_body))

    def _check_aborted(self) -> None:
        if self._abort.is_set():
            raise ToolAbortedError("Run aborted by user")

    def _loop_tracker_count(self) -> int:
        """Read the loop tracker's consecutive count."""

        return self._loop_tracker.consecutive_identical_count

    def _is_terminal_tool(self, name: str) -> bool:
        tool = self._registry.get(name)
        if tool is None:
            return False
        return tool.completes_run

    def _accumulate_usage(self, usage: dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self._total_usage[key] = self._total_usage.get(key, 0) + int(
                usage.get(key, 0)
            )

    def _completed_result(self, content: str) -> RunResult:
        return RunResult(
            status="completed",
            final_content=content,
            iterations=self._iteration,
            messages=list(self._messages),
            total_usage=dict(self._total_usage),
        )

    def _aborted_result(self, reason: str) -> RunResult:
        return RunResult(
            status="aborted",
            error=reason,
            iterations=self._iteration,
            messages=list(self._messages),
            total_usage=dict(self._total_usage),
            stop_reason="aborted",
        )

    async def _emit_event(self, event: RuntimeEvent) -> None:
        if self._emit is not None:
            await self._emit(event)

    async def _emit_terminal_event(self, result: RunResult) -> None:
        """Emit the appropriate terminal event (completed/aborted/failed).

        Called by :meth:`run` and :meth:`continue_run` after
        :meth:`_run_loop` returns, so the UI can print the run summary,
        change diff, and final stats.
        """

        if result.status == "completed":
            await self._emit_event(
                RunCompletedEvent(
                    type="run-completed",
                    run_id=self._run_id,
                    final_content=result.final_content,
                    iterations=result.iterations,
                    total_usage=dict(self._total_usage),
                )
            )
        elif result.status == "aborted":
            await self._emit_event(
                RunAbortedEvent(
                    type="run-aborted",
                    run_id=self._run_id,
                    reason=result.error or result.stop_reason,
                )
            )
        elif result.status == "failed":
            await self._emit_event(
                RunFailedEvent(
                    type="run-failed",
                    run_id=self._run_id,
                    error=result.error,
                    iterations=result.iterations,
                )
            )


__all__ = [
    "AgentMode",
    "AgentRuntime",
    "AgentRuntimeConfig",
    "AssistantMessageEvent",
    "ChangeTracker",
    "EventEmitter",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "LoopWarningEvent",
    "Message",
    "MistakeLimitHitEvent",
    "RunAbortedEvent",
    "RunCompletedEvent",
    "RunFailedEvent",
    "RunResult",
    "RunStartedEvent",
    "RuntimeEvent",
    "Session",
    "SessionMetadata",
    "SessionStatus",
    "SessionStore",
    "ToolCall",
    "ToolFinishedEvent",
    "ToolResultPart",
    "ToolStartedEvent",
    "TurnStartedEvent",
]
