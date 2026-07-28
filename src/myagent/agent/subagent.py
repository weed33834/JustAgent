"""Subagents — parallel read-only research agents with isolated context.

A subagent is a lightweight agent spawned by the main agent to
investigate a research question. Subagents:

* Run with **read-only tools** only (read_file, search, web_fetch,
  ask_question). They cannot write, edit, or run commands.
* Have their own **isolated conversation context** — they don't see the
  parent's history, and the parent only sees the subagent's final summary.
* Can run **in parallel** via asyncio.gather.
* Return a **text summary** of their findings to the parent.

The parent agent invokes a subagent via the ``dispatch_subagent`` tool.
The subagent's summary is returned as the tool result.

Reference: Cline's ``Task`` tool (``competitors/cline/.../tools/task.ts``)
and OpenCode's subagent support.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from myagent.exceptions import MyAgentError

# ---------------------------------------------------------------------------
# No-op callbacks (named so mypy can infer their types — lambdas can't)
# ---------------------------------------------------------------------------


def _noop_emit(event: dict[str, Any]) -> None:
    """Default no-op event sink for :class:`SubagentManager`."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SubagentError(MyAgentError):
    """Raised when a subagent operation fails."""


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class SubagentStatus(str, Enum):  # noqa: UP042
    """Lifecycle status of a subagent task.

    * ``PENDING`` — created but not yet started.
    * ``RUNNING`` — currently executing.
    * ``COMPLETED`` — finished successfully; ``summary`` is populated.
    * ``FAILED`` — terminated by an exception; ``error`` is populated.
    * ``ABORTED`` — cancelled by the user via :meth:`SubagentManager.abort`.
    * ``TIMED_OUT`` — exceeded its ``timeout_seconds`` budget.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    TIMED_OUT = "timed_out"


# ---------------------------------------------------------------------------
# Task & result value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubagentTask:
    """A single subagent research task.

    Attributes:
        id: Unique identifier (12-char hex). Generated when empty.
        prompt: The research question / instruction for the subagent.
        description: Short human-readable label (shown in UIs).
        max_iterations: Maximum agent-loop iterations.
        timeout_seconds: Wall-clock budget for the task.
        created_at: ``time.time()`` when the task was created.
    """

    id: str
    prompt: str
    description: str = ""
    max_iterations: int = 10
    timeout_seconds: float = 120.0
    created_at: float = 0.0

    def __post_init__(self) -> None:
        # Frozen dataclasses need ``object.__setattr__`` to mutate fields
        # post-construction. Auto-generate a 12-char hex id when the
        # caller didn't supply one.
        if not self.id:
            object.__setattr__(self, "id", uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class SubagentResult:
    """Outcome of running a :class:`SubagentTask`.

    Attributes:
        task_id: The id of the task this result corresponds to.
        status: Final :class:`SubagentStatus`.
        summary: The subagent's text summary (final assistant message),
            truncated to ``SubagentConfig.max_summary_chars``.
        error: Error message when ``status`` is ``FAILED`` /
            ``TIMED_OUT`` / ``ABORTED``; empty otherwise.
        iterations: Number of agent-loop iterations executed.
        elapsed_seconds: Wall-clock duration of the run.
        tokens_used: Total tokens consumed (best-effort, may be 0 when
            the LLM client does not report usage).
    """

    task_id: str
    status: SubagentStatus
    summary: str = ""
    error: str = ""
    iterations: int = 0
    elapsed_seconds: float = 0.0
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _default_allowed_tools() -> list[str]:
    """Default read-only tool whitelist for subagents."""

    return ["read_file", "search_files", "list_files", "web_fetch", "ask_question"]


@dataclass
class SubagentConfig:
    """Configuration for :class:`SubagentManager`.

    Attributes:
        max_concurrent: Maximum number of subagents that may run in
            parallel. Enforced via an :class:`asyncio.Semaphore`.
        default_max_iterations: Default ``max_iterations`` for tasks
            that do not specify one.
        default_timeout_seconds: Default ``timeout_seconds`` for tasks
            that do not specify one.
        max_summary_chars: Hard cap on the length of a subagent's
            summary; longer summaries are truncated.
        allowed_tools: Tool names a subagent is permitted to call. Any
            tool call outside this list is rejected and surfaced to the
            subagent as an error so it can correct itself.
    """

    max_concurrent: int = 3
    default_max_iterations: int = 10
    default_timeout_seconds: float = 120.0
    max_summary_chars: int = 4000
    allowed_tools: list[str] = field(default_factory=_default_allowed_tools)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


_SUBAGENT_SYSTEM_PROMPT = """\
You are a read-only research subagent. Your job is to investigate a
question and return a concise summary of your findings.

Constraints:
- You can ONLY use read-only tools: read_file, search, web_fetch,
  ask_question. You cannot write, edit, or run commands.
- You do NOT see the parent agent's conversation history. Work only from
  the prompt you are given.
- When you have enough information, respond with a final text message
  containing your summary. Do not call any more tools.
- Keep the summary focused and factual. Include file paths, search
  results, and concrete findings. Avoid speculation.

Your final assistant message (with no tool calls) becomes the summary
returned to the parent agent.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def filter_readonly_tools(tool_registry: Any, allowed: list[str]) -> dict[str, Any]:
    """Return only the tools whose names are in the ``allowed`` list.

    The ``tool_registry`` is duck-typed: any object with a ``get(name)``
    method (returning the tool or ``None``) works. The returned dict
    maps each allowed tool name to its tool object. Names that are not
    present in the registry are silently skipped.
    """

    filtered: dict[str, Any] = {}
    for name in allowed:
        tool = tool_registry.get(name)
        if tool is not None:
            filtered[name] = tool
    return filtered


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SubagentManager:
    """Spawns and tracks read-only research subagents.

    A manager owns:

    * a :class:`SubagentConfig` (defaults, concurrency cap, allowed
      tools),
    * a duck-typed ``llm_client`` with an ``async chat(messages, tools=None)``
      method returning a dict ``{"content": str, "tool_calls": list}``,
    * a duck-typed ``tool_registry`` with a ``get(name)`` method
      returning a tool with an ``async execute(input, context)`` callable,
    * an optional ``emit`` callback for streaming events.

    Subagents can be run one at a time (:meth:`run`) or several in
    parallel (:meth:`run_many`). Concurrency is bounded by
    :attr:`SubagentConfig.max_concurrent` via an
    :class:`asyncio.Semaphore`.
    """

    def __init__(
        self,
        config: SubagentConfig | None = None,
        llm_client: Any | None = None,
        tool_registry: Any | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._config = config or SubagentConfig()
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._emit: Callable[[dict[str, Any]], None] = emit or _noop_emit

        # Per-task state. ``_statuses`` is the source of truth for
        # status queries; ``_abort_events`` holds the asyncio.Event
        # used to cancel a running task.
        self._statuses: dict[str, SubagentStatus] = {}
        self._abort_events: dict[str, asyncio.Event] = {}
        self._active: set[str] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SubagentConfig:
        """The manager's configuration."""

        return self._config

    @property
    def llm_client(self) -> Any | None:
        """The LLM client (or ``None`` if not configured)."""

        return self._llm_client

    @property
    def tool_registry(self) -> Any | None:
        """The tool registry (or ``None`` if not configured)."""

        return self._tool_registry

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------

    def create_task(
        self,
        prompt: str,
        description: str = "",
        max_iterations: int | None = None,
        timeout_seconds: float | None = None,
    ) -> SubagentTask:
        """Create a :class:`SubagentTask` (does not start it).

        Defaults for ``max_iterations`` and ``timeout_seconds`` come
        from the manager's :class:`SubagentConfig`. A unique id is
        generated when the caller does not supply one.
        """

        task_id = uuid.uuid4().hex[:12]
        task = SubagentTask(
            id=task_id,
            prompt=prompt,
            description=description,
            max_iterations=(
                max_iterations
                if max_iterations is not None
                else self._config.default_max_iterations
            ),
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else self._config.default_timeout_seconds
            ),
            created_at=time.time(),
        )
        self._statuses[task_id] = SubagentStatus.PENDING
        return task

    # ------------------------------------------------------------------
    # Single-task run
    # ------------------------------------------------------------------

    async def run(self, task: SubagentTask) -> SubagentResult:
        """Run a single subagent to completion.

        Returns a :class:`SubagentResult` with the final status. The
        status will be one of ``COMPLETED``, ``FAILED``, ``ABORTED``,
        or ``TIMED_OUT``. The status tracker is updated regardless of
        outcome.
        """

        if self._llm_client is None:
            raise SubagentError("No LLM client configured")

        # Register state for this task.
        abort_event = asyncio.Event()
        self._abort_events[task.id] = abort_event
        self._statuses[task.id] = SubagentStatus.RUNNING
        self._active.add(task.id)

        self._emit(
            {
                "type": "subagent-started",
                "task_id": task.id,
                "description": task.description,
            }
        )

        start = time.monotonic()
        # Mutable holders so exception handlers can still report how far
        # the loop got before the failure/abort/timeout.
        iterations_holder: list[int] = [0]
        tokens_holder: list[int] = [0]
        try:
            summary = await asyncio.wait_for(
                self._run_loop(task, abort_event, iterations_holder, tokens_holder),
                timeout=task.timeout_seconds,
            )
            elapsed = time.monotonic() - start
            summary = self._truncate(summary)
            self._statuses[task.id] = SubagentStatus.COMPLETED
            self._emit(
                {
                    "type": "subagent-completed",
                    "task_id": task.id,
                    "iterations": iterations_holder[0],
                    "elapsed_seconds": elapsed,
                }
            )
            return SubagentResult(
                task_id=task.id,
                status=SubagentStatus.COMPLETED,
                summary=summary,
                iterations=iterations_holder[0],
                elapsed_seconds=elapsed,
                tokens_used=tokens_holder[0],
            )
        except TimeoutError:
            elapsed = time.monotonic() - start
            self._statuses[task.id] = SubagentStatus.TIMED_OUT
            self._emit(
                {
                    "type": "subagent-timed-out",
                    "task_id": task.id,
                    "elapsed_seconds": elapsed,
                }
            )
            return SubagentResult(
                task_id=task.id,
                status=SubagentStatus.TIMED_OUT,
                error=f"Subagent timed out after {task.timeout_seconds}s",
                iterations=iterations_holder[0],
                elapsed_seconds=elapsed,
                tokens_used=tokens_holder[0],
            )
        except _AbortedError as exc:
            elapsed = time.monotonic() - start
            self._statuses[task.id] = SubagentStatus.ABORTED
            self._emit(
                {
                    "type": "subagent-aborted",
                    "task_id": task.id,
                    "elapsed_seconds": elapsed,
                }
            )
            return SubagentResult(
                task_id=task.id,
                status=SubagentStatus.ABORTED,
                error=str(exc) or "Subagent aborted",
                iterations=iterations_holder[0],
                elapsed_seconds=elapsed,
                tokens_used=tokens_holder[0],
            )
        except SubagentError:
            elapsed = time.monotonic() - start
            self._statuses[task.id] = SubagentStatus.FAILED
            self._emit(
                {
                    "type": "subagent-failed",
                    "task_id": task.id,
                    "elapsed_seconds": elapsed,
                }
            )
            raise
        except Exception as exc:  # noqa: BLE001 — surface any failure to caller
            elapsed = time.monotonic() - start
            self._statuses[task.id] = SubagentStatus.FAILED
            self._emit(
                {
                    "type": "subagent-failed",
                    "task_id": task.id,
                    "error": str(exc),
                    "elapsed_seconds": elapsed,
                }
            )
            return SubagentResult(
                task_id=task.id,
                status=SubagentStatus.FAILED,
                error=str(exc) or exc.__class__.__name__,
                iterations=iterations_holder[0],
                elapsed_seconds=elapsed,
                tokens_used=tokens_holder[0],
            )
        finally:
            self._active.discard(task.id)
            self._abort_events.pop(task.id, None)

    # ------------------------------------------------------------------
    # Multi-task run
    # ------------------------------------------------------------------

    async def run_many(self, tasks: list[SubagentTask]) -> list[SubagentResult]:
        """Run multiple subagents in parallel.

        Concurrency is bounded by :attr:`SubagentConfig.max_concurrent`
        via an :class:`asyncio.Semaphore` shared across all tasks in
        this batch. Results are returned in the same order as ``tasks``.
        """

        # Create one semaphore per batch so each batch gets its own
        # budget (a previous batch's in-flight tasks should not count
        # against this batch).
        sem = asyncio.Semaphore(self._config.max_concurrent)

        async def _bounded(task: SubagentTask) -> SubagentResult:
            async with sem:
                return await self.run(task)

        return list(await asyncio.gather(*(_bounded(t) for t in tasks), return_exceptions=False))

    # ------------------------------------------------------------------
    # Status / control
    # ------------------------------------------------------------------

    def get_status(self, task_id: str) -> SubagentStatus | None:
        """Return the current status of ``task_id``, or ``None`` if unknown."""

        return self._statuses.get(task_id)

    def abort(self, task_id: str) -> bool:
        """Abort a running task.

        Sets the task's abort event; the loop checks it between
        iterations and bails out. Returns ``True`` if the task was
        running (and thus the abort signal was delivered), ``False``
        if the task was not running.
        """

        event = self._abort_events.get(task_id)
        if event is None or task_id not in self._active:
            return False
        event.set()
        return True

    def list_active(self) -> list[str]:
        """Return the IDs of currently running tasks (insertion order)."""

        return [tid for tid in self._statuses if tid in self._active]

    # ------------------------------------------------------------------
    # Internal: agent loop
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        task: SubagentTask,
        abort_event: asyncio.Event,
        iterations_holder: list[int],
        tokens_holder: list[int],
    ) -> str:
        """Run the subagent's iterative LLM ↔ tool loop.

        Returns the final summary string. ``iterations_holder`` and
        ``tokens_holder`` are mutable one-element lists updated each
        iteration so the caller (:meth:`run`) can still report how far
        the loop got when an exception propagates.

        The loop:

        1. Builds an isolated message list (system prompt + user prompt).
        2. Calls the LLM.
        3. If the response contains tool calls, executes only the
           allowed ones, feeds the results back, and repeats.
        4. Stops when the LLM responds without tool calls (the content
           is the summary), when ``max_iterations`` is reached (the
           latest assistant content is used as the summary), or when
           the abort event fires (checked between iterations and right
           after each LLM call returns).
        """

        if not task.prompt:
            raise SubagentError("Subagent prompt must not be empty")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SUBAGENT_SYSTEM_PROMPT},
            {"role": "user", "content": task.prompt},
        ]

        # Build the allowed-tools map once. Tool calls outside this map
        # are surfaced to the LLM as an error result so it can retry.
        allowed_tools: dict[str, Any] = (
            filter_readonly_tools(self._tool_registry, self._config.allowed_tools)
            if self._tool_registry is not None
            else {}
        )

        last_assistant_content = ""
        max_iter = task.max_iterations
        while iterations_holder[0] < max_iter:
            # Abort check between iterations.
            if abort_event.is_set():
                raise _AbortedError("Subagent aborted by user")

            iterations_holder[0] += 1
            response = await self._llm_client.chat(  # type: ignore[union-attr]
                messages, tools=list(allowed_tools.values()) or None
            )
            tokens_holder[0] += self._extract_tokens(response)

            # Abort check right after the LLM call returns — so an abort
            # fired during the (potentially long) call takes effect
            # immediately rather than waiting for the next iteration.
            if abort_event.is_set():
                raise _AbortedError("Subagent aborted by user")

            content = str(response.get("content") or "")
            raw_tool_calls = response.get("tool_calls") or []

            # Append the assistant turn to the conversation. Even when
            # the response only contains tool calls, we keep the content
            # (often empty) so the message history stays well-formed.
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": raw_tool_calls or None,
                }
            )
            if content:
                last_assistant_content = content

            # No tool calls → final summary.
            if not raw_tool_calls:
                return content

            # Execute each tool call sequentially and feed results back.
            for call in raw_tool_calls:
                tool_name = str(call.get("name") or "")
                tool_input = call.get("input") or {}
                call_id = str(call.get("id") or tool_name)

                if tool_name not in allowed_tools:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": (
                                f"Error: tool {tool_name!r} is not allowed "
                                f"in subagent context. Available: "
                                f"{', '.join(sorted(allowed_tools)) or '(none)'}"
                            ),
                        }
                    )
                    continue

                tool = allowed_tools[tool_name]
                try:
                    result = await tool.execute(tool_input, None)
                    output = self._tool_output(result)
                except Exception as exc:  # noqa: BLE001 — surface to LLM
                    output = f"Error executing {tool_name}: {exc}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": output,
                    }
                )

        # Iteration cap reached — return the latest assistant content.
        return last_assistant_content

    # ------------------------------------------------------------------
    # Internal: small helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        """Truncate ``text`` to ``max_summary_chars``."""

        cap = self._config.max_summary_chars
        if cap <= 0 or len(text) <= cap:
            return text
        # Keep the head; the tail is less useful for a research summary.
        return text[:cap]

    @staticmethod
    def _extract_tokens(response: dict[str, Any]) -> int:
        """Best-effort token accounting from an LLM response dict."""

        usage = response.get("usage") or {}
        if not isinstance(usage, dict):
            return 0
        for key in ("total_tokens", "tokens", "token_count"):
            value = usage.get(key)
            if isinstance(value, int) and value > 0:
                return value
        # Fall back to summing prompt + completion if both are present.
        prompt = usage.get("prompt_tokens") or 0
        completion = usage.get("completion_tokens") or 0
        if isinstance(prompt, int) and isinstance(completion, int):
            return prompt + completion
        return 0

    @staticmethod
    def _tool_output(result: Any) -> str:
        """Render a ToolResult-like object into a string for the LLM."""

        # ToolResult dataclass-style: ``output`` + ``error``.
        output = getattr(result, "output", None)
        error = getattr(result, "error", None)
        if error:
            return f"Error: {error}"
        if isinstance(output, str):
            return output
        if output is None:
            return ""
        return str(output)


# ---------------------------------------------------------------------------
# Internal abort signal
# ---------------------------------------------------------------------------


class _AbortedError(Exception):
    """Internal sentinel raised when a subagent's abort event fires."""


# ---------------------------------------------------------------------------
# Sync convenience wrapper
# ---------------------------------------------------------------------------


def run_research_sync(
    prompt: str,
    llm_client: Any,
    tool_registry: Any,
    config: SubagentConfig | None = None,
) -> SubagentResult:
    """Run a single research subagent synchronously (blocking).

    Convenience wrapper around :class:`SubagentManager` for simple use
    cases. Uses :func:`asyncio.run` internally, so it must not be called
    from within a running event loop.
    """

    manager = SubagentManager(
        config=config,
        llm_client=llm_client,
        tool_registry=tool_registry,
    )
    task = manager.create_task(prompt=prompt)
    return asyncio.run(manager.run(task))


__all__ = [
    "SubagentConfig",
    "SubagentError",
    "SubagentManager",
    "SubagentResult",
    "SubagentStatus",
    "SubagentTask",
    "filter_readonly_tools",
    "run_research_sync",
]
