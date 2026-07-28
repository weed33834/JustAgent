"""Context compaction for the justagent agent.

When a conversation grows close to the model's context-window budget,
the compactor reclaims tokens by either truncating old messages
(``basic`` mode) or replacing them with an LLM-generated summary
(``agentic`` mode). The system prompt and the ``keep_recent_messages``
most recent messages are always preserved.

Design parallels Cline's "autoCompact" / OpenCode's "summarize" path:
once ``should_compact`` trips, :meth:`Compactor.compact` partitions the
history into (system prefix, removable middle, recent tail), summarizes
or drops the middle, and returns a fresh message list the runtime can
swap in.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from justagent.agent.runtime import LLMClient, LLMRequest, Message

__all__ = [
    "CompactionConfig",
    "CompactionResult",
    "Compactor",
]


# ---------------------------------------------------------------------------
# Configuration & result
# ---------------------------------------------------------------------------


@dataclass
class CompactionConfig:
    """Configuration for :class:`Compactor`.

    Attributes:
        trigger_ratio: Compact when token usage reaches this fraction of
            ``max_context_tokens``.
        max_context_tokens: The context-window budget (in tokens).
        mode: ``"basic"`` truncates old messages; ``"agentic"`` asks the
            LLM to summarize them.
        keep_recent_messages: Always preserve the N most recent messages.
        keep_system_prompt: When True, preserve leading ``role=system``
            messages verbatim.
    """

    trigger_ratio: float = 0.9
    max_context_tokens: int = 128_000
    mode: Literal["basic", "agentic"] = "basic"
    keep_recent_messages: int = 6
    keep_system_prompt: bool = True


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of :meth:`Compactor.compact`.

    Attributes:
        compacted_messages: The new message list after compaction.
        removed_count: How many messages were removed/summarized.
        summary: The summary text if agentic mode was used, empty for
            basic mode (a placeholder is still inserted into
            ``compacted_messages`` when messages are removed).
        tokens_before: Estimated token count before compaction.
        tokens_after: Estimated token count after compaction.
    """

    compacted_messages: list[Message]
    removed_count: int
    summary: str
    tokens_before: int
    tokens_after: int


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------


class Compactor:
    """Compacts a conversation to reclaim context-window tokens.

    Two modes are supported:

    * ``basic`` — keep the system prompt (optionally) plus the last
      ``keep_recent_messages`` messages; insert a placeholder summary
      describing how many messages were dropped.
    * ``agentic`` — like ``basic``, but the dropped messages are first
      summarized by an :class:`LLMClient` and the summary is inserted as
      a ``role=system`` message instead of a placeholder.

    Example::

        config = CompactionConfig(max_context_tokens=8000, mode="agentic")
        compactor = Compactor(config, llm_client=client)
        if compactor.should_compact(messages, current_tokens=7500):
            result = compactor.compact(messages)
            messages = result.compacted_messages
    """

    def __init__(
        self,
        config: CompactionConfig,
        *,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize the compactor.

        Args:
            config: Compaction settings.
            llm_client: Required for ``agentic`` mode; ignored in
                ``basic`` mode. If ``None`` and ``mode="agentic"``, the
                compactor falls back to a placeholder summary.
        """

        self._config = config
        self._llm_client = llm_client

    def should_compact(
        self,
        messages: list[Message],
        current_tokens: int,
    ) -> bool:
        """Return True if compaction should run.

        Compaction triggers when ``current_tokens`` reaches
        ``max_context_tokens * trigger_ratio``.

        Args:
            messages: The current conversation (reserved for future
                heuristics; not used in the threshold calculation).
            current_tokens: The current token usage of ``messages``.
        """

        threshold = int(self._config.max_context_tokens * self._config.trigger_ratio)
        return current_tokens >= threshold

    def compact(self, messages: list[Message]) -> CompactionResult:
        """Compact ``messages`` and return the result.

        In ``basic`` mode the removed messages are replaced by a
        placeholder summary; in ``agentic`` mode they are summarized by
        the LLM client. The system prompt (when kept) and the last
        ``keep_recent_messages`` messages are always preserved.

        If there are no removable messages (e.g. the history is shorter
        than ``keep_recent_messages``), the messages are returned
        unchanged with ``removed_count=0``.
        """

        tokens_before = self.estimate_tokens(messages)
        system_messages, removable, recent = self._partition(messages)
        removed_count = len(removable)

        if removed_count == 0:
            compacted: list[Message] = list(messages)
            return CompactionResult(
                compacted_messages=compacted,
                removed_count=0,
                summary="",
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        if self._config.mode == "agentic":
            summary = self._summarize(removable)
        else:
            summary = self._placeholder(removed_count)

        compacted = []
        if self._config.keep_system_prompt:
            compacted.extend(system_messages)
        compacted.append(Message(role="system", content=summary))
        compacted.extend(recent)

        tokens_after = self.estimate_tokens(compacted)

        return CompactionResult(
            compacted_messages=compacted,
            removed_count=removed_count,
            summary=summary,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def estimate_tokens(self, messages: list[Message]) -> int:
        """Return a rough token estimate (4 chars ≈ 1 token).

        Counts the text length of each message's ``content``,
        ``tool_calls`` (name + input repr), and ``tool_result`` output,
        then divides by 4. This mirrors the common "4 chars per token"
        heuristic used by Cline and OpenAI's tiktoken approximator.
        """

        total_chars = 0
        for msg in messages:
            total_chars += len(msg.content)
            for tc in msg.tool_calls:
                total_chars += len(tc.name)
                total_chars += len(str(tc.input))
            if msg.tool_result is not None:
                total_chars += len(msg.tool_result.output)
        return total_chars // 4

    # -- internal helpers -------------------------------------------------

    def _partition(
        self,
        messages: list[Message],
    ) -> tuple[list[Message], list[Message], list[Message]]:
        """Split ``messages`` into (system, removable, recent).

        * ``system`` — leading ``role=system`` messages (only when
          ``keep_system_prompt`` is set).
        * ``removable`` — the middle block that will be summarized/dropped.
        * ``recent`` — the last ``keep_recent_messages`` messages (from
          the non-system tail).
        """

        if self._config.keep_system_prompt:
            system_messages: list[Message] = []
            idx = 0
            for msg in messages:
                if msg.role == "system":
                    system_messages.append(msg)
                    idx += 1
                else:
                    break
            rest = list(messages[idx:])
        else:
            system_messages = []
            rest = list(messages)

        keep_recent = self._config.keep_recent_messages
        if keep_recent >= len(rest):
            return system_messages, [], rest

        split = len(rest) - keep_recent
        removable = rest[:split]
        recent = rest[split:]
        return system_messages, removable, recent

    def _placeholder(self, removed_count: int) -> str:
        """Return a basic-mode placeholder summary."""

        return (
            f"[context compacted: {removed_count} earlier message(s) "
            f"removed from history]"
        )

    def _summarize(self, removable: list[Message]) -> str:
        """Ask the LLM to summarize ``removable`` and return the text.

        Falls back to a placeholder when no LLM client is configured or
        when the LLM call cannot be executed (e.g. we are already inside
        a running event loop).
        """

        if self._llm_client is None:
            return self._placeholder(len(removable))

        transcript = self._render_transcript(removable)
        request = LLMRequest(
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are a context-compaction assistant. Summarize "
                        "the following conversation fragment so that a "
                        "downstream agent can continue the work without "
                        "losing key context. Preserve: decisions made, "
                        "file paths touched, errors seen, and any open "
                        "todos. Be concise."
                    ),
                ),
                Message(role="user", content=transcript),
            ],
            tools=[],
        )

        try:
            response = asyncio.run(self._llm_client.complete(request))
        except RuntimeError:
            # Already inside a running event loop — can't nest
            # asyncio.run(). Fall back to the placeholder so compaction
            # still succeeds.
            return self._placeholder(len(removable))

        text = response.content.strip()
        if not text:
            return self._placeholder(len(removable))
        return text

    @staticmethod
    def _render_transcript(messages: list[Message]) -> str:
        """Render ``messages`` as a compact text transcript for the LLM."""

        lines: list[str] = []
        for msg in messages:
            body = msg.content
            if msg.tool_calls:
                calls = ", ".join(
                    f"{tc.name}({tc.input})" for tc in msg.tool_calls
                )
                body = f"{body} [tool_calls: {calls}]".strip()
            if msg.tool_result is not None:
                body = (
                    f"{body} "
                    f"[tool_result({msg.tool_result.name}): "
                    f"{msg.tool_result.output}]"
                ).strip()
            lines.append(f"[{msg.role}] {body}")
        return "\n".join(lines)
