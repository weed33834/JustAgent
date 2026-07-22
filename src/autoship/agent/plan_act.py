"""Plan / Act mode switching.

Ports Cline's dual-mode concept (``mode: "act" | "plan" | "yolo"``)
to autoship. In **Plan mode** the agent explores and proposes a plan
without making changes; in **Act mode** it executes. **Yolo mode**
disables permission prompts entirely.

Design (following Cline's approach, with OpenCode's plan-file idea):

* :class:`AgentMode` is a string enum.
* :class:`ModeConfig` holds the current mode + optional plan file path.
* :func:`filter_tools_for_mode` removes edit tools in Plan mode.
* :func:`build_system_prompt` appends Plan-mode instructions when active.
* :func:`format_user_message` wraps user input with ``<user_input mode="...">``
  so the LLM sees the mode on every turn.
* :func:`format_mode_switch_notice` produces a ``<mode_notice>`` block
  when the user toggles between modes.
* :class:`ModeSwitchTracker` records pending switch notices so they
  fire on the next user message (mirrors Cline's
  ``createModeSwitchNoticeTracker``).

Reference:
``competitors/cline/sdk/packages/shared/src/prompt/cline.ts`` and
``competitors/cline/sdk/packages/shared/src/prompt/format.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from autoship.agent.tools.base import Tool


class AgentMode(str, Enum):  # noqa: UP042 - match existing codebase style
    """The agent's execution mode."""

    ACT = "act"
    PLAN = "plan"
    YOLO = "yolo"

    @property
    def allows_edits(self) -> bool:
        """Whether file-editing tools are available in this mode."""

        return self in (AgentMode.ACT, AgentMode.YOLO)

    @property
    def allows_commands(self) -> bool:
        """Whether shell commands are available in this mode.

        Cline intentionally keeps ``run_commands`` available in Plan
        mode for read-only investigation (grep, ls, git log, etc.) —
        the constraint is enforced via prompting rather than tool
        removal.
        """

        return True

    @property
    def requires_permission(self) -> bool:
        """Whether side-effecting tools need user approval.

        Yolo mode skips the permission prompt entirely.
        """

        return self is not AgentMode.YOLO


# ---------------------------------------------------------------------------
# Tool filtering
# ---------------------------------------------------------------------------


#: Tool ids that modify files and must be hidden in Plan mode.
_EDIT_TOOL_IDS: frozenset[str] = frozenset(
    {
        "write_to_file",
        "replace_in_file",
        "apply_patch",
    }
)

#: Tool ids that run shell commands (kept in Plan mode, but the prompt
#: instructs the LLM to keep them read-only).
_COMMAND_TOOL_IDS: frozenset[str] = frozenset({"run_command"})


def filter_tools_for_mode(tools: list[Tool], mode: AgentMode) -> list[Tool]:
    """Return the subset of ``tools`` available in ``mode``.

    In Plan mode, edit tools (``write_to_file``, ``replace_in_file``,
    ``apply_patch``) are removed. In Act and Yolo mode, all tools are
    returned.
    """

    if mode.allows_edits:
        return list(tools)
    return [t for t in tools if t.id not in _EDIT_TOOL_IDS]


def is_edit_tool(tool_id: str) -> bool:
    """Return True if ``tool_id`` is a file-editing tool."""

    return tool_id in _EDIT_TOOL_IDS


def is_command_tool(tool_id: str) -> bool:
    """Return True if ``tool_id`` runs shell commands."""

    return tool_id in _COMMAND_TOOL_IDS


# ---------------------------------------------------------------------------
# System prompt building
# ---------------------------------------------------------------------------


#: The mode-tag instructions are always included (mirrors Cline's
#: ``MODE_TAG_INSTRUCTIONS``) so the LLM understands the
#: ``<user_input mode="...">`` wrapper.
MODE_TAG_INSTRUCTIONS = """# Plan / Act Modes

User messages arrive wrapped in a `<user_input mode="...">` tag. The mode attribute is the interaction mode the user was in when they sent that message: "plan" means plan-mode constraints applied (explore, analyze, and align on a plan -- no edits or state-changing commands), while "act" (or "yolo") means implementation was allowed. If the mode attribute changes between messages, the user switched modes -- the newest message's mode is what governs right now, regardless of what earlier messages allowed. A <mode_notice> block inside a message marks exactly when such a switch happened."""


#: Appended only in Plan mode (mirrors Cline's ``PLAN_MODE_INSTRUCTIONS``).
PLAN_MODE_INSTRUCTIONS = """# Plan Mode

You are in Plan mode. Your role is to explore, analyze, and plan -- not to execute.

- Read files, search the codebase, and gather context to understand the problem
- Ask clarifying questions when requirements are ambiguous
- Present your plan as a structured outline with clear steps
- Explain tradeoffs between different approaches when they exist
- Do NOT edit files, write code, run destructive commands, or make any changes
- Do NOT implement anything -- focus on understanding and alignment first

The run_command tool remains available in plan mode strictly for read-only inspection -- listing files, searching (grep), reading configs, inspecting git history and diffs, checking tool versions, and the like. Never use it to change anything: no creating, modifying, or deleting files, no writing scripts that make changes, and no state-changing commands (installs, migrations, database or schema changes, container commands that mutate state, etc.). If the task requires a mutation, put it in the plan; it happens only after the user switches to act mode.

Once the user has reviewed your plan and explicitly approved it in a follow-up message, the user will switch you to act mode manually. Do not attempt to switch modes yourself."""


#: Appended in Yolo mode.
YOLO_MODE_INSTRUCTIONS = """# Yolo Mode

You are in Yolo mode. Permission prompts are disabled — all tool calls
execute immediately without user approval. Use this for trusted,
autonomous workflows (e.g. CI fixes, scripted bulk operations). Be
extra careful with destructive operations since no human confirmation
will be requested."""


def build_system_prompt(
    base_prompt: str,
    mode: AgentMode,
    *,
    extra_rules: str = "",
) -> str:
    """Compose the full system prompt for ``mode``.

    ``base_prompt`` is the agent's core instructions. Mode-specific
    instructions are appended in order: extra rules → mode tag →
    mode-specific instructions (Plan/Yolo only).
    """

    parts: list[str] = [base_prompt.strip()]
    if extra_rules:
        parts.append(extra_rules.strip())
    parts.append(MODE_TAG_INSTRUCTIONS)
    if mode is AgentMode.PLAN:
        parts.append(PLAN_MODE_INSTRUCTIONS)
    elif mode is AgentMode.YOLO:
        parts.append(YOLO_MODE_INSTRUCTIONS)
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# User message formatting
# ---------------------------------------------------------------------------


def format_user_message(content: str, mode: AgentMode) -> str:
    """Wrap ``content`` in a ``<user_input mode="...">`` block.

    Mirrors Cline's ``formatUserInputBlock``. The wrapper tells the
    LLM which mode was active when the user sent the message.
    """

    return f'<user_input mode="{mode.value}">{content}</user_input>'


def format_mode_switch_notice(
    from_mode: AgentMode,
    to_mode: AgentMode,
) -> str:
    """Return a ``<mode_notice>`` block describing a mode switch.

    Returns an empty string when ``from_mode == to_mode`` (no switch).
    """

    if from_mode == to_mode:
        return ""
    return (
        f"<mode_notice>The user switched from {from_mode.value} mode "
        f"to {to_mode.value} mode before sending this message."
        "</mode_notice>"
    )


# ---------------------------------------------------------------------------
# ModeSwitchTracker
# ---------------------------------------------------------------------------


@dataclass
class ModeSwitch:
    """A pending mode-switch notice."""

    from_mode: AgentMode
    to_mode: AgentMode


@dataclass
class ModeSwitchTracker:
    """Tracks pending mode switches so they fire on the next user message.

    Mirrors Cline's ``createModeSwitchNoticeTracker``. Only UI-initiated
    switches are tracked — the model does not switch modes itself in
    autoship (unlike Cline's ``switch_to_act_mode`` tool).

    Consecutive switches coalesce: ``plan → act → plan`` is treated as
    no switch (we end up where we started).
    """

    _pending: ModeSwitch | None = None

    def record(self, from_mode: AgentMode, to_mode: AgentMode) -> None:
        """Record a mode switch (called when the user toggles the UI)."""

        if from_mode == to_mode:
            return
        if self._pending is None:
            self._pending = ModeSwitch(from_mode, to_mode)
            return
        # Coalesce: if we had A→B pending and now go B→A, cancel.
        if self._pending.to_mode == from_mode:
            # Net effect is pending.from → to_mode.
            self._pending = ModeSwitch(self._pending.from_mode, to_mode)
        else:
            self._pending = ModeSwitch(self._pending.from_mode, to_mode)
        # If we ended up where we started, drop the notice.
        if self._pending.from_mode == self._pending.to_mode:
            self._pending = None

    def consume(self) -> ModeSwitch | None:
        """Return and clear the pending switch, if any."""

        pending = self._pending
        self._pending = None
        return pending

    @property
    def pending(self) -> ModeSwitch | None:
        """The pending switch (without consuming)."""

        return self._pending


# ---------------------------------------------------------------------------
# ModeConfig
# ---------------------------------------------------------------------------


@dataclass
class ModeConfig:
    """Per-session mode configuration.

    Holds the current mode and a :class:`ModeSwitchTracker` for pending
    UI-initiated switches. The runtime consults this before each LLM
    call to decide which tools are available and which prompt fragment
    to append.
    """

    mode: AgentMode = AgentMode.ACT
    tracker: ModeSwitchTracker = field(default_factory=ModeSwitchTracker)

    def switch_to(self, new_mode: AgentMode) -> None:
        """Record a mode switch (called by the UI layer)."""

        self.tracker.record(self.mode, new_mode)
        self.mode = new_mode

    def consume_switch_notice(self) -> str:
        """Return the pending switch notice as a string, or empty."""

        pending = self.tracker.consume()
        if pending is None:
            return ""
        return format_mode_switch_notice(pending.from_mode, pending.to_mode)


# ---------------------------------------------------------------------------
# Plan file path helper (OpenCode-inspired)
# ---------------------------------------------------------------------------


def default_plan_file_path(cwd: str | Path) -> Path:
    """Return the default plan file path for ``cwd``.

    Following OpenCode's convention, plans live in
    ``.autoship/plans/plan.md`` within the project root. The directory
    is *not* created here — callers should ``mkdir(parents=True,
    exist_ok=True)`` before writing.
    """

    return Path(cwd) / ".autoship" / "plans" / "plan.md"


__all__ = [
    "AgentMode",
    "MODE_TAG_INSTRUCTIONS",
    "ModeConfig",
    "ModeSwitch",
    "ModeSwitchTracker",
    "PLAN_MODE_INSTRUCTIONS",
    "YOLO_MODE_INSTRUCTIONS",
    "build_system_prompt",
    "default_plan_file_path",
    "filter_tools_for_mode",
    "format_mode_switch_notice",
    "format_user_message",
    "is_command_tool",
    "is_edit_tool",
]
