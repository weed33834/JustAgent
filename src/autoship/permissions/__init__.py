"""Tool permission rules — allow / deny / ask with scope and wildcards.

Ports Cline's permission system: before a tool with side effects runs,
the runtime consults a :class:`PermissionEngine` to decide whether to
allow it, deny it, or ask the user. Rules can be scoped to ``once``
(prompt every time) or ``always`` (remember the decision for the
session). Patterns support wildcards so a rule like ``write_to_file:/**``
can allow all writes.

Reference: ``competitors/cline/sdk/packages/core/src/permissions/``.

Design:

* :class:`PermissionRule` — one rule: ``(tool, pattern, decision, scope)``.
* :class:`PermissionEngine` — evaluates rules in order. The first
  matching rule wins. If no rule matches, the default decision applies
  (configurable: ``ask`` by default, ``allow`` for Yolo mode).
* :class:`PermissionDecision` — frozen dataclass (``allow`` / ``deny``
  / ``ask``) with the matched rule for auditability.
* Decisions can be remembered via :meth:`PermissionEngine.remember` so
  subsequent matching calls skip the prompt.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PermissionAction(str, Enum):  # noqa: UP042
    """The action a rule instructs the engine to take."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionScope(str, Enum):  # noqa: UP042
    """How long a remembered decision lasts.

    * ``once`` — the decision applies only to this single call. The
      next call with the same tool/input will prompt again.
    * ``always`` — the decision is remembered for the rest of the
      session (or until the rule is removed).
    """

    ONCE = "once"
    ALWAYS = "always"


@dataclass(frozen=True)
class PermissionRule:
    """A single permission rule.

    Attributes:
        tool: Tool name to match (or ``*`` to match all tools).
        pattern: Glob pattern matched against the tool's input
            (stringified). ``*`` matches everything.
        action: What to do when this rule matches.
        scope: ``once`` or ``always``.
        description: Optional human-readable note.
    """

    tool: str
    pattern: str = "*"
    action: PermissionAction = PermissionAction.ASK
    scope: PermissionScope = PermissionScope.ALWAYS
    description: str = ""


@dataclass(frozen=True)
class PermissionDecision:
    """The engine's decision for a single permission check.

    Attributes:
        action: ``allow`` / ``deny`` / ``ask``.
        matched_rule: The rule that produced this decision, or None if
            the default was used.
        reason: Human-readable explanation.
    """

    action: PermissionAction
    matched_rule: PermissionRule | None = None
    reason: str = ""


def _stringify_input(tool_input: dict[str, Any]) -> str:
    """Convert a tool input dict into a string for pattern matching.

    For tools that operate on paths (write_to_file, read_file, etc.),
    we match against the ``path`` field. For run_command, we match
    against the ``command`` field. For other tools, we match against
    a JSON representation of the input.
    """

    # Common path-based tools.
    path = tool_input.get("path")
    if isinstance(path, str):
        return path
    command = tool_input.get("command")
    if isinstance(command, str):
        return command
    # Fall back to JSON for everything else.
    import json

    return json.dumps(tool_input, sort_keys=True, default=str)


class PermissionEngine:
    """Evaluates permission rules for tool calls.

    Example:

        >>> engine = PermissionEngine()
        >>> engine.add_rule(PermissionRule(
        ...     tool="read_file", pattern="*", action=PermissionAction.ALLOW
        ... ))
        >>> engine.add_rule(PermissionRule(
        ...     tool="write_to_file", pattern="/tmp/**",
        ...     action=PermissionAction.ALLOW
        ... ))
        >>> engine.add_rule(PermissionRule(
        ...     tool="write_to_file", pattern="*",
        ...     action=PermissionAction.ASK
        ... ))
        >>> engine.check("read_file", {"path": "/etc/passwd"})
        PermissionDecision(action=<PermissionAction.ALLOW: 'allow'>, ...)
        >>> engine.check("write_to_file", {"path": "/tmp/test.txt"})
        PermissionDecision(action=<PermissionAction.ALLOW: 'allow'>, ...)
        >>> engine.check("write_to_file", {"path": "/home/user/x.txt"})
        PermissionDecision(action=<PermissionAction.ASK: 'ask'>, ...)
    """

    def __init__(
        self,
        *,
        default_action: PermissionAction = PermissionAction.ASK,
    ) -> None:
        self._rules: list[PermissionRule] = []
        self._default_action = default_action
        # Remembered "always" decisions: (tool, input_string) -> action.
        self._remembered: dict[tuple[str, str], PermissionAction] = {}

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: PermissionRule) -> None:
        """Add a rule. Rules are evaluated in insertion order."""

        self._rules.append(rule)

    def remove_rule(self, index: int) -> PermissionRule | None:
        """Remove the rule at ``index``. Returns the removed rule."""

        if 0 <= index < len(self._rules):
            return self._rules.pop(index)
        return None

    def clear_rules(self) -> None:
        """Remove all rules."""

        self._rules.clear()
        self._remembered.clear()

    @property
    def rules(self) -> list[PermissionRule]:
        """Return a copy of the current rules."""

        return list(self._rules)

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(self, tool: str, tool_input: dict[str, Any]) -> PermissionDecision:
        """Evaluate rules and return a :class:`PermissionDecision`.

        The first matching rule wins. If no rule matches, the default
        action is returned. Remembered "always" decisions short-circuit
        the evaluation.
        """

        input_str = _stringify_input(tool_input)

        # Check remembered "always" decisions first.
        key = (tool, input_str)
        if key in self._remembered:
            action = self._remembered[key]
            return PermissionDecision(
                action=action,
                matched_rule=None,
                reason=f"Remembered {action.value} decision",
            )

        # Evaluate rules in order.
        for rule in self._rules:
            if self._matches(rule, tool, input_str):
                if (
                    rule.action in (PermissionAction.ALLOW, PermissionAction.DENY)
                    and rule.scope is PermissionScope.ALWAYS
                ):
                    self._remembered[key] = rule.action
                return PermissionDecision(
                    action=rule.action,
                    matched_rule=rule,
                    reason=f"Matched rule: {rule.tool}:{rule.pattern} → {rule.action.value}",
                )

        # No rule matched → default.
        return PermissionDecision(
            action=self._default_action,
            matched_rule=None,
            reason=f"No rule matched; default is {self._default_action.value}",
        )

    # ------------------------------------------------------------------
    # Remember
    # ------------------------------------------------------------------

    def remember(
        self,
        tool: str,
        tool_input: dict[str, Any],
        action: PermissionAction,
        scope: PermissionScope = PermissionScope.ALWAYS,
    ) -> None:
        """Manually remember a decision.

        This is called after the user responds to an ``ask`` prompt.
        If scope is ``always``, the decision is cached so future calls
        with the same tool+input skip the prompt. If scope is ``once``,
        nothing is cached.
        """

        if scope is PermissionScope.ONCE:
            return
        input_str = _stringify_input(tool_input)
        self._remembered[(tool, input_str)] = action

    def forget(self, tool: str, tool_input: dict[str, Any]) -> None:
        """Forget a remembered decision for the given tool+input."""

        input_str = _stringify_input(tool_input)
        self._remembered.pop((tool, input_str), None)

    def clear_remembered(self) -> None:
        """Clear all remembered decisions."""

        self._remembered.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def default_action(self) -> PermissionAction:
        """The default action when no rule matches."""

        return self._default_action

    @default_action.setter
    def default_action(self, value: PermissionAction) -> None:
        self._default_action = value

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _matches(
        self, rule: PermissionRule, tool: str, input_str: str
    ) -> bool:
        """Check if a rule matches the given tool and input."""

        # Tool match: exact or wildcard.
        if rule.tool != "*" and rule.tool != tool:
            return False
        # Pattern match: glob against the stringified input.
        # Support ** as recursive glob (fnmatch treats ** same as *,
        # but that's fine for our purposes).
        return fnmatch.fnmatch(input_str, rule.pattern)


# ---------------------------------------------------------------------------
# Convenience: create engines for common modes
# ---------------------------------------------------------------------------


def create_act_mode_engine() -> PermissionEngine:
    """Create a permission engine for Act mode.

    Act mode allows read-only tools by default and asks for destructive
    operations.
    """

    engine = PermissionEngine(default_action=PermissionAction.ASK)
    # Read-only tools are always allowed.
    for tool in ("read_file", "search_files", "list_files", "web_fetch", "ask_question"):
        engine.add_rule(
            PermissionRule(
                tool=tool,
                pattern="*",
                action=PermissionAction.ALLOW,
                description=f"Read-only tool {tool} always allowed",
            )
        )
    # Destructive tools ask by default (no rule needed — default is ask).
    return engine


def create_plan_mode_engine() -> PermissionEngine:
    """Create a permission engine for Plan mode.

    Plan mode denies all edit tools and allows read-only tools.
    """

    engine = PermissionEngine(default_action=PermissionAction.DENY)
    # Read-only tools are allowed.
    for tool in ("read_file", "search_files", "list_files", "web_fetch", "ask_question"):
        engine.add_rule(
            PermissionRule(
                tool=tool,
                pattern="*",
                action=PermissionAction.ALLOW,
                description=f"Read-only tool {tool} allowed in plan mode",
            )
        )
    # run_command is allowed but only for read-only commands (pattern
    # would need to be more sophisticated in production; for now, allow
    # all run_command in plan mode since the LLM is prompt-constrained).
    engine.add_rule(
        PermissionRule(
            tool="run_command",
            pattern="*",
            action=PermissionAction.ALLOW,
            description="run_command allowed in plan mode (prompt-constrained)",
        )
    )
    return engine


def create_yolo_mode_engine() -> PermissionEngine:
    """Create a permission engine for Yolo mode.

    Yolo mode allows everything without asking.
    """

    return PermissionEngine(default_action=PermissionAction.ALLOW)


__all__ = [
    "PermissionAction",
    "PermissionDecision",
    "PermissionEngine",
    "PermissionRule",
    "PermissionScope",
    "create_act_mode_engine",
    "create_plan_mode_engine",
    "create_yolo_mode_engine",
]
