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
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("myagent.permissions")


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
    against the ``command`` field. For tools whose input is a large blob
    (``apply_patch`` carries ``patch``, ``replace_in_file`` carries
    ``diff``), the full blob is a poor key for the remembered-decision
    cache (it would make the cache balloon and every distinct patch would
    re-prompt), so we collapse it to a short stable string. For other
    tools, we match against a JSON representation of the input.
    """

    # Tools whose input is a large blob — collapse to a short stable key
    # so the remembered-decision cache does not bloat.
    if "patch" in tool_input:
        return "patch"
    if "diff" in tool_input:
        return "diff"
    # Common path-based tools.
    path = tool_input.get("path")
    if isinstance(path, str):
        return path
    command = tool_input.get("command")
    if isinstance(command, str):
        return command
    # Fall back to JSON for everything else.
    return json.dumps(tool_input, sort_keys=True, default=str)


# Compiled-glob cache for patterns that contain ``**``. Patterns without
# ``**`` keep using ``fnmatch`` (which already caches internally).
_REGEX_CACHE: dict[str, re.Pattern[str]] = {}


def _glob_to_regex(pattern: str) -> str:
    """Translate a glob pattern into a regex.

    Translation rules (used when the pattern contains ``**``):

    * ``**`` -> ``.*``  (matches any character, including ``/`` — the
      recursive glob that ``fnmatch`` does not provide).
    * ``*``  -> ``[^/]*`` (matches a single path segment, no ``/``).
    * ``?``  -> ``[^/]`` (matches a single character, no ``/``).
    * every other character is regex-escaped.
    """

    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*" and i + 1 < n and pattern[i + 1] == "*":
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _glob_match(pattern: str, text: str) -> bool:
    """Glob-match ``text`` against ``pattern``.

    Patterns containing ``**`` use a custom translator where ``**``
    matches any character including ``/`` (so ``/tmp/**`` matches
    ``/tmp/a/b/c``). Patterns without ``**`` keep ``fnmatch`` behaviour
    for backward compatibility (``fnmatch``'s ``*`` already crosses ``/``,
    which existing ``pattern="*"`` rules rely on). Compiled regexes are
    cached for performance.
    """

    if "**" not in pattern:
        return fnmatch.fnmatch(text, pattern)
    regex = _REGEX_CACHE.get(pattern)
    if regex is None:
        regex = re.compile(_glob_to_regex(pattern))
        _REGEX_CACHE[pattern] = regex
    return regex.fullmatch(text) is not None


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
        key = (tool, input_str)

        decision: PermissionDecision
        # Check remembered "always" decisions first.
        if key in self._remembered:
            action = self._remembered[key]
            decision = PermissionDecision(
                action=action,
                matched_rule=None,
                reason=f"Remembered {action.value} decision",
            )
        else:
            # Evaluate rules in order; the first match wins.
            matched_rule: PermissionRule | None = None
            for rule in self._rules:
                if self._matches(rule, tool, input_str):
                    matched_rule = rule
                    break
            if matched_rule is not None:
                if (
                    matched_rule.action in (PermissionAction.ALLOW, PermissionAction.DENY)
                    and matched_rule.scope is PermissionScope.ALWAYS
                ):
                    self._remembered[key] = matched_rule.action
                decision = PermissionDecision(
                    action=matched_rule.action,
                    matched_rule=matched_rule,
                    reason=(
                        f"Matched rule: {matched_rule.tool}:{matched_rule.pattern} "
                        f"→ {matched_rule.action.value}"
                    ),
                )
            else:
                # No rule matched → default.
                decision = PermissionDecision(
                    action=self._default_action,
                    matched_rule=None,
                    reason=f"No rule matched; default is {self._default_action.value}",
                )

        logger.debug(
            "Permission check: tool=%s input=%s decision=%s reason=%s",
            tool,
            input_str,
            decision.action,
            decision.reason,
        )
        return decision

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

    def _matches(self, rule: PermissionRule, tool: str, input_str: str) -> bool:
        """Check if a rule matches the given tool and input."""

        # Tool match: exact or wildcard.
        if rule.tool != "*" and rule.tool != tool:
            return False
        # Pattern match: glob against the stringified input. ``**`` is a
        # recursive glob (matches ``/``); see :func:`_glob_match`.
        return _glob_match(rule.pattern, input_str)


# ---------------------------------------------------------------------------
# Convenience: create engines for common modes
# ---------------------------------------------------------------------------


def create_act_mode_engine() -> PermissionEngine:
    """Create a permission engine for Act mode.

    Act mode allows read-only tools by default and asks for destructive
    operations.
    """

    engine = PermissionEngine(default_action=PermissionAction.ASK)
    # Read-only tools are always allowed. ``read_file`` doubles as the
    # directory-listing tool (there is no separate ``list_files`` tool),
    # and the built-in content-search tool is ``search``.
    for tool in ("read_file", "search", "web_fetch", "ask_question"):
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
    # Read-only tools are allowed. ``read_file`` doubles as the
    # directory-listing tool (there is no separate ``list_files`` tool),
    # and the built-in content-search tool is ``search``.
    for tool in ("read_file", "search", "web_fetch", "ask_question"):
        engine.add_rule(
            PermissionRule(
                tool=tool,
                pattern="*",
                action=PermissionAction.ALLOW,
                description=f"Read-only tool {tool} allowed in plan mode",
            )
        )
    # Plan mode promises read-only behaviour. Rather than blanket-allowing
    # ``run_command`` (which would let destructive commands through at the
    # engine level), every command asks the user so the read-only promise
    # is enforced by the engine, not only by LLM prompt constraints.
    engine.add_rule(
        PermissionRule(
            tool="run_command",
            pattern="*",
            action=PermissionAction.ASK,
            description="run_command asks in plan mode (read-only guarantee)",
        )
    )
    return engine


def create_yolo_mode_engine() -> PermissionEngine:
    """Create a permission engine for Yolo mode.

    Yolo mode allows everything without asking.
    """

    return PermissionEngine(default_action=PermissionAction.ALLOW)


# Re-export enterprise extensions for convenience.
from myagent.permissions.enterprise import (
    ClearanceLevel,
    DataOperation,
    DataPermission,
    DataResource,
    EnterpriseDecision,
    EnterprisePermissionEngine,
    HardwareResource,
    ResourceCriticality,
    ResourceOperation,
    ResourcePermission,
    UserContext,
    create_enterprise_engine,
)

__all__ = [
    "PermissionAction",
    "PermissionDecision",
    "PermissionEngine",
    "PermissionRule",
    "PermissionScope",
    "create_act_mode_engine",
    "create_plan_mode_engine",
    "create_yolo_mode_engine",
    # Enterprise extensions
    "ClearanceLevel",
    "DataOperation",
    "DataPermission",
    "DataResource",
    "EnterpriseDecision",
    "EnterprisePermissionEngine",
    "HardwareResource",
    "ResourceCriticality",
    "ResourceOperation",
    "ResourcePermission",
    "UserContext",
    "create_enterprise_engine",
]
