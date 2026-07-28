"""Tests for the permissions rules engine."""

from __future__ import annotations

import pytest

from justagent.permissions import (
    PermissionAction,
    PermissionDecision,
    PermissionEngine,
    PermissionRule,
    PermissionScope,
    create_act_mode_engine,
    create_plan_mode_engine,
    create_yolo_mode_engine,
)

# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class TestPermissionAction:
    def test_values(self) -> None:
        assert PermissionAction.ALLOW.value == "allow"
        assert PermissionAction.DENY.value == "deny"
        assert PermissionAction.ASK.value == "ask"

    def test_from_string(self) -> None:
        assert PermissionAction("allow") is PermissionAction.ALLOW
        assert PermissionAction("deny") is PermissionAction.DENY


class TestPermissionScope:
    def test_values(self) -> None:
        assert PermissionScope.ONCE.value == "once"
        assert PermissionScope.ALWAYS.value == "always"


class TestPermissionRule:
    def test_defaults(self) -> None:
        rule = PermissionRule(tool="write_to_file")
        assert rule.pattern == "*"
        assert rule.action is PermissionAction.ASK
        assert rule.scope is PermissionScope.ALWAYS
        assert rule.description == ""

    def test_is_frozen(self) -> None:
        rule = PermissionRule(tool="x")
        with pytest.raises(AttributeError):
            rule.tool = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PermissionEngine — basic check
# ---------------------------------------------------------------------------


class TestEngineBasicCheck:
    def test_no_rules_returns_default(self) -> None:
        engine = PermissionEngine(default_action=PermissionAction.ASK)
        decision = engine.check("write_to_file", {"path": "/tmp/test.txt"})
        assert decision.action is PermissionAction.ASK
        assert decision.matched_rule is None

    def test_default_allow(self) -> None:
        engine = PermissionEngine(default_action=PermissionAction.ALLOW)
        decision = engine.check("any_tool", {"x": 1})
        assert decision.action is PermissionAction.ALLOW

    def test_default_deny(self) -> None:
        engine = PermissionEngine(default_action=PermissionAction.DENY)
        decision = engine.check("any_tool", {"x": 1})
        assert decision.action is PermissionAction.DENY

    def test_rule_allows_specific_tool(self) -> None:
        engine = PermissionEngine()
        engine.add_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        decision = engine.check("read_file", {"path": "/etc/passwd"})
        assert decision.action is PermissionAction.ALLOW
        assert decision.matched_rule is not None

    def test_rule_denies_specific_tool(self) -> None:
        engine = PermissionEngine()
        engine.add_rule(
            PermissionRule(tool="write_to_file", action=PermissionAction.DENY)
        )
        decision = engine.check("write_to_file", {"path": "/x"})
        assert decision.action is PermissionAction.DENY

    def test_first_matching_rule_wins(self) -> None:
        engine = PermissionEngine()
        engine.add_rule(
            PermissionRule(tool="write_to_file", pattern="/tmp/*", action=PermissionAction.ALLOW)
        )
        engine.add_rule(
            PermissionRule(tool="write_to_file", pattern="*", action=PermissionAction.DENY)
        )
        assert engine.check("write_to_file", {"path": "/tmp/x"}).action is PermissionAction.ALLOW
        assert engine.check("write_to_file", {"path": "/home/x"}).action is PermissionAction.DENY

    def test_wildcard_tool_matches_all(self) -> None:
        engine = PermissionEngine()
        engine.add_rule(
            PermissionRule(tool="*", action=PermissionAction.ALLOW)
        )
        assert engine.check("any_tool", {"x": 1}).action is PermissionAction.ALLOW
        assert engine.check("other_tool", {"y": 2}).action is PermissionAction.ALLOW

    def test_non_matching_tool_uses_default(self) -> None:
        engine = PermissionEngine(default_action=PermissionAction.ASK)
        engine.add_rule(
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        )
        # write_to_file has no rule → default ASK
        assert engine.check("write_to_file", {"path": "/x"}).action is PermissionAction.ASK


# ---------------------------------------------------------------------------
# PermissionEngine — pattern matching
# ---------------------------------------------------------------------------


class TestPatternMatching:
    def test_glob_pattern_matches_path(self) -> None:
        engine = PermissionEngine(default_action=PermissionAction.DENY)
        engine.add_rule(
            PermissionRule(
                tool="write_to_file", pattern="/tmp/*", action=PermissionAction.ALLOW
            )
        )
        assert engine.check("write_to_file", {"path": "/tmp/test.txt"}).action is PermissionAction.ALLOW
        assert engine.check("write_to_file", {"path": "/home/test.txt"}).action is PermissionAction.DENY

    def test_glob_pattern_matches_command(self) -> None:
        engine = PermissionEngine(default_action=PermissionAction.DENY)
        engine.add_rule(
            PermissionRule(
                tool="run_command", pattern="ls *", action=PermissionAction.ALLOW
            )
        )
        assert engine.check("run_command", {"command": "ls -la"}).action is PermissionAction.ALLOW
        assert engine.check("run_command", {"command": "rm -rf /"}).action is PermissionAction.DENY

    def test_star_pattern_matches_everything(self) -> None:
        engine = PermissionEngine(default_action=PermissionAction.DENY)
        engine.add_rule(
            PermissionRule(tool="*", pattern="*", action=PermissionAction.ALLOW)
        )
        assert engine.check("any_tool", {"path": "/anything"}).action is PermissionAction.ALLOW
        assert engine.check("other", {"command": "anything"}).action is PermissionAction.ALLOW


# ---------------------------------------------------------------------------
# PermissionEngine — remembering decisions
# ---------------------------------------------------------------------------


class TestRemembering:
    def test_always_scope_remembers_allow(self) -> None:
        engine = PermissionEngine()
        engine.add_rule(
            PermissionRule(
                tool="write_to_file",
                pattern="*",
                action=PermissionAction.ALLOW,
                scope=PermissionScope.ALWAYS,
            )
        )
        # First check matches the rule and remembers.
        d1 = engine.check("write_to_file", {"path": "/tmp/x"})
        assert d1.action is PermissionAction.ALLOW
        # Second check should hit the remembered cache.
        d2 = engine.check("write_to_file", {"path": "/tmp/x"})
        assert d2.action is PermissionAction.ALLOW
        assert "Remembered" in d2.reason

    def test_remember_manual_decision(self) -> None:
        engine = PermissionEngine(default_action=PermissionAction.ASK)
        # Simulate user saying "allow always"
        engine.remember(
            "write_to_file",
            {"path": "/tmp/x"},
            PermissionAction.ALLOW,
            PermissionScope.ALWAYS,
        )
        decision = engine.check("write_to_file", {"path": "/tmp/x"})
        assert decision.action is PermissionAction.ALLOW
        assert "Remembered" in decision.reason

    def test_once_scope_does_not_remember(self) -> None:
        engine = PermissionEngine()
        engine.remember(
            "write_to_file",
            {"path": "/tmp/x"},
            PermissionAction.ALLOW,
            PermissionScope.ONCE,
        )
        # Should still use the default (ASK) since once doesn't cache.
        decision = engine.check("write_to_file", {"path": "/tmp/x"})
        assert decision.action is PermissionAction.ASK

    def test_forget_removes_remembered(self) -> None:
        engine = PermissionEngine()
        engine.remember(
            "write_to_file",
            {"path": "/tmp/x"},
            PermissionAction.ALLOW,
        )
        engine.forget("write_to_file", {"path": "/tmp/x"})
        decision = engine.check("write_to_file", {"path": "/tmp/x"})
        assert decision.action is PermissionAction.ASK  # default

    def test_clear_remembered(self) -> None:
        engine = PermissionEngine()
        engine.remember("tool1", {"path": "a"}, PermissionAction.ALLOW)
        engine.remember("tool2", {"path": "b"}, PermissionAction.DENY)
        engine.clear_remembered()
        assert engine.check("tool1", {"path": "a"}).action is PermissionAction.ASK
        assert engine.check("tool2", {"path": "b"}).action is PermissionAction.ASK

    def test_different_inputs_not_confused(self) -> None:
        engine = PermissionEngine()
        engine.remember("write_to_file", {"path": "/a"}, PermissionAction.ALLOW)
        # Different path should not be remembered.
        assert engine.check("write_to_file", {"path": "/b"}).action is PermissionAction.ASK


# ---------------------------------------------------------------------------
# PermissionEngine — rule management
# ---------------------------------------------------------------------------


class TestRuleManagement:
    def test_add_and_list_rules(self) -> None:
        engine = PermissionEngine()
        r1 = PermissionRule(tool="a", action=PermissionAction.ALLOW)
        r2 = PermissionRule(tool="b", action=PermissionAction.DENY)
        engine.add_rule(r1)
        engine.add_rule(r2)
        rules = engine.rules
        assert len(rules) == 2
        assert rules[0] == r1
        assert rules[1] == r2

    def test_remove_rule_by_index(self) -> None:
        engine = PermissionEngine()
        r1 = PermissionRule(tool="a")
        engine.add_rule(r1)
        removed = engine.remove_rule(0)
        assert removed == r1
        assert len(engine.rules) == 0

    def test_remove_invalid_index_returns_none(self) -> None:
        engine = PermissionEngine()
        assert engine.remove_rule(0) is None
        assert engine.remove_rule(-1) is None

    def test_clear_rules(self) -> None:
        engine = PermissionEngine()
        engine.add_rule(PermissionRule(tool="a"))
        engine.add_rule(PermissionRule(tool="b"))
        engine.clear_rules()
        assert len(engine.rules) == 0

    def test_clear_also_clears_remembered(self) -> None:
        engine = PermissionEngine()
        engine.remember("x", {"path": "a"}, PermissionAction.ALLOW)
        engine.clear_rules()
        assert engine.check("x", {"path": "a"}).action is PermissionAction.ASK

    def test_default_action_setter(self) -> None:
        engine = PermissionEngine(default_action=PermissionAction.ASK)
        assert engine.default_action is PermissionAction.ASK
        engine.default_action = PermissionAction.ALLOW
        assert engine.default_action is PermissionAction.ALLOW
        assert engine.check("x", {}).action is PermissionAction.ALLOW


# ---------------------------------------------------------------------------
# Mode-specific engines
# ---------------------------------------------------------------------------


class TestModeEngines:
    def test_act_mode_allows_read_tools(self) -> None:
        engine = create_act_mode_engine()
        assert engine.check("read_file", {"path": "/x"}).action is PermissionAction.ALLOW
        assert engine.check("search", {"pattern": "x"}).action is PermissionAction.ALLOW
        assert engine.check("web_fetch", {"url": "http://x"}).action is PermissionAction.ALLOW
        assert engine.check("ask_question", {"question": "x"}).action is PermissionAction.ALLOW

    def test_act_mode_asks_for_writes(self) -> None:
        engine = create_act_mode_engine()
        assert engine.check("write_to_file", {"path": "/x"}).action is PermissionAction.ASK
        assert engine.check("apply_patch", {"patch": "x"}).action is PermissionAction.ASK

    def test_plan_mode_allows_read_tools(self) -> None:
        engine = create_plan_mode_engine()
        assert engine.check("read_file", {"path": "/x"}).action is PermissionAction.ALLOW

    def test_plan_mode_denies_writes(self) -> None:
        engine = create_plan_mode_engine()
        assert engine.check("write_to_file", {"path": "/x"}).action is PermissionAction.DENY
        assert engine.check("apply_patch", {"patch": "x"}).action is PermissionAction.DENY

    def test_plan_mode_asks_for_run_command(self) -> None:
        engine = create_plan_mode_engine()
        # Plan mode promises read-only behaviour: every command must ask
        # the user rather than being auto-allowed at the engine level.
        assert engine.check("run_command", {"command": "ls"}).action is PermissionAction.ASK

    def test_yolo_mode_allows_everything(self) -> None:
        engine = create_yolo_mode_engine()
        assert engine.check("write_to_file", {"path": "/x"}).action is PermissionAction.ALLOW
        assert engine.check("run_command", {"command": "rm -rf /"}).action is PermissionAction.ALLOW
        assert engine.check("any_tool", {}).action is PermissionAction.ALLOW

    def test_yolo_mode_has_no_rules(self) -> None:
        engine = create_yolo_mode_engine()
        assert len(engine.rules) == 0

    def test_act_mode_has_read_rules(self) -> None:
        engine = create_act_mode_engine()
        assert len(engine.rules) >= 4  # read_file, search, web_fetch, ask_question

    def test_plan_mode_has_read_and_command_rules(self) -> None:
        engine = create_plan_mode_engine()
        tool_names = {r.tool for r in engine.rules}
        assert "read_file" in tool_names
        assert "run_command" in tool_names


# ---------------------------------------------------------------------------
# PermissionDecision
# ---------------------------------------------------------------------------


class TestPermissionDecision:
    def test_fields(self) -> None:
        rule = PermissionRule(tool="x", action=PermissionAction.ALLOW)
        d = PermissionDecision(
            action=PermissionAction.ALLOW,
            matched_rule=rule,
            reason="test",
        )
        assert d.action is PermissionAction.ALLOW
        assert d.matched_rule == rule
        assert d.reason == "test"

    def test_is_frozen(self) -> None:
        d = PermissionDecision(action=PermissionAction.ALLOW)
        with pytest.raises(AttributeError):
            d.action = PermissionAction.DENY  # type: ignore[misc]
