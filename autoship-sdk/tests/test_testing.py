"""Tests for autoship_sdk.testing."""

from __future__ import annotations

from pathlib import Path

from autoship_sdk import Plugin, hook
from autoship_sdk.testing import PluginTestHarness


class ContextCapturePlugin(Plugin):
    """Plugin that captures the context it receives."""

    def __init__(self) -> None:
        self.contexts: list[object] = []

    @hook
    def pre_commit(self, context) -> None:  # type: ignore[no-untyped-def]
        self.contexts.append(context)


def test_harness_creates_default_context() -> None:
    harness = PluginTestHarness()
    context = harness.make_context("verify")
    assert context.command == "verify"
    assert context.trace_id == "test-trace"
    assert context.project_root == Path(".")


def test_harness_creates_context_with_overrides(tmp_path: Path) -> None:
    harness = PluginTestHarness()
    context = harness.make_context(
        "upload",
        project_root=tmp_path,
        trace_id="custom-trace",
        extras={"fix": True},
    )
    assert context.command == "upload"
    assert context.project_root == tmp_path
    assert context.trace_id == "custom-trace"
    assert context.extras["fix"] is True


def test_harness_invokes_registered_plugin() -> None:
    harness = PluginTestHarness()
    plugin = ContextCapturePlugin()
    harness.register(plugin)
    context = harness.make_context("commit")
    harness.call("pre_commit", context)
    assert plugin.contexts == [context]


def test_harness_unregister_removes_plugin() -> None:
    harness = PluginTestHarness()
    plugin = ContextCapturePlugin()
    harness.register(plugin)
    harness.unregister(plugin)
    context = harness.make_context("commit")
    harness.call("pre_commit", context)
    assert plugin.contexts == []
