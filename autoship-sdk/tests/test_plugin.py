"""Tests for autoship_sdk.plugin."""

from __future__ import annotations

from autoship.core.context import CommandContext
from autoship.core.fix import FixSuggestion

from autoship_sdk import Plugin, hook


class SamplePlugin(Plugin):
    """Test plugin."""

    def __init__(self) -> None:
        self.ran_hooks: list[str] = []

    @hook
    def pre_commit(self, context: CommandContext) -> None:
        self.ran_hooks.append("pre_commit")

    @hook
    def on_error(self, context: CommandContext, error: Exception) -> FixSuggestion | None:
        return FixSuggestion(description="sample fix")


def test_plugin_name_defaults_to_class_name() -> None:
    plugin = SamplePlugin()
    assert plugin.name == "SamplePlugin"


def test_hook_decorator_marks_method_as_hookimpl() -> None:
    assert "autoship_impl" in SamplePlugin.pre_commit.__dict__


def test_plugin_register_calls_dispatcher_register(mocker) -> None:
    plugin = SamplePlugin()
    dispatcher = mocker.Mock()
    plugin.register(dispatcher)
    dispatcher.register.assert_called_once_with(plugin)


def test_plugin_hook_is_invoked_by_harness() -> None:
    from autoship_sdk.testing import PluginTestHarness

    plugin = SamplePlugin()
    harness = PluginTestHarness()
    harness.register(plugin)
    context = harness.make_context("commit")
    harness.call("pre_commit", context)
    assert plugin.ran_hooks == ["pre_commit"]
