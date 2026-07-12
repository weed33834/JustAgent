"""Testing utilities for AutoShip plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoship.core.context import CommandContext
from autoship.core.hook_dispatcher import HookDispatcher
from autoship.models.config import AppConfig


class PluginTestHarness:
    """Helper to test plugin hooks in isolation.

    Example:
        >>> from autoship_sdk import Plugin, hook
        >>> from autoship_sdk.testing import PluginTestHarness
        >>> class MyPlugin(Plugin):
        ...     @hook
        ...     def pre_commit(self, context):
        ...         context.extras["ran"] = True
        >>> harness = PluginTestHarness()
        >>> plugin = MyPlugin()
        >>> harness.register(plugin)
        >>> context = harness.make_context("commit")
        >>> harness.call("pre_commit", context)
        >>> assert context.extras["ran"]
    """

    def __init__(self) -> None:
        self.dispatcher = HookDispatcher(load_builtins=False, load_entry_points=False)

    def register(self, plugin: Any) -> None:
        """Register a plugin instance with the internal dispatcher."""
        self.dispatcher.pm.register(plugin)

    def unregister(self, plugin: Any) -> None:
        """Unregister a plugin instance from the internal dispatcher."""
        self.dispatcher.pm.unregister(plugin)

    def make_context(
        self,
        command: str,
        project_root: Path | None = None,
        config: AppConfig | None = None,
        trace_id: str = "test-trace",
        extras: dict[str, Any] | None = None,
    ) -> CommandContext:
        """Create a ``CommandContext`` suitable for hook tests."""
        return CommandContext(
            command=command,
            project_root=project_root or Path("."),
            config=config or AppConfig(),
            trace_id=trace_id,
            extras=extras or {},
        )

    def call(
        self,
        hook_name: str,
        context: CommandContext,
        *,
        fail_fast: bool = True,
        **kwargs: Any,
    ) -> list[Any]:
        """Invoke ``hook_name`` on all registered plugins."""
        return self.dispatcher.call(hook_name, context, fail_fast=fail_fast, **kwargs)
