"""Plugin hook specifications for MyAgent-CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pluggy

from myagent.core.context import CommandContext

if TYPE_CHECKING:
    # 仅用于类型注解；运行时避免导入，防止与 plugins.defaults 形成循环导入。
    from myagent.plugins.defaults import FixSuggestion

hookspec = pluggy.HookspecMarker("myagent")
hookimpl = pluggy.HookimplMarker("myagent")


class MyAgentHookSpec:
    """Lifecycle hooks available to plugins."""

    @hookspec
    def pre_init(self, context: CommandContext) -> None:
        """Called before ``myagent init`` writes the config file."""

    @hookspec
    def post_init(self, context: CommandContext) -> None:
        """Called after ``myagent init`` writes the config file."""

    @hookspec
    def pre_clean(self, context: CommandContext) -> None:
        """Called before ``myagent clean`` runs formatters."""

    @hookspec
    def post_clean(self, context: CommandContext) -> None:
        """Called after ``myagent clean`` runs formatters."""

    @hookspec
    def pre_commit(self, context: CommandContext) -> None:
        """Called before ``myagent commit`` generates/uses a message."""

    @hookspec
    def post_commit(self, context: CommandContext) -> None:
        """Called after ``myagent commit`` completes."""

    @hookspec
    def pre_verify(self, context: CommandContext) -> None:
        """Called before ``myagent verify`` runs the verification command."""

    @hookspec
    def post_verify(self, context: CommandContext) -> None:
        """Called after ``myagent verify`` completes."""

    @hookspec
    def pre_upload(self, context: CommandContext) -> None:
        """Called before ``myagent upload`` publishes artifacts."""

    @hookspec
    def post_upload(self, context: CommandContext) -> None:
        """Called after ``myagent upload`` completes."""

    @hookspec
    def on_error(self, context: CommandContext, error: Exception) -> FixSuggestion | None:
        """Called when a command raises an error; may return a fix suggestion."""
