"""Plugin hook specifications for JustAgent-CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pluggy

from justagent.core.context import CommandContext

if TYPE_CHECKING:
    # 仅用于类型注解；运行时避免导入，防止与 plugins.defaults 形成循环导入。
    from justagent.plugins.defaults import FixSuggestion

hookspec = pluggy.HookspecMarker("justagent")
hookimpl = pluggy.HookimplMarker("justagent")


class MyAgentHookSpec:
    """Lifecycle hooks available to plugins."""

    @hookspec
    def pre_init(self, context: CommandContext) -> None:
        """Called before ``justagent init`` writes the config file."""

    @hookspec
    def post_init(self, context: CommandContext) -> None:
        """Called after ``justagent init`` writes the config file."""

    @hookspec
    def pre_clean(self, context: CommandContext) -> None:
        """Called before ``justagent clean`` runs formatters."""

    @hookspec
    def post_clean(self, context: CommandContext) -> None:
        """Called after ``justagent clean`` runs formatters."""

    @hookspec
    def pre_commit(self, context: CommandContext) -> None:
        """Called before ``justagent commit`` generates/uses a message."""

    @hookspec
    def post_commit(self, context: CommandContext) -> None:
        """Called after ``justagent commit`` completes."""

    @hookspec
    def pre_verify(self, context: CommandContext) -> None:
        """Called before ``justagent verify`` runs the verification command."""

    @hookspec
    def post_verify(self, context: CommandContext) -> None:
        """Called after ``justagent verify`` completes."""

    @hookspec
    def pre_upload(self, context: CommandContext) -> None:
        """Called before ``justagent upload`` publishes artifacts."""

    @hookspec
    def post_upload(self, context: CommandContext) -> None:
        """Called after ``justagent upload`` completes."""

    @hookspec
    def on_error(self, context: CommandContext, error: Exception) -> FixSuggestion | None:
        """Called when a command raises an error; may return a fix suggestion."""
