"""$description"""

from __future__ import annotations

from myagent.core.context import CommandContext
from myagent.plugins.defaults import FixSuggestion
from myagent_sdk import Plugin, hook


class $class_name(Plugin):
    """$description"""

    @hook
    def pre_commit(self, context: CommandContext) -> None:
        """Called before ``myagent commit`` runs."""

    @hook
    def on_error(self, context: CommandContext, error: Exception) -> FixSuggestion | None:
        """Called when a command raises an error; may return a fix suggestion."""
        return None


def register() -> $class_name:
    """Factory used by the ``myagent.plugins`` entry point."""
    return $class_name()
