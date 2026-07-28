"""$description"""

from __future__ import annotations

from justagent.core.context import CommandContext
from justagent.plugins.defaults import FixSuggestion
from justagent_sdk import Plugin, hook


class $class_name(Plugin):
    """$description"""

    @hook
    def pre_commit(self, context: CommandContext) -> None:
        """Called before ``justagent commit`` runs."""

    @hook
    def on_error(self, context: CommandContext, error: Exception) -> FixSuggestion | None:
        """Called when a command raises an error; may return a fix suggestion."""
        return None


def register() -> $class_name:
    """Factory used by the ``justagent.plugins`` entry point."""
    return $class_name()
