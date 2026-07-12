"""$description"""

from __future__ import annotations

from autoship.core.context import CommandContext
from autoship.core.fix import FixSuggestion
from autoship_sdk import Plugin, hook


class $class_name(Plugin):
    """$description"""

    @hook
    def pre_commit(self, context: CommandContext) -> None:
        """Called before ``autoship commit`` runs."""

    @hook
    def on_error(self, context: CommandContext, error: Exception) -> FixSuggestion | None:
        """Called when a command raises an error; may return a fix suggestion."""
        return None


def register() -> $class_name:
    """Factory used by the ``autoship.plugins`` entry point."""
    return $class_name()
