"""Base classes and decorators for AutoShip plugins."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from autoship.core.context import CommandContext
from autoship.core.fix import FixSuggestion
from autoship.hookspec import hookimpl


class Plugin:
    """Base class for AutoShip plugins.

    Subclasses define hook methods and decorate them with :func:`hook`. The
    dispatcher will discover and invoke these methods at the appropriate
    lifecycle points.

    Example:
        >>> from autoship_sdk import Plugin, hook
        >>> from autoship.core.context import CommandContext
        >>> class MyPlugin(Plugin):
        ...     @hook
        ...     def pre_commit(self, context: CommandContext) -> None:
        ...         print("About to commit")
    """

    @property
    def name(self) -> str:
        """Return the plugin name, defaults to the class name."""
        return self.__class__.__name__

    def register(self, dispatcher: Any) -> None:
        """Register this plugin instance with a hook dispatcher.

        Args:
            dispatcher: An object with a ``register`` method, such as
                :class:`autoship.core.hook_dispatcher.HookDispatcher`.
        """
        dispatcher.register(self)


def hook(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a plugin method as a hook implementation.

    This is a thin wrapper around ``autoship.hookspec.hookimpl`` that can be
    used on methods of a :class:`Plugin` subclass.
    """
    return hookimpl(func)


__all__ = ["Plugin", "hook", "CommandContext", "FixSuggestion"]
