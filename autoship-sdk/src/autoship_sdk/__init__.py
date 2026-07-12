"""AutoShip plugin development SDK.

This package provides helpers, base classes, and testing utilities for building
AutoShip plugins.
"""

from __future__ import annotations

from autoship.core.sso import SsoProvider
from autoship.exceptions import VerifyError

from autoship_sdk.plugin import CommandContext, FixSuggestion, Plugin, hook
from autoship_sdk.templates import TemplateError, create_plugin
from autoship_sdk.testing import PluginTestHarness

__version__ = "1.2.0"

__all__ = [
    "Plugin",
    "hook",
    "PluginTestHarness",
    "create_plugin",
    "TemplateError",
    "SsoProvider",
    "CommandContext",
    "FixSuggestion",
    "VerifyError",
]
