"""JustAgent plugin development SDK.

This package provides helpers, base classes, and testing utilities for building
JustAgent plugins.
"""

from __future__ import annotations

from justagent.exceptions import VerifyError

from justagent_sdk.plugin import CommandContext, FixSuggestion, Plugin, hook
from justagent_sdk.templates import TemplateError, create_plugin
from justagent_sdk.testing import PluginTestHarness

__version__ = "2.0.0"

__all__ = [
    "Plugin",
    "hook",
    "PluginTestHarness",
    "create_plugin",
    "TemplateError",
    "CommandContext",
    "FixSuggestion",
    "VerifyError",
]
