"""MyAgent plugin development SDK.

This package provides helpers, base classes, and testing utilities for building
MyAgent plugins.
"""

from __future__ import annotations

from myagent.core.sso import SsoProvider
from myagent.exceptions import VerifyError

from myagent_sdk.plugin import CommandContext, FixSuggestion, Plugin, hook
from myagent_sdk.templates import TemplateError, create_plugin
from myagent_sdk.testing import PluginTestHarness

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
