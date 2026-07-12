"""Auto-discover and register all Typer subcommands."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules

import typer


def register_all(parent: typer.Typer) -> None:
    """Register every module-level ``register`` function as a Typer subcommand."""
    package = __package__ or "autoship.cli.commands"
    for _, module_name, _ in iter_modules(__path__, prefix=package + "."):
        mod = import_module(module_name)
        register = getattr(mod, "register", None)
        if callable(register):
            register(parent)
