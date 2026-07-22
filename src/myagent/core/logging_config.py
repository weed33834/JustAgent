"""Structlog configuration: Rich console in verbose mode, JSON to audit file.

Provides a single ``configure_structlog()`` entry point that
configures the root structlog logger. Console output uses
``rich`` for colorized rendering when ``--verbose`` is active;
audit logs are always written as JSON Lines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console


def _add_audit_context(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Strip internal structlog keys before writing JSON."""
    return event_dict


def configure_structlog(
    *,
    verbose: bool = False,
    audit_log_path: Path | None = None,
) -> None:
    """Configure structlog with Rich console + JSON file outputs.

    - Console: Rich-enhanced colored output (verbose) or plain (default).
    - Audit: JSON Lines written to *audit_log_path* when set.
    """
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: structlog.types.Processor
    if verbose:
        _console = Console(stderr=True, highlight=True)
        structlog.stdlib.recreate_defaults(log_level=logging.DEBUG)
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        structlog.stdlib.recreate_defaults(log_level=logging.WARNING)
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors + [renderer],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure audit JSON file output if path is provided.
    if audit_log_path is not None:
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        audit_handler = logging.FileHandler(str(audit_log_path), encoding="utf-8")
        audit_handler.setLevel(logging.INFO)
        _json_processor = structlog.processors.JSONRenderer()
        _json_chain = shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
        _formatter = structlog.stdlib.ProcessorFormatter(
            processor=_json_processor,
            foreign_pre_chain=_json_chain,
        )
        audit_handler.setFormatter(_formatter)
        root_logger = structlog.get_logger()
        root_logger.addHandler(audit_handler)
