"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoship.core.audit_logger import AuditLogger
from autoship.core.context import CommandContext
from autoship.core.i18n import get_i18n
from autoship.models.config import AppConfig

os.environ.setdefault("LANG", "en_US.UTF-8")


@pytest.fixture
def i18n():
    """Return an English I18n instance for tests."""
    return get_i18n("en")


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Return a temporary project root."""
    return tmp_path


@pytest.fixture
def app_config(project_root: Path) -> AppConfig:
    """Return a default AppConfig rooted in a temp directory."""
    return AppConfig(project_root=project_root)


@pytest.fixture
def audit_logger(app_config: AppConfig):
    """Return an AuditLogger using a temp-bound config; close on teardown."""
    audit = AuditLogger(app_config)
    yield audit
    audit.close()


@pytest.fixture
def command_context(app_config: AppConfig, audit_logger: AuditLogger) -> CommandContext:
    """Return a populated CommandContext for testing."""
    return CommandContext(
        command="test",
        project_root=app_config.project_root,
        config=app_config,
        trace_id=audit_logger.trace_id,
    )


@pytest.fixture
def mock_config(project_root: Path) -> AppConfig:
    """Return an AppConfig with mocked model backends."""
    return AppConfig(
        project_root=project_root,
        model={
            "default_tier": 2,
            "fallback": False,
            "backends": [],
        },
    )


@pytest.fixture
def typer_context(app_config: AppConfig, audit_logger: AuditLogger, i18n) -> MagicMock:
    """Return a mocked typer.Context with AutoShip state."""
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "config_path": None,
        "i18n": i18n,
        "audit_logger": audit_logger,
        "dry_run": False,
        "yes": False,
        "verbose": False,
    }
    return ctx
