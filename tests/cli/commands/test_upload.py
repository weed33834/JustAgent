"""Tests for the upload command."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autoship.cli.commands import upload
from autoship.exceptions import ConfigError
from autoship.models.config import AppConfig


def test_upload_pypi_dry_run(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": True,
        "yes": True,
        "verbose": False,
    }
    upload.upload(ctx, target="pypi")


def test_upload_unknown_target_dry_run(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": True,
        "yes": True,
        "verbose": False,
    }
    # In dry-run mode, unknown targets print a friendly message instead of raising.
    upload.upload(ctx, target="unknown")


def test_upload_unknown_target_no_dry_run(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with pytest.raises(ConfigError):
        upload.upload(ctx, target="unknown")


def test_upload_user_abort(project_root, app_config: AppConfig, monkeypatch) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": False,
        "verbose": False,
    }
    monkeypatch.setattr(upload.typer, "confirm", lambda _: False)
    with pytest.raises(upload.typer.Exit) as exc:
        upload.upload(ctx, target="pypi")
    assert exc.value.exit_code == 0


def test_format_dry_run_details_formats_list() -> None:
    details = {"target": "pypi", "artifacts": ["dist/a.whl", "dist/b.whl"]}
    formatted = upload._format_dry_run_details(details)
    assert "target: pypi" in formatted
    assert "artifacts: dist/a.whl, dist/b.whl" in formatted


def test_format_dry_run_details_ignores_dry_run_key() -> None:
    details = {"dry_run": True, "repository": "pypi"}
    formatted = upload._format_dry_run_details(details)
    assert "dry_run" not in formatted
    assert "repository: pypi" in formatted


def test_format_dry_run_details_returns_empty_for_none() -> None:
    assert upload._format_dry_run_details(None) == ""
