"""Tests for the commit command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoship.cli.commands import commit
from autoship.exceptions import GitError, ModelGatewayError
from autoship.models.config import AppConfig


def test_commit_no_changes(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with (
        patch.object(commit.GitAdapter, "is_git_repo", return_value=True),
        patch.object(commit.GitAdapter, "has_changes", return_value=False),
    ):
        result = commit.commit(ctx)
    assert result is None


def test_commit_with_message_dry_run(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": True,
        "yes": True,
        "verbose": False,
    }
    with (
        patch.object(commit.GitAdapter, "is_git_repo", return_value=True),
        patch.object(commit.GitAdapter, "has_changes", return_value=True),
        patch.object(commit.GitAdapter, "diff", return_value="diff"),
        patch.object(commit.GitAdapter, "stats", return_value="stats"),
    ):
        commit.commit(ctx, message="fix: bug")


def test_commit_generates_message_and_commits(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with (
        patch.object(commit.GitAdapter, "is_git_repo", return_value=True),
        patch.object(commit.GitAdapter, "has_changes", return_value=True),
        patch.object(commit.GitAdapter, "diff", return_value="diff"),
        patch.object(commit.GitAdapter, "stats", return_value="stats"),
        patch.object(
            commit.ModelRouter, "generate_commit_message", return_value="feat: add feature"
        ),
        patch.object(commit.GitAdapter, "commit") as mock_commit,
    ):
        commit.commit(ctx, message=None)
    mock_commit.assert_called_once_with("feat: add feature")


def test_commit_model_failure_fallback(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with (
        patch.object(commit.GitAdapter, "is_git_repo", return_value=True),
        patch.object(commit.GitAdapter, "has_changes", return_value=True),
        patch.object(commit.GitAdapter, "diff", return_value="diff"),
        patch.object(commit.GitAdapter, "stats", return_value="stats"),
        patch.object(
            commit.ModelRouter, "generate_commit_message", side_effect=ModelGatewayError("down")
        ),
        patch.object(commit.GitAdapter, "commit") as mock_commit,
    ):
        commit.commit(ctx, message=None)
    mock_commit.assert_called_once_with("Update files")


def test_commit_git_error_raises(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with (
        patch.object(commit.GitAdapter, "is_git_repo", return_value=True),
        patch.object(commit.GitAdapter, "has_changes", return_value=True),
        patch.object(commit.GitAdapter, "diff", return_value="diff"),
        patch.object(commit.GitAdapter, "stats", return_value="stats"),
        patch.object(commit.ModelRouter, "generate_commit_message", return_value="feat: update"),
        patch.object(commit.GitAdapter, "commit", side_effect=GitError("boom")),
        pytest.raises(GitError),
    ):
        commit.commit(ctx, message=None)


def test_validate_editor_accepts_allowed_editor(i18n) -> None:
    executable = commit._validate_editor("vim", ["vim", "nvim"], i18n)
    assert executable == "vim"


def test_validate_editor_accepts_allowed_editor_with_args(i18n) -> None:
    executable = commit._validate_editor("code --wait", ["code", "vim"], i18n)
    assert executable == "code"


def test_validate_editor_rejects_semicolon_injection(i18n) -> None:
    with pytest.raises(GitError, match="not allowed"):
        commit._validate_editor("vim; rm -rf /", ["vim", "nvim"], i18n)


def test_validate_editor_rejects_path_traversal(i18n) -> None:
    with pytest.raises(GitError, match="not allowed"):
        commit._validate_editor("../bin/evil-editor", ["evil-editor"], i18n)


def test_validate_editor_rejects_unknown_editor(i18n) -> None:
    with pytest.raises(GitError, match="not in the allowlist"):
        commit._validate_editor("malicious", ["vim", "nvim"], i18n)


def test_validate_editor_uses_configurable_allowlist(i18n) -> None:
    executable = commit._validate_editor("custom-editor", ["custom-editor"], i18n)
    assert executable == "custom-editor"


def test_open_editor_rejects_injected_editor(i18n, monkeypatch) -> None:
    monkeypatch.setenv("EDITOR", "vim; rm -rf /")
    with pytest.raises(GitError, match="not allowed"):
        commit._open_editor(i18n, "initial", ["vim", "nvim"])
