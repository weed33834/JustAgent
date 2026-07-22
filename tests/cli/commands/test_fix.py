"""Tests for the LLM fix command."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from myagent.cli.commands import fix as fix_module
from myagent.cli.main import app
from myagent.exceptions import ConfigError
from myagent.models.config import ToolConfig, ToolsConfig

runner = CliRunner()


def test_fix_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    """The fix command must fail when no model backend is available."""
    monkeypatch.chdir(tmp_path)
    error_log = tmp_path / "error.txt"
    error_log.write_text("SyntaxError", encoding="utf-8")
    # API keys cannot be set via environment (SENSITIVE_ENV_KEYS blocks them),
    # so a project without configured backends should fail cleanly.
    result = runner.invoke(app, ["fix", str(error_log)])
    assert result.exit_code != 0


def test_fix_requires_error_log(tmp_path: Path) -> None:
    missing_log = tmp_path / "missing.txt"
    result = runner.invoke(app, ["fix", str(missing_log)])
    assert result.exit_code != 0
    assert "No error log" in result.output


def test_fix_empty_error_log(tmp_path: Path) -> None:
    empty_log = tmp_path / "empty.txt"
    empty_log.write_text("   ", encoding="utf-8")
    result = runner.invoke(app, ["fix", str(empty_log)])
    assert result.exit_code != 0
    assert "empty" in result.output


def _write_config_with_api_key(root: Path) -> None:
    """Create a project config that supplies an LLM API key for tests."""
    config_file = root / ".myagent.toml"
    config_file.write_text('[llm]\napi_key = "fake-key"\n', encoding="utf-8")


def test_fix_calls_llm_and_shows_suggestion(tmp_path: Path, monkeypatch) -> None:
    _write_config_with_api_key(tmp_path)
    monkeypatch.chdir(tmp_path)
    error_log = tmp_path / "error.txt"
    error_log.write_text("NameError: name 'x' is not defined", encoding="utf-8")

    mock_router = MagicMock()
    mock_router.chat.return_value = "Add `x = 0` before usage."
    with patch("myagent.cli.commands.fix._model_router", return_value=mock_router):
        result = runner.invoke(app, ["fix", str(error_log)])

    # The LLM suggestion is still echoed to the user...
    assert "Add `x = 0`" in result.output
    # ...but the command exits non-zero because no patch was produced (dogfood
    # fix: ``fix`` must surface "did not actually fix" to callers).
    assert result.exit_code == 1


def test_fix_yes_flag_applies_patch(tmp_path: Path, monkeypatch) -> None:
    _write_config_with_api_key(tmp_path)
    monkeypatch.chdir(tmp_path)
    error_log = tmp_path / "error.txt"
    error_log.write_text("error", encoding="utf-8")

    patch_text = "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new"

    mock_router = MagicMock()
    mock_router.chat.return_value = f"```diff\n{patch_text}\n```"
    with (
        patch("myagent.cli.commands.fix._model_router", return_value=mock_router),
        patch("myagent.cli.commands.fix._apply_patch") as mock_apply,
    ):
        result = runner.invoke(app, ["fix", str(error_log), "--yes"])

    assert result.exit_code == 0
    mock_apply.assert_called_once()


def test_extract_patch_from_diff_block() -> None:
    response = "```diff\n--- a\n+++ b\n```"
    assert fix_module._extract_patch(response) == "--- a\n+++ b\n"


def test_extract_patch_from_plain_diff() -> None:
    response = "Some text\n--- a/file.txt\n+++ b/file.txt"
    assert fix_module._extract_patch(response) == "--- a/file.txt\n+++ b/file.txt\n"


def test_extract_patch_returns_none_when_no_patch() -> None:
    assert fix_module._extract_patch("No patch here") is None


def test_collect_relevant_files(tmp_path: Path) -> None:
    py_file = tmp_path / "module.py"
    py_file.write_text("x = 1", encoding="utf-8")
    error_context = f"Error in {py_file}"
    files, read_paths = fix_module._collect_relevant_files(tmp_path, error_context)
    assert "module.py" in files
    assert "module.py" in read_paths


def test_collect_relevant_files_rejects_outside_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1", encoding="utf-8")
    files, read_paths = fix_module._collect_relevant_files(tmp_path, str(outside))
    assert not files
    assert not read_paths


def test_collect_relevant_files_rejects_traversal(tmp_path: Path) -> None:
    parent_file = tmp_path.parent / "parent.py"
    parent_file.write_text("x = 1", encoding="utf-8")
    files, read_paths = fix_module._collect_relevant_files(tmp_path, "../parent.py")
    assert not files
    assert not read_paths


def test_collect_relevant_files_rejects_bad_extension(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("ssh-key", encoding="utf-8")
    files, _ = fix_module._collect_relevant_files(tmp_path, str(secret))
    assert "secret.txt" not in files


def test_collect_relevant_files_respects_size_limit(tmp_path: Path) -> None:
    big = tmp_path / "big.py"
    big.write_text("x" * (fix_module.MAX_FILE_SIZE + 1), encoding="utf-8")
    files, _ = fix_module._collect_relevant_files(tmp_path, str(big))
    assert "big.py" not in files


def test_collect_relevant_files_rejects_symlink_outside_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(outside)
    files, _ = fix_module._collect_relevant_files(tmp_path, str(link))
    assert not files


def test_collect_relevant_files_prefers_traceback_source_over_test_noise(
    tmp_path: Path,
) -> None:
    """The real failing source file must win over pytest collection noise.

    Reproduces the e2e failure: the error log lists test files first (during
    pytest collection) and the real source file only appears later in a
    traceback frame. The traceback source must be collected even though test
    files appear earlier and would have filled the old 3-slot budget.
    """
    src = tmp_path / "src" / "module.py"
    src.parent.mkdir(parents=True)
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    test_a = tmp_path / "tests" / "test_a.py"
    test_a.parent.mkdir()
    test_a.write_text("from src.module import f\n", encoding="utf-8")
    test_b = tmp_path / "tests" / "test_b.py"
    test_b.write_text("from src.module import f\n", encoding="utf-8")

    # Simulate a real pytest log: collection lists test files first, the
    # traceback frame pointing at the real source comes later.
    error_context = (
        f"tests/test_a.py ....\n"
        f"tests/test_b.py ....\n"
        f'  File "{src}", line 2, in f\n'
        f"    return 1\n"
        f"AssertionError: boom"
    )
    files, read_paths = fix_module._collect_relevant_files(tmp_path, error_context)

    # The implementation file must be present (it is the fix target).
    assert "src/module.py" in files
    assert "src/module.py" in read_paths


def test_collect_relevant_files_demotes_test_files(tmp_path: Path) -> None:
    """Test files only fill remaining slots after implementation files.

    When the error log references both test and implementation files (none in
    a traceback frame), implementation files take all slots first; test files
    are added only to fill the remaining budget.
    """
    impl = tmp_path / "impl.py"
    impl.write_text("x = 1\n", encoding="utf-8")
    test_file = tmp_path / "test_impl.py"
    test_file.write_text("assert True\n", encoding="utf-8")

    # No traceback frames — both files appear as plain tokens. Implementation
    # file must be collected; test file fills the second slot.
    error_context = f"Error in {impl} referenced by {test_file}"
    files, read_paths = fix_module._collect_relevant_files(tmp_path, error_context)

    assert "impl.py" in files
    assert "test_impl.py" in files
    # Implementation file comes first.
    assert read_paths[0] == "impl.py"


def test_is_test_path_classifies_correctly() -> None:
    assert fix_module._is_test_path("tests/test_foo.py")
    assert fix_module._is_test_path("src/tests/test_bar.py")
    assert fix_module._is_test_path("test_module.py")
    assert fix_module._is_test_path("module_test.py")
    assert fix_module._is_test_path("subdir/test.py")
    assert not fix_module._is_test_path("src/module.py")
    assert not fix_module._is_test_path("utils/helpers.py")


def test_apply_patch_with_git(tmp_path: Path) -> None:
    patch_text = "diff content"
    i18n_mock = MagicMock()
    i18n_mock._ = MagicMock(return_value="ok")

    check = MagicMock(returncode=0)
    apply = MagicMock(returncode=0)

    with (
        patch("shutil.which", side_effect=["git", None]),
        patch("subprocess.run", side_effect=[check, apply]) as mock_run,
    ):
        fix_module._apply_patch(tmp_path, patch_text, i18n_mock)

    assert mock_run.call_count == 2
    assert mock_run.call_args_list[0][0][0] == ["git", "apply", "--check"]


def test_apply_patch_with_patch_command(tmp_path: Path) -> None:
    patch_text = "diff content"
    i18n_mock = MagicMock()
    i18n_mock._ = MagicMock(return_value="ok")

    proc = MagicMock(returncode=0)

    with (
        patch("shutil.which", side_effect=[None, "patch"]),
        patch("subprocess.run", return_value=proc) as mock_run,
    ):
        fix_module._apply_patch(tmp_path, patch_text, i18n_mock)

    assert mock_run.call_args[0][0] == ["patch", "-p1", "--batch"]


def test_apply_patch_cleans_up_orig_and_rej_on_patch_failure(tmp_path: Path) -> None:
    """patch(1) failure must not leave ``.orig`` / ``.rej`` scratch files behind.

    Reproduces the real-world scenario observed during e2e testing: when an
    LLM-proposed patch is malformed and ``patch -p1 --batch`` rejects it,
    patch(1) writes ``<file>.orig`` (backup) and ``<file>.rej`` (rejected
    hunks) next to the target. These leak into ``git status`` as untracked
    files. ``apply_patch`` must clean them up on failure.
    """
    target = tmp_path / "src.py"
    target.write_text("original\n")
    # Simulate the scratch files that ``patch -p1`` writes on rejection.
    (tmp_path / "src.py.orig").write_text("backup")
    (tmp_path / "src.py.rej").write_text("rejected hunk")

    patch_text = "--- a/src.py\n+++ b/src.py\n@@ -1 +1 @@\n-original\n+modified\n"
    i18n_mock = MagicMock()
    i18n_mock._ = MagicMock(return_value="failed")

    proc = MagicMock(returncode=1, stderr="patch: malformed patch")
    with (
        patch("shutil.which", side_effect=[None, "patch"]),
        patch("subprocess.run", return_value=proc),
    ):
        applied = fix_module._apply_patch(tmp_path, patch_text, i18n_mock)

    assert applied is False
    # The scratch files must be removed so they don't pollute the worktree.
    assert not (tmp_path / "src.py.orig").exists()
    assert not (tmp_path / "src.py.rej").exists()


def test_apply_patch_no_tool(tmp_path: Path) -> None:
    patch_text = "diff content"
    i18n_mock = MagicMock()
    i18n_mock._ = MagicMock(return_value="no tool")

    with patch("shutil.which", return_value=None):
        fix_module._apply_patch(tmp_path, patch_text, i18n_mock)


def test_patch_paths_are_safe_accepts_relative_paths(tmp_path: Path) -> None:
    patch = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-old\n+new"
    assert fix_module._patch_paths_are_safe(tmp_path, patch) is True


def test_patch_paths_are_safe_rejects_absolute_paths(tmp_path: Path) -> None:
    patch = "--- /etc/passwd\n+++ /etc/passwd\n@@ -1 +1 @@\n-old\n+new"
    assert fix_module._patch_paths_are_safe(tmp_path, patch) is False


def test_patch_paths_are_safe_rejects_traversal(tmp_path: Path) -> None:
    patch = "--- a/../etc/passwd\n+++ b/../etc/passwd\n@@ -1 +1 @@\n-old\n+new"
    assert fix_module._patch_paths_are_safe(tmp_path, patch) is False


def _make_fake_binary(path: Path) -> Path:
    """Create a small executable file used as a stand-in for a pinned tool."""
    path.write_bytes(b"#!/bin/sh\necho ok\n")
    path.chmod(0o755)
    return path


def test_apply_patch_with_pinned_git_path_uses_pinned_binary(tmp_path: Path) -> None:
    """A pinned ``tools.git.path`` must head the git subprocess command."""
    fake_git = _make_fake_binary(tmp_path / "fake-git")
    tools = ToolsConfig(git=ToolConfig(path=str(fake_git)))

    patch_text = "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new"
    i18n_mock = MagicMock()
    i18n_mock._ = MagicMock(return_value="ok")

    check = MagicMock(returncode=0)
    apply = MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=[check, apply]) as mock_run:
        fix_module._apply_patch(tmp_path, patch_text, i18n_mock, tools)

    assert mock_run.call_count == 2
    git_check_cmd = mock_run.call_args_list[0][0][0]
    assert git_check_cmd[0] == str(fake_git.resolve())
    assert git_check_cmd[1:] == ["apply", "--check"]
    git_apply_cmd = mock_run.call_args_list[1][0][0]
    assert git_apply_cmd[0] == str(fake_git.resolve())


def test_apply_patch_with_wrong_sha256_skips_git_and_falls_back(tmp_path: Path) -> None:
    """A wrong ``tools.git.sha256`` skips git and falls back to patch."""
    fake_git = _make_fake_binary(tmp_path / "fake-git")
    real_digest = hashlib.sha256(fake_git.read_bytes()).hexdigest()
    wrong_digest = "0" * 64
    assert real_digest != wrong_digest

    tools = ToolsConfig(git=ToolConfig(path=str(fake_git), sha256=wrong_digest))

    patch_text = "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new"
    i18n_mock = MagicMock()
    i18n_mock._ = MagicMock(return_value="ok")

    proc = MagicMock(returncode=0)

    with (
        patch("shutil.which", return_value="/usr/bin/patch"),
        patch("subprocess.run", return_value=proc) as mock_run,
    ):
        fix_module._apply_patch(tmp_path, patch_text, i18n_mock, tools)

    # git was never invoked because the pinned hash mismatched.
    assert mock_run.call_count == 1
    patch_cmd = mock_run.call_args[0][0]
    assert patch_cmd[0] == "patch"
    assert patch_cmd[1:] == ["-p1", "--batch"]


def test_apply_patch_with_wrong_sha256_and_no_patch_warns(tmp_path: Path) -> None:
    """When both git (bad hash) and patch are unavailable, warn instead of raising."""
    fake_git = _make_fake_binary(tmp_path / "fake-git")
    wrong_digest = "0" * 64
    tools = ToolsConfig(git=ToolConfig(path=str(fake_git), sha256=wrong_digest))

    patch_text = "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new"
    i18n_mock = MagicMock()
    i18n_mock._ = MagicMock(return_value="no tool")

    with (
        patch("shutil.which", return_value=None),
        patch("subprocess.run") as mock_run,
    ):
        # Must not raise: the ConfigError from the sha256 mismatch is caught.
        fix_module._apply_patch(tmp_path, patch_text, i18n_mock, tools)

    mock_run.assert_not_called()


def test_apply_patch_wrong_sha256_resolve_raises_config_error(tmp_path: Path) -> None:
    """A direct ``ToolVerifier.resolve`` raises ConfigError on a hash mismatch.

    This documents the contract that ``_apply_patch`` relies on: a wrong
    ``sha256`` produces a ``ConfigError`` (which ``_apply_patch`` then swallows
    so the patch fallback can run).
    """
    from myagent.core.tool_verifier import ToolVerifier

    fake_git = _make_fake_binary(tmp_path / "fake-git")
    tools = ToolsConfig(git=ToolConfig(path=str(fake_git), sha256="0" * 64))

    with pytest.raises(ConfigError, match="SHA-256 mismatch"):
        ToolVerifier(tools).resolve("git")


def test_fix_redacts_absolute_project_root_from_prompt(tmp_path: Path, monkeypatch) -> None:
    """The prompt sent to the model must not contain the absolute project root."""
    _write_config_with_api_key(tmp_path)
    monkeypatch.chdir(tmp_path)

    error_log = tmp_path / "error.txt"
    absolute_ref = f"{tmp_path}/tests/test_x.py:123 failed"
    error_log.write_text(absolute_ref, encoding="utf-8")

    mock_router = MagicMock()
    mock_router.chat.return_value = "No patch needed."
    with patch("myagent.cli.commands.fix._model_router", return_value=mock_router):
        result = runner.invoke(app, ["fix", str(error_log), "--yes"])

    # Exit code is 1 because no patch was produced (dogfood fix).
    assert result.exit_code == 1
    mock_router.chat.assert_called_once()
    messages = mock_router.chat.call_args[0][0]
    user_prompt = messages[1].content
    assert str(tmp_path) not in user_prompt
    assert "./tests/test_x.py:123 failed" in user_prompt


def test_build_prompt_redacts_when_called_with_redacted_context(tmp_path: Path) -> None:
    """``_build_prompt`` itself does not reintroduce absolute project paths.

    This complements ``test_fix_redacts_absolute_project_root_from_prompt`` by
    pinning the contract at the helper level: once the caller has run
    ``redact_paths``, the resulting prompt must stay free of the absolute root.
    """
    redacted = "./tests/test_x.py:123 failed"
    prompt, _ = fix_module._build_prompt(redacted, tmp_path)
    assert str(tmp_path) not in prompt
    assert "./tests/test_x.py:123 failed" in prompt
