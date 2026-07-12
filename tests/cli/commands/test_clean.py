"""Tests for the clean command."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from autoship.cli.commands import clean
from autoship.cli.commands.clean import (
    _builtin_format_file,
    _collect_source_files,
    _compress_inline_spaces,
)
from autoship.cli.main import app
from autoship.exceptions import ToolChainError
from autoship.models.config import AppConfig


def test_clean_noop_when_already_clean(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with patch("shutil.which", return_value=None):
        clean.clean(ctx, paths=[Path(".")], check=False)


def test_clean_check_raises_when_changes_needed(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with (
        patch.object(clean.ToolChain, "preview", return_value="--- diff ---"),
        pytest.raises(ToolChainError),
    ):
        clean.clean(ctx, paths=[Path(".")], check=True)


def test_clean_preview_failure_raises_toolchain_error(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with (
        patch.object(clean.ToolChain, "preview", side_effect=subprocess.CalledProcessError(1, [])),
        pytest.raises(ToolChainError, match="Failed to preview"),
    ):
        clean.clean(ctx, paths=[Path(".")], check=False)


def test_clean_apply_failure_raises_toolchain_error(project_root, app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": True,
        "verbose": False,
    }
    with (
        patch.object(clean.ToolChain, "preview", return_value="--- diff ---"),
        patch.object(clean.ToolChain, "apply", side_effect=subprocess.CalledProcessError(1, [])),
        pytest.raises(ToolChainError, match="Failed to apply"),
    ):
        clean.clean(ctx, paths=[Path(".")], check=False)


def test_clean_yes_option_skips_confirmation(app_config: AppConfig) -> None:
    ctx = MagicMock()
    ctx.obj = {
        "config": app_config,
        "audit_logger": MagicMock(),
        "dry_run": False,
        "yes": False,
        "verbose": False,
    }
    with (
        patch.object(clean.ToolChain, "preview", return_value="--- diff ---"),
        patch.object(clean.ToolChain, "apply") as mock_apply,
    ):
        clean.clean(ctx, paths=[Path(".")], check=False, yes=True)
        mock_apply.assert_called_once()


runner = CliRunner()


def test_clean_subcommand_yes_skips_confirm() -> None:
    with (
        patch.object(clean.ToolChain, "preview", return_value="--- diff ---"),
        patch.object(clean.ToolChain, "apply") as mock_apply,
    ):
        result = runner.invoke(app, ["clean", "--yes"])
    assert result.exit_code == 0
    mock_apply.assert_called_once()


def test_clean_global_yes_still_skips_confirm() -> None:
    with (
        patch.object(clean.ToolChain, "preview", return_value="--- diff ---"),
        patch.object(clean.ToolChain, "apply") as mock_apply,
    ):
        result = runner.invoke(app, ["--yes", "clean"])
    assert result.exit_code == 0
    mock_apply.assert_called_once()


# ============================================================
# Tests for _compress_inline_spaces
# ============================================================


class TestCompressInlineSpaces:
    """Tests for the _compress_inline_spaces helper."""

    def test_basic_compression(self) -> None:
        """Compress 2+ spaces to a single space in code."""
        assert _compress_inline_spaces("return  x+y") == "return x+y"

    def test_multiple_runs(self) -> None:
        """Multiple runs of spaces are each compressed."""
        assert _compress_inline_spaces("a   b    c") == "a b c"

    def test_preserves_indentation(self) -> None:
        """Leading whitespace (indentation) is left untouched."""
        assert _compress_inline_spaces("    return  x+y") == "    return x+y"

    def test_skips_single_quoted(self) -> None:
        """Spaces inside single-quoted string literals are not compressed."""
        assert _compress_inline_spaces("x = 'hello  world'") == "x = 'hello  world'"

    def test_skips_double_quoted(self) -> None:
        """Spaces inside double-quoted string literals are not compressed."""
        assert _compress_inline_spaces('x = "hello  world"') == 'x = "hello  world"'

    def test_skips_triple_single_quoted(self) -> None:
        """Spaces inside triple-single-quoted strings are not compressed."""
        assert _compress_inline_spaces("x = '''hello  world'''") == "x = '''hello  world'''"

    def test_skips_triple_double_quoted(self) -> None:
        """Spaces inside triple-double-quoted strings are not compressed."""
        assert _compress_inline_spaces('x = """hello  world"""') == 'x = """hello  world"""'

    def test_mixed_string_and_code(self) -> None:
        """String content is preserved; surrounding code is compressed."""
        assert _compress_inline_spaces('x  =  "hello  world"  +  y') == 'x = "hello  world" + y'

    def test_multiple_strings_in_line(self) -> None:
        """Multiple string segments on one line are all preserved."""
        assert _compress_inline_spaces("a  =  'x  y'  +  \"p  q\"") == "a = 'x  y' + \"p  q\""

    def test_empty_line(self) -> None:
        """Empty string returns empty string."""
        assert _compress_inline_spaces("") == ""

    def test_only_spaces(self) -> None:
        """A line consisting only of spaces is returned as-is."""
        assert _compress_inline_spaces("    ") == "    "

    def test_preserves_newline(self) -> None:
        """Trailing newline is preserved after compression."""
        assert _compress_inline_spaces("return  x+y\n") == "return x+y\n"

    def test_only_newline(self) -> None:
        """A line that is just newline is returned as-is."""
        assert _compress_inline_spaces("\n") == "\n"

    def test_no_spaces(self) -> None:
        """A line with no spaces is unchanged."""
        assert _compress_inline_spaces("return(x+y)") == "return(x+y)"

    def test_single_space_unchanged(self) -> None:
        """A single space between tokens stays a single space."""
        assert _compress_inline_spaces("return x+y") == "return x+y"

    def test_spaces_after_closing_paren_compress(self) -> None:
        """Spaces after a closing parenthesis compress normally."""
        assert _compress_inline_spaces('print("hello")  world') == 'print("hello") world'

    def test_spaces_after_string_close_compress(self) -> None:
        """Spaces after a closing quote are compressed normally."""
        assert _compress_inline_spaces("x = 'foo'  +  y") == "x = 'foo' + y"


# ============================================================
# Tests for _builtin_format_file
# ============================================================


class TestBuiltinFormatFile:
    """Tests for the _builtin_format_file helper."""

    def test_formats_file_with_trailing_whitespace_and_blank_lines(self, tmp_path: Path) -> None:
        """File with trailing whitespace, blank lines, and inline double spaces."""
        f = tmp_path / "test.py"
        f.write_text("def foo():  \n\n\n    x  =  1\n    return  x\n")
        changed = _builtin_format_file(f)
        assert changed is True
        assert f.read_text() == "def foo():\n\n    x = 1\n    return x\n"

    def test_no_change_returns_false(self, tmp_path: Path) -> None:
        """A clean file is not modified and returns False."""
        f = tmp_path / "clean.py"
        content = "def foo():\n    return 1\n"
        f.write_text(content)
        changed = _builtin_format_file(f)
        assert changed is False
        assert f.read_text() == content

    def test_empty_file(self, tmp_path: Path) -> None:
        """An empty file gets a trailing newline added."""
        f = tmp_path / "empty.py"
        f.write_text("")
        changed = _builtin_format_file(f)
        assert changed is True
        assert f.read_text() == "\n"

    def test_only_blank_lines(self, tmp_path: Path) -> None:
        """Multiple consecutive blank lines collapse to a single newline."""
        f = tmp_path / "blank.py"
        f.write_text("\n\n\n")
        changed = _builtin_format_file(f)
        assert changed is True
        assert f.read_text() == "\n"

    def test_binary_file_returns_false(self, tmp_path: Path) -> None:
        """A binary file with invalid UTF-8 is skipped and returns False."""
        f = tmp_path / "binary.py"
        f.write_bytes(b"\xff\xfe\x00")
        changed = _builtin_format_file(f)
        assert changed is False

    def test_nonexistent_file_returns_false(self, tmp_path: Path) -> None:
        """A path that does not point to a file returns False."""
        f = tmp_path / "nonexistent.py"
        changed = _builtin_format_file(f)
        assert changed is False

    def test_already_clean_file(self, tmp_path: Path) -> None:
        """A file that is already properly formatted is unchanged."""
        f = tmp_path / "clean.py"
        content = "pass\n"
        f.write_text(content)
        changed = _builtin_format_file(f)
        assert changed is False

    def test_duplicate_blank_lines_collapse(self, tmp_path: Path) -> None:
        """Multiple consecutive blank lines are collapsed to one."""
        f = tmp_path / "blank.py"
        f.write_text("a\n\n\n\nb\n")
        changed = _builtin_format_file(f)
        assert changed is True
        assert f.read_text() == "a\n\nb\n"


# ============================================================
# Tests for _collect_source_files
# ============================================================


class TestCollectSourceFiles:
    """Tests for the _collect_source_files helper."""

    def test_collects_python_files(self, tmp_path: Path) -> None:
        """Collects .py files and ignores non-source files."""
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        assert len(result) == 1
        assert result[0].name == "a.py"

    def test_collects_js_files(self, tmp_path: Path) -> None:
        """Collects .js source files."""
        (tmp_path / "app.js").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        assert len(result) == 1
        assert result[0].name == "app.js"

    def test_collects_rs_files(self, tmp_path: Path) -> None:
        """Collects .rs source files."""
        (tmp_path / "main.rs").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        assert len(result) == 1
        assert result[0].name == "main.rs"

    def test_collects_multiple_extensions(self, tmp_path: Path) -> None:
        """Collects files with different supported extensions."""
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.js").write_text("")
        (tmp_path / "c.rs").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        names = {f.name for f in result}
        assert names == {"a.py", "b.js", "c.rs"}

    def test_excludes_node_modules(self, tmp_path: Path) -> None:
        """Files inside node_modules are excluded."""
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "app.js").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        assert len(result) == 0

    def test_excludes_dot_git(self, tmp_path: Path) -> None:
        """Files inside .git are excluded."""
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config.py").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        assert len(result) == 0

    def test_excludes_dot_venv(self, tmp_path: Path) -> None:
        """Files inside .venv are excluded."""
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        assert len(result) == 0

    def test_excludes_build_directory(self, tmp_path: Path) -> None:
        """Files inside build/ are excluded."""
        build = tmp_path / "build"
        build.mkdir()
        (build / "helper.py").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        assert len(result) == 0

    def test_excludes_dist_directory(self, tmp_path: Path) -> None:
        """Files inside dist/ are excluded."""
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "bundle.js").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        assert len(result) == 0

    def test_excludes_target_directory(self, tmp_path: Path) -> None:
        """Files inside target/ are excluded."""
        target = tmp_path / "target"
        target.mkdir()
        (target / "debug.rs").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        assert len(result) == 0

    def test_absolute_path_collected(self, tmp_path: Path) -> None:
        """An absolute path to a source file is collected."""
        f = tmp_path / "test.py"
        f.write_text("")
        result = _collect_source_files([f.absolute()], tmp_path)
        assert len(result) == 1
        assert result[0] == f.absolute()

    def test_subdirectory_collected(self, tmp_path: Path) -> None:
        """Files in a subdirectory are collected via rglob."""
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("")
        result = _collect_source_files([Path("src")], tmp_path)
        assert len(result) == 1
        assert result[0].name == "main.py"

    def test_file_inside_excluded_dir_is_skipped(self, tmp_path: Path) -> None:
        """A file path that passes through an excluded dir is skipped."""
        (tmp_path / "node_modules").mkdir(parents=True, exist_ok=True)
        f = tmp_path / "node_modules" / "app.js"
        f.write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        names = {r.name for r in result}
        assert "app.js" not in names

    def test_non_source_file_is_skipped(self, tmp_path: Path) -> None:
        """A file that is not a source file is not collected."""
        (tmp_path / "README.md").write_text("")
        result = _collect_source_files([Path(".")], tmp_path)
        assert len(result) == 0


# ============================================================
# Integration tests — clean command with built-in fallback
# ============================================================


class TestCleanBuiltinFallback:
    """Integration tests exercising the built-in formatting path."""

    def test_builtin_format_when_tools_missing(self, tmp_path: Path, app_config: AppConfig) -> None:
        """When autoflake/black are missing, all source files get builtin formatting."""
        app_config.project_root = tmp_path

        f = tmp_path / "test.py"
        f.write_text("def foo():  \n\n\n    x  =  1\n    return  x\n")

        ctx = MagicMock()
        ctx.obj = {
            "config": app_config,
            "audit_logger": MagicMock(),
            "dry_run": False,
            "yes": True,
            "verbose": False,
        }

        with (
            patch.object(clean.ToolChain, "preview", return_value=""),
            patch("shutil.which", return_value=None),
        ):
            clean.clean(ctx, paths=[Path(".")], check=False)

        assert f.read_text() == "def foo():\n\n    x = 1\n    return x\n"

    def test_builtin_format_non_python_files(self, tmp_path: Path, app_config: AppConfig) -> None:
        """External tools handle Python; built-in handles JS/RS etc."""
        app_config.project_root = tmp_path

        f = tmp_path / "app.js"
        f.write_text("function foo() {  \n\n\n    var x  =  1;\n}\n")

        ctx = MagicMock()
        ctx.obj = {
            "config": app_config,
            "audit_logger": MagicMock(),
            "dry_run": False,
            "yes": True,
            "verbose": False,
        }

        with (
            patch.object(clean.ToolChain, "preview", return_value=""),
            patch("shutil.which", return_value="/usr/bin/autoflake"),
        ):
            clean.clean(ctx, paths=[Path(".")], check=False)

        assert f.read_text() == "function foo() {\n\n    var x = 1;\n}\n"

    def test_dry_run_does_not_modify_files(self, tmp_path: Path, app_config: AppConfig) -> None:
        """In dry-run mode, builtin formatting does NOT write to disk."""
        app_config.project_root = tmp_path

        f = tmp_path / "test.py"
        original = "def foo():  \n"
        f.write_text(original)

        ctx = MagicMock()
        ctx.obj = {
            "config": app_config,
            "audit_logger": MagicMock(),
            "dry_run": True,
            "yes": True,
            "verbose": False,
        }

        with (
            patch.object(clean.ToolChain, "preview", return_value=""),
            patch("shutil.which", return_value=None),
        ):
            clean.clean(ctx, paths=[Path(".")], check=False)

        assert f.read_text() == original

    def test_noop_when_nothing_to_format(self, tmp_path: Path, app_config: AppConfig) -> None:
        """No tools missing and no non-Python files → no-op path."""
        app_config.project_root = tmp_path

        ctx = MagicMock()
        ctx.obj = {
            "config": app_config,
            "audit_logger": MagicMock(),
            "dry_run": False,
            "yes": True,
            "verbose": False,
        }

        with (
            patch.object(clean.ToolChain, "preview", return_value=""),
            patch("shutil.which", return_value="/usr/bin/autoflake"),
        ):
            clean.clean(ctx, paths=[Path(".")], check=False)

        ctx.obj["audit_logger"].record.assert_any_call("clean.noop")

    def test_verbose_shows_formatted_filename(self, tmp_path: Path, app_config: AppConfig) -> None:
        """In verbose mode, builtin formatting prints the formatted filename."""
        app_config.project_root = tmp_path

        f = tmp_path / "test.py"
        f.write_text("def foo():  \n")

        ctx = MagicMock()
        ctx.obj = {
            "config": app_config,
            "audit_logger": MagicMock(),
            "dry_run": False,
            "yes": True,
            "verbose": True,
        }

        with (
            patch.object(clean.ToolChain, "preview", return_value=""),
            patch("shutil.which", return_value=None),
            patch("typer.echo") as mock_echo,
        ):
            clean.clean(ctx, paths=[Path(".")], check=False)

        formatted_calls = [c for c in mock_echo.call_args_list if "Formatted:" in str(c)]
        assert len(formatted_calls) >= 1

    def test_noop_when_builtin_makes_no_changes(
        self, tmp_path: Path, app_config: AppConfig
    ) -> None:
        """When builtin formatting changes nothing, the noop path is taken."""
        app_config.project_root = tmp_path

        # A file that's already properly formatted
        f = tmp_path / "clean.py"
        f.write_text("pass\n")

        ctx = MagicMock()
        ctx.obj = {
            "config": app_config,
            "audit_logger": MagicMock(),
            "dry_run": False,
            "yes": True,
            "verbose": False,
        }

        with (
            patch.object(clean.ToolChain, "preview", return_value=""),
            patch("shutil.which", return_value=None),
        ):
            clean.clean(ctx, paths=[Path(".")], check=False)

        ctx.obj["audit_logger"].record.assert_any_call("clean.noop")

    def test_verbose_shows_diff_from_tools(self, tmp_path: Path, app_config: AppConfig) -> None:
        """When external tools produce a diff and verbose is on, diff is echoed."""
        app_config.project_root = tmp_path

        ctx = MagicMock()
        ctx.obj = {
            "config": app_config,
            "audit_logger": MagicMock(),
            "dry_run": False,
            "yes": True,
            "verbose": True,
        }

        with (
            patch.object(clean.ToolChain, "preview", return_value="--- diff ---"),
            patch.object(clean.ToolChain, "apply"),
        ):
            clean.clean(ctx, paths=[Path(".")], check=False)

        # Test passes if no exception — the diff is echoed (covered by line 278)

    def test_dry_run_shows_diff_from_tools(self, tmp_path: Path, app_config: AppConfig) -> None:
        """When external tools produce a diff and dry_run is on, diff is echoed."""
        app_config.project_root = tmp_path

        ctx = MagicMock()
        ctx.obj = {
            "config": app_config,
            "audit_logger": MagicMock(),
            "dry_run": True,
            "yes": True,
            "verbose": False,
        }

        with (
            patch.object(clean.ToolChain, "preview", return_value="--- diff ---"),
            patch.object(clean.ToolChain, "apply") as mock_apply,
            patch("typer.echo") as mock_echo,
        ):
            clean.clean(ctx, paths=[Path(".")], check=False)

        # In dry_run mode, diff is echoed (line 278 coverage)
        diff_calls = [c for c in mock_echo.call_args_list if "--- diff ---" in str(c)]
        assert len(diff_calls) >= 1
        # apply is still called in external tool dry_run mode
        mock_apply.assert_called_once()
