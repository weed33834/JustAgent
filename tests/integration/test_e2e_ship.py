"""End-to-end integration test for the init -> clean -> verify -> commit flow."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from autoship.cli.main import app

runner = CliRunner()


def _conventional_commit_pattern() -> re.Pattern[str]:
    """Return a regex matching conventional commit messages."""
    return re.compile(
        r"^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
        r"(\([a-zA-Z0-9_\-]+\))?!?: .+"
    )


def test_e2e_init_clean_verify_commit(tmp_path: Path, monkeypatch) -> None:
    """Exercise the full AutoShip main flow in a mocked temporary project."""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    # Simulated Python project.
    pyproject = project / "pyproject.toml"
    pyproject.write_text('[build-system]\nrequires = ["hatchling"]\n', encoding="utf-8")
    src = project / "src"
    src.mkdir()
    py_file = src / "hello.py"
    original_content = "import os\nimport sys\n\nx=1+2\n"
    py_file.write_text(original_content, encoding="utf-8")

    config_path = project / ".autoship.toml"
    expected_message = "feat(src): format hello module"

    hardware_mock = MagicMock()
    hardware_mock.recommended_tier = 2

    def _fake_apply(_paths: list[Path]) -> None:
        """Simulate a formatter changing the source file."""
        content = py_file.read_text(encoding="utf-8")
        py_file.write_text(content.replace("x=1+2", "x = 1 + 2"), encoding="utf-8")

    with (
        patch("autoship.cli.commands.init.detect_hardware", return_value=hardware_mock),
        patch("autoship.cli.commands.clean.ToolChain") as mock_toolchain_cls,
        patch("autoship.cli.commands.verify.shutil.which", return_value="/bin/echo"),
        patch("autoship.cli.commands.verify.subprocess.run") as mock_verify_run,
        patch("autoship.cli.commands.commit.ModelRouter") as mock_router_cls,
        patch("autoship.cli.commands.commit.GitAdapter") as mock_git_cls,
    ):
        toolchain_instance = mock_toolchain_cls.return_value
        toolchain_instance.preview.return_value = "--- diff ---"
        toolchain_instance.apply.side_effect = _fake_apply

        mock_verify_run.return_value.returncode = 0
        mock_verify_run.return_value.stdout = "ok"
        mock_verify_run.return_value.stderr = ""

        mock_router_cls.return_value.__enter__.return_value = mock_router_cls.return_value
        mock_router_cls.return_value.__exit__.return_value = False
        mock_router_cls.return_value.generate_commit_message.return_value = expected_message

        git_instance = mock_git_cls.return_value
        git_instance.has_changes.return_value = True
        git_instance.diff.return_value = "diff"
        git_instance.stats.return_value = "stats"

        # 1. init: generate the AutoShip configuration file.
        init_result = runner.invoke(
            app,
            ["--yes", "init", "--output", str(config_path)],
        )
        assert init_result.exit_code == 0
        assert config_path.exists()
        config_text = config_path.read_text(encoding="utf-8")
        assert "project_type" in config_text

        # 2. clean: format/clean the project files.
        clean_result = runner.invoke(
            app,
            ["--yes", "clean", str(src)],
        )
        assert clean_result.exit_code == 0
        assert py_file.read_text(encoding="utf-8") != original_content

        # 3. verify: run a mocked external verification command.
        verify_result = runner.invoke(
            app,
            ["--yes", "verify", "python -c \"print('ok')\""],
        )
        assert verify_result.exit_code == 0
        assert "Verified" in verify_result.output
        mock_verify_run.assert_called()

        # 4. commit: generate a conventional commit message and commit changes.
        commit_result = runner.invoke(
            app,
            ["--yes", "commit"],
        )
        assert commit_result.exit_code == 0
        git_instance.commit.assert_called_once_with(expected_message)
        assert _conventional_commit_pattern().match(expected_message)
