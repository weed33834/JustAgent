"""Tests for the path-redaction helper in ``autoship.utils.redaction``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from autoship.utils.redaction import redact_paths


def test_redact_paths_replaces_project_root_prefix_with_dot(tmp_path: Path) -> None:
    """Absolute paths inside the project root become relative (``./...``)."""
    project = tmp_path / "autoship-cli"
    project.mkdir()
    text = f"{project}/tests/test_x.py:123 failed"

    result = redact_paths(text, project)

    assert result.startswith("./tests/test_x.py:123 failed")
    assert str(project) not in result


def test_redact_paths_replaces_home_prefix_with_tilde(tmp_path: Path) -> None:
    """Paths under the user's home directory are masked with ``~``."""
    fake_home = tmp_path / "home" / "user"
    fake_home.mkdir(parents=True)
    text = f"{fake_home}/.config/secret.toml is missing"

    with patch("autoship.utils.redaction.Path.home", return_value=fake_home):
        result = redact_paths(text, project_root=None)

    assert result.startswith("~/.config/secret.toml is missing")
    assert str(fake_home) not in result


def test_redact_paths_applies_both_project_and_home(tmp_path: Path) -> None:
    """Project root replacement runs first, then the home prefix."""
    project = tmp_path / "proj"
    project.mkdir()
    fake_home = tmp_path
    text = f"{project}/src/app.py and {fake_home}/.cache/lost"

    with patch("autoship.utils.redaction.Path.home", return_value=fake_home):
        result = redact_paths(text, project)

    assert "./src/app.py" in result
    assert "~/.cache/lost" in result
    assert str(project) not in result
    assert str(fake_home) not in result


def test_redact_paths_handles_none_project_root() -> None:
    """``project_root=None`` must not raise and should still redact home."""
    fake_home = Path("/very/unlikely/home/path")
    text = f"{fake_home}/file.txt"

    with patch("autoship.utils.redaction.Path.home", return_value=fake_home):
        result = redact_paths(text, project_root=None)

    assert result == "~/file.txt"


def test_redact_paths_no_replacement_when_nothing_matches(tmp_path: Path) -> None:
    """Unrelated text is returned unchanged."""
    project = tmp_path / "proj"
    project.mkdir()
    fake_home = tmp_path / "other-home"
    fake_home.mkdir()
    text = "no paths here at all"

    with patch("autoship.utils.redaction.Path.home", return_value=fake_home):
        result = redact_paths(text, project)

    assert result == text


def test_redact_paths_can_disable_home_redaction(tmp_path: Path) -> None:
    """``redact_home=False`` leaves the home prefix intact."""
    fake_home = tmp_path
    text = f"{fake_home}/.bashrc"

    with patch("autoship.utils.redaction.Path.home", return_value=fake_home):
        result = redact_paths(text, project_root=None, redact_home=False)

    assert result == text
