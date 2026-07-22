"""CLI tests for the ``myagent project scan`` and ``batch-*`` commands."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from myagent.cli.main import app
from myagent.core.project_store import ProjectStore
from myagent.models.project import ManagedProject

runner = CliRunner()


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal ``.myagent.toml`` rooted in ``tmp_path``."""
    config_path = tmp_path / ".myagent.toml"
    config_path.write_text(
        f'schema_version = 1\nproject_root = "{tmp_path}"\n'
        f'[audit]\nlog_dir = "{tmp_path / "audit"}"\n',
        encoding="utf-8",
    )
    return config_path


def _make_project(tmp_path: Path, name: str) -> Path:
    """Create a project directory and return its path."""
    directory = tmp_path / name
    directory.mkdir()
    return directory


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> CompletedProcess[str]:
    """Build a fake :class:`subprocess.CompletedProcess`."""
    return CompletedProcess(
        args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def test_scan_dry_run(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    proj = _make_project(tmp_path, "myproj")
    (proj / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    config_path = _write_config(tmp_path)
    with patch("myagent.cli.commands.project.ProjectStore", return_value=store):
        result = runner.invoke(
            app,
            ["--config", str(config_path), "project", "scan", str(tmp_path), "--dry-run"],
        )
    assert result.exit_code == 0, result.output
    assert "myproj" in result.output
    assert "dry run" in result.output.lower()
    assert store.list_all() == []


def test_scan_finds_multiple_projects(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    for name in ("alpha", "beta"):
        directory = _make_project(tmp_path, name)
        (directory / "pyproject.toml").write_text(
            "[project]\nname='x'\n", encoding="utf-8"
        )
    result = runner.invoke(
        app,
        ["--config", str(config_path), "project", "scan", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "alpha" in result.output
    assert "beta" in result.output


def test_scan_register(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    proj = _make_project(tmp_path, "myproj")
    (proj / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    config_path = _write_config(tmp_path)
    with patch("myagent.cli.commands.project.ProjectStore", return_value=store):
        result = runner.invoke(
            app,
            ["--config", str(config_path), "project", "scan", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    assert "Registered" in result.output
    fetched = store.get("myproj")
    assert fetched is not None
    assert fetched.name == "myproj"
    assert Path(fetched.path).exists()


def test_scan_with_tag(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    proj = _make_project(tmp_path, "myproj")
    (proj / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    config_path = _write_config(tmp_path)
    with patch("myagent.cli.commands.project.ProjectStore", return_value=store):
        result = runner.invoke(
            app,
            [
                "--config", str(config_path),
                "project", "scan", str(tmp_path),
                "--tag", "web", "--tag", "py",
            ],
        )
    assert result.exit_code == 0, result.output
    fetched = store.get("myproj")
    assert fetched is not None
    assert fetched.tags == ["web", "py"]


def test_scan_nonexistent_root(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["--config", str(config_path), "project", "scan", str(tmp_path / "nope")],
    )
    assert result.exit_code == 1, result.output
    assert "does not exist" in result.output.lower()


def test_scan_no_projects_found(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / "empty").mkdir()
    result = runner.invoke(
        app,
        ["--config", str(config_path), "project", "scan", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "no projects" in result.output.lower()


# ---------------------------------------------------------------------------
# batch-status
# ---------------------------------------------------------------------------


def test_batch_status(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    proj = _make_project(tmp_path, "proj")
    store.add(ManagedProject(name="proj", path=str(proj), added_at=1.0))
    config_path = _write_config(tmp_path)
    with (
        patch("myagent.cli.commands.project.ProjectStore", return_value=store),
        patch(
            "myagent.core.batch_ops.subprocess.run",
            return_value=_completed(0, "", ""),
        ),
    ):
        result = runner.invoke(
            app, ["--config", str(config_path), "project", "batch-status"]
        )
    assert result.exit_code == 0, result.output
    assert "proj" in result.output
    assert "OK" in result.output


def test_batch_status_no_projects(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    config_path = _write_config(tmp_path)
    with patch("myagent.cli.commands.project.ProjectStore", return_value=store):
        result = runner.invoke(
            app, ["--config", str(config_path), "project", "batch-status"]
        )
    assert result.exit_code == 0, result.output
    assert "Total: 0" in result.output


# ---------------------------------------------------------------------------
# batch-run
# ---------------------------------------------------------------------------


def test_batch_run(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    proj = _make_project(tmp_path, "proj")
    store.add(ManagedProject(name="proj", path=str(proj), added_at=1.0))
    config_path = _write_config(tmp_path)
    with patch("myagent.cli.commands.project.ProjectStore", return_value=store):
        result = runner.invoke(
            app,
            ["--config", str(config_path), "project", "batch-run", "echo", "hello"],
        )
    assert result.exit_code == 0, result.output
    assert "proj" in result.output
    assert "OK" in result.output


def test_batch_run_no_command(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    config_path = _write_config(tmp_path)
    with patch("myagent.cli.commands.project.ProjectStore", return_value=store):
        result = runner.invoke(
            app, ["--config", str(config_path), "project", "batch-run"]
        )
    assert result.exit_code == 1, result.output
    assert "No command" in result.output


# ---------------------------------------------------------------------------
# batch-ship
# ---------------------------------------------------------------------------


def test_batch_ship(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    proj = _make_project(tmp_path, "proj")
    store.add(ManagedProject(name="proj", path=str(proj), added_at=1.0))
    config_path = _write_config(tmp_path)
    fake_completed = MagicMock(returncode=0, stdout="", stderr="")
    with (
        patch("myagent.cli.commands.project.ProjectStore", return_value=store),
        patch(
            "myagent.core.batch_ops.subprocess.run",
            return_value=fake_completed,
        ) as mock_run,
    ):
        result = runner.invoke(
            app,
            [
                "--config", str(config_path),
                "project", "batch-ship",
                "--stages", "clean",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "proj" in result.output
    assert "OK" in result.output
    args, _ = mock_run.call_args
    assert args[0] == ["myagent", "clean"]


def test_batch_ship_invalid_stage(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    config_path = _write_config(tmp_path)
    with patch("myagent.cli.commands.project.ProjectStore", return_value=store):
        result = runner.invoke(
            app,
            ["--config", str(config_path), "project", "batch-ship", "--stages", "bogus"],
        )
    assert result.exit_code == 1, result.output
    assert "Invalid stage" in result.output
