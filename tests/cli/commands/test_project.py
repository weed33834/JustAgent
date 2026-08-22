"""Tests for the ``justagent project`` command and :class:`ProjectStore`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from justagent.cli.main import app
from justagent.core.project_store import ProjectStore
from justagent.models.project import ManagedProject

runner = CliRunner()


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal ``.justagent.toml`` rooted in ``tmp_path``.

    Pinning ``project_root`` and ``audit.log_dir`` to a temp directory keeps
    the test from touching the user's real ``~/.justagent`` tree.
    """
    config_path = tmp_path / ".justagent.toml"
    config_path.write_text(
        f'schema_version = 1\nproject_root = "{tmp_path}"\n'
        f'[audit]\nlog_dir = "{tmp_path / "audit"}"\n',
        encoding="utf-8",
    )
    return config_path


# ---------------------------------------------------------------------------
# ProjectStore unit tests
# ---------------------------------------------------------------------------


def test_store_add_and_get(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    store.add(ManagedProject(name="alpha", path="/tmp/alpha", added_at=1000.0))
    fetched = store.get("alpha")
    assert fetched is not None
    assert fetched.name == "alpha"
    assert fetched.path == "/tmp/alpha"
    assert fetched.added_at == 1000.0


def test_store_add_overwrites_existing(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    store.add(ManagedProject(name="alpha", path="/tmp/alpha", added_at=1000.0))
    store.add(ManagedProject(name="alpha", path="/tmp/alpha-v2", added_at=2000.0, tags=["x"]))
    fetched = store.get("alpha")
    assert fetched is not None
    assert fetched.path == "/tmp/alpha-v2"
    assert fetched.tags == ["x"]
    assert fetched.added_at == 2000.0


def test_store_remove_found(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    store.add(ManagedProject(name="alpha", path="/tmp/alpha", added_at=1.0))
    assert store.remove("alpha") is True
    assert store.get("alpha") is None


def test_store_remove_missing_returns_false(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    assert store.remove("ghost") is False


def test_store_list_all_sorted_by_name(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    store.add(ManagedProject(name="zeta", path="/tmp/zeta", added_at=1.0))
    store.add(ManagedProject(name="alpha", path="/tmp/alpha", added_at=2.0))
    store.add(ManagedProject(name="mid", path="/tmp/mid", added_at=3.0))
    assert [p.name for p in store.list_all()] == ["alpha", "mid", "zeta"]


def test_store_list_all_empty_when_file_missing(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "missing.json")
    assert store.list_all() == []


def test_store_load_handles_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = ProjectStore(store_path=path)
    assert store.list_all() == []


def test_store_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    ProjectStore(store_path=path).add(
        ManagedProject(
            name="alpha",
            path="/tmp/alpha",
            added_at=1.0,
            tags=["a", "b"],
            description="d",
        )
    )
    fresh = ProjectStore(store_path=path)
    fetched = fresh.get("alpha")
    assert fetched is not None
    assert fetched.tags == ["a", "b"]
    assert fetched.description == "d"


def test_store_skips_invalid_entries(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    path.write_text(
        '{"good": {"name": "good", "path": "/tmp/good", "added_at": 1.0}, "bad": {"name": "bad"}}',
        encoding="utf-8",
    )
    store = ProjectStore(store_path=path)
    assert [p.name for p in store.list_all()] == ["good"]


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_project_list_empty(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    config_path = _write_config(tmp_path)
    with patch("justagent.cli.commands.project.ProjectStore", return_value=store):
        result = runner.invoke(app, ["--config", str(config_path), "project", "list"])
    assert result.exit_code == 0, result.output
    assert "no managed projects" in result.output.lower()


def test_project_add_then_list(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    config_path = _write_config(tmp_path)
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()

    with patch("justagent.cli.commands.project.ProjectStore", return_value=store):
        add_result = runner.invoke(
            app,
            [
                "--config",
                str(config_path),
                "project",
                "add",
                str(project_dir),
                "--tag",
                "web",
                "--tag",
                "py",
            ],
        )
    assert add_result.exit_code == 0, add_result.output
    assert "myproj" in add_result.output

    with patch("justagent.cli.commands.project.ProjectStore", return_value=store):
        list_result = runner.invoke(app, ["--config", str(config_path), "project", "list"])
    assert list_result.exit_code == 0, list_result.output
    assert "myproj" in list_result.output
    assert "web,py" in list_result.output


def test_project_remove(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    store.add(ManagedProject(name="alpha", path="/tmp/alpha", added_at=1.0))
    config_path = _write_config(tmp_path)

    with patch("justagent.cli.commands.project.ProjectStore", return_value=store):
        remove_result = runner.invoke(
            app, ["--config", str(config_path), "project", "remove", "alpha"]
        )
    assert remove_result.exit_code == 0, remove_result.output
    assert "Removed project 'alpha'" in remove_result.output
    assert store.get("alpha") is None


def test_project_remove_missing_exits_one(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    config_path = _write_config(tmp_path)

    with patch("justagent.cli.commands.project.ProjectStore", return_value=store):
        remove_result = runner.invoke(
            app, ["--config", str(config_path), "project", "remove", "ghost"]
        )
    assert remove_result.exit_code == 1, remove_result.output
    assert "No project named 'ghost'" in remove_result.output


def test_project_run_invokes_subprocess_in_project_cwd(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    project_dir = tmp_path / "runproj"
    project_dir.mkdir()
    store.add(ManagedProject(name="runproj", path=str(project_dir), added_at=1.0))
    config_path = _write_config(tmp_path)

    fake_result = MagicMock(returncode=0)
    with (
        patch("justagent.cli.commands.project.ProjectStore", return_value=store),
        patch(
            "justagent.cli.commands.project.subprocess.run", return_value=fake_result
        ) as run_mock,
    ):
        run_cli_result = runner.invoke(
            app,
            ["--config", str(config_path), "project", "run", "runproj", "echo", "hi"],
        )
    assert run_cli_result.exit_code == 0, run_cli_result.output
    run_mock.assert_called_once()
    called_args, called_kwargs = run_mock.call_args
    assert called_args[0] == ["echo", "hi"]
    assert called_kwargs["cwd"] == str(project_dir)


def test_project_run_missing_project_exits_one(tmp_path: Path) -> None:
    store = ProjectStore(store_path=tmp_path / "projects.json")
    config_path = _write_config(tmp_path)

    with patch("justagent.cli.commands.project.ProjectStore", return_value=store):
        run_cli_result = runner.invoke(
            app,
            ["--config", str(config_path), "project", "run", "ghost", "echo", "hi"],
        )
    assert run_cli_result.exit_code == 1, run_cli_result.output
    assert "No project named 'ghost'" in run_cli_result.output
