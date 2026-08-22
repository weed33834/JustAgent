"""Tests for the ``justagent session`` CLI commands."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from justagent.agent.session import SessionStore
from justagent.cli.main import app

runner = CliRunner()


def _invoke(args: list[str], *, sessions_dir: Path, monkeypatch: pytest.MonkeyPatch):
    # Set the canonical env var explicitly. Config loading runs
    # ``_migrate_legacy_env()`` which would otherwise copy the legacy
    # ``MYAGENT_*`` value into ``JUSTAGENT_*`` as an untracked side effect
    # and leak across tests. Setting it here keeps each test isolated.
    monkeypatch.setenv("JUSTAGENT_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setenv("MYAGENT_SESSIONS_DIR", str(sessions_dir))
    return runner.invoke(app, ["session", *args])


def _seed_session(
    store: SessionStore,
    *,
    prompt: str = "do a thing",
    model: str = "gpt-4o",
    mode: str = "act",
) -> str:
    s = store.create_session(mode=mode, model=model, cwd="/proj", initial_prompt=prompt)
    s.messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "done"},
    ]
    s.usage = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
    store.save(s)
    return s.metadata.id


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestSessionList:
    def test_list_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _invoke(["list"], sessions_dir=tmp_path / "s", monkeypatch=monkeypatch)
        assert result.exit_code == 0
        assert "没有保存的会话" in result.output or "No saved sessions" in result.output

    def test_list_with_sessions(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sessions_dir = tmp_path / "s"
        store = SessionStore(sessions_dir)
        sid = _seed_session(store, prompt="refactor the module")

        result = _invoke(["list"], sessions_dir=sessions_dir, monkeypatch=monkeypatch)
        assert result.exit_code == 0
        assert sid in result.output
        assert "refactor the module" in result.output
        # Header columns present.
        assert "ID" in result.output
        assert "Mode" in result.output

    def test_list_shows_multiple_sessions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sessions_dir = tmp_path / "s"
        store = SessionStore(sessions_dir)
        sid1 = _seed_session(store, prompt="first task")
        sid2 = _seed_session(store, prompt="second task")

        result = _invoke(["list"], sessions_dir=sessions_dir, monkeypatch=monkeypatch)
        assert result.exit_code == 0
        assert sid1 in result.output
        assert sid2 in result.output


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


class TestSessionShow:
    def test_show_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sessions_dir = tmp_path / "s"
        store = SessionStore(sessions_dir)
        sid = _seed_session(store, prompt="hello there", model="claude-3")

        result = _invoke(["show", sid], sessions_dir=sessions_dir, monkeypatch=monkeypatch)
        assert result.exit_code == 0
        assert sid in result.output
        assert "claude-3" in result.output
        assert "act" in result.output
        assert "hello there" in result.output
        # Message count derived from saved messages.
        assert "Messages:" in result.output

    def test_show_nonexistent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _invoke(
            ["show", "no-such-session"],
            sessions_dir=tmp_path / "s",
            monkeypatch=monkeypatch,
        )
        assert result.exit_code == 1
        assert "no-such-session" in result.output or "找不到" in result.output


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestSessionDelete:
    def test_delete_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sessions_dir = tmp_path / "s"
        store = SessionStore(sessions_dir)
        sid = _seed_session(store)

        result = _invoke(["delete", sid], sessions_dir=sessions_dir, monkeypatch=monkeypatch)
        assert result.exit_code == 0
        assert sid in result.output
        # Actually removed from the store.
        assert not store.exists(sid)

    def test_delete_nonexistent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _invoke(
            ["delete", "missing-id"],
            sessions_dir=tmp_path / "s",
            monkeypatch=monkeypatch,
        )
        assert result.exit_code == 1
        assert "missing-id" in result.output or "找不到" in result.output


# ---------------------------------------------------------------------------
# resume (launches the interactive agent with --resume <id>)
# ---------------------------------------------------------------------------


class TestSessionResume:
    def test_resume_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sessions_dir = tmp_path / "s"
        store = SessionStore(sessions_dir)
        sid = _seed_session(store)

        captured: dict[str, object] = {}

        class _Completed:
            returncode = 0

        def _fake_run(cmd, *args, **kwargs):
            captured["cmd"] = list(cmd)
            return _Completed()

        # Don't actually spawn an interactive subprocess in the unit test.
        monkeypatch.setattr("subprocess.run", _fake_run)

        result = _invoke(["resume", sid], sessions_dir=sessions_dir, monkeypatch=monkeypatch)
        assert result.exit_code == 0
        assert sid in result.output
        # The interactive agent is launched with the session restored.
        cmd = list(captured.get("cmd", []))
        assert "agent" in cmd
        assert "-i" in cmd
        assert "--resume" in cmd
        assert sid in cmd

    def test_resume_nonexistent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _invoke(
            ["resume", "ghost-id"],
            sessions_dir=tmp_path / "s",
            monkeypatch=monkeypatch,
        )
        assert result.exit_code == 1
        assert "ghost-id" in result.output or "找不到" in result.output


# ---------------------------------------------------------------------------
# Registration smoke test
# ---------------------------------------------------------------------------


class TestSessionRegistration:
    def test_session_command_registered(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "session" in result.output

    def test_session_help_lists_subcommands(self) -> None:
        result = runner.invoke(app, ["session", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "resume" in result.output
        assert "delete" in result.output

    def test_agent_help_lists_resume_flag(self) -> None:
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0
        out = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--resume" in out
