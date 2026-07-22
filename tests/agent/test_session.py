"""Tests for ``autoship.agent.session`` (session persistence)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from autoship.agent.runtime import Message, ToolCall, ToolResultPart
from autoship.agent.session import (
    Session,
    SessionError,
    SessionMetadata,
    SessionStatus,
    SessionStore,
    default_store_dir,
    deserialize_message,
    get_session_store,
    serialize_message,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    *,
    id: str = "sess-1",
    updated_at: float = 1000.0,
    created_at: float = 900.0,
    status: SessionStatus = SessionStatus.ACTIVE,
    files_changed: list[str] | None = None,
) -> SessionMetadata:
    return SessionMetadata(
        id=id,
        created_at=created_at,
        updated_at=updated_at,
        status=status,
        mode="act",
        model="gpt-4o",
        cwd="/tmp",
        prompt_preview="hello world",
        iterations=3,
        total_tokens=42,
        message_count=5,
        files_changed=files_changed if files_changed is not None else [],
    )


# ---------------------------------------------------------------------------
# SessionMetadata
# ---------------------------------------------------------------------------


class TestSessionMetadata:
    def test_construction(self) -> None:
        m = _make_metadata()
        assert m.id == "sess-1"
        assert m.status is SessionStatus.ACTIVE
        assert m.mode == "act"
        assert m.total_tokens == 42
        assert m.message_count == 5
        assert m.files_changed == []

    def test_frozen_immutable(self) -> None:
        m = _make_metadata()
        with pytest.raises(FrozenInstanceError):
            m.id = "other"  # type: ignore[misc]

    def test_files_changed_default_isolated(self) -> None:
        """Two metadata instances must not share the files_changed list."""

        m1 = SessionMetadata(
            id="a",
            created_at=0,
            updated_at=0,
            status=SessionStatus.ACTIVE,
            mode="act",
            model="m",
            cwd=".",
            prompt_preview="",
            iterations=0,
            total_tokens=0,
            message_count=0,
        )
        m2 = SessionMetadata(
            id="b",
            created_at=0,
            updated_at=0,
            status=SessionStatus.ACTIVE,
            mode="act",
            model="m",
            cwd=".",
            prompt_preview="",
            iterations=0,
            total_tokens=0,
            message_count=0,
        )
        m1.files_changed.append("x.py")
        assert m2.files_changed == []

    def test_status_is_str_enum(self) -> None:
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus("completed") is SessionStatus.COMPLETED


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class TestSession:
    def test_construction_defaults(self) -> None:
        m = _make_metadata()
        s = Session(metadata=m)
        assert s.metadata is m
        assert s.messages == []
        assert s.usage == {}

    def test_construction_with_data(self) -> None:
        m = _make_metadata()
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        s = Session(metadata=m, messages=msgs, usage={"total_tokens": 7})
        assert s.messages == msgs
        assert s.usage == {"total_tokens": 7}

    def test_session_mutable(self) -> None:
        s = Session(metadata=_make_metadata())
        s.messages.append({"role": "user", "content": "x"})
        s.usage["total_tokens"] = 1
        assert len(s.messages) == 1
        assert s.usage["total_tokens"] == 1

    def test_messages_default_isolated(self) -> None:
        s1 = Session(metadata=_make_metadata())
        s2 = Session(metadata=_make_metadata())
        s1.messages.append({"role": "user", "content": "x"})
        assert s2.messages == []


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------


class TestSerializeMessage:
    def test_round_trip_simple(self) -> None:
        msg = Message(role="user", content="hello there")
        data = serialize_message(msg)
        assert data["role"] == "user"
        assert data["content"] == "hello there"
        restored = deserialize_message(data)
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.tool_calls == []
        assert restored.tool_result is None

    def test_round_trip_system_message(self) -> None:
        msg = Message(role="system", content="you are an agent")
        restored = deserialize_message(serialize_message(msg))
        assert restored.role == "system"
        assert restored.content == "you are an agent"

    def test_with_tool_calls(self) -> None:
        msg = Message(
            role="assistant",
            content="calling tool",
            tool_calls=[
                ToolCall(id="call-1", name="read_file", input={"path": "a.py"}),
                ToolCall(id="call-2", name="run_command", input={"command": "ls"}),
            ],
        )
        data = serialize_message(msg)
        assert data["tool_calls"] == [
            {"id": "call-1", "name": "read_file", "input": {"path": "a.py"}},
            {"id": "call-2", "name": "run_command", "input": {"command": "ls"}},
        ]
        restored = deserialize_message(data)
        assert restored.role == "assistant"
        assert restored.content == "calling tool"
        assert len(restored.tool_calls) == 2
        assert restored.tool_calls[0].id == "call-1"
        assert restored.tool_calls[0].name == "read_file"
        assert restored.tool_calls[0].input == {"path": "a.py"}
        assert restored.tool_calls[1].input == {"command": "ls"}

    def test_with_tool_result(self) -> None:
        tr = ToolResultPart(
            tool_call_id="call-1",
            name="read_file",
            output="file contents",
            is_error=False,
        )
        msg = Message(role="tool", content="", tool_result=tr, name="read_file")
        data = serialize_message(msg)
        assert data["tool_result"] == {
            "tool_call_id": "call-1",
            "name": "read_file",
            "output": "file contents",
            "is_error": False,
        }
        assert data["name"] == "read_file"
        restored = deserialize_message(data)
        assert restored.role == "tool"
        assert restored.tool_result is not None
        assert restored.tool_result.tool_call_id == "call-1"
        assert restored.tool_result.name == "read_file"
        assert restored.tool_result.output == "file contents"
        assert restored.tool_result.is_error is False
        assert restored.name == "read_file"

    def test_with_error_tool_result(self) -> None:
        tr = ToolResultPart(
            tool_call_id="c2", name="run", output="boom", is_error=True
        )
        msg = Message(role="tool", tool_result=tr)
        data = serialize_message(msg)
        assert data["tool_result"]["is_error"] is True
        restored = deserialize_message(data)
        assert restored.tool_result is not None
        assert restored.tool_result.is_error is True

    def test_with_metadata(self) -> None:
        msg = Message(
            role="assistant", content="ok", metadata={"iteration": 4}
        )
        data = serialize_message(msg)
        assert data["metadata"] == {"iteration": 4}
        restored = deserialize_message(data)
        assert restored.metadata == {"iteration": 4}

    def test_empty_tool_calls_not_serialized(self) -> None:
        msg = Message(role="assistant", content="done")
        data = serialize_message(msg)
        assert "tool_calls" not in data
        assert "tool_result" not in data

    def test_deserialize_missing_optional_fields(self) -> None:
        restored = deserialize_message({"role": "user", "content": "hi"})
        assert restored.tool_calls == []
        assert restored.tool_result is None
        assert restored.name is None
        assert restored.metadata == {}


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class TestSessionStore:
    def test_default_store_dir(self) -> None:
        d = default_store_dir()
        assert d.name == "sessions"
        assert d.parent.name == ".autoship"

    def test_create_session(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        s = store.create_session(
            mode="act", model="gpt-4o", cwd="/proj", initial_prompt="do thing"
        )
        assert s.metadata.id
        assert len(s.metadata.id) > 0
        assert s.metadata.status is SessionStatus.ACTIVE
        assert s.metadata.mode == "act"
        assert s.metadata.model == "gpt-4o"
        assert s.metadata.cwd == "/proj"
        assert s.metadata.prompt_preview == "do thing"
        assert s.metadata.iterations == 0
        assert s.metadata.total_tokens == 0
        assert s.metadata.message_count == 0
        assert s.metadata.files_changed == []
        assert s.messages == []
        assert s.usage == {}

    def test_create_session_truncates_preview(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        long_prompt = "x" * 500
        s = store.create_session(
            mode="act", model="m", cwd=".", initial_prompt=long_prompt
        )
        assert len(s.metadata.prompt_preview) == 200

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        s = store.create_session(
            mode="plan", model="claude", cwd="/proj", initial_prompt="hello"
        )
        s.messages = [serialize_message(Message(role="user", content="hi"))]
        s.usage = {"prompt_tokens": 10, "total_tokens": 10}
        store.save(s)

        assert store.exists(s.metadata.id)
        loaded = store.load(s.metadata.id)
        assert loaded.metadata.id == s.metadata.id
        assert loaded.metadata.mode == "plan"
        assert loaded.metadata.model == "claude"
        assert loaded.metadata.prompt_preview == "hello"
        assert loaded.messages == s.messages
        assert loaded.usage == s.usage

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        with pytest.raises(SessionError):
            store.load("does-not-exist")

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "deep" / "nested" / "sessions")
        s = store.create_session("act", "m", ".", "p")
        store.save(s)
        assert store.exists(s.metadata.id)

    def test_list_sessions_empty(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        assert store.list_sessions() == []

    def test_list_sessions_empty_when_dir_missing(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        # Directory does not exist yet.
        assert store.list_sessions() == []

    def test_list_sessions_sorted_by_updated_desc(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        old = store.create_session("act", "m", ".", "old")
        old.metadata = SessionMetadata(
            id=old.metadata.id,
            created_at=100.0,
            updated_at=100.0,
            status=SessionStatus.COMPLETED,
            mode="act",
            model="m",
            cwd=".",
            prompt_preview="old",
            iterations=1,
            total_tokens=1,
            message_count=1,
            files_changed=[],
        )
        new = store.create_session("act", "m", ".", "new")
        new.metadata = SessionMetadata(
            id=new.metadata.id,
            created_at=200.0,
            updated_at=200.0,
            status=SessionStatus.ACTIVE,
            mode="act",
            model="m",
            cwd=".",
            prompt_preview="new",
            iterations=2,
            total_tokens=2,
            message_count=2,
            files_changed=[],
        )
        store.save(old)
        store.save(new)

        listed = store.list_sessions()
        assert len(listed) == 2
        assert listed[0].id == new.metadata.id  # newer first
        assert listed[1].id == old.metadata.id

    def test_list_sessions_skips_corrupt_files(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        good = store.create_session("act", "m", ".", "good")
        store.save(good)

        # Write a corrupt JSON file alongside the good one.
        (tmp_path / "sessions" / "broken.json").write_text("{not valid json")

        listed = store.list_sessions()
        assert len(listed) == 1
        assert listed[0].id == good.metadata.id

    def test_list_sessions_skips_non_session_json(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        good = store.create_session("act", "m", ".", "good")
        store.save(good)
        # A JSON file that parses but lacks session structure.
        (tmp_path / "sessions" / "other.json").write_text('{"hello": "world"}')

        listed = store.list_sessions()
        assert len(listed) == 1
        assert listed[0].id == good.metadata.id

    def test_delete_existing(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        s = store.create_session("act", "m", ".", "p")
        store.save(s)
        assert store.delete(s.metadata.id) is True
        assert not store.exists(s.metadata.id)
        # Deleting again returns False.
        assert store.delete(s.metadata.id) is False

    def test_delete_missing_returns_false(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        assert store.delete("nope") is False

    def test_exists(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        assert store.exists("nope") is False
        s = store.create_session("act", "m", ".", "p")
        store.save(s)
        assert store.exists(s.metadata.id) is True

    def test_update_session_persists_and_returns(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        s = store.create_session("act", "m", ".", "p")
        msgs = [serialize_message(Message(role="user", content="hi"))]
        updated = store.update_session(
            s,
            messages=msgs,
            usage={"total_tokens": 99},
            status=SessionStatus.COMPLETED,
            files_changed=["a.py", "b.py"],
        )
        assert updated.metadata.status is SessionStatus.COMPLETED
        assert updated.metadata.total_tokens == 99
        assert updated.metadata.message_count == 1
        assert updated.metadata.files_changed == ["a.py", "b.py"]
        # Persisted to disk.
        reloaded = store.load(s.metadata.id)
        assert reloaded.metadata.status is SessionStatus.COMPLETED
        assert reloaded.metadata.total_tokens == 99
        assert reloaded.metadata.files_changed == ["a.py", "b.py"]
        assert len(reloaded.messages) == 1

    def test_update_session_keeps_existing_files(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        s = store.create_session("act", "m", ".", "p")
        # Seed files_changed via a manual metadata update.
        s.metadata = SessionMetadata(
            id=s.metadata.id,
            created_at=s.metadata.created_at,
            updated_at=s.metadata.updated_at,
            status=s.metadata.status,
            mode=s.metadata.mode,
            model=s.metadata.model,
            cwd=s.metadata.cwd,
            prompt_preview=s.metadata.prompt_preview,
            iterations=s.metadata.iterations,
            total_tokens=s.metadata.total_tokens,
            message_count=s.metadata.message_count,
            files_changed=["kept.py"],
        )
        updated = store.update_session(
            s,
            messages=[],
            usage={"total_tokens": 0},
            status=SessionStatus.ACTIVE,
        )
        assert updated.metadata.files_changed == ["kept.py"]


# ---------------------------------------------------------------------------
# get_session_store / env override
# ---------------------------------------------------------------------------


class TestGetSessionStore:
    def test_explicit_store_dir_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AUTOSHIP_SESSIONS_DIR", str(tmp_path / "env"))
        store = get_session_store(tmp_path / "explicit")
        assert store.store_dir == tmp_path / "explicit"

    def test_env_var_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_dir = tmp_path / "env-sessions"
        monkeypatch.setenv("AUTOSHIP_SESSIONS_DIR", str(env_dir))
        store = get_session_store()
        assert store.store_dir == env_dir
        s = store.create_session("act", "m", ".", "p")
        store.save(s)
        assert (env_dir / f"{s.metadata.id}.json").exists()

    def test_default_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUTOSHIP_SESSIONS_DIR", raising=False)
        store = get_session_store()
        assert store.store_dir == default_store_dir()


# ---------------------------------------------------------------------------
# Disk format sanity
# ---------------------------------------------------------------------------


class TestDiskFormat:
    def test_save_uses_indented_utf8_json(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        s = store.create_session("act", "m", ".", "中文预览")
        store.save(s)
        path = tmp_path / "sessions" / f"{s.metadata.id}.json"
        raw = path.read_text(encoding="utf-8")
        # Indented (two-space) and non-ASCII preserved.
        assert "\n  " in raw
        assert "中文预览" in raw
        data = json.loads(raw)
        assert data["metadata"]["id"] == s.metadata.id
        assert data["metadata"]["status"] == "active"
        assert data["messages"] == []
        assert data["usage"] == {}
