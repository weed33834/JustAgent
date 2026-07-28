"""Session persistence — save and resume agent conversations.

Sessions are saved to ``~/.justagent/sessions/`` as JSON files. Each
session contains the conversation messages, usage stats, mode, model,
and metadata. Sessions can be listed, resumed, and deleted.

Reference: Cline's task history and OpenCode's session persistence.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from justagent.exceptions import MyAgentError
from justagent.utils.atomic_write import atomic_write_text

# ---------------------------------------------------------------------------
# Module-level named defaults (lambdas can't satisfy mypy defaults)
# ---------------------------------------------------------------------------


def _default_files_changed() -> list[str]:
    return []


def _default_messages() -> list[dict[str, Any]]:
    return []


def _default_usage() -> dict[str, Any]:
    return {}


def default_store_dir() -> Path:
    """Return the default session store directory (``~/.justagent/sessions``)."""

    return Path.home() / ".justagent" / "sessions"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SessionError(MyAgentError):
    """Raised when a session operation fails (not found, corrupt, etc.)."""


# ---------------------------------------------------------------------------
# Status & dataclasses
# ---------------------------------------------------------------------------


class SessionStatus(str, Enum):  # noqa: UP042
    """Lifecycle status of a session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass(frozen=True)
class SessionMetadata:
    """Immutable snapshot of a session's high-level state.

    Mirrors Cline's task metadata and OpenCode's session summary. The
    ``files_changed`` list is the unique set of paths touched during the
    run (in first-seen order).
    """

    id: str
    created_at: float
    updated_at: float
    status: SessionStatus
    mode: str
    model: str
    cwd: str
    prompt_preview: str
    iterations: int
    total_tokens: int
    message_count: int
    files_changed: list[str] = field(default_factory=_default_files_changed)


@dataclass
class Session:
    """A persisted agent conversation.

    ``messages`` are serialized :class:`justagent.agent.runtime.Message`
    dicts (see :func:`serialize_message`). ``usage`` is the accumulated
    token-usage mapping (``prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens``).
    """

    metadata: SessionMetadata
    messages: list[dict[str, Any]] = field(default_factory=_default_messages)
    usage: dict[str, Any] = field(default_factory=_default_usage)


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------


def serialize_message(msg: Any) -> dict[str, Any]:
    """Convert a runtime Message to a JSON-serializable dict.

    Handles ``role`` / ``content`` (always), ``tool_calls`` (assistant
    messages that request tools), ``tool_result`` (``role="tool"``
    messages), and the optional ``name`` / ``metadata`` fields.
    """

    result: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        result["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "input": tc.input}
            for tc in msg.tool_calls
        ]
    if msg.tool_result is not None:
        result["tool_result"] = {
            "tool_call_id": msg.tool_result.tool_call_id,
            "name": msg.tool_result.name,
            "output": msg.tool_result.output,
            "is_error": msg.tool_result.is_error,
        }
    if msg.name is not None:
        result["name"] = msg.name
    if msg.metadata:
        result["metadata"] = dict(msg.metadata)
    return result


def deserialize_message(data: dict[str, Any]) -> Any:
    """Convert a dict back to a runtime Message.

    Inverse of :func:`serialize_message`. Imports the runtime message
    classes lazily to avoid a circular import at module load time.
    """

    from justagent.agent.runtime import Message, ToolCall, ToolResultPart

    tool_calls = [
        ToolCall(id=tc["id"], name=tc["name"], input=dict(tc["input"]))
        for tc in data.get("tool_calls", [])
    ]
    tool_result: ToolResultPart | None = None
    raw_tr = data.get("tool_result")
    if raw_tr is not None:
        tool_result = ToolResultPart(
            tool_call_id=raw_tr["tool_call_id"],
            name=raw_tr["name"],
            output=raw_tr["output"],
            is_error=bool(raw_tr.get("is_error", False)),
        )
    return Message(
        role=data["role"],
        content=data.get("content", ""),
        tool_calls=tool_calls,
        tool_result=tool_result,
        name=data.get("name"),
        metadata=dict(data.get("metadata", {})),
    )


# ---------------------------------------------------------------------------
# Session <-> dict (disk format)
# ---------------------------------------------------------------------------


def _metadata_to_dict(m: SessionMetadata) -> dict[str, Any]:
    return {
        "id": m.id,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "status": m.status.value,
        "mode": m.mode,
        "model": m.model,
        "cwd": m.cwd,
        "prompt_preview": m.prompt_preview,
        "iterations": m.iterations,
        "total_tokens": m.total_tokens,
        "message_count": m.message_count,
        "files_changed": list(m.files_changed),
    }


def _metadata_from_dict(d: dict[str, Any]) -> SessionMetadata:
    return SessionMetadata(
        id=str(d["id"]),
        created_at=float(d["created_at"]),
        updated_at=float(d["updated_at"]),
        status=SessionStatus(str(d["status"])),
        mode=str(d["mode"]),
        model=str(d["model"]),
        cwd=str(d["cwd"]),
        prompt_preview=str(d.get("prompt_preview", "")),
        iterations=int(d.get("iterations", 0)),
        total_tokens=int(d.get("total_tokens", 0)),
        message_count=int(d.get("message_count", 0)),
        files_changed=list(d.get("files_changed", [])),
    )


def _session_to_dict(session: Session) -> dict[str, Any]:
    return {
        "metadata": _metadata_to_dict(session.metadata),
        "messages": list(session.messages),
        "usage": dict(session.usage),
    }


def _session_from_dict(d: dict[str, Any]) -> Session:
    return Session(
        metadata=_metadata_from_dict(d["metadata"]),
        messages=list(d.get("messages", [])),
        usage=dict(d.get("usage", {})),
    )


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """Filesystem-backed store for agent sessions.

    Each session is one JSON file at ``{store_dir}/{session_id}.json``.
    Writes are atomic via :func:`atomic_write_text`. Corrupt files are
    skipped (best-effort) when listing, never crashing the CLI.
    """

    def __init__(self, store_dir: Path | None = None) -> None:
        self._store_dir = store_dir or default_store_dir()

    @property
    def store_dir(self) -> Path:
        return self._store_dir

    def _path_for(self, session_id: str) -> Path:
        return self._store_dir / f"{session_id}.json"

    def save(self, session: Session) -> None:
        """Persist ``session`` to ``{store_dir}/{id}.json`` atomically."""

        path = self._path_for(session.metadata.id)
        data = _session_to_dict(session)
        atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))

    def load(self, session_id: str) -> Session:
        """Load a session by id. Raise :class:`SessionError` if missing/corrupt."""

        path = self._path_for(session_id)
        if not path.exists():
            raise SessionError(f"Session not found: {session_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionError(
                f"Failed to load session {session_id}: {exc}"
            ) from exc
        try:
            return _session_from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionError(
                f"Corrupt session file {session_id}: {exc}"
            ) from exc

    def list_sessions(self) -> list[SessionMetadata]:
        """Return all sessions, sorted by ``updated_at`` descending.

        Corrupt or unreadable files are skipped gracefully.
        """

        if not self._store_dir.exists():
            return []
        result: list[SessionMetadata] = []
        for path in self._store_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                result.append(_session_from_dict(data).metadata)
            except Exception:  # noqa: BLE001 - skip corrupt files
                continue
        result.sort(key=lambda m: m.updated_at, reverse=True)
        return result

    def delete(self, session_id: str) -> bool:
        """Delete a session file. Return ``True`` if deleted, ``False`` if absent."""

        path = self._path_for(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def exists(self, session_id: str) -> bool:
        return self._path_for(session_id).exists()

    def create_session(
        self,
        mode: str,
        model: str,
        cwd: str,
        initial_prompt: str,
    ) -> Session:
        """Create a new empty session with a generated id."""

        now = time.time()
        session_id = uuid.uuid4().hex[:16]
        metadata = SessionMetadata(
            id=session_id,
            created_at=now,
            updated_at=now,
            status=SessionStatus.ACTIVE,
            mode=mode,
            model=model,
            cwd=cwd,
            prompt_preview=initial_prompt[:200],
            iterations=0,
            total_tokens=0,
            message_count=0,
            files_changed=[],
        )
        return Session(metadata=metadata, messages=[], usage={})

    def update_session(
        self,
        session: Session,
        messages: list[dict[str, Any]],
        usage: dict[str, Any],
        status: SessionStatus,
        files_changed: list[str] | None = None,
    ) -> Session:
        """Update a session's data, persist it, and return the new session."""

        new_files = (
            list(files_changed)
            if files_changed is not None
            else list(session.metadata.files_changed)
        )
        new_metadata = replace(
            session.metadata,
            updated_at=time.time(),
            status=status,
            total_tokens=int(usage.get("total_tokens", 0)),
            message_count=len(messages),
            files_changed=new_files,
        )
        updated = Session(
            metadata=new_metadata,
            messages=list(messages),
            usage=dict(usage),
        )
        self.save(updated)
        return updated


def get_session_store(store_dir: Path | None = None) -> SessionStore:
    """Build a :class:`SessionStore`, honoring the ``MYAGENT_SESSIONS_DIR`` env var.

    Explicit ``store_dir`` wins; otherwise the ``MYAGENT_SESSIONS_DIR``
    environment variable is consulted (used by tests); otherwise the
    default ``~/.justagent/sessions`` directory is used.
    """

    if store_dir is not None:
        return SessionStore(store_dir)
    env_dir = os.environ.get("MYAGENT_SESSIONS_DIR")
    if env_dir:
        return SessionStore(Path(env_dir))
    return SessionStore()


__all__ = [
    "Session",
    "SessionError",
    "SessionMetadata",
    "SessionStatus",
    "SessionStore",
    "default_store_dir",
    "deserialize_message",
    "get_session_store",
    "serialize_message",
]
