"""Immutable audit log — append-only storage with cryptographic hash chaining.

Every operation in the communication module (messages sent, notifications
delivered, broadcasts fired, meetings scheduled) is recorded here. Entries
are *append-only*: there is no ``update`` or ``delete`` method. Each entry
carries a SHA-256 ``entry_hash`` computed over its content plus the
``prev_hash`` of the preceding entry, forming a tamper-evident hash chain.
Modifying or removing any entry breaks the chain and is immediately
detectable via :meth:`AuditStore.verify_chain`.

Storage is JSONL (one JSON object per line) on the local filesystem. The
store loads existing entries on startup so chains survive process restarts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("myagent.communication.audit")

#: SHA-256 of an empty string — sentinel for the genesis (first) entry.
_GENESIS_PREV_HASH = "0" * 64


class AuditLevel(str, Enum):  # noqa: UP042 - match existing codebase style
    """Severity level for an audit entry."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(str, Enum):  # noqa: UP042
    """Functional category grouping related audit events."""

    AUTH = "auth"
    MESSAGE = "message"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    MEETING = "meeting"
    CHANNEL = "channel"
    SYSTEM = "system"
    USER = "user"
    SECURITY = "security"


class AuditEntry(BaseModel):
    """A single immutable audit record linked into a hash chain.

    Attributes:
        sequence: Monotonically increasing 0-based index.
        timestamp: UTC instant the entry was created.
        level: Severity of the event.
        category: Functional grouping.
        actor: Identifier of the user/service that performed the action.
        action: Short verb describing the operation (e.g. ``"message.send"``).
        target: Identifier of the object acted upon.
        details: Arbitrary structured payload (redacted at storage time
            by callers if needed).
        prev_hash: ``entry_hash`` of the previous entry (genesis sentinel
            for the first entry).
        entry_hash: SHA-256 over every field above. Computed by the store
            and verified on read-back.
    """

    sequence: int
    timestamp: datetime
    level: AuditLevel
    category: AuditCategory
    actor: str
    action: str
    target: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = _GENESIS_PREV_HASH
    entry_hash: str = ""

    # ------------------------------------------------------------------
    # Hash computation
    # ------------------------------------------------------------------

    def _signing_payload(self) -> str:
        """Canonical JSON of every field *except* ``entry_hash``."""

        payload: dict[str, Any] = {
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "details": self.details,
            "prev_hash": self.prev_hash,
        }
        return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)

    def compute_hash(self) -> str:
        """Return the SHA-256 hex digest of this entry's signing payload."""

        return hashlib.sha256(self._signing_payload().encode("utf-8")).hexdigest()

    def seal(self) -> AuditEntry:
        """Return a copy with ``entry_hash`` populated (idempotent)."""

        digest = self.compute_hash()
        if self.entry_hash == digest:
            return self
        return self.model_copy(update={"entry_hash": digest})

    def verify(self) -> bool:
        """True when the stored ``entry_hash`` matches a recomputation."""

        if not self.entry_hash:
            return False
        return self.compute_hash() == self.entry_hash


# ---------------------------------------------------------------------------
# Query filter
# ---------------------------------------------------------------------------


class AuditQuery(BaseModel):
    """Filter criteria for :meth:`AuditStore.query`.

    All fields are optional; ``None`` means "no filter on this dimension".
    """

    level: AuditLevel | None = None
    category: AuditCategory | None = None
    actor: str | None = None
    action: str | None = None
    target: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    text: str | None = Field(
        default=None,
        description="Substring search across actor, action, target and details.",
    )

    def matches(self, entry: AuditEntry) -> bool:
        """Return True when *entry* satisfies every non-None criterion."""

        if self.level is not None and entry.level is not self.level:
            return False
        if self.category is not None and entry.category is not self.category:
            return False
        if self.actor is not None and entry.actor != self.actor:
            return False
        if self.action is not None and entry.action != self.action:
            return False
        if self.target is not None and entry.target != self.target:
            return False
        if self.since is not None and entry.timestamp < self.since:
            return False
        if self.until is not None and entry.timestamp > self.until:
            return False
        if self.text is not None:
            needle = self.text.lower()
            haystack_parts = [
                entry.actor.lower(),
                entry.action.lower(),
                entry.target.lower(),
                json.dumps(entry.details, default=str).lower(),
            ]
            if not any(needle in part for part in haystack_parts):
                return False
        return True


# ---------------------------------------------------------------------------
# Chain verification result
# ---------------------------------------------------------------------------


class ChainVerification(BaseModel):
    """Outcome of a full hash-chain integrity check."""

    valid: bool
    checked: int
    broken_at: int | None = Field(
        default=None,
        description="Sequence index of the first entry whose link is broken.",
    )
    reason: str = ""

    def __bool__(self) -> bool:
        return self.valid


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AuditStore:
    """Append-only, hash-chained audit log backed by a JSONL file.

    The store is safe for concurrent use within a single asyncio event
    loop (an :class:`asyncio.Lock` serialises appends). On construction it
    loads any pre-existing entries from disk so that the chain continues
    across process restarts.

    Example::

        store = AuditStore(Path("/var/log/myagent/audit.jsonl"))
        await store.append(
            actor="alice",
            action="message.send",
            category=AuditCategory.MESSAGE,
            level=AuditLevel.INFO,
            target="channel#general",
            details={"message_id": "msg-1"},
        )
        report = await store.verify_chain()
        assert report.valid
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._entries: list[AuditEntry] = []
        self._lock = asyncio.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Read existing entries from disk (idempotent, called once)."""

        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            if self._path is not None and self._path.exists():
                await asyncio.to_thread(self._load_sync)
            self._loaded = True
            logger.debug("Loaded %d audit entries from %s", len(self._entries), self._path)

    def _load_sync(self) -> None:
        assert self._path is not None
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed audit line in %s", self._path)
                    continue
                try:
                    entry = AuditEntry.model_validate(data)
                except Exception:  # noqa: BLE001 - skip unparseable, keep chain
                    logger.warning("Skipping unparseable audit entry in %s", self._path)
                    continue
                self._entries.append(entry)

    async def flush(self) -> None:
        """Ensure all in-memory entries are persisted to disk."""

        if self._path is None:
            return
        async with self._lock:
            await asyncio.to_thread(self._flush_sync)

    def _flush_sync(self) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            for entry in self._entries:
                fh.write(entry.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    # Append (the only mutation)
    # ------------------------------------------------------------------

    async def append(
        self,
        *,
        actor: str,
        action: str,
        category: AuditCategory,
        level: AuditLevel = AuditLevel.INFO,
        target: str = "",
        details: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> AuditEntry:
        """Create, seal, persist and return a new audit entry.

        Raises :class:`RuntimeError` if called before :meth:`load`.
        """

        if not self._loaded:
            await self.load()
        async with self._lock:
            sequence = len(self._entries)
            prev_hash = self._entries[-1].entry_hash if self._entries else _GENESIS_PREV_HASH
            entry = AuditEntry(
                sequence=sequence,
                timestamp=timestamp or datetime.now(UTC),
                level=level,
                category=category,
                actor=actor,
                action=action,
                target=target,
                details=details or {},
                prev_hash=prev_hash,
            )
            sealed = entry.seal()
            self._entries.append(sealed)
            if self._path is not None:
                await asyncio.to_thread(self._append_sync, sealed)
            logger.debug(
                "Audit entry #%d sealed: %s/%s by %s",
                sealed.sequence,
                sealed.category.value,
                sealed.action,
                sealed.actor,
            )
            return sealed

    def _append_sync(self, entry: AuditEntry) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(self, filter_: AuditQuery | None = None) -> list[AuditEntry]:
        """Return entries matching *filter_* (or all entries when ``None``)."""

        if not self._loaded:
            await self.load()
        async with self._lock:
            snapshot = list(self._entries)
        if filter_ is None:
            return snapshot
        return [e for e in snapshot if filter_.matches(e)]

    async def get(self, sequence: int) -> AuditEntry | None:
        """Return the entry at *sequence*, or ``None`` if out of range."""

        if not self._loaded:
            await self.load()
        async with self._lock:
            if 0 <= sequence < len(self._entries):
                return self._entries[sequence]
        return None

    async def iterate(self) -> AsyncIterator[AuditEntry]:
        """Asynchronously yield entries in sequence order."""

        if not self._loaded:
            await self.load()
        async with self._lock:
            snapshot = list(self._entries)
        for entry in snapshot:
            yield entry

    # ------------------------------------------------------------------
    # Chain verification
    # ------------------------------------------------------------------

    async def verify_chain(self) -> ChainVerification:
        """Verify the integrity of the entire hash chain.

        Checks two invariants for every entry *N* (N > 0):

        1. ``entry.entry_hash`` matches a fresh recomputation.
        2. ``entry.prev_hash`` equals ``entries[N-1].entry_hash``.
        """

        if not self._loaded:
            await self.load()
        async with self._lock:
            snapshot = list(self._entries)

        if not snapshot:
            return ChainVerification(valid=True, checked=0, reason="empty store")

        for idx, entry in enumerate(snapshot):
            if not entry.verify():
                return ChainVerification(
                    valid=False,
                    checked=idx,
                    broken_at=idx,
                    reason=f"Entry #{idx} hash mismatch (stored != recomputed)",
                )
            if idx == 0:
                if entry.prev_hash != _GENESIS_PREV_HASH:
                    return ChainVerification(
                        valid=False,
                        checked=1,
                        broken_at=0,
                        reason="Genesis entry prev_hash is not the zero sentinel",
                    )
            else:
                prev = snapshot[idx - 1]
                if entry.prev_hash != prev.entry_hash:
                    return ChainVerification(
                        valid=False,
                        checked=idx + 1,
                        broken_at=idx,
                        reason=(
                            f"Entry #{idx} prev_hash does not link to entry #{idx - 1} entry_hash"
                        ),
                    )
        return ChainVerification(valid=True, checked=len(snapshot))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    async def export_jsonl(self, output: Path, filter_: AuditQuery | None = None) -> Path:
        """Export matching entries to a JSONL file. Returns the output path."""

        entries = await self.query(filter_)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [e.model_dump_json() for e in entries]
        await asyncio.to_thread(
            output.write_text, "\n".join(lines) + ("\n" if lines else ""), "utf-8"
        )
        logger.info("Exported %d audit entries to %s", len(entries), output)
        return output

    async def export_json(self, output: Path, filter_: AuditQuery | None = None) -> Path:
        """Export matching entries to a single JSON array file."""

        entries = await self.query(filter_)
        data = [json.loads(e.model_dump_json()) for e in entries]
        payload = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(output.write_text, payload + "\n", "utf-8")
        logger.info("Exported %d audit entries to %s", len(entries), output)
        return output

    async def export_csv(self, output: Path, filter_: AuditQuery | None = None) -> Path:
        """Export matching entries to a CSV file with a fixed column set."""

        import csv
        import io

        entries = await self.query(filter_)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "sequence",
                "timestamp",
                "level",
                "category",
                "actor",
                "action",
                "target",
                "prev_hash",
                "entry_hash",
            ]
        )
        for e in entries:
            writer.writerow(
                [
                    e.sequence,
                    e.timestamp.isoformat(),
                    e.level.value,
                    e.category.value,
                    e.actor,
                    e.action,
                    e.target,
                    e.prev_hash,
                    e.entry_hash,
                ]
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(output.write_text, buf.getvalue(), "utf-8")
        logger.info("Exported %d audit entries to %s", len(entries), output)
        return output

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of entries currently held in memory."""

        return len(self._entries)

    @property
    def path(self) -> Path | None:
        """The backing JSONL file path, if any."""

        return self._path

    def __iter__(self) -> Iterator[AuditEntry]:
        """Synchronous iteration over the in-memory entries."""

        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_audit_store(path: Path | None = None) -> AuditStore:
    """Create an :class:`AuditStore` and load any existing entries lazily.

    The actual disk read happens on the first :meth:`AuditStore.append` or
    :meth:`AuditStore.query` call, so this factory never performs I/O.
    """

    return AuditStore(path=path)


__all__ = [
    "AuditCategory",
    "AuditEntry",
    "AuditLevel",
    "AuditQuery",
    "AuditStore",
    "ChainVerification",
    "create_audit_store",
]
