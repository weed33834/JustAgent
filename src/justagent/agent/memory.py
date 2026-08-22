"""Long-term memory management — cross-session context retention for JustAgent.

While :mod:`justagent.agent.session` provides *short-term* memory (the
messages of a single conversation) and :mod:`justagent.agent.compaction`
reclaims context-window tokens within a session, this module provides
*long-term* memory that persists across sessions. Memorable facts,
decisions and user preferences extracted from one conversation can be
recalled in a later one by injecting them into the system prompt.

Design parallels Cline's "memory bank" / OpenCode's persistent context
and MemGPT-style tiered memory:

* :class:`MemoryEntry` — a Pydantic v2 model describing one memory: its
  content, a short summary, origin session, importance score (0–1),
  tags and an optional embedding vector.
* :class:`MemoryStore` — a thread-safe, JSON-file-backed persistent
  store (one file at ``~/.justagent/memories/memories.json``). Supports
  add / get / semantic search / list / forget / decay / consolidate /
  export / import. Semantic search uses cosine similarity over stored
  embeddings when an :class:`~justagent.knowledge.vector.EmbeddingProvider`
  is configured, and falls back to keyword (token-overlap) matching
  otherwise — so the store is fully functional with zero dependencies.
* :class:`MemoryManager` — higher-level orchestration that extracts
  memorable information from a conversation (heuristically by default,
  or via an LLM when an :class:`~justagent.agent.runtime.LLMClient` is
  supplied), builds a context string for prompt injection, and adjusts
  importance based on usage.

Example::

    from justagent.agent.memory import MemoryManager

    manager = MemoryManager()
    manager.extract_memories(messages, session_id="abc123")
    context = manager.build_context("How does the user prefer to run tests?")
    # inject `context` into the system prompt of the next session
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from justagent.exceptions import JustAgentError
from justagent.utils.atomic_write import atomic_write_text

if TYPE_CHECKING:  # pragma: no cover - import-only type hints
    from justagent.agent.runtime import LLMClient, Message
    from justagent.knowledge.vector import EmbeddingProvider

logger = logging.getLogger("justagent.agent.memory")


#: Cached cosine-similarity callable (or ``None`` when unavailable).
_cosine_similarity: Any = None
#: Whether ``_cosine_similarity`` has been resolved yet.
_vector_checked: bool = False


def _resolve_cosine_similarity() -> Any:
    """Lazily resolve :func:`cosine_similarity` from the vector module.

    The knowledge/vector stack (and its transitive imports) is optional.
    Importing it at module load time would make this module fail to import
    whenever that stack or one of its dependencies is unavailable. We
    therefore resolve ``cosine_similarity`` on first use and cache the
    result. Returns ``None`` when unavailable, in which case callers fall
    back to pure keyword matching.
    """

    global _cosine_similarity, _vector_checked
    if not _vector_checked:
        _vector_checked = True
        try:
            from justagent.knowledge.vector import cosine_similarity

            _cosine_similarity = cosine_similarity
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.debug(
                "vector module unavailable; semantic search disabled: %s", exc
            )
            _cosine_similarity = None
    return _cosine_similarity


# ---------------------------------------------------------------------------
# Module-level named defaults (lambdas can't satisfy mypy defaults)
# ---------------------------------------------------------------------------

# Filename inside the store directory that holds every memory.
_MEMORIES_FILENAME = "memories.json"

# Decay constant: seconds per day.
_SECONDS_PER_DAY = 86_400.0


def default_store_dir() -> Path:
    """Return the default memory store directory (``~/.justagent/memories``).

    Returns:
        The :class:`~pathlib.Path` to the default memories directory.
    """

    return Path.home() / ".justagent" / "memories"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MemoryStoreError(JustAgentError):
    """Raised when a memory-store operation fails (corrupt file, bad import)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class MemoryEntry(BaseModel):
    """A single long-term memory.

    Attributes:
        id: Unique memory identifier (16-char hex by default).
        content: The full text of the memory.
        summary: A short human-readable summary (``<=120`` chars). When
            not supplied at creation, :class:`MemoryStore` derives one
            from ``content``.
        timestamp: Unix timestamp (seconds) of when the memory was
            created.
        session_id: The originating session id (empty for ad-hoc
            memories).
        importance_score: Salience in ``[0.0, 1.0]``. Higher values are
            retained longer by :meth:`MemoryStore.decay` and surfaced
            first by :meth:`MemoryStore.search`.
        tags: Free-form labels used for grouping/consolidation and
            keyword search (e.g. ``["preference", "python"]``).
        embedding: Optional embedding vector for semantic search.
            ``None`` when no embedder is configured.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    content: str
    summary: str = ""
    timestamp: float = Field(default_factory=lambda: time.time())
    session_id: str = ""
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None

    @field_validator("importance_score")
    @classmethod
    def _clamp_importance(cls, value: float) -> float:
        """Coerce importance into the valid ``[0, 1]`` range."""

        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore:
    """Persistent, thread-safe store for :class:`MemoryEntry` objects.

    All memories live in a single JSON file at
    ``{store_dir}/memories.json`` (an array of :class:`MemoryEntry`
    dicts). Writes are atomic via :func:`atomic_write_text`; reads are
    cached in memory and lazily loaded on first access. All public
    methods are guarded by a :class:`threading.RLock` and return deep
    copies so callers cannot mutate internal state.

    Semantic search uses cosine similarity over stored embeddings when
    an ``embedder`` is configured and a memory has an embedding;
    otherwise (or for memories lacking an embedding) it falls back to
    token-overlap keyword matching. A hybrid score blends the two so
    stores with a mix of embedded/non-embedded memories still rank
    sensibly.

    Example::

        store = MemoryStore()
        store.add("User prefers pytest over unittest", session_id="s1",
                  importance=0.8, tags=["preference", "testing"])
        hits = store.search("testing framework preference", limit=3)
    """

    #: Maximum length of an auto-generated summary.
    _SUMMARY_MAX_LEN = 120
    #: Regex for tokenising text for keyword search.
    _TOKEN_RE = re.compile(r"\b\w+\b")

    def __init__(
        self,
        store_dir: Path | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize the store.

        Args:
            store_dir: Directory holding ``memories.json``. Defaults to
                :func:`default_store_dir` (``~/.justagent/memories``).
            embedder: Optional embedding provider. When set, embeddings
                are computed on :meth:`add` and used for semantic
                :meth:`search`. When ``None``, only keyword matching is
                used.
        """

        self._store_dir = store_dir or default_store_dir()
        self._embedder = embedder
        self._lock = threading.RLock()
        self._memories: dict[str, MemoryEntry] = {}
        self._loaded = False

    # -- properties -------------------------------------------------------

    @property
    def store_dir(self) -> Path:
        """The on-disk directory backing this store."""

        return self._store_dir

    @property
    def embedder(self) -> EmbeddingProvider | None:
        """The configured embedding provider, if any."""

        return self._embedder

    @property
    def path(self) -> Path:
        """The full path to the JSON memory file."""

        return self._store_dir / _MEMORIES_FILENAME

    # -- loading / saving ------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazily load memories from disk on first access.

        Must be called while holding ``self._lock``.
        """

        if self._loaded:
            return
        self._loaded = True
        path = self.path
        if not path.exists():
            self._memories = {}
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load memories from %s: %s", path, exc)
            self._memories = {}
            return
        if not isinstance(data, list):
            logger.warning("Memory file %s is not a JSON array; ignoring", path)
            self._memories = {}
            return
        loaded: dict[str, MemoryEntry] = {}
        for item in data:
            try:
                entry = MemoryEntry.model_validate(item)
            except Exception as exc:  # noqa: BLE001 - skip corrupt entries
                logger.warning("Skipping invalid memory entry: %s", exc)
                continue
            loaded[entry.id] = entry
        self._memories = loaded
        logger.debug("Loaded %d memories from %s", len(loaded), path)

    def _save_locked(self) -> None:
        """Persist all memories to disk atomically.

        Must be called while holding ``self._lock``.
        """

        data = [entry.model_dump() for entry in self._memories.values()]
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        atomic_write_text(self.path, payload)
        logger.debug("Saved %d memories to %s", len(data), self.path)

    # -- public API ------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        session_id: str = "",
        importance: float = 0.5,
        tags: Sequence[str] | None = None,
        summary: str | None = None,
    ) -> MemoryEntry:
        """Store a new memory and return a copy of it.

        When an embedder is configured, an embedding vector is computed
        for ``content`` (outside the store lock, so slow remote
        embedders do not block other readers). A ``summary`` is derived
        from ``content`` when not provided.

        Args:
            content: The full memory text.
            session_id: Originating session id.
            importance: Importance in ``[0, 1]``; clamped to range.
            tags: Optional labels for grouping/search.
            summary: Optional short summary; auto-derived if ``None``.

        Returns:
            A deep copy of the stored :class:`MemoryEntry`.
        """

        if not content or not content.strip():
            raise MemoryStoreError("memory content must not be empty")

        importance = max(0.0, min(1.0, float(importance)))
        entry = MemoryEntry(
            content=content.strip(),
            summary=(summary or self._auto_summary(content)).strip(),
            session_id=session_id,
            importance_score=importance,
            tags=list(tags) if tags else [],
        )
        # Compute the embedding outside the lock — embedding calls (esp.
        # remote ones) can be slow and should not block concurrent reads.
        if self._embedder is not None:
            try:
                entry.embedding = self._embedder.embed(entry.content)
            except Exception as exc:  # noqa: BLE001 - degrade to keyword search
                logger.warning("Failed to embed memory %s: %s", entry.id, exc)
                entry.embedding = None

        with self._lock:
            self._ensure_loaded()
            self._memories[entry.id] = entry
            self._save_locked()
        logger.info(
            "Added memory %s (importance=%.2f, session=%s, tags=%s)",
            entry.id, importance, session_id or "-", entry.tags,
        )
        return entry.model_copy(deep=True)

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a memory by id.

        Args:
            memory_id: The memory's unique id.

        Returns:
            A deep copy of the memory, or ``None`` if not found.
        """

        with self._lock:
            self._ensure_loaded()
            entry = self._memories.get(memory_id)
            return entry.model_copy(deep=True) if entry is not None else None

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_importance: float = 0.0,
    ) -> list[MemoryEntry]:
        """Search memories for relevance to ``query``.

        Uses a hybrid score: when an embedder is configured and a memory
        has an embedding, cosine similarity contributes ``0.7`` of the
        score and token-overlap ``0.3``; memories without embeddings use
        token-overlap alone. Falls back to pure keyword matching when no
        embedder is configured.

        Args:
            query: The search query text.
            limit: Maximum number of results to return.
            min_importance: Exclude memories whose importance is below
                this threshold.

        Returns:
            Up to ``limit`` memories ranked by descending relevance
            (each a deep copy).
        """

        if limit <= 0 or not query or not query.strip():
            return []

        with self._lock:
            self._ensure_loaded()
            candidates = [
                e for e in self._memories.values()
                if e.importance_score >= min_importance
            ]

        if not candidates:
            return []

        query_tokens = self._tokenize(query)
        query_vec: list[float] | None = None
        if self._embedder is not None and any(e.embedding for e in candidates):
            try:
                query_vec = self._embedder.embed(query)
            except Exception as exc:  # noqa: BLE001 - fall back to keywords
                logger.warning("Query embedding failed; using keywords: %s", exc)
                query_vec = None

        scored: list[tuple[float, MemoryEntry]] = [
            (self._score_memory(query_vec, query_tokens, entry), entry)
            for entry in candidates
        ]
        # Drop memories with no relevance: a score <= 0 means there is no
        # token overlap and no semantic similarity, so returning them would
        # inject irrelevant context (especially via :meth:`build_context`).
        scored = [pair for pair in scored if pair[0] > 0.0]
        # Stable sort by descending score; break ties by recency.
        scored.sort(key=lambda pair: (-pair[0], -pair[1].timestamp))

        return [entry.model_copy(deep=True) for _, entry in scored[:limit]]

    def list_recent(
        self,
        *,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[MemoryEntry]:
        """List recent memories, newest first.

        Args:
            limit: Maximum number of memories to return.
            session_id: When set, restrict to memories from this session.

        Returns:
            Up to ``limit`` memories sorted by descending timestamp
            (each a deep copy).
        """

        if limit <= 0:
            return []
        with self._lock:
            self._ensure_loaded()
            entries = list(self._memories.values())
        if session_id is not None:
            entries = [e for e in entries if e.session_id == session_id]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return [e.model_copy(deep=True) for e in entries[:limit]]

    def forget(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: The memory's unique id.

        Returns:
            ``True`` if a memory was deleted, ``False`` if it was
            absent.
        """

        with self._lock:
            self._ensure_loaded()
            if memory_id not in self._memories:
                return False
            del self._memories[memory_id]
            self._save_locked()
        logger.info("Forgot memory %s", memory_id)
        return True

    def decay(
        self,
        importance_threshold: float,
        older_than_days: float,
    ) -> int:
        """Prune old, low-importance memories.

        Removes every memory that is *both* older than
        ``older_than_days`` *and* has an importance below
        ``importance_threshold``. This implements an Ebbinghaus-style
        forgetting curve gate: only stale, unimportant memories are
        dropped, while recent or salient memories are preserved.

        Args:
            importance_threshold: Memories with importance strictly
                below this value are eligible for decay.
            older_than_days: Only memories older than this many days are
                eligible for decay.

        Returns:
            The number of memories removed.
        """

        cutoff = time.time() - older_than_days * _SECONDS_PER_DAY
        with self._lock:
            self._ensure_loaded()
            doomed = [
                mid for mid, entry in self._memories.items()
                if entry.timestamp < cutoff
                and entry.importance_score < importance_threshold
            ]
            if not doomed:
                return 0
            for mid in doomed:
                del self._memories[mid]
            self._save_locked()
        logger.info(
            "Decayed %d memories older than %.1f days (threshold=%.2f)",
            len(doomed), older_than_days, importance_threshold,
        )
        return len(doomed)

    def consolidate(self, session_id: str) -> int:
        """Consolidate a session's memories into merged summaries.

        Memories belonging to ``session_id`` that share at least one tag
        are treated as "related" (connected components over the tag
        graph) and merged into a single memory whose content joins the
        originals, summary concatenates them, importance takes the max,
        and tags are unioned. The originals are then forgotten.

        Memories with no tags, or groups of a single memory, are left
        untouched. Embeddings for merged memories are recomputed when an
        embedder is configured.

        Args:
            session_id: The session whose memories to consolidate.

        Returns:
            The number of original memories that were merged away.
        """

        with self._lock:
            self._ensure_loaded()
            session_entries = [
                e for e in self._memories.values() if e.session_id == session_id
            ]

        if len(session_entries) < 2:
            return 0

        groups = self._group_related(session_entries)
        consolidated = 0
        for group in groups:
            if len(group) < 2:
                continue
            merged = self._merge_entries(group, session_id)
            with self._lock:
                self._ensure_loaded()
                for entry in group:
                    self._memories.pop(entry.id, None)
                self._memories[merged.id] = merged
                self._save_locked()
            consolidated += len(group)

        if consolidated:
            logger.info(
                "Consolidated %d memories into %d summary/summaries for session %s",
                consolidated, sum(1 for g in groups if len(g) >= 2), session_id,
            )
        return consolidated

    def export(self) -> list[dict[str, Any]]:
        """Export all memories as a list of JSON-serializable dicts.

        Returns:
            A list of :meth:`MemoryEntry.model_dump` dicts (one per
            memory), suitable for :meth:`import_` or external storage.
        """

        with self._lock:
            self._ensure_loaded()
            return [entry.model_dump() for entry in self._memories.values()]

    def import_(self, data: Sequence[dict[str, Any]]) -> int:
        """Import memories from a list of dicts, replacing by id.

        Each dict is validated into a :class:`MemoryEntry`; invalid
        entries are skipped with a warning. Existing memories with the
        same id are overwritten.

        Args:
            data: A sequence of memory dicts (as produced by
                :meth:`export`).

        Returns:
            The number of memories successfully imported.

        Raises:
            MemoryStoreError: If ``data`` is not a list/sequence of
                dicts.
        """

        if not isinstance(data, (list, tuple)):
            raise MemoryStoreError("import data must be a list of memory dicts")

        count = 0
        with self._lock:
            self._ensure_loaded()
            for item in data:
                if not isinstance(item, dict):
                    logger.warning("Skipping non-dict memory entry: %r", item)
                    continue
                try:
                    entry = MemoryEntry.model_validate(item)
                except Exception as exc:  # noqa: BLE001 - skip invalid entries
                    logger.warning("Skipping invalid memory entry: %s", exc)
                    continue
                self._memories[entry.id] = entry
                count += 1
            self._save_locked()
        logger.info("Imported %d memories", count)
        return count

    # -- internal: scoring & helpers -------------------------------------

    def _score_memory(
        self,
        query_vec: list[float] | None,
        query_tokens: list[str],
        entry: MemoryEntry,
    ) -> float:
        """Compute a hybrid relevance score for one memory.

        Blends cosine similarity (``0.7`` weight) with token-overlap
        (``0.3`` weight) when an embedding is available; otherwise uses
        token-overlap alone.
        """

        keyword = self._keyword_score(query_tokens, entry)
        if query_vec is not None and entry.embedding:
            cosine = _resolve_cosine_similarity()
            if cosine is not None:
                vec = cosine(query_vec, entry.embedding)
                blended: float = 0.7 * vec + 0.3 * keyword
                return blended
        return keyword

    def _keyword_score(self, query_tokens: list[str], entry: MemoryEntry) -> float:
        """Token-overlap score between the query and a memory.

        Uses the overlap coefficient
        ``|Q ∩ H| / min(|Q|, |H|)`` over the query tokens and the
        memory's ``content`` + ``summary`` + ``tags``.
        """

        if not query_tokens:
            return 0.0
        haystack = " ".join(
            [entry.content, entry.summary, *entry.tags]
        )
        haystack_tokens = set(self._tokenize(haystack))
        if not haystack_tokens:
            return 0.0
        matched = sum(1 for token in query_tokens if token in haystack_tokens)
        if matched == 0:
            return 0.0
        denom = min(len(query_tokens), len(haystack_tokens))
        return matched / denom if denom else 0.0

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase word tokens of ``text``."""

        return [t.lower() for t in self._TOKEN_RE.findall(text)]

    @classmethod
    def _auto_summary(cls, content: str) -> str:
        """Derive a short summary from ``content``.

        Takes the first sentence (up to :attr:`_SUMMARY_MAX_LEN` chars);
        truncates with an ellipsis otherwise.
        """

        text = content.strip()
        if not text:
            return ""
        boundary = re.search(r"[.!?。！？]\s", text)
        if boundary and boundary.start() + 1 <= cls._SUMMARY_MAX_LEN:
            return text[: boundary.start() + 1].strip()
        if len(text) <= cls._SUMMARY_MAX_LEN:
            return text
        return text[: cls._SUMMARY_MAX_LEN].rstrip() + "…"

    @staticmethod
    def _group_related(
        entries: Sequence[MemoryEntry],
    ) -> list[list[MemoryEntry]]:
        """Group entries that share at least one tag (connected components).

        Memories with no tags form singleton groups. Used by
        :meth:`consolidate` to decide which memories to merge.
        """

        parent: dict[str, str] = {e.id: e.id for e in entries}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: str, b: str) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_a] = root_b

        tag_to_ids: dict[str, list[str]] = {}
        for entry in entries:
            for tag in entry.tags:
                tag_to_ids.setdefault(tag.lower(), []).append(entry.id)
        for ids in tag_to_ids.values():
            for other in ids[1:]:
                union(ids[0], other)

        groups: dict[str, list[MemoryEntry]] = {}
        for entry in entries:
            groups.setdefault(find(entry.id), []).append(entry)
        return list(groups.values())

    def _merge_entries(
        self,
        entries: Sequence[MemoryEntry],
        session_id: str,
    ) -> MemoryEntry:
        """Merge a group of related entries into one :class:`MemoryEntry`."""

        ordered = sorted(entries, key=lambda e: e.timestamp)
        merged_content = "\n---\n".join(e.content for e in ordered)
        merged_summary = "; ".join(e.summary for e in ordered if e.summary)
        merged_tags = sorted({t for e in ordered for t in e.tags})
        merged_importance = max(e.importance_score for e in ordered)
        merged = MemoryEntry(
            content=merged_content,
            summary=merged_summary or self._auto_summary(merged_content),
            session_id=session_id,
            importance_score=merged_importance,
            tags=merged_tags,
        )
        if self._embedder is not None:
            try:
                merged.embedding = self._embedder.embed(
                    merged.summary or merged_content
                )
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning("Failed to embed merged memory %s: %s", merged.id, exc)
                merged.embedding = None
        return merged

    def _modify_entry(
        self,
        memory_id: str,
        modifier: Callable[[MemoryEntry], MemoryEntry],
    ) -> MemoryEntry | None:
        """Apply ``modifier`` to a memory under the lock and persist it.

        Internal helper used by :class:`MemoryManager` to mutate a
        memory atomically (e.g. adjusting importance). ``modifier``
        receives the current entry and must return a new (or updated)
        :class:`MemoryEntry`.

        Args:
            memory_id: The memory to modify.
            modifier: A function ``(entry) -> entry``.

        Returns:
            A deep copy of the modified memory, or ``None`` if the id is
            unknown.
        """

        with self._lock:
            self._ensure_loaded()
            entry = self._memories.get(memory_id)
            if entry is None:
                return None
            updated = modifier(entry)
            self._memories[memory_id] = updated
            self._save_locked()
            return updated.model_copy(deep=True)


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

# Keyword patterns used by the heuristic extractor to classify sentences.
_PREFERENCE_PATTERNS = [
    re.compile(r"\bi (?:prefer|like|use|want|need|always|usually|typically)\b"),
    re.compile(r"\bmy (?:favorite|favourite|preferred|default|usual|normal)\b"),
    re.compile(r"\bremember (?:that|i|to|this)\b"),
    re.compile(r"\bdon'?t (?:want|like|use|need)\b"),
    re.compile(r"\bnever\b"),
    re.compile(r"\bplease (?:always|never|remember)\b"),
]
_DECISION_PATTERNS = [
    re.compile(
        r"\b(?:decided|decision|will|let'?s|we should|going to|plan to|"
        r"chose|choose|adopted|agreed|final answer)\b"
    ),
    re.compile(r"\bshould (?:use|do|go with|adopt|pick)\b"),
]
_FACT_PATTERNS = [
    re.compile(r"\bis (?:a|an|the|defined as|located)\b"),
    re.compile(r"\bdefined as\b"),
    re.compile(r"\blocated (?:at|in)\b"),
    re.compile(r"\bworks by\b"),
    re.compile(r"\bmeans\b"),
]

# System prompt for LLM-assisted extraction.
_EXTRACTION_SYSTEM_PROMPT = (
    "You are a memory-extraction assistant. Read the conversation and extract "
    "memorable information worth persisting across sessions: key decisions, "
    "important facts, user preferences, and notable context. Respond with ONLY "
    "a JSON array (no prose, no markdown fences) where each element is an "
    "object with keys: "
    '"content" (string, the memorable fact/decision), '
    '"summary" (string, <=120 chars), '
    '"importance" (float 0.0-1.0), '
    '"tags" (array of short strings, e.g. ["preference","python"]). '
    "Omit trivial or transient details. Return [] if nothing is memorable."
)


class MemoryManager:
    """Higher-level orchestration over a :class:`MemoryStore`.

    Bridges conversations and long-term memory:

    * :meth:`extract_memories` pulls memorable items out of a
      conversation (heuristically by default; via an LLM when an
      :class:`~justagent.agent.runtime.LLMClient` is configured) and
      persists them.
    * :meth:`build_context` retrieves the memories most relevant to a
      query and renders them as a string for prompt injection.
    * :meth:`update_importance` reinforces or demotes a memory, feeding
      a usage-based recall signal.

    Example::

        manager = MemoryManager(llm_client=client)
        manager.extract_memories(messages, session_id="s1")
        ctx = manager.build_context("the user's testing preference")
    """

    #: Maximum characters of transcript sent to the LLM extractor.
    _TRANSCRIPT_MAX_CHARS = 8000

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        llm_client: LLMClient | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            store: An existing :class:`MemoryStore`. When ``None``, one
                is created via :func:`get_memory_store` (honoring the
                ``JUSTAGENT_MEMORIES_DIR`` env var), configured with
                ``embedder``.
            llm_client: Optional LLM client for higher-quality memory
                extraction. When ``None``, heuristic extraction is used.
            embedder: Optional embedding provider, used to seed a new
                store when ``store`` is ``None``. Ignored when ``store``
                is provided (use the store's own embedder instead).
        """

        if store is not None:
            self._store = store
            self._embedder = embedder or store.embedder
        else:
            self._store = get_memory_store(embedder=embedder)
            self._embedder = embedder or self._store.embedder
        self._llm_client = llm_client

    # -- properties -------------------------------------------------------

    @property
    def store(self) -> MemoryStore:
        """The underlying :class:`MemoryStore`."""

        return self._store

    @property
    def llm_client(self) -> LLMClient | None:
        """The configured LLM client, if any."""

        return self._llm_client

    # -- extraction -------------------------------------------------------

    def extract_memories(
        self,
        messages: Sequence[Message | dict[str, Any]],
        session_id: str,
    ) -> list[MemoryEntry]:
        """Extract memorable information from a conversation.

        When an LLM client is configured, the conversation transcript is
        sent to the LLM which returns structured memories (content,
        summary, importance, tags). Otherwise a heuristic extractor
        scans user/assistant messages for preference, decision and fact
        patterns. Extracted memories are persisted to the store and
        returned.

        Any LLM failure (network error, nested event loop, unparseable
        output) silently falls back to the heuristic extractor so
        extraction never blocks the caller.

        Args:
            messages: The conversation — either runtime
                :class:`~justagent.agent.runtime.Message` objects or
                their serialized dict form (as stored by
                :mod:`justagent.agent.session`).
            session_id: The originating session id.

        Returns:
            The list of newly created :class:`MemoryEntry` objects
            (each a deep copy).
        """

        if not messages:
            return []

        if self._llm_client is not None:
            try:
                return self._llm_extract(messages, session_id)
            except Exception as exc:  # noqa: BLE001 - never block on LLM
                logger.warning(
                    "LLM memory extraction failed; using heuristics: %s", exc
                )
        return self._heuristic_extract(messages, session_id)

    def _llm_extract(
        self,
        messages: Sequence[Message | dict[str, Any]],
        session_id: str,
    ) -> list[MemoryEntry]:
        """Use the LLM to extract structured memories."""

        import asyncio

        from justagent.agent.runtime import LLMRequest, Message

        transcript = self._render_transcript(messages)
        if not transcript.strip():
            return []
        assert self._llm_client is not None  # guarded by caller

        request = LLMRequest(
            messages=[
                Message(role="system", content=_EXTRACTION_SYSTEM_PROMPT),
                Message(role="user", content=transcript),
            ],
            tools=[],
            temperature=0.2,
        )

        try:
            response = asyncio.run(self._llm_client.complete(request))
        except RuntimeError:
            # Already inside a running event loop — can't nest
            # asyncio.run(). Fall back to heuristics.
            logger.debug("Cannot run LLM extraction inside an event loop")
            raise
        except Exception as exc:  # noqa: BLE001 - network/API errors
            logger.warning("LLM extraction call failed: %s", exc)
            raise

        parsed = self._parse_llm_memories(response.content)
        if not parsed:
            logger.debug("LLM returned no parseable memories; using heuristics")
            raise MemoryStoreError("LLM returned no memories")

        extracted: list[MemoryEntry] = []
        for item in parsed:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            entry = self._store.add(
                content,
                session_id=session_id,
                importance=float(item.get("importance", 0.5)),
                tags=item.get("tags") or [],
                summary=(item.get("summary") or "").strip() or None,
            )
            extracted.append(entry)
        logger.info(
            "LLM-extracted %d memories for session %s", len(extracted), session_id
        )
        return extracted

    def _heuristic_extract(
        self,
        messages: Sequence[Message | dict[str, Any]],
        session_id: str,
    ) -> list[MemoryEntry]:
        """Heuristically extract preference/decision/fact memories."""

        extracted: list[MemoryEntry] = []
        seen: set[str] = set()
        for msg in messages:
            role = self._msg_role(msg)
            content = self._msg_content(msg)
            if not content or role not in ("user", "assistant"):
                continue
            for sentence in self._split_sentences(content):
                text = sentence.strip()
                if len(text) < 10:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                category, importance = self._classify(text)
                if category is None:
                    continue
                seen.add(key)
                entry = self._store.add(
                    text,
                    session_id=session_id,
                    importance=importance,
                    tags=[category],
                )
                extracted.append(entry)
        if extracted:
            logger.info(
                "Heuristically extracted %d memories for session %s",
                len(extracted), session_id,
            )
        return extracted

    @staticmethod
    def _parse_llm_memories(text: str) -> list[dict[str, Any]]:
        """Parse a JSON array of memory objects out of an LLM response.

        Tolerates surrounding prose and `````json`` fences by extracting
        the first ``[...]`` block.
        """

        if not text:
            return []
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [
            item for item in data
            if isinstance(item, dict) and item.get("content")
        ]

    @staticmethod
    def _classify(text: str) -> tuple[str | None, float]:
        """Classify a sentence as preference/decision/fact.

        Returns ``(category, importance)`` or ``(None, 0.0)`` when the
        sentence carries nothing memorable.
        """

        lower = text.lower()
        for pattern in _PREFERENCE_PATTERNS:
            if pattern.search(lower):
                return "preference", 0.75
        for pattern in _DECISION_PATTERNS:
            if pattern.search(lower):
                return "decision", 0.8
        for pattern in _FACT_PATTERNS:
            if pattern.search(lower):
                return "fact", 0.45
        return None, 0.0

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split ``text`` into sentences on punctuation or newlines."""

        parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        return [p.strip() for p in parts if p.strip()]

    # -- context building -------------------------------------------------

    def build_context(
        self,
        query: str,
        *,
        max_memories: int = 5,
    ) -> str:
        """Build a context string from relevant memories for prompt injection.

        When ``query`` is non-empty, memories are retrieved via semantic
        :meth:`MemoryStore.search`; when empty, the most recent memories
        are used instead. The result is a ready-to-paste block listing
        each memory's importance and summary.

        Args:
            query: The current user turn or topic to find memories for.
            max_memories: Maximum number of memories to include.

        Returns:
            A newline-delimited context string. Empty when no memories
            are available.
        """

        if max_memories <= 0:
            return ""

        if query and query.strip():
            memories = self._store.search(
                query, limit=max_memories, min_importance=0.1
            )
        else:
            memories = self._store.list_recent(limit=max_memories)

        if not memories:
            return ""

        lines = ["Relevant long-term memories:"]
        for memory in memories:
            label = memory.summary or memory.content
            lines.append(f"- [{memory.importance_score:.2f}] {label}")
        return "\n".join(lines)

    # -- importance reinforcement ----------------------------------------

    def update_importance(
        self,
        memory_id: str,
        delta: float,
    ) -> MemoryEntry | None:
        """Adjust a memory's importance by ``delta`` (clamped to ``[0, 1]``).

        Positive deltas reinforce frequently-recalled memories; negative
        deltas demote stale or incorrect ones. The new score is clamped
        to the valid range before persistence.

        Args:
            memory_id: The memory to adjust.
            delta: The signed change to apply to the importance score.

        Returns:
            A deep copy of the updated memory, or ``None`` if the id is
            unknown.
        """

        def modifier(entry: MemoryEntry) -> MemoryEntry:
            new_score = max(0.0, min(1.0, entry.importance_score + delta))
            return entry.model_copy(update={"importance_score": new_score})

        result = self._store._modify_entry(memory_id, modifier)
        if result is not None:
            logger.info(
                "Adjusted importance of %s by %+.2f -> %.2f",
                memory_id, delta, result.importance_score,
            )
        return result

    # -- message helpers --------------------------------------------------

    @staticmethod
    def _msg_role(msg: Message | dict[str, Any]) -> str:
        """Extract the role from a Message object or dict."""

        role = getattr(msg, "role", None)
        if role is not None:
            return str(role)
        if isinstance(msg, dict):
            return str(msg.get("role", ""))
        return ""

    @staticmethod
    def _msg_content(msg: Message | dict[str, Any]) -> str:
        """Extract the text content from a Message object or dict."""

        content = getattr(msg, "content", None)
        if content is not None:
            return str(content)
        if isinstance(msg, dict):
            return str(msg.get("content", "") or "")
        return ""

    @classmethod
    def _render_transcript(
        cls,
        messages: Sequence[Message | dict[str, Any]],
    ) -> str:
        """Render a conversation as a compact ``[role] text`` transcript.

        Only ``user``/``assistant``/``system`` turns are included, and
        the total length is capped at :attr:`_TRANSCRIPT_MAX_CHARS`.
        """

        lines: list[str] = []
        total = 0
        for msg in messages:
            role = cls._msg_role(msg)
            content = cls._msg_content(msg)
            if not content or role not in ("user", "assistant", "system"):
                continue
            line = f"[{role}] {content}"
            if total + len(line) > cls._TRANSCRIPT_MAX_CHARS:
                remaining = cls._TRANSCRIPT_MAX_CHARS - total
                if remaining > 0:
                    lines.append(line[:remaining] + "…")
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_memory_store(
    store_dir: Path | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
) -> MemoryStore:
    """Build a :class:`MemoryStore`, honoring the ``JUSTAGENT_MEMORIES_DIR`` env var.

    Explicit ``store_dir`` wins; otherwise the ``JUSTAGENT_MEMORIES_DIR``
    (or legacy ``MYAGENT_MEMORIES_DIR``) environment variable is
    consulted; otherwise the default ``~/.justagent/memories`` directory
    is used.

    Args:
        store_dir: Optional explicit store directory.
        embedder: Optional embedding provider passed to the store.

    Returns:
        A ready-to-use :class:`MemoryStore`.
    """

    if store_dir is not None:
        return MemoryStore(store_dir, embedder=embedder)
    env_dir = (
        os.environ.get("JUSTAGENT_MEMORIES_DIR")
        or os.environ.get("MYAGENT_MEMORIES_DIR")
    )
    if env_dir:
        return MemoryStore(Path(env_dir), embedder=embedder)
    return MemoryStore(embedder=embedder)


__all__ = [
    "MemoryEntry",
    "MemoryManager",
    "MemoryStore",
    "MemoryStoreError",
    "default_store_dir",
    "get_memory_store",
]
