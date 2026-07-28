"""Vector store abstraction with embedding generation.

Provides a dependency-free vector store for semantic retrieval. The
default embedder (:class:`HashingEmbedder`) uses feature hashing — it is
deterministic and zero-dependency but not semantically meaningful. When
numpy is available, :class:`NumpyVectorStore` uses it for fast batch
cosine similarity; otherwise the pure-Python implementation is used.

Design:

* :class:`SearchResult` — a ranked search hit containing the chunk,
  its parent document metadata, and a relevance score.
* :class:`EmbeddingProvider` — abstract interface for text-to-vector
  conversion. Subclasses implement :meth:`embed` and :meth:`embed_batch`.
* :class:`HashingEmbedder` — pure-Python bag-of-tokens embedder using
  feature hashing. Always available; serves as the default fallback.
* :class:`VectorStore` — abstract interface for add / search / remove.
* :class:`InMemoryVectorStore` — in-memory store with cosine similarity.
  Uses numpy when available for fast distance computation, with a
  pure-Python fallback when numpy is absent.
* :class:`FileVectorStore` — persists the in-memory store to disk as
  JSON for local-first deployments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

from myagent.knowledge.document import Chunk

logger = logging.getLogger("myagent.knowledge")

# ---------------------------------------------------------------------------
# Detect numpy availability once at import time.
# ---------------------------------------------------------------------------

try:
    import numpy as np  # type: ignore[import-untyped]

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False
    logger.debug("numpy not available; using pure-Python vector operations")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class VectorRecord(BaseModel):
    """An indexed vector with its associated chunk metadata.

    Attributes:
        id: Unique record ID (same as the chunk ID by default).
        chunk: The :class:`Chunk` this vector was generated from.
        embedding: The embedding vector (list of floats).
        document_id: ID of the parent document.
        document_title: Title of the parent document (for citation).
        created_at: Unix timestamp of when the record was indexed.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    chunk: Chunk
    embedding: list[float] = Field(default_factory=list)
    document_id: str = ""
    document_title: str = ""
    created_at: float = Field(default_factory=lambda: __import__("time").time())

    def model_post_init(self, __context: Any) -> None:
        """Derive ``document_id`` from the chunk if not explicitly set."""
        if not self.document_id:
            self.document_id = self.chunk.document_id
        if not self.document_title:
            self.document_title = self.chunk.metadata.get("title", "")


class SearchResult(BaseModel):
    """A ranked search result.

    Attributes:
        chunk: The matching :class:`Chunk`.
        document_id: ID of the parent document.
        document_title: Title of the parent document.
        score: Cosine similarity score in ``[-1, 1]``. Higher is better.
        rank: 1-based rank in the result list.
    """

    chunk: Chunk
    document_id: str
    document_title: str = ""
    score: float = 0.0
    rank: int = 0


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract interface for text-to-vector embedding generation.

    Concrete implementations may use hashing (dependency-free), a local
    model (e.g. sentence-transformers), or a remote API.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of vectors produced by this provider."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for ``text``."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Default implementation calls :meth:`embed` for each text.
        Subclasses may override for efficiency.
        """
        return [self.embed(t) for t in texts]


class HashingEmbedder(EmbeddingProvider):
    """Pure-Python bag-of-tokens embedder using feature hashing.

    This embedder tokenises text into lowercase word n-grams (1 and 2
    grams), hashes each token into a fixed-dimensional vector, and
    L2-normalises the result. It is **deterministic** and
    **zero-dependency**, making it suitable as a fallback when no real
    embedding model is available.

    Note:
        Hashing embeddings are not semantically meaningful (they capture
        lexical overlap, not semantic similarity). For production RAG,
        substitute a proper embedding model (e.g. sentence-transformers
        or an API-based embedder).
    """

    _TOKEN_PATTERN = re.compile(r"\b\w+\b")
    _BIGRAM_SEP = "\x00"

    def __init__(
        self,
        dim: int = 256,
        use_bigrams: bool = True,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._use_bigrams = use_bigrams

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Generate a normalised hashing embedding for ``text``."""
        tokens = self._tokenize(text)
        vec = [0.0] * self._dim
        for token in tokens:
            idx = self._hash(token) % self._dim
            vec[idx] += 1.0
        if self._use_bigrams and len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                bigram = tokens[i] + self._BIGRAM_SEP + tokens[i + 1]
                idx = self._hash(bigram) % self._dim
                vec[idx] += 0.5
        # L2 normalise.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        return [self.embed(t) for t in texts]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Extract lowercase word tokens from ``text``."""
        return [t.lower() for t in self._TOKEN_PATTERN.findall(text)]

    @staticmethod
    def _hash(token: str) -> int:
        """Deterministic hash of a token using MD5 (stable across runs)."""
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


class NumpyHashingEmbedder(HashingEmbedder):
    """Numpy-accelerated version of :class:`HashingEmbedder`.

    Uses numpy arrays internally for faster batch embedding. Falls back
    to the pure-Python :meth:`embed` when numpy is not available.
    """

    def embed(self, text: str) -> list[float]:
        if not _HAS_NUMPY:
            return super().embed(text)
        tokens = self._tokenize(text)
        vec = np.zeros(self._dim, dtype=np.float64)
        for token in tokens:
            idx = self._hash(token) % self._dim
            vec[idx] += 1.0
        if self._use_bigrams and len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                bigram = tokens[i] + self._BIGRAM_SEP + tokens[i + 1]
                idx = self._hash(bigram) % self._dim
                vec[idx] += 0.5
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not _HAS_NUMPY:
            return super().embed_batch(texts)
        if not texts:
            return []
        # Build the full token list, then construct the matrix.
        all_tokens: list[list[str]] = [self._tokenize(t) for t in texts]
        results: list[list[float]] = []
        for tokens in all_tokens:
            vec = np.zeros(self._dim, dtype=np.float64)
            for token in tokens:
                idx = self._hash(token) % self._dim
                vec[idx] += 1.0
            if self._use_bigrams and len(tokens) >= 2:
                for i in range(len(tokens) - 1):
                    bigram = tokens[i] + self._BIGRAM_SEP + tokens[i + 1]
                    idx = self._hash(bigram) % self._dim
                    vec[idx] += 0.5
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            results.append(vec.tolist())
        return results


def create_default_embedder(dim: int = 256) -> EmbeddingProvider:
    """Create the default embedding provider.

    Uses :class:`NumpyHashingEmbedder` when numpy is available,
    otherwise falls back to :class:`HashingEmbedder`.
    """
    if _HAS_NUMPY:
        return NumpyHashingEmbedder(dim=dim)
    return HashingEmbedder(dim=dim)


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses numpy when available for performance; falls back to pure Python.
    Returns 0.0 if either vector has zero norm.
    """
    if _HAS_NUMPY:
        arr_a = np.asarray(a, dtype=np.float64)
        arr_b = np.asarray(b, dtype=np.float64)
        norm_a = float(np.linalg.norm(arr_a))
        norm_b = float(np.linalg.norm(arr_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))
    # Pure-Python fallback.
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def batch_cosine_similarity(
    query: Sequence[float], vectors: Sequence[Sequence[float]]
) -> list[float]:
    """Compute cosine similarity between ``query`` and each vector.

    Uses numpy for efficient batch computation when available.
    """
    if not vectors:
        return []
    if _HAS_NUMPY:
        q = np.asarray(query, dtype=np.float64)
        mat = np.asarray(vectors, dtype=np.float64)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        q_norm = np.linalg.norm(q)
        mat_norms = np.linalg.norm(mat, axis=1)
        # Avoid division by zero.
        denom = q_norm * mat_norms
        denom[denom == 0.0] = 1.0
        scores = (mat @ q) / denom
        # Zero out rows where either norm is zero.
        zero_mask = (q_norm == 0.0) | (mat_norms == 0.0)
        scores[zero_mask] = 0.0
        return scores.tolist()
    # Pure-Python fallback.
    return [cosine_similarity(query, v) for v in vectors]


# ---------------------------------------------------------------------------
# Vector store interface
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Abstract interface for a vector store.

    A vector store maps chunks to their embedding vectors and supports
    cosine-similarity search. Records can be filtered by document ID.
    """

    @abstractmethod
    def add(self, record: VectorRecord) -> None:
        """Add or replace a single vector record."""

    @abstractmethod
    def add_batch(self, records: list[VectorRecord]) -> None:
        """Add or replace multiple vector records."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        document_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for the top-k most similar records.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results to return.
            document_ids: Optional filter — only search within these
                document IDs. If None, all documents are searched.
            min_score: Minimum cosine similarity score for inclusion.

        Returns:
            List of :class:`SearchResult` sorted by descending score.
        """

    @abstractmethod
    def get(self, record_id: str) -> VectorRecord | None:
        """Return a record by ID, or None if not found."""

    @abstractmethod
    def remove(self, record_id: str) -> bool:
        """Remove a record by ID. Returns True if removed."""

    @abstractmethod
    def remove_by_document(self, document_id: str) -> int:
        """Remove all records belonging to ``document_id``.

        Returns the number of records removed.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored records."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all records."""

    @abstractmethod
    def list_records(self) -> list[VectorRecord]:
        """Return all stored records (for inspection / export)."""

    # ------------------------------------------------------------------
    # Convenience helpers (shared logic)
    # ------------------------------------------------------------------

    def _rank_results(
        self,
        scores: list[float],
        records: list[VectorRecord],
        top_k: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Rank records by score and return :class:`SearchResult` list."""
        paired = list(zip(scores, records))
        paired.sort(key=lambda pair: pair[0], reverse=True)
        results: list[SearchResult] = []
        for rank, (score, record) in enumerate(paired, start=1):
            if score < min_score:
                break
            results.append(
                SearchResult(
                    chunk=record.chunk,
                    document_id=record.document_id,
                    document_title=record.document_title,
                    score=score,
                    rank=rank,
                )
            )
            if len(results) >= top_k:
                break
        return results


# ---------------------------------------------------------------------------
# In-memory vector store
# ---------------------------------------------------------------------------


class InMemoryVectorStore(VectorStore):
    """In-memory vector store with cosine similarity search.

    Records are stored in a dict keyed by record ID. When numpy is
    available, similarity search uses vectorised matrix multiplication;
    otherwise a pure-Python loop is used.

    Example::

        >>> store = InMemoryVectorStore()
        >>> embedder = HashingEmbedder(dim=64)
        >>> chunk = Chunk(document_id="d1", content="hello world")
        >>> record = VectorRecord(chunk=chunk, embedding=embedder.embed("hello world"))
        >>> store.add(record)
        >>> results = store.search(embedder.embed("hello"), top_k=1)
        >>> len(results)
        1
    """

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}
        # Cache document_id -> set of record_ids for fast removal.
        self._doc_index: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Add / remove
    # ------------------------------------------------------------------

    def add(self, record: VectorRecord) -> None:
        """Add or replace a single vector record."""
        existing = self._records.get(record.id)
        if existing is not None and existing.document_id != record.document_id:
            # Document changed; clean up old index.
            self._doc_index.setdefault(existing.document_id, set()).discard(record.id)
        self._records[record.id] = record
        self._doc_index.setdefault(record.document_id, set()).add(record.id)

    def add_batch(self, records: list[VectorRecord]) -> None:
        """Add or replace multiple vector records."""
        for record in records:
            self.add(record)

    def get(self, record_id: str) -> VectorRecord | None:
        """Return a record by ID, or None if not found."""
        return self._records.get(record_id)

    def remove(self, record_id: str) -> bool:
        """Remove a record by ID. Returns True if removed."""
        record = self._records.pop(record_id, None)
        if record is None:
            return False
        doc_set = self._doc_index.get(record.document_id)
        if doc_set is not None:
            doc_set.discard(record_id)
            if not doc_set:
                del self._doc_index[record.document_id]
        return True

    def remove_by_document(self, document_id: str) -> int:
        """Remove all records belonging to ``document_id``."""
        record_ids = self._doc_index.pop(document_id, set())
        for rid in record_ids:
            self._records.pop(rid, None)
        if record_ids:
            logger.debug(
                "Removed %d records for document %s", len(record_ids), document_id
            )
        return len(record_ids)

    def count(self) -> int:
        """Return the total number of stored records."""
        return len(self._records)

    def clear(self) -> None:
        """Remove all records."""
        self._records.clear()
        self._doc_index.clear()

    def list_records(self) -> list[VectorRecord]:
        """Return all stored records."""
        return list(self._records.values())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        document_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for the top-k most similar records.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results to return.
            document_ids: Optional filter — only search within these
                document IDs. If None, all documents are searched.
            min_score: Minimum cosine similarity score for inclusion.

        Returns:
            List of :class:`SearchResult` sorted by descending score.
        """
        if not self._records or top_k <= 0:
            return []

        # Filter records by document_ids if specified.
        if document_ids is not None:
            doc_set = set(document_ids)
            candidates = [
                rec for rec in self._records.values() if rec.document_id in doc_set
            ]
        else:
            candidates = list(self._records.values())

        if not candidates:
            return []

        embeddings = [rec.embedding for rec in candidates]
        scores = batch_cosine_similarity(query_embedding, embeddings)
        return self._rank_results(scores, candidates, top_k, min_score)


# ---------------------------------------------------------------------------
# File-based vector store
# ---------------------------------------------------------------------------


class FileVectorStore(InMemoryVectorStore):
    """File-persisted vector store backed by JSON.

    Extends :class:`InMemoryVectorStore` with :meth:`save` and
    :meth:`load` for local-first persistence. The file format is a JSON
    array of :class:`VectorRecord` objects.

    Example::

        >>> store = FileVectorStore(Path("vectors.json"))
        >>> store.add(record)
        >>> store.save()
        >>> store2 = FileVectorStore(Path("vectors.json"))
        >>> store2.load()
        >>> store2.count()
        1
    """

    def __init__(self, path: Path | str) -> None:
        super().__init__()
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def save(self) -> None:
        """Persist all records to the JSON file atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [rec.model_dump() for rec in self._records.values()]
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        logger.debug("Saved %d vector records to %s", len(data), self._path)

    def load(self) -> int:
        """Load records from the JSON file.

        Returns the number of records loaded. If the file does not exist
        or is corrupted, the store remains empty and 0 is returned.
        """
        if not self._path.exists():
            return 0
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load vector store from %s: %s", self._path, exc)
            return 0
        if not isinstance(data, list):
            logger.warning("Vector store file %s is not a JSON array", self._path)
            return 0
        self.clear()
        count = 0
        for item in data:
            try:
                record = VectorRecord.model_validate(item)
                self.add(record)
                count += 1
            except Exception as exc:
                logger.warning("Skipping invalid vector record: %s", exc)
        logger.debug("Loaded %d vector records from %s", count, self._path)
        return count


# ---------------------------------------------------------------------------
# Convenience: index a document's chunks
# ---------------------------------------------------------------------------


def index_document_chunks(
    store: VectorStore,
    chunks: list[Chunk],
    embedder: EmbeddingProvider,
    *,
    document_title: str = "",
) -> list[VectorRecord]:
    """Embed and index a list of chunks into the vector store.

    Args:
        store: The target :class:`VectorStore`.
        chunks: List of :class:`Chunk` objects to index.
        embedder: The :class:`EmbeddingProvider` to use.
        document_title: Optional document title for citation.

    Returns:
        List of :class:`VectorRecord` objects that were added.
    """
    if not chunks:
        return []
    texts = [c.content for c in chunks]
    embeddings = embedder.embed_batch(texts)
    records: list[VectorRecord] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        record = VectorRecord(
            id=chunk.id,
            chunk=chunk,
            embedding=embedding,
            document_id=chunk.document_id,
            document_title=document_title or chunk.metadata.get("title", ""),
        )
        records.append(record)
    store.add_batch(records)
    logger.debug(
        "Indexed %d chunks for document %s", len(records), chunks[0].document_id
    )
    return records


__all__ = [
    "EmbeddingProvider",
    "FileVectorStore",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "NumpyHashingEmbedder",
    "SearchResult",
    "VectorRecord",
    "VectorStore",
    "batch_cosine_similarity",
    "cosine_similarity",
    "create_default_embedder",
    "index_document_chunks",
]
