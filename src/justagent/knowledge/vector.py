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

Optional real embedding backends are available when their dependencies
are installed. All external libraries are imported lazily so the module
remains importable (and fully functional via :class:`HashingEmbedder`)
when they are absent:

* :class:`SentenceTransformersEmbedder` — local sentence-transformers
  model (BGE / M3E etc.), well suited to Chinese legal text.
* :class:`OpenAIEmbedder` — OpenAI embeddings API accessed via
  ``litellm`` (supports ``text-embedding-3-small`` and friends).
* :class:`HuggingFaceEmbedder` — Hugging Face Inference API.
* :class:`EmbedderConfig` / :func:`create_embedder` — Pydantic config
  and a factory that auto-selects a backend by priority.
* :class:`ChromaVectorStore` — ChromaDB-backed persistent vector store
  implementing the same :class:`VectorStore` interface.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

from justagent.knowledge.document import Chunk

logger = logging.getLogger("justagent.knowledge")

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
# Real embedding backends (all dependencies imported lazily)
# ---------------------------------------------------------------------------


def _can_import(module_name: str) -> bool:
    """Return ``True`` if *module_name* can be imported without errors."""

    return importlib.util.find_spec(module_name) is not None


def _flatten_embedding(vec: Any) -> list[float]:
    """Flatten a possibly nested embedding into a 1-D list of floats.

    Remote inference APIs occasionally return ``(1, D)`` shaped arrays
    (or numpy arrays) rather than a plain list of floats. This helper
    normalises any of those shapes to a flat ``list[float]``.
    """

    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if isinstance(vec, list):
        if not vec:
            return []
        if isinstance(vec[0], (int, float)):
            return [float(x) for x in vec]
        flat: list[float] = []
        for sub in vec:
            flat.extend(_flatten_embedding(sub))
        return flat
    if isinstance(vec, (int, float)):
        return [float(vec)]
    return []


class EmbedderProvider(str, Enum):  # noqa: UP042 - match existing codebase style
    """Selects which embedding backend :func:`create_embedder` builds.

    Attributes:
        AUTO: Choose automatically by priority — sentence-transformers,
            then OpenAI, then the hashing fallback.
        HASHING: Force the zero-dependency :class:`HashingEmbedder`.
        SENTENCE_TRANSFORMERS: Force a local sentence-transformers model.
        OPENAI: Force the OpenAI embeddings API (via ``litellm``).
        HUGGINGFACE: Force the Hugging Face Inference API.
    """

    AUTO = "auto"
    HASHING = "hashing"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class EmbedderConfig(BaseModel):
    """Pydantic configuration model for :func:`create_embedder`.

    Attributes:
        provider: Which backend to build (:attr:`EmbedderProvider.AUTO`
            selects by priority).
        model_name: Model identifier — a sentence-transformers model id
            (e.g. ``BAAI/bge-small-zh-v1.5``), an OpenAI model name
            (e.g. ``text-embedding-3-small``) or a Hugging Face model id.
        api_key: API key for remote backends. When empty, OpenAI reads
            ``OPENAI_API_KEY`` and Hugging Face reads ``HF_TOKEN`` /
            ``HUGGINGFACEHUB_API_TOKEN`` from the environment.
        api_base: Optional override of the API base URL.
        dimension: For OpenAI v3 models, the truncated output dimension.
            For Hugging Face, an explicit dimension to avoid a probe
            call. Ignored by sentence-transformers (auto-detected).
        device: Device for sentence-transformers (e.g. ``"cpu"``,
            ``"cuda"``). Empty string lets the library choose.
        normalize_embeddings: L2-normalise embeddings (recommended).
        batch_size: Number of texts per API / encode call.
        hashing_dim: Dimension used by the hashing fallback.
        extra: Escape hatch for backend-specific keyword arguments.
    """

    provider: EmbedderProvider = EmbedderProvider.AUTO
    model_name: str = ""
    api_key: str = ""
    api_base: str = ""
    dimension: int | None = None
    device: str = ""
    normalize_embeddings: bool = True
    batch_size: int = 32
    hashing_dim: int = 256
    extra: dict[str, Any] = Field(default_factory=dict)


class SentenceTransformersEmbedder(EmbeddingProvider):
    """Embedder backed by a local sentence-transformers model.

    Loads a model such as ``BAAI/bge-small-zh-v1.5`` or
    ``moka-ai/m3e-base`` (well suited to Chinese legal text) into memory
    and produces semantically meaningful embeddings. The
    ``sentence-transformers`` package is imported lazily so the module
    remains importable when it is not installed.

    Example::

        >>> embedder = SentenceTransformersEmbedder("BAAI/bge-small-zh-v1.5")
        >>> vec = embedder.embed("合同纠纷")
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        *,
        device: str = "",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be a non-empty string")
        self._model_name = model_name
        self._device = device or None
        self._normalize = normalize_embeddings
        self._batch_size = max(1, batch_size)
        self._model: Any = None
        self._dim: int | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> Any:
        """Lazily load and cache the sentence-transformers model."""

        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        logger.info("Loading sentence-transformers model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._ensure_model()
        assert self._dim is not None  # noqa: S101 - invariant after load
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for ``text`` using the local model."""
        model = self._ensure_model()
        vec = model.encode(
            text,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return _flatten_embedding(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts in fixed-size chunks."""
        if not texts:
            return []
        model = self._ensure_model()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vecs = model.encode(
                batch,
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            results.extend(_flatten_embedding(v) for v in vecs)
        return results


class OpenAIEmbedder(EmbeddingProvider):
    """Embedder backed by the OpenAI embeddings API via ``litellm``.

    Supports models such as ``text-embedding-3-small`` and
    ``text-embedding-3-large``. ``litellm`` is imported lazily so the
    module degrades gracefully when it (or network access) is absent.

    The API key may be supplied explicitly or read from the
    ``OPENAI_API_KEY`` environment variable by ``litellm``.
    """

    #: Known output dimensions for common OpenAI embedding models,
    #: used to avoid a probe API call in :attr:`dimension`.
    _KNOWN_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str = "",
        api_base: str = "",
        dimensions: int | None = None,
        batch_size: int = 100,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        self._model = model
        self._api_key = api_key or None
        self._api_base = api_base or None
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)
        self._dim: int | None = None

    # ------------------------------------------------------------------
    # API access
    # ------------------------------------------------------------------

    def _ensure_litellm(self) -> Any:
        """Lazily import and return the ``litellm`` module."""

        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "litellm is not installed. "
                "Install it with: pip install litellm"
            ) from exc
        return litellm

    def _call_api(self, inputs: list[str]) -> list[list[float]]:
        """Call the embeddings API and return embeddings in input order."""

        litellm = self._ensure_litellm()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": inputs,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response = litellm.embedding(**kwargs)
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data", [])

        pairs: list[tuple[int, list[float]]] = []
        for item in data:
            emb = getattr(item, "embedding", None)
            if emb is None and isinstance(item, dict):
                emb = item.get("embedding")
            idx = getattr(item, "index", None)
            if idx is None:
                idx = item.get("index", 0) if isinstance(item, dict) else 0
            pairs.append((int(idx), _flatten_embedding(emb)))
        pairs.sort(key=lambda pair: pair[0])
        return [emb for _, emb in pairs]

    @property
    def dimension(self) -> int:
        if self._dim is not None:
            return self._dim
        # An explicit ``dimensions`` param truncates the OpenAI v3 output.
        if self._dimensions is not None:
            self._dim = self._dimensions
            return self._dim
        known = self._KNOWN_DIMENSIONS.get(self._model)
        if known is not None:
            self._dim = known
            return self._dim
        # Last resort: probe the API with a minimal input.
        self._dim = len(self._call_api(["probe"])[0])
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for ``text`` via the OpenAI API."""
        return self._call_api([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts in fixed-size chunks."""
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._call_api(batch))
        return results


class HuggingFaceEmbedder(EmbeddingProvider):
    """Embedder backed by the Hugging Face Inference API.

    Uses ``huggingface_hub.InferenceClient.feature_extraction`` to
    generate embeddings for any model hosted on the Hub (e.g.
    ``BAAI/bge-large-zh-v1.5``). The ``huggingface-hub`` package is
    imported lazily.

    The API token may be supplied explicitly or read from the
    ``HF_TOKEN`` / ``HUGGINGFACEHUB_API_TOKEN`` environment variable.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-zh-v1.5",
        *,
        api_key: str = "",
        dimension: int | None = None,
        batch_size: int = 32,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        self._model = model
        self._api_key = api_key or None
        self._batch_size = max(1, batch_size)
        self._dim = dimension
        self._client: Any = None

    # ------------------------------------------------------------------
    # Client access
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazily create and cache the Hugging Face inference client."""

        if self._client is not None:
            return self._client
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "huggingface-hub is not installed. "
                "Install it with: pip install huggingface-hub"
            ) from exc
        self._client = InferenceClient(model=self._model, token=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Probe the API to detect the model's embedding dimension.
            self._dim = len(self.embed("dimension probe"))
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for ``text`` via the Hugging Face API."""
        client = self._ensure_client()
        result = client.feature_extraction(text)
        return _flatten_embedding(result)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        The Hugging Face Inference API does not support batched input, so
        each text is embedded individually.
        """
        if not texts:
            return []
        return [self.embed(t) for t in texts]


def create_embedder(
    config: EmbedderConfig | None = None,
    **overrides: Any,
) -> EmbeddingProvider:
    """Create an embedding provider from an :class:`EmbedderConfig`.

    Selection priority when ``provider`` is :attr:`EmbedderProvider.AUTO`:

    1. :class:`SentenceTransformersEmbedder` — when ``sentence-transformers``
       is importable **and** a ``model_name`` is configured.
    2. :class:`OpenAIEmbedder` — when an API key is available (from
       ``config.api_key`` or the ``OPENAI_API_KEY`` environment variable)
       and ``litellm`` is importable.
    3. :class:`HashingEmbedder` — the always-available zero-dependency
       fallback (via :func:`create_default_embedder`).

    Args:
        config: An :class:`EmbedderConfig`. ``None`` uses defaults.
        **overrides: Field overrides merged into *config* before use.

    Returns:
        A ready-to-use :class:`EmbeddingProvider`.

    Raises:
        ImportError: If a forced backend's dependency is not installed.
    """
    cfg = config or EmbedderConfig()
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    provider = cfg.provider

    if provider is EmbedderProvider.HASHING:
        return create_default_embedder(dim=cfg.hashing_dim)

    if provider is EmbedderProvider.SENTENCE_TRANSFORMERS:
        return SentenceTransformersEmbedder(
            cfg.model_name or "BAAI/bge-small-zh-v1.5",
            device=cfg.device,
            normalize_embeddings=cfg.normalize_embeddings,
            batch_size=cfg.batch_size,
        )

    if provider is EmbedderProvider.OPENAI:
        return OpenAIEmbedder(
            cfg.model_name or "text-embedding-3-small",
            api_key=cfg.api_key,
            api_base=cfg.api_base,
            dimensions=cfg.dimension,
            batch_size=cfg.batch_size,
        )

    if provider is EmbedderProvider.HUGGINGFACE:
        return HuggingFaceEmbedder(
            cfg.model_name or "BAAI/bge-small-zh-v1.5",
            api_key=cfg.api_key,
            dimension=cfg.dimension,
            batch_size=cfg.batch_size,
        )

    # ------------------------------------------------------------------
    # AUTO: try backends in priority order.
    # ------------------------------------------------------------------
    if _can_import("sentence_transformers") and cfg.model_name:
        logger.info("Auto-selected sentence-transformers embedder (%s)", cfg.model_name)
        return SentenceTransformersEmbedder(
            cfg.model_name,
            device=cfg.device,
            normalize_embeddings=cfg.normalize_embeddings,
            batch_size=cfg.batch_size,
        )

    api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY", "")
    if api_key and _can_import("litellm"):
        logger.info("Auto-selected OpenAI embedder (%s)", cfg.model_name or "text-embedding-3-small")
        return OpenAIEmbedder(
            cfg.model_name or "text-embedding-3-small",
            api_key=api_key,
            api_base=cfg.api_base,
            dimensions=cfg.dimension,
            batch_size=cfg.batch_size,
        )

    logger.info(
        "No real embedding backend available; falling back to HashingEmbedder"
    )
    return create_default_embedder(dim=cfg.hashing_dim)


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
# ChromaDB-backed vector store
# ---------------------------------------------------------------------------


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB.

    Uses ChromaDB as the storage and similarity-search backend. The
    client may be in-memory, persistent (local disk) or remote (HTTP).
    The ``chromadb`` package is imported lazily so the module remains
    importable without it.

    Each :class:`VectorRecord` is stored with the chunk content as the
    Chroma document and a JSON-serialised :class:`Chunk` plus
    ``document_id`` / ``document_title`` / ``created_at`` in metadata,
    enabling full round-trip reconstruction and ``document_ids`` filtering
    via Chroma's ``where`` clause. The collection uses cosine distance.

    Example::

        >>> store = ChromaVectorStore(persist_directory="./chroma")
        >>> store.add(record)
        >>> results = store.search(query_vec, top_k=5)
    """

    _CHUNK_KEY = "_chunk_json"
    _DOC_ID_KEY = "document_id"
    _DOC_TITLE_KEY = "document_title"
    _CREATED_AT_KEY = "created_at"

    def __init__(
        self,
        *,
        collection_name: str = "justagent",
        persist_directory: Path | str | None = None,
        host: str | None = None,
        port: int | None = None,
        embedding_function: Any = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_directory = (
            str(persist_directory) if persist_directory is not None else None
        )
        self._host = host
        self._port = port
        self._embedding_function = embedding_function
        self._client: Any = None
        self._collection: Any = None

    # ------------------------------------------------------------------
    # Lazy client / collection
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazily create the ChromaDB client and collection."""

        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "chromadb is not installed. "
                "Install it with: pip install chromadb"
            ) from exc

        if self._host is not None:
            self._client = chromadb.HttpClient(
                host=self._host, port=self._port
            )
        elif self._persist_directory is not None:
            self._client = chromadb.PersistentClient(path=self._persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "Initialised ChromaDB collection %r (persist=%s, http=%s)",
            self._collection_name,
            self._persist_directory is not None,
            self._host is not None,
        )
        return self._collection

    def _recreate_collection(self) -> Any:
        """Drop and recreate the collection (used by :meth:`clear`)."""

        assert self._client is not None  # noqa: S101 - invariant after _ensure
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    # ------------------------------------------------------------------
    # Record <-> Chroma mapping
    # ------------------------------------------------------------------

    @classmethod
    def _record_to_metadata(cls, record: VectorRecord) -> dict[str, Any]:
        """Build the Chroma metadata dict for *record*."""

        return {
            cls._DOC_ID_KEY: record.document_id,
            cls._DOC_TITLE_KEY: record.document_title,
            cls._CREATED_AT_KEY: record.created_at,
            cls._CHUNK_KEY: record.chunk.model_dump_json(),
        }

    @classmethod
    def _chroma_to_record(
        cls,
        record_id: str,
        metadata: dict[str, Any] | None,
    ) -> VectorRecord | None:
        """Reconstruct a :class:`VectorRecord` from ChromaDB metadata."""

        if not metadata:
            return None
        chunk_json = metadata.get(cls._CHUNK_KEY)
        if not chunk_json:
            return None
        try:
            chunk = Chunk.model_validate_json(chunk_json)
        except Exception as exc:  # noqa: BLE001 - best-effort decode
            logger.warning("Failed to decode chunk from ChromaDB: %s", exc)
            return None
        return VectorRecord(
            id=record_id,
            chunk=chunk,
            embedding=[],
            document_id=metadata.get(cls._DOC_ID_KEY, ""),
            document_title=metadata.get(cls._DOC_TITLE_KEY, ""),
            created_at=float(metadata.get(cls._CREATED_AT_KEY, 0.0)),
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, record: VectorRecord) -> None:
        """Add or replace a single vector record."""
        collection = self._ensure_client()
        collection.upsert(
            ids=[record.id],
            embeddings=[record.embedding] if record.embedding else None,
            documents=[record.chunk.content],
            metadatas=[self._record_to_metadata(record)],
        )

    def add_batch(self, records: list[VectorRecord]) -> None:
        """Add or replace multiple vector records."""
        if not records:
            return
        collection = self._ensure_client()
        has_embeddings = bool(records[0].embedding)
        collection.upsert(
            ids=[r.id for r in records],
            embeddings=[r.embedding for r in records] if has_embeddings else None,
            documents=[r.chunk.content for r in records],
            metadatas=[self._record_to_metadata(r) for r in records],
        )

    def get(self, record_id: str) -> VectorRecord | None:
        """Return a record by ID, or None if not found."""
        collection = self._ensure_client()
        result = collection.get(
            ids=[record_id],
            include=["metadatas", "documents", "embeddings"],
        )
        ids = result.get("ids", []) if isinstance(result, dict) else []
        if not ids:
            return None
        metas = result.get("metadatas", []) if isinstance(result, dict) else []
        embs = result.get("embeddings", []) if isinstance(result, dict) else []
        record = self._chroma_to_record(
            ids[0], metas[0] if metas else None
        )
        if record is None:
            return None
        if embs:
            record.embedding = list(embs[0])
        return record

    def remove(self, record_id: str) -> bool:
        """Remove a record by ID. Returns True if removed."""
        collection = self._ensure_client()
        existing = collection.get(ids=[record_id])
        existing_ids = (
            existing.get("ids", []) if isinstance(existing, dict) else []
        )
        if not existing_ids:
            return False
        collection.delete(ids=[record_id])
        return True

    def remove_by_document(self, document_id: str) -> int:
        """Remove all records belonging to ``document_id``."""
        collection = self._ensure_client()
        existing = collection.get(
            where={self._DOC_ID_KEY: document_id},
        )
        ids = existing.get("ids", []) if isinstance(existing, dict) else []
        if not ids:
            return 0
        collection.delete(ids=list(ids))
        return len(ids)

    def count(self) -> int:
        """Return the total number of stored records."""
        collection = self._ensure_client()
        return int(collection.count())

    def clear(self) -> None:
        """Remove all records."""
        self._ensure_client()
        self._recreate_collection()

    def list_records(self) -> list[VectorRecord]:
        """Return all stored records (for inspection / export)."""
        collection = self._ensure_client()
        result = collection.get(
            include=["metadatas", "documents", "embeddings"]
        )
        if not isinstance(result, dict):
            return []
        ids = result.get("ids", [])
        metas = result.get("metadatas", [])
        embs = result.get("embeddings", [])
        records: list[VectorRecord] = []
        for idx, rid in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else None
            record = self._chroma_to_record(rid, meta)
            if record is None:
                continue
            if embs and idx < len(embs):
                record.embedding = list(embs[idx])
            records.append(record)
        return records

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
                document IDs (mapped to a Chroma ``where`` clause).
            min_score: Minimum cosine similarity score for inclusion.

        Returns:
            List of :class:`SearchResult` sorted by descending score.
        """
        if top_k <= 0:
            return []
        collection = self._ensure_client()
        where: dict[str, Any] | None = None
        if document_ids is not None:
            where = {self._DOC_ID_KEY: {"$in": list(document_ids)}}

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        if not isinstance(result, dict):
            return []

        ids_batch = result.get("ids", [[]])
        metas_batch = result.get("metadatas", [[]])
        dists_batch = result.get("distances", [[]])
        if not ids_batch:
            return []

        ids = ids_batch[0]
        metas = metas_batch[0] if metas_batch else []
        dists = dists_batch[0] if dists_batch else []

        results: list[SearchResult] = []
        rank = 0
        for idx, rid in enumerate(ids):
            # ChromaDB cosine distance == 1 - cosine_similarity.
            dist = float(dists[idx]) if idx < len(dists) else 1.0
            score = 1.0 - dist
            if score < min_score:
                continue
            meta = metas[idx] if idx < len(metas) else None
            record = self._chroma_to_record(rid, meta)
            if record is None:
                continue
            rank += 1
            results.append(
                SearchResult(
                    chunk=record.chunk,
                    document_id=record.document_id,
                    document_title=record.document_title,
                    score=score,
                    rank=rank,
                )
            )
        return results


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
    "ChromaVectorStore",
    "EmbedderConfig",
    "EmbedderProvider",
    "EmbeddingProvider",
    "FileVectorStore",
    "HashingEmbedder",
    "HuggingFaceEmbedder",
    "InMemoryVectorStore",
    "NumpyHashingEmbedder",
    "OpenAIEmbedder",
    "SearchResult",
    "SentenceTransformersEmbedder",
    "VectorRecord",
    "VectorStore",
    "batch_cosine_similarity",
    "cosine_similarity",
    "create_default_embedder",
    "create_embedder",
    "index_document_chunks",
]
