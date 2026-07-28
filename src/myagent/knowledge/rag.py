"""RAG pipeline — retrieve, augment, generate with source citation.

Implements a retrieval-augmented generation pipeline that combines the
vector store (:mod:`myagent.knowledge.vector`) with the model gateway
(:mod:`myagent.adapters.model_gateway`) for LLM-based answer synthesis.

The pipeline:

1. **Ingest** — a :class:`~myagent.knowledge.document.Document` is
   chunked (by the document parser), embedded, and indexed in the
   vector store.
2. **Retrieve** — given a user question, the query is embedded and the
   top-k most similar chunks are retrieved via cosine similarity.
3. **Augment** — the retrieved chunks are formatted into a context
   prompt with numbered source citations.
4. **Generate** — the LLM (via :class:`ModelGateway`) generates an
   answer grounded in the retrieved context.
5. **Cite** — the answer is returned with a list of
   :class:`Citation` objects referencing the source documents, chunk
   indices, and relevance scores.

Design:

* :class:`Citation` — a single source reference (document name, chunk
  index, score, snippet).
* :class:`RAGAnswer` — the full RAG response (answer text + citations).
* :class:`RAGPipeline` — orchestrates ingest / retrieve / generate.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from myagent.adapters.model_gateway import (
    ChatCompletionRequest,
    ChatMessage,
    ModelGateway,
)
from myagent.knowledge.document import (
    Document,
    DocumentParser,
    DocumentStatus,
    TextChunker,
)
from myagent.knowledge.vector import (
    EmbeddingProvider,
    SearchResult,
    VectorRecord,
    VectorStore,
    create_default_embedder,
    index_document_chunks,
)

logger = logging.getLogger("myagent.knowledge")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A source citation for a RAG answer.

    Attributes:
        document_id: ID of the source document.
        document_title: Human-readable title of the source document.
        chunk_index: Index of the cited chunk within the document.
        chunk_id: Unique ID of the cited chunk.
        score: Relevance (cosine similarity) score in ``[-1, 1]``.
        snippet: A short text snippet from the chunk.
    """

    document_id: str
    document_title: str = ""
    chunk_index: int = 0
    chunk_id: str = ""
    score: float = 0.0
    snippet: str = ""

    def format(self) -> str:
        """Format the citation as a human-readable string."""
        title = self.document_title or self.document_id
        return f"[{title}#{self.chunk_index}] (score: {self.score:.3f})"


class RAGAnswer(BaseModel):
    """A complete RAG response with answer text and source citations.

    Attributes:
        query: The original user question.
        answer: The generated answer text.
        citations: List of :class:`Citation` objects for the sources used.
        retrieval_scores: Raw similarity scores of all retrieved chunks.
        latency_ms: Total pipeline latency in milliseconds.
        metadata: Additional pipeline metadata (e.g. model name, token usage).
    """

    query: str
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_answer(self) -> bool:
        """True if the answer is non-empty."""
        return bool(self.answer.strip())

    @property
    def num_sources(self) -> int:
        """Number of source citations."""
        return len(self.citations)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant answering questions based on the "
    "provided context. Follow these rules:\n"
    "1. Answer the question using ONLY the information in the context.\n"
    "2. If the context does not contain enough information to answer, "
    "say 'I cannot answer this question based on the provided context.'\n"
    "3. When you use information from a source, cite it using the source "
    "number in square brackets, e.g. [1] or [2].\n"
    "4. Be concise and factual.\n"
)

_CONTEXT_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

_SOURCE_TEMPLATE = "[{idx}] {title} (chunk {chunk_index}, score {score:.3f}):\n{content}"


def _build_context(results: list[SearchResult], max_chars: int = 8000) -> str:
    """Build a numbered context string from search results.

    Each result is numbered so the LLM can cite it. The total context
    is truncated to ``max_chars`` to stay within token budgets.
    """
    parts: list[str] = []
    total = 0
    for i, result in enumerate(results, start=1):
        title = result.document_title or result.document_id
        snippet = result.chunk.content.strip()
        # Truncate individual snippets.
        max_snippet = 2000
        if len(snippet) > max_snippet:
            snippet = snippet[:max_snippet] + "..."
        entry = _SOURCE_TEMPLATE.format(
            idx=i,
            title=title,
            chunk_index=result.chunk.index,
            score=result.score,
            content=snippet,
        )
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """Retrieval-augmented generation pipeline.

    Orchestrates document ingestion (chunking + embedding + indexing),
    semantic retrieval, and LLM-based answer generation with source
    citations.

    Args:
        vector_store: The :class:`VectorStore` for chunk storage and
            similarity search.
        embedder: The :class:`EmbeddingProvider` for query and chunk
            embedding. If None, a default :class:`HashingEmbedder` is
            used.
        gateway: The :class:`ModelGateway` for LLM answer generation.
            If None, the pipeline can still retrieve and return context
            snippets without LLM-generated answers (useful for testing).
        parser: Optional :class:`DocumentParser` for file ingestion.
            If None, a default parser is created.
        top_k: Default number of chunks to retrieve per query.
        min_score: Minimum cosine similarity score for inclusion.
        max_context_chars: Maximum total characters of context sent to
            the LLM.
        temperature: LLM sampling temperature.
        max_tokens: Maximum tokens for the LLM response.

    Example::

        >>> from myagent.adapters.model_gateway import ModelGateway
        >>> store = InMemoryVectorStore()
        >>> pipeline = RAGPipeline(
        ...     vector_store=store,
        ...     embedder=HashingEmbedder(),
        ...     gateway=my_gateway,
        ... )
        >>> pipeline.ingest_text("Python is a programming language.", title="intro")
        >>> answer = pipeline.query("What is Python?")
        >>> answer.has_answer
        True
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        embedder: EmbeddingProvider | None = None,
        gateway: ModelGateway | None = None,
        parser: DocumentParser | None = None,
        top_k: int = 5,
        min_score: float = 0.01,
        max_context_chars: int = 8000,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._store = vector_store
        self._embedder = embedder or create_default_embedder()
        self._gateway = gateway
        self._parser = parser or DocumentParser()
        self._top_k = top_k
        self._min_score = min_score
        self._max_context_chars = max_context_chars
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Track document IDs that have been indexed.
        self._indexed_docs: set[str] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    @property
    def embedder(self) -> EmbeddingProvider:
        return self._embedder

    @property
    def gateway(self) -> ModelGateway | None:
        return self._gateway

    @gateway.setter
    def gateway(self, value: ModelGateway | None) -> None:
        self._gateway = value

    @property
    def indexed_document_count(self) -> int:
        """Number of documents currently indexed."""
        return len(self._indexed_docs)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(self, document: Document) -> int:
        """Chunk, embed, and index a document into the vector store.

        If the document has no chunks, it will be chunked using the
        pipeline's parser chunker. Previously indexed chunks for the
        same document are removed first (re-indexing).

        Args:
            document: The :class:`Document` to ingest.

        Returns:
            Number of chunks indexed.
        """
        if not document.content.strip():
            logger.warning("Document %s has no content; skipping", document.id)
            return 0

        # Remove old chunks if re-indexing.
        if document.id in self._indexed_docs:
            removed = self._store.remove_by_document(document.id)
            logger.debug(
                "Removed %d old chunks for document %s", removed, document.id
            )

        # Chunk if needed.
        chunks = document.chunks
        if not chunks:
            chunks = self._parser.chunker.chunk_document(document)
            document.chunks = chunks

        # Embed and index.
        records = index_document_chunks(
            self._store,
            chunks,
            self._embedder,
            document_title=document.title,
        )
        self._indexed_docs.add(document.id)
        logger.info(
            "Ingested document %s (%s): %d chunks indexed",
            document.id,
            document.title,
            len(records),
        )
        return len(records)

    def ingest_file(self, path: str, *, title: str | None = None) -> int:
        """Parse and ingest a file into the vector store.

        Args:
            path: Path to the file to ingest.
            title: Optional title override.

        Returns:
            Number of chunks indexed.
        """
        document = self._parser.parse_file(path, title=title)
        return self.ingest_document(document)

    def ingest_text(
        self,
        text: str,
        *,
        title: str = "Untitled",
        source: str = "",
        doc_id: str | None = None,
    ) -> int:
        """Parse and ingest raw text into the vector store.

        Args:
            text: The text to ingest.
            title: Document title.
            source: Optional source identifier (e.g. URL or path).
            doc_id: Optional explicit document ID.

        Returns:
            Number of chunks indexed.
        """
        document = self._parser.parse_text(text, title=title, source=source)
        if doc_id:
            document.id = doc_id
        return self.ingest_document(document)

    def remove_document(self, document_id: str) -> int:
        """Remove a document and all its chunks from the vector store.

        Returns the number of chunks removed.
        """
        self._indexed_docs.discard(document_id)
        return self._store.remove_by_document(document_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Retrieve the top-k most relevant chunks for ``query``.

        Args:
            query: The search query.
            top_k: Override the default top_k.
            document_ids: Optional filter to search within specific documents.
            min_score: Override the default minimum score.

        Returns:
            List of :class:`SearchResult` sorted by descending score.
        """
        k = top_k or self._top_k
        threshold = min_score if min_score is not None else self._min_score
        query_embedding = self._embedder.embed(query)
        return self._store.search(
            query_embedding,
            top_k=k,
            document_ids=document_ids,
            min_score=threshold,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        min_score: float | None = None,
    ) -> RAGAnswer:
        """Answer a question using retrieval-augmented generation.

        Retrieves relevant chunks, builds a context prompt, and calls
        the LLM to generate an answer with source citations.

        If no LLM gateway is configured, returns the retrieved context
        snippets as the answer (useful for testing retrieval without an
        LLM).

        Args:
            question: The user's question.
            top_k: Override the default top_k.
            document_ids: Optional filter to search within specific docs.
            min_score: Override the default minimum score.

        Returns:
            A :class:`RAGAnswer` with the generated answer and citations.
        """
        start = time.time()

        # 1. Retrieve.
        results = self.retrieve(
            question,
            top_k=top_k,
            document_ids=document_ids,
            min_score=min_score,
        )
        logger.debug(
            "Retrieved %d chunks for query: %s", len(results), question[:80]
        )

        # Build citations.
        citations = [
            Citation(
                document_id=r.document_id,
                document_title=r.document_title,
                chunk_index=r.chunk.index,
                chunk_id=r.chunk.id,
                score=r.score,
                snippet=r.chunk.content[:300],
            )
            for r in results
        ]
        retrieval_scores = [r.score for r in results]

        # 2. Generate.
        answer_text = ""
        metadata: dict[str, Any] = {
            "num_retrieved": len(results),
            "top_k": top_k or self._top_k,
        }

        if not results:
            answer_text = (
                "I cannot answer this question based on the provided context. "
                "No relevant documents were found."
            )
            metadata["no_results"] = True
        elif self._gateway is not None:
            answer_text = self._generate(question, results)
        else:
            # No LLM gateway — return the context snippets as the answer.
            answer_text = self._format_context_only(results)
            metadata["no_llm"] = True

        latency_ms = (time.time() - start) * 1000

        return RAGAnswer(
            query=question,
            answer=answer_text,
            citations=citations,
            retrieval_scores=retrieval_scores,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def query_batch(
        self,
        questions: list[str],
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RAGAnswer]:
        """Answer multiple questions in sequence.

        Returns a list of :class:`RAGAnswer` objects, one per question.
        """
        return [
            self.query(q, top_k=top_k, document_ids=document_ids)
            for q in questions
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate(
        self, question: str, results: list[SearchResult]
    ) -> str:
        """Call the LLM gateway to generate an answer."""
        context = _build_context(results, max_chars=self._max_context_chars)
        user_message = _CONTEXT_TEMPLATE.format(
            context=context, question=question
        )
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_message),
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        try:
            response = self._gateway.chat(request)  # type: ignore[union-attr]
            return response.content.strip()
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return (
                f"I encountered an error while generating the answer: {exc}. "
                f"Here are the retrieved context snippets:\n\n"
                + self._format_context_only(results)
            )

    @staticmethod
    def _format_context_only(results: list[SearchResult]) -> str:
        """Format search results as a context-only answer (no LLM)."""
        if not results:
            return "No relevant documents found."
        parts: list[str] = []
        for i, result in enumerate(results, start=1):
            title = result.document_title or result.document_id
            parts.append(
                f"[{i}] {title} (chunk {result.chunk.index}, "
                f"score {result.score:.3f}):\n{result.chunk.content.strip()}"
            )
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Convenience: create a default pipeline
# ---------------------------------------------------------------------------


def create_pipeline(
    *,
    gateway: ModelGateway | None = None,
    embedder: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    top_k: int = 5,
) -> RAGPipeline:
    """Create a RAG pipeline with sensible defaults.

    Uses an :class:`InMemoryVectorStore` and
    :class:`HashingEmbedder` (or numpy-accelerated variant) by default.
    """
    from myagent.knowledge.vector import InMemoryVectorStore

    store = vector_store or InMemoryVectorStore()
    return RAGPipeline(
        vector_store=store,
        embedder=embedder or create_default_embedder(),
        gateway=gateway,
        top_k=top_k,
    )


__all__ = [
    "Citation",
    "RAGAnswer",
    "RAGPipeline",
    "create_pipeline",
]
