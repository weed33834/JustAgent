"""Enterprise knowledge management module for the JustAgent platform.

Provides a local-first knowledge management system with:

* **Document parsing** — PDF, Word, Excel, PPT, Markdown, HTML, and
  plain text with format auto-detection. Binary format parsers use
  lazy imports with graceful fallback.
* **Vector search / semantic retrieval** — an in-memory vector store
  with cosine similarity search. Numpy is used when available for fast
  batch computation; a pure-Python fallback is used otherwise. No hard
  dependency on external vector databases.
* **Knowledge graph** — entity-relation extraction and storage with
  rule-based patterns and optional LLM-assisted extraction.
* **RAG pipeline** — retrieval-augmented generation with source
  citations, integrating with
  :mod:`justagent.adapters.model_gateway` for LLM calls.
* **Document lifecycle management** — versioning, archival, and
  soft-deletion.
* **ETL pipeline** — multi-source data ingestion (filesystem, database,
  HTTP API) with incremental sync.

Modules:

* :mod:`justagent.knowledge.document` — document models, parsing, lifecycle.
* :mod:`justagent.knowledge.vector` — vector store, embeddings, similarity.
* :mod:`justagent.knowledge.graph` — knowledge graph, entity/relation extraction.
* :mod:`justagent.knowledge.rag` — RAG pipeline with citations.
* :mod:`justagent.knowledge.etl` — ETL pipeline for data ingestion.

Quick start::

    from justagent.knowledge import (
        DocumentParser,
        InMemoryVectorStore,
        RAGPipeline,
        HashingEmbedder,
    )

    # Parse a file.
    parser = DocumentParser()
    doc = parser.parse_file("README.md")

    # Set up the RAG pipeline.
    store = InMemoryVectorStore()
    pipeline = RAGPipeline(
        vector_store=store,
        embedder=HashingEmbedder(),
        gateway=my_model_gateway,
    )

    # Ingest and query.
    pipeline.ingest_document(doc)
    answer = pipeline.query("What is this project about?")
    for citation in answer.citations:
        print(citation.format())
"""

from __future__ import annotations

import logging

# Re-export all public symbols.
from justagent.knowledge.document import (
    Chunk,
    Document,
    DocumentLifecycleManager,
    DocumentParser,
    DocumentStatus,
    DocumentType,
    DocumentVersion,
    TextChunker,
    detect_type,
    format_timestamp,
    parse_html,
    parse_markdown,
    utcnow,
)
from justagent.knowledge.etl import (
    APISource,
    DatabaseSource,
    ETLPipeline,
    ETLSource,
    FilesystemSource,
    RawItem,
    SourceType,
    SyncResult,
    SyncState,
)
from justagent.knowledge.graph import (
    Entity,
    EntityType,
    KnowledgeGraph,
    Relation,
    extract_entities,
    extract_relations,
)
from justagent.knowledge.rag import (
    Citation,
    RAGAnswer,
    RAGPipeline,
    create_pipeline,
)
from justagent.knowledge.vector import (
    EmbeddingProvider,
    FileVectorStore,
    HashingEmbedder,
    InMemoryVectorStore,
    NumpyHashingEmbedder,
    SearchResult,
    VectorRecord,
    VectorStore,
    batch_cosine_similarity,
    cosine_similarity,
    create_default_embedder,
    index_document_chunks,
)

logger = logging.getLogger("justagent.knowledge")

__all__ = [
    # document.py
    "Chunk",
    "Document",
    "DocumentLifecycleManager",
    "DocumentParser",
    "DocumentStatus",
    "DocumentType",
    "DocumentVersion",
    "TextChunker",
    "detect_type",
    "format_timestamp",
    "parse_html",
    "parse_markdown",
    "utcnow",
    # vector.py
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
    # graph.py
    "Entity",
    "EntityType",
    "KnowledgeGraph",
    "Relation",
    "extract_entities",
    "extract_relations",
    # rag.py
    "Citation",
    "RAGAnswer",
    "RAGPipeline",
    "create_pipeline",
    # etl.py
    "APISource",
    "DatabaseSource",
    "ETLPipeline",
    "ETLSource",
    "FilesystemSource",
    "RawItem",
    "SourceType",
    "SyncResult",
    "SyncState",
]
