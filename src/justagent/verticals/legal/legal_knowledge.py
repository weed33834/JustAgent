"""Legal knowledge base — statutes, precedents, and concept lookup.

Provides a structured legal knowledge base that stores statutory articles
(:class:`LegalArticle`) and precedent cases (:class:`LegalCase`), and
supports semantic retrieval over both via integration with the platform's
vector store (:mod:`justagent.knowledge.vector`) and knowledge graph
(:mod:`justagent.knowledge.graph`).

The module supports multiple legal domains (criminal, civil, administrative,
procedural, commercial, labor) and offers three core capabilities:

1. **Statute search** — keyword and semantic search over legal articles,
   returning ranked results with relevance scores.
2. **Similar-case retrieval** — find precedent cases whose facts or legal
   issues are similar to a query, using vector similarity.
3. **Concept explanation** — look up legal concepts and their definitions,
   leveraging the knowledge graph to surface related concepts and the
   statutes that define them.

Design:

* :class:`LegalDomain` — enum of supported legal domains.
* :class:`LegalArticle` — a statutory article (law name, article number,
  content, effective date, domain, keywords).
* :class:`LegalCase` — a precedent case (case number, cause of action,
  ruling essence, ruling result, applied articles).
* :class:`ArticleSearchResult` / :class:`CaseSearchResult` — ranked
  retrieval results.
* :class:`LegalKnowledgeBase` — the thread-safe, async-capable manager
  that owns the article/case registries and delegates semantic search to
  the injected :class:`VectorStore` and :class:`KnowledgeGraph`.

All registry mutations are protected by a ``threading.RLock``. Async
variants (``asyncio.to_thread``) are provided for I/O-bound and LLM-backed
operations so the knowledge base can be used from async orchestration
workflows without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from justagent.knowledge.graph import (
    EntityType,
    KnowledgeGraph,
)
from justagent.knowledge.vector import (
    EmbeddingProvider,
    VectorStore,
    create_default_embedder,
)
from justagent.utils import now

logger = logging.getLogger("justagent.verticals.legal.legal_knowledge")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LegalKnowledgeError(Exception):
    """Raised for invalid legal-knowledge operations."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LegalDomain(str, Enum):  # noqa: UP042 - match existing codebase style
    """Supported legal domains.

    Attributes:
        CRIMINAL: Criminal law (刑法).
        CIVIL: Civil law (民法).
        ADMINISTRATIVE: Administrative law (行政法).
        PROCEDURAL: Procedural law (程序法 — civil, criminal, administrative
            procedure).
        COMMERCIAL: Commercial law (商法).
        LABOR: Labor / employment law (劳动法).
        CONSTITUTIONAL: Constitutional law (宪法).
        CUSTOM: A user-defined domain not covered by the built-in values.
    """

    CRIMINAL = "criminal"
    CIVIL = "civil"
    ADMINISTRATIVE = "administrative"
    PROCEDURAL = "procedural"
    COMMERCIAL = "commercial"
    LABOR = "labor"
    CONSTITUTIONAL = "constitutional"
    CUSTOM = "custom"


class ArticleStatus(str, Enum):  # noqa: UP042
    """Effectiveness status of a legal article.

    Attributes:
        EFFECTIVE: Currently in force.
        AMENDED: Superseded by a newer version (retained for history).
        REPEALED: No longer in force.
    """

    EFFECTIVE = "effective"
    AMENDED = "amended"
    REPEALED = "repealed"


class CaseLevel(str, Enum):  # noqa: UP042
    """Guiding / precedential level of a case.

    Attributes:
        GUIDING: Supreme Court guiding case (指导性案例).
        TYPICAL: Typical / reference case (典型案例).
        ORDINARY: Ordinary published judgement.
    """

    GUIDING = "guiding"
    TYPICAL = "typical"
    ORDINARY = "ordinary"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class LegalArticle(BaseModel):
    """A statutory article in the legal knowledge base.

    Attributes:
        id: Unique article identifier (auto-generated UUID4 hex).
        law_name: Name of the law (e.g. ``"中华人民共和国民法典"``).
        article_number: Article number or range (e.g. ``"第一百四十三条"``
            or ``"第143条"``).
        chapter: Chapter / section the article belongs to.
        content: Full text of the article.
        domain: The :class:`LegalDomain` this article belongs to.
        effective_date: Date the article came into force (ISO ``YYYY-MM-DD``).
        amended_date: Date of the most recent amendment, if any.
        status: Current :class:`ArticleStatus`.
        keywords: Indexable keywords for keyword search.
        metadata: Arbitrary key-value metadata.
        created_at: Unix timestamp of when the record was created.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    law_name: str
    article_number: str
    chapter: str = ""
    content: str
    domain: LegalDomain = LegalDomain.CIVIL
    effective_date: str = ""
    amended_date: str = ""
    status: ArticleStatus = ArticleStatus.EFFECTIVE
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=now)

    @property
    def is_effective(self) -> bool:
        """True if the article is currently in force."""

        return self.status is ArticleStatus.EFFECTIVE

    @property
    def citation(self) -> str:
        """A human-readable citation string (e.g. ``《民法典》第143条``)."""

        return f"《{self.law_name}》{self.article_number}"

    def searchable_text(self) -> str:
        """Return a concatenated text suitable for embedding / keyword search."""

        parts = [self.law_name, self.article_number, self.content]
        if self.chapter:
            parts.append(self.chapter)
        if self.keywords:
            parts.append(" ".join(self.keywords))
        return "\n".join(parts)


class LegalCase(BaseModel):
    """A precedent case in the legal knowledge base.

    Attributes:
        id: Unique case identifier (auto-generated UUID4 hex).
        case_number: Official case number (e.g. ``"(2023)京01民终123号"``).
        cause_of_action: Cause of action / case type (e.g. ``"合同纠纷"``).
        court: Name of the adjudicating court.
        judgment_date: Date of judgment (ISO ``YYYY-MM-DD``).
        ruling_essence: The court's key legal ruling / holding (裁判要旨).
        ruling_result: The outcome (e.g. ``"驳回上诉，维持原判"``).
        applied_article_ids: IDs of :class:`LegalArticle` records applied.
        applied_articles: Human-readable citations of applied articles.
        domain: The :class:`LegalDomain` this case belongs to.
        level: The :class:`CaseLevel` (guiding / typical / ordinary).
        keywords: Indexable keywords.
        summary: Optional factual summary of the case.
        metadata: Arbitrary key-value metadata.
        created_at: Unix timestamp of when the record was created.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    case_number: str
    cause_of_action: str = ""
    court: str = ""
    judgment_date: str = ""
    ruling_essence: str = ""
    ruling_result: str = ""
    applied_article_ids: list[str] = Field(default_factory=list)
    applied_articles: list[str] = Field(default_factory=list)
    domain: LegalDomain = LegalDomain.CIVIL
    level: CaseLevel = CaseLevel.ORDINARY
    keywords: list[str] = Field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=now)

    @property
    def is_guiding(self) -> bool:
        """True if this is a guiding case (指导性案例)."""

        return self.level is CaseLevel.GUIDING

    def searchable_text(self) -> str:
        """Return a concatenated text suitable for embedding / keyword search."""

        parts = [
            self.case_number,
            self.cause_of_action,
            self.ruling_essence,
            self.ruling_result,
        ]
        if self.summary:
            parts.append(self.summary)
        if self.keywords:
            parts.append(" ".join(self.keywords))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Search result models
# ---------------------------------------------------------------------------


class ArticleSearchResult(BaseModel):
    """A ranked statute-search result.

    Attributes:
        article: The matched :class:`LegalArticle`.
        score: Relevance score in ``[0, 1]``.
        match_type: How the match was found — ``"semantic"`` or
            ``"keyword"``.
    """

    article: LegalArticle
    score: float = 0.0
    match_type: str = "semantic"


class CaseSearchResult(BaseModel):
    """A ranked similar-case search result.

    Attributes:
        case: The matched :class:`LegalCase`.
        score: Relevance score in ``[0, 1]``.
        match_type: How the match was found — ``"semantic"`` or
            ``"keyword"``.
    """

    case: LegalCase
    score: float = 0.0
    match_type: str = "semantic"


class ConceptExplanation(BaseModel):
    """A legal-concept explanation result.

    Attributes:
        concept: The queried legal concept / term.
        definition: The explanation text.
        related_concepts: Names of related legal concepts.
        defining_articles: Citations of articles that define this concept.
        source: Where the explanation came from (``"graph"``,
            ``"articles"``, ``"llm"``).
    """

    concept: str
    definition: str = ""
    related_concepts: list[str] = Field(default_factory=list)
    defining_articles: list[str] = Field(default_factory=list)
    source: str = "articles"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _keyword_score(query_terms: list[str], text: str) -> float:
    """Compute a simple keyword-overlap score in ``[0, 1]``.

    Returns the fraction of *query_terms* that appear (case-insensitive)
    in *text*. An empty query returns 0.
    """

    if not query_terms:
        return 0.0
    lower = text.lower()
    hits = sum(1 for term in query_terms if term.lower() in lower)
    return hits / len(query_terms)


def _tokenize_query(query: str) -> list[str]:
    """Tokenize a search query, using jieba for Chinese text."""

    try:
        import jieba

        tokens = [t.strip() for t in jieba.cut(query) if t.strip()]
        # Also include the full query as a token for exact match
        if query.strip() and query.strip() not in tokens:
            tokens.append(query.strip())
        return tokens
    except ImportError:
        return [t for t in query.split() if t.strip()]


# ---------------------------------------------------------------------------
# Legal knowledge base
# ---------------------------------------------------------------------------


class LegalKnowledgeBase:
    """Thread-safe legal knowledge base with semantic and keyword retrieval.

    Owns in-memory registries of :class:`LegalArticle` and
    :class:`LegalCase` records. When a :class:`VectorStore` and
    :class:`EmbeddingProvider` are injected, articles and cases are also
    embedded and indexed for cosine-similarity search. When a
    :class:`KnowledgeGraph` is injected, legal concepts extracted from
    article content are added as graph entities, enabling concept lookup
    and relation traversal.

    All registry mutations are protected by a re-entrant lock. The class
    also exposes async variants of the search methods (via
    :func:`asyncio.to_thread`) so it can be used from the async
    orchestration layer without blocking the event loop.

    Args:
        vector_store: Optional :class:`VectorStore` for semantic search.
            If None, only keyword search is available.
        embedder: Optional :class:`EmbeddingProvider`. Defaults to
            :func:`create_default_embedder` when a vector store is given.
        knowledge_graph: Optional :class:`KnowledgeGraph` for concept
            extraction and relation lookup.

    Example::

        >>> from justagent.knowledge.vector import InMemoryVectorStore
        >>> from justagent.verticals.legal.legal_knowledge import (
        ...     LegalKnowledgeBase, LegalArticle, LegalDomain,
        ... )
        >>> kb = LegalKnowledgeBase(vector_store=InMemoryVectorStore())
        >>> kb.add_article(LegalArticle(
        ...     law_name="民法典", article_number="第143条",
        ...     content="具备下列条件的民事法律行为有效...",
        ...     domain=LegalDomain.CIVIL,
        ... ))
        >>> results = kb.search_articles("民事法律行为有效条件")
        >>> len(results) >= 1
        True
    """

    #: Metadata namespace prefixes used to distinguish article vs. case
    #: chunks inside the shared vector store.
    _ARTICLE_PREFIX = "legal_article"
    _CASE_PREFIX = "legal_case"

    def __init__(
        self,
        *,
        vector_store: VectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> None:
        self._articles: dict[str, LegalArticle] = {}
        self._cases: dict[str, LegalCase] = {}
        self._article_number_index: dict[str, str] = {}
        self._case_number_index: dict[str, str] = {}
        self._store = vector_store
        self._embedder = embedder
        if vector_store is not None and embedder is None:
            self._embedder = create_default_embedder()
        self._graph = knowledge_graph
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vector_store(self) -> VectorStore | None:
        """The backing vector store, or ``None`` if semantic search is disabled."""

        return self._store

    @property
    def embedder(self) -> EmbeddingProvider | None:
        """The embedding provider, or ``None``."""

        return self._embedder

    @property
    def knowledge_graph(self) -> KnowledgeGraph | None:
        """The backing knowledge graph, or ``None``."""

        return self._graph

    @property
    def article_count(self) -> int:
        """Total number of registered articles."""

        with self._lock:
            return len(self._articles)

    @property
    def case_count(self) -> int:
        """Total number of registered cases."""

        with self._lock:
            return len(self._cases)

    # ------------------------------------------------------------------
    # Article management
    # ------------------------------------------------------------------

    def add_article(self, article: LegalArticle) -> LegalArticle:
        """Register a legal article (or update an existing one by ID).

        The article is embedded and indexed in the vector store (if
        configured) and its concepts are extracted into the knowledge
        graph (if configured).

        Raises:
            LegalKnowledgeError: If an article with the same law name and
                article number already exists (different ID).
        """

        with self._lock:
            key = self._article_key(article.law_name, article.article_number)
            existing_id = self._article_number_index.get(key)
            if existing_id is not None and existing_id != article.id:
                raise LegalKnowledgeError(f"Article already exists: {article.citation}")
            self._articles[article.id] = article
            self._article_number_index[key] = article.id

            # Index in vector store.
            self._index_article(article)

            # Extract concepts into the knowledge graph.
            self._extract_article_concepts(article)
        logger.info("Added legal article %s (%s)", article.id, article.citation)
        return article

    def add_articles(self, articles: list[LegalArticle]) -> int:
        """Register multiple articles. Returns the number added."""

        count = 0
        for article in articles:
            try:
                self.add_article(article)
                count += 1
            except LegalKnowledgeError as exc:
                logger.warning("Skipping duplicate article: %s", exc)
        return count

    def get_article(self, article_id: str) -> LegalArticle | None:
        """Return an article by ID, or ``None``."""

        with self._lock:
            return self._articles.get(article_id)

    def find_article(self, law_name: str, article_number: str) -> LegalArticle | None:
        """Find an article by law name and article number."""

        with self._lock:
            key = self._article_key(law_name, article_number)
            aid = self._article_number_index.get(key)
            return self._articles.get(aid) if aid else None

    def list_articles(
        self,
        *,
        domain: LegalDomain | None = None,
        law_name: str | None = None,
        status: ArticleStatus | None = None,
    ) -> list[LegalArticle]:
        """List articles, optionally filtered by domain, law name, or status."""

        with self._lock:
            result = list(self._articles.values())
        if domain is not None:
            result = [a for a in result if a.domain is domain]
        if law_name is not None:
            result = [a for a in result if a.law_name == law_name]
        if status is not None:
            result = [a for a in result if a.status is status]
        return result

    def remove_article(self, article_id: str) -> LegalArticle | None:
        """Remove an article and its vector record. Returns the removed article."""

        with self._lock:
            article = self._articles.pop(article_id, None)
            if article is None:
                return None
            key = self._article_key(article.law_name, article.article_number)
            self._article_number_index.pop(key, None)
            if self._store is not None:
                self._store.remove(article_id)
        logger.info("Removed legal article %s", article_id)
        return article

    # ------------------------------------------------------------------
    # Case management
    # ------------------------------------------------------------------

    def add_case(self, case: LegalCase) -> LegalCase:
        """Register a precedent case (or update an existing one by ID).

        Raises:
            LegalKnowledgeError: If a case with the same case number
                already exists (different ID).
        """

        with self._lock:
            existing_id = self._case_number_index.get(case.case_number)
            if existing_id is not None and existing_id != case.id:
                raise LegalKnowledgeError(f"Case already exists: {case.case_number}")
            self._cases[case.id] = case
            self._case_number_index[case.case_number] = case.id
            self._index_case(case)
        logger.info("Added legal case %s (%s)", case.id, case.case_number)
        return case

    def add_cases(self, cases: list[LegalCase]) -> int:
        """Register multiple cases. Returns the number added."""

        count = 0
        for case in cases:
            try:
                self.add_case(case)
                count += 1
            except LegalKnowledgeError as exc:
                logger.warning("Skipping duplicate case: %s", exc)
        return count

    def get_case(self, case_id: str) -> LegalCase | None:
        """Return a case by ID, or ``None``."""

        with self._lock:
            return self._cases.get(case_id)

    def find_case(self, case_number: str) -> LegalCase | None:
        """Find a case by its official case number."""

        with self._lock:
            cid = self._case_number_index.get(case_number)
            return self._cases.get(cid) if cid else None

    def list_cases(
        self,
        *,
        domain: LegalDomain | None = None,
        cause_of_action: str | None = None,
        level: CaseLevel | None = None,
    ) -> list[LegalCase]:
        """List cases, optionally filtered by domain, cause, or level."""

        with self._lock:
            result = list(self._cases.values())
        if domain is not None:
            result = [c for c in result if c.domain is domain]
        if cause_of_action is not None:
            result = [c for c in result if c.cause_of_action == cause_of_action]
        if level is not None:
            result = [c for c in result if c.level is level]
        return result

    def remove_case(self, case_id: str) -> LegalCase | None:
        """Remove a case and its vector record. Returns the removed case."""

        with self._lock:
            case = self._cases.pop(case_id, None)
            if case is None:
                return None
            self._case_number_index.pop(case.case_number, None)
            if self._store is not None:
                self._store.remove(case_id)
        logger.info("Removed legal case %s", case_id)
        return case

    # ------------------------------------------------------------------
    # Statute search
    # ------------------------------------------------------------------

    def search_articles(
        self,
        query: str,
        *,
        top_k: int = 10,
        domain: LegalDomain | None = None,
        law_name: str | None = None,
        min_score: float = 0.0,
    ) -> list[ArticleSearchResult]:
        """Search legal articles by semantic similarity and keyword overlap.

        When a vector store is configured, semantic search is performed
        first; keyword overlap is then blended in to boost exact-term
        matches. When no vector store is configured, only keyword search
        is used.

        Args:
            query: The search query (natural language or keywords).
            top_k: Maximum number of results.
            domain: Optional domain filter.
            law_name: Optional law-name filter.
            min_score: Minimum blended score for inclusion.

        Returns:
            List of :class:`ArticleSearchResult` sorted by descending score.
        """

        with self._lock:
            candidates = list(self._articles.values())
        if domain is not None:
            candidates = [a for a in candidates if a.domain is domain]
        if law_name is not None:
            candidates = [a for a in candidates if a.law_name == law_name]
        if not candidates:
            return []

        query_terms = _tokenize_query(query)

        # Semantic search via vector store.
        semantic_scores: dict[str, float] = {}
        if self._store is not None and self._embedder is not None:
            qvec = self._embedder.embed(query)
            results = self._store.search(
                qvec,
                top_k=top_k * 3,
                document_ids=[a.id for a in candidates],
                min_score=0.0,
            )
            for r in results:
                semantic_scores[r.document_id] = r.score

        # Blend semantic + keyword scores.
        scored: list[ArticleSearchResult] = []
        for article in candidates:
            sem = semantic_scores.get(article.id, 0.0)
            kw = _keyword_score(query_terms, article.searchable_text())
            blended = max(sem, kw * 0.8) if sem > 0 else kw
            if blended >= min_score:
                match_type = "semantic" if sem >= kw else "keyword"
                scored.append(
                    ArticleSearchResult(
                        article=article,
                        score=round(blended, 4),
                        match_type=match_type,
                    )
                )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Similar-case search
    # ------------------------------------------------------------------

    def search_cases(
        self,
        query: str,
        *,
        top_k: int = 10,
        domain: LegalDomain | None = None,
        cause_of_action: str | None = None,
        min_score: float = 0.0,
    ) -> list[CaseSearchResult]:
        """Search precedent cases by semantic similarity and keyword overlap.

        Args:
            query: The search query (case facts, legal issues, keywords).
            top_k: Maximum number of results.
            domain: Optional domain filter.
            cause_of_action: Optional cause-of-action filter.
            min_score: Minimum blended score for inclusion.

        Returns:
            List of :class:`CaseSearchResult` sorted by descending score.
        """

        with self._lock:
            candidates = list(self._cases.values())
        if domain is not None:
            candidates = [c for c in candidates if c.domain is domain]
        if cause_of_action is not None:
            candidates = [c for c in candidates if c.cause_of_action == cause_of_action]
        if not candidates:
            return []

        query_terms = _tokenize_query(query)

        semantic_scores: dict[str, float] = {}
        if self._store is not None and self._embedder is not None:
            qvec = self._embedder.embed(query)
            results = self._store.search(
                qvec,
                top_k=top_k * 3,
                document_ids=[c.id for c in candidates],
                min_score=0.0,
            )
            for r in results:
                semantic_scores[r.document_id] = r.score

        scored: list[CaseSearchResult] = []
        for case in candidates:
            sem = semantic_scores.get(case.id, 0.0)
            kw = _keyword_score(query_terms, case.searchable_text())
            blended = max(sem, kw * 0.8) if sem > 0 else kw
            if blended >= min_score:
                match_type = "semantic" if sem >= kw else "keyword"
                scored.append(
                    CaseSearchResult(
                        case=case,
                        score=round(blended, 4),
                        match_type=match_type,
                    )
                )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Concept explanation
    # ------------------------------------------------------------------

    def explain_concept(
        self,
        concept: str,
        *,
        domain: LegalDomain | None = None,
    ) -> ConceptExplanation:
        """Explain a legal concept using the knowledge graph and articles.

        The method first looks up the concept as an entity in the
        knowledge graph (if configured), then searches articles whose
        content mentions the concept, and finally assembles a
        :class:`ConceptExplanation` with the definition, related
        concepts, and defining-article citations.

        Args:
            concept: The legal concept / term to explain.
            domain: Optional domain filter for the article search.

        Returns:
            A :class:`ConceptExplanation`. If nothing is found, the
            ``definition`` will be empty.
        """

        related: list[str] = []
        defining: list[str] = []
        definition = ""
        source = "articles"

        # 1. Look up the concept in the knowledge graph.
        if self._graph is not None:
            entity = self._graph.find_entity(concept)
            if entity is not None:
                # Use entity metadata for a definition if available.
                definition = entity.metadata.get("definition", "")
                if definition:
                    source = "graph"
                # Find related entities via graph neighbours.
                for neighbour in self._graph.neighbors(entity.id):
                    if neighbour.name.lower() != concept.lower():
                        related.append(neighbour.name)

        # 2. Search articles that mention the concept.
        article_results = self.search_articles(concept, top_k=5, domain=domain, min_score=0.0)
        for ar in article_results:
            defining.append(ar.article.citation)
            if not definition and ar.article.content:
                definition = ar.article.content

        return ConceptExplanation(
            concept=concept,
            definition=definition,
            related_concepts=related,
            defining_articles=defining,
            source=source,
        )

    # ------------------------------------------------------------------
    # Async variants
    # ------------------------------------------------------------------

    async def search_articles_async(
        self,
        query: str,
        *,
        top_k: int = 10,
        domain: LegalDomain | None = None,
        law_name: str | None = None,
        min_score: float = 0.0,
    ) -> list[ArticleSearchResult]:
        """Async wrapper for :meth:`search_articles`."""

        return await asyncio.to_thread(
            self.search_articles,
            query,
            top_k=top_k,
            domain=domain,
            law_name=law_name,
            min_score=min_score,
        )

    async def search_cases_async(
        self,
        query: str,
        *,
        top_k: int = 10,
        domain: LegalDomain | None = None,
        cause_of_action: str | None = None,
        min_score: float = 0.0,
    ) -> list[CaseSearchResult]:
        """Async wrapper for :meth:`search_cases`."""

        return await asyncio.to_thread(
            self.search_cases,
            query,
            top_k=top_k,
            domain=domain,
            cause_of_action=cause_of_action,
            min_score=min_score,
        )

    async def explain_concept_async(
        self,
        concept: str,
        *,
        domain: LegalDomain | None = None,
    ) -> ConceptExplanation:
        """Async wrapper for :meth:`explain_concept`."""

        return await asyncio.to_thread(self.explain_concept, concept, domain=domain)

    # ------------------------------------------------------------------
    # Aggregate / summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for dashboards."""

        with self._lock:
            domains: dict[str, int] = {}
            for article in self._articles.values():
                domains[article.domain.value] = domains.get(article.domain.value, 0) + 1
            return {
                "articles": len(self._articles),
                "cases": len(self._cases),
                "domains": domains,
                "has_vector_store": self._store is not None,
                "has_knowledge_graph": self._graph is not None,
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _article_key(law_name: str, article_number: str) -> str:
        """Build a deduplication key from law name + article number."""

        return f"{law_name}::{article_number}"

    def _index_article(self, article: LegalArticle) -> None:
        """Embed and index an article chunk in the vector store."""

        if self._store is None or self._embedder is None:
            return
        from justagent.knowledge.document import Chunk
        from justagent.knowledge.vector import VectorRecord

        text = article.searchable_text()
        embedding = self._embedder.embed(text)
        chunk = Chunk(
            document_id=article.id,
            content=text,
            index=0,
            metadata={
                "type": self._ARTICLE_PREFIX,
                "law_name": article.law_name,
                "article_number": article.article_number,
                "domain": article.domain.value,
            },
        )
        record = VectorRecord(
            id=article.id,
            chunk=chunk,
            embedding=embedding,
            document_id=article.id,
            document_title=article.citation,
        )
        self._store.add(record)

    def _index_case(self, case: LegalCase) -> None:
        """Embed and index a case chunk in the vector store."""

        if self._store is None or self._embedder is None:
            return
        from justagent.knowledge.document import Chunk
        from justagent.knowledge.vector import VectorRecord

        text = case.searchable_text()
        embedding = self._embedder.embed(text)
        chunk = Chunk(
            document_id=case.id,
            content=text,
            index=0,
            metadata={
                "type": self._CASE_PREFIX,
                "case_number": case.case_number,
                "domain": case.domain.value,
            },
        )
        record = VectorRecord(
            id=case.id,
            chunk=chunk,
            embedding=embedding,
            document_id=case.id,
            document_title=case.case_number,
        )
        self._store.add(record)

    def _extract_article_concepts(self, article: LegalArticle) -> None:
        """Extract legal concepts from article content into the graph."""

        if self._graph is None:
            return
        # Use the graph's built-in entity extraction, supplemented with
        # the article's keywords as explicit concept entities.
        self._graph.extract_from_text(
            article.content,
            document_id=article.id,
            include_capitalized=False,
        )
        for kw in article.keywords:
            entity = self._graph.add_entity(
                name=kw,
                entity_type=EntityType.CONCEPT.value,
                source_documents=[article.id],
                metadata={
                    "domain": article.domain.value,
                    "source_article": article.citation,
                },
            )
            # Record a 'defined_by' relation to the law-name entity.
            law_entity = self._graph.add_entity(
                name=article.law_name,
                entity_type=EntityType.CONCEPT.value,
                source_documents=[article.id],
            )
            try:
                self._graph.add_relation(
                    entity.id,
                    law_entity.id,
                    "defined_by",
                    weight=0.9,
                    source_documents=[article.id],
                )
            except KeyError as exc:
                logger.debug("Entity not found, skipping definition relation: %s", exc)


__all__ = [
    "ArticleSearchResult",
    "ArticleStatus",
    "CaseLevel",
    "CaseSearchResult",
    "ConceptExplanation",
    "LegalArticle",
    "LegalCase",
    "LegalDomain",
    "LegalKnowledgeBase",
    "LegalKnowledgeError",
]
