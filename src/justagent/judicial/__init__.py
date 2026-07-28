"""Judicial business module for the JustAgent platform.

This package provides four integrated subsystems that together deliver
the core judicial AI capabilities of the platform:

* **Case Management** (:mod:`justagent.judicial.case_manager`) —
  structured case files with parties, facts, evidence references,
  disputed issues, claims, and a chronological timeline. Multi-format
  material import (PDF, Word, Excel, PPT, Markdown, HTML, plain text)
  via :class:`~justagent.knowledge.document.DocumentParser`, with
  rule-based structured extraction (parties, claims, facts, timeline)
  and context assembly for downstream document generation.

* **Evidence Review** (:mod:`justagent.judicial.evidence`) —
  evidence modelling across the seven statutory categories (documentary,
  physical, testimony, expert opinion, inspection record, audio-visual,
  electronic data), with legality (admissibility) checks, relevance
  assessment, probative-value rating, evidence-chain completeness
  analysis, contradiction detection, and gap identification. Integrates
  with :class:`~justagent.knowledge.graph.KnowledgeGraph` for graph-based
  evidence-chain representation.

* **Legal Document Generation**
  (:mod:`justagent.judicial.document_generator`) — generates formal
  legal documents (indictments, defense statements, judgments, rulings,
  mediation agreements, agency opinions, legal opinions, evidence
  lists, cross-examination opinions) from case context, evidence
  analysis, and RAG-retrieved legal context. Includes a template
  manager with built-in defaults and a citation-verification mechanism
  that validates statute references against the legal knowledge base.
  Integrates with :class:`~justagent.knowledge.rag.RAGPipeline` and
  uses LangChain prompt templates (lazy import with fallback).

* **Legal Knowledge Base** (:mod:`justagent.judicial.legal_knowledge`)
  — stores statutory articles and precedent cases, supporting semantic
  and keyword search, similar-case retrieval, and legal-concept
  explanation. Integrates with
  :class:`~justagent.knowledge.vector.VectorStore` and
  :class:`~justagent.knowledge.graph.KnowledgeGraph` for retrieval and
  concept-graph construction. Supports criminal, civil, administrative,
  procedural, commercial, labor, and constitutional law domains.

All subsystems use Pydantic v2 for data models, are thread-safe
(``threading.RLock``), provide async variants (``asyncio.to_thread``)
for integration with the async orchestration layer, and follow the
``justagent.judicial.<submodule>`` logging namespace. External
frameworks (LangChain prompt templates) are used via lazy imports with
graceful fallback, avoiding hard dependencies.

Architecture overview::

    +-------------------+     +-------------------+
    |   Case Manager    |     |  Evidence Review  |
    |  (CaseFile,       |     |  (Evidence,       |
    |   CaseManager)    |     |   EvidenceChain,  |
    |                   |     |   EvidenceReviewer)|
    +-------------------+     +-------------------+
             |                         |
             v                         v
    +-------------------+     +-------------------+
    | Document Generator|     | Legal Knowledge   |
    | (LegalDocument    |     | Base              |
    |  Generator,       |     | (LegalKnowledgeBase|
    |  TemplateManager) |     |  LegalArticle,    |
    |                   |     |  LegalCase)       |
    +-------------------+     +-------------------+
             |                         |
             v                         v
    +-------------------------------------------+
    |          Platform Knowledge Layer          |
    |  (DocumentParser, VectorStore, RAG,       |
    |   KnowledgeGraph, ModelGateway)           |
    +-------------------------------------------+

Quick start::

    from justagent.judicial import (
        # Case management
        CaseManager, CaseFile, Party, PartyRole, CaseStatus,
        Claim, FactElement, DisputedIssue, TimelineEvent,
        # Evidence review
        Evidence, EvidenceType, EvidenceChain, EvidenceReviewer,
        Admissibility, ProbativeStrength,
        # Document generation
        LegalDocumentGenerator, LegalDocumentType,
        LegalDocumentTemplateManager,
        # Legal knowledge
        LegalKnowledgeBase, LegalArticle, LegalCase, LegalDomain,
    )

    # --- Case management ---
    mgr = CaseManager()
    case = mgr.create_case(
        case_number="(2024)京01民初1号",
        cause_of_action="买卖合同纠纷",
    )
    mgr.add_party(case.id, Party(name="甲公司", role=PartyRole.PLAINTIFF))

    # --- Evidence review ---
    chain = EvidenceChain()
    reviewer = EvidenceReviewer(chain)
    chain.add_evidence(Evidence(
        name="购销合同", type=EvidenceType.DOCUMENTARY,
        source="当事人提供", collector="张律师",
        collection_method="当事人提供",
        proving_object="合同关系成立",
        case_id=case.id,
    ))
    result = reviewer.review(chain.list_evidence()[0].id)
    chain_analysis = chain.analyze(case.id)

    # --- Legal knowledge ---
    kb = LegalKnowledgeBase()
    kb.add_article(LegalArticle(
        law_name="民法典", article_number="第143条",
        content="具备下列条件的民事法律行为有效...",
        domain=LegalDomain.CIVIL,
    ))
    articles = kb.search_articles("民事法律行为有效条件")

    # --- Document generation ---
    gen = LegalDocumentGenerator(
        case_manager=mgr,
        evidence_chain=chain,
        knowledge_base=kb,
    )
    doc = gen.generate(case.id, LegalDocumentType.INDICTMENT)
    print(doc.content)
    for v in doc.citation_verifications:
        print(f"  {v.citation}: valid={v.is_valid}")
"""

from __future__ import annotations

import logging

from justagent.judicial.case_manager import (
    CaseContext,
    CaseFile,
    CaseManager,
    CaseManagerError,
    CaseMaterial,
    CaseStatus,
    Claim,
    DisputedIssue,
    FactElement,
    MaterialType,
    Party,
    PartyRole,
    TimelineEvent,
)
from justagent.judicial.document_generator import (
    CitationVerification,
    DocumentGenerationError,
    GeneratedDocument,
    GeneratedDocumentSection,
    LegalDocumentGenerator,
    LegalDocumentSection,
    LegalDocumentTemplate,
    LegalDocumentTemplateManager,
    LegalDocumentType,
)
from justagent.judicial.evidence import (
    Admissibility,
    ChainAnalysisResult,
    Evidence,
    EvidenceChain,
    EvidenceError,
    EvidenceRelation,
    EvidenceRelationType,
    EvidenceReviewer,
    EvidenceType,
    ProbativeStrength,
    ReviewResult,
)
from justagent.judicial.legal_knowledge import (
    ArticleSearchResult,
    ArticleStatus,
    CaseLevel,
    CaseSearchResult,
    ConceptExplanation,
    LegalArticle,
    LegalCase,
    LegalDomain,
    LegalKnowledgeBase,
    LegalKnowledgeError,
)

logger = logging.getLogger("justagent.judicial")

__all__ = [
    # case_manager.py
    "CaseContext",
    "CaseFile",
    "CaseManager",
    "CaseManagerError",
    "CaseMaterial",
    "CaseStatus",
    "Claim",
    "DisputedIssue",
    "FactElement",
    "MaterialType",
    "Party",
    "PartyRole",
    "TimelineEvent",
    # evidence.py
    "Admissibility",
    "ChainAnalysisResult",
    "Evidence",
    "EvidenceChain",
    "EvidenceError",
    "EvidenceRelation",
    "EvidenceRelationType",
    "EvidenceReviewer",
    "EvidenceType",
    "ProbativeStrength",
    "ReviewResult",
    # document_generator.py
    "CitationVerification",
    "DocumentGenerationError",
    "GeneratedDocument",
    "GeneratedDocumentSection",
    "LegalDocumentGenerator",
    "LegalDocumentSection",
    "LegalDocumentTemplate",
    "LegalDocumentTemplateManager",
    "LegalDocumentType",
    # legal_knowledge.py
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
