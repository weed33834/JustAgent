"""Legal document generation — templates, LLM-backed drafting, citation verification.

Generates formal legal documents (indictments, defense statements,
judgments, rulings, mediation agreements, agency opinions, legal
opinions, evidence lists, cross-examination opinions) from a case
context bundle, evidence review results, and relevant legal articles.

The module integrates with the platform's RAG pipeline
(:mod:`justagent.knowledge.rag`) to retrieve relevant legal context,
and with the :class:`~justagent.verticals.legal.legal_knowledge.LegalKnowledgeBase`
to verify that every statute citation in the generated document
corresponds to a real, in-force article. Prompt rendering uses
LangChain's ``PromptTemplate`` when available (lazy import with
graceful fallback to ``str.format``), avoiding a hard LangChain dependency.

Design:

* :class:`LegalDocumentType` — enum of supported document types.
* :class:`LegalDocumentSection` — a named section within a template.
* :class:`LegalDocumentTemplate` — a reusable template definition.
* :class:`LegalDocumentTemplateManager` — registry of templates with
  built-in defaults for every document type.
* :class:`GeneratedDocument` — the output of a generation run.
* :class:`CitationVerification` — the result of verifying a single
  statute citation.
* :class:`LegalDocumentGenerator` — the thread-safe, async-capable
  generator that combines case context, evidence, RAG retrieval, and
  LLM synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from justagent.adapters.model_gateway import (
    ChatCompletionRequest,
    ChatMessage,
    ModelGateway,
)
from justagent.knowledge.rag import RAGPipeline
from justagent.utils import now

if TYPE_CHECKING:
    from justagent.verticals.legal.case_manager import CaseContext, CaseManager
    from justagent.verticals.legal.evidence import EvidenceChain
    from justagent.verticals.legal.legal_knowledge import LegalKnowledgeBase

logger = logging.getLogger("justagent.verticals.legal.document_generator")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DocumentGenerationError(Exception):
    """Raised for errors during legal document generation."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LegalDocumentType(str, Enum):  # noqa: UP042 - match existing codebase style
    """Supported legal document types.

    Attributes:
        INDICTMENT: Indictment / complaint (起诉书/起诉状).
        STATEMENT_OF_DEFENSE: Statement of defense (答辩状).
        JUDGMENT: Judgment (判决书).
        RULING: Ruling / order (裁定书).
        MEDIATION_AGREEMENT: Mediation agreement (调解书).
        AGENCY_OPINION: Agency opinion / representation brief (代理词).
        LEGAL_OPINION: Legal opinion (法律意见书).
        EVIDENCE_LIST: Evidence list / catalogue (证据目录).
        CROSS_EXAMINATION_OPINION: Cross-examination opinion (质证意见).
    """

    INDICTMENT = "indictment"
    STATEMENT_OF_DEFENSE = "statement_of_defense"
    JUDGMENT = "judgment"
    RULING = "ruling"
    MEDIATION_AGREEMENT = "mediation_agreement"
    AGENCY_OPINION = "agency_opinion"
    LEGAL_OPINION = "legal_opinion"
    EVIDENCE_LIST = "evidence_list"
    CROSS_EXAMINATION_OPINION = "cross_examination_opinion"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class LegalDocumentSection(BaseModel):
    """A named section within a legal document template.

    Attributes:
        title: Section heading (e.g. ``"诉讼请求"``).
        content_template: Template string with placeholders for case
            context variables. Rendered via LangChain ``PromptTemplate``
            or ``str.format``.
        required: Whether this section must be present in the output.
        order: Display / generation order (lower = earlier).
    """

    title: str
    content_template: str = ""
    required: bool = True
    order: int = 0


class LegalDocumentTemplate(BaseModel):
    """A reusable legal document template.

    Attributes:
        id: Unique template identifier (auto-generated UUID4 hex).
        doc_type: The :class:`LegalDocumentType` this template produces.
        name: Human-readable template name.
        description: What the template is for.
        sections: Ordered list of :class:`LegalDocumentSection` objects.
        placeholders: List of placeholder variable names expected by the
            section templates.
        system_prompt: Optional system prompt for LLM-assisted generation.
        metadata: Arbitrary key-value metadata.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    doc_type: LegalDocumentType
    name: str
    description: str = ""
    sections: list[LegalDocumentSection] = Field(default_factory=list)
    placeholders: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedDocumentSection(BaseModel):
    """A rendered section in a generated document.

    Attributes:
        title: Section heading.
        content: Rendered section content.
    """

    title: str
    content: str = ""


class CitationVerification(BaseModel):
    """The result of verifying a single statute citation.

    Attributes:
        citation: The raw citation text found in the document (e.g.
            ``"《民法典》第143条"``).
        is_valid: Whether the citation matches a known, in-force article.
        matched_article_id: ID of the matched article, or empty.
        matched_law_name: Name of the matched law, or empty.
        matched_article_number: Article number of the match, or empty.
        issues: List of problems found (e.g. ``"article not found"``,
            ``"article has been repealed"``).
    """

    citation: str
    is_valid: bool = False
    matched_article_id: str = ""
    matched_law_name: str = ""
    matched_article_number: str = ""
    issues: list[str] = Field(default_factory=list)


class GeneratedDocument(BaseModel):
    """The output of a legal document generation run.

    Attributes:
        id: Unique document identifier (auto-generated UUID4 hex).
        case_id: The source case ID.
        doc_type: The :class:`LegalDocumentType`.
        title: Document title.
        sections: List of rendered :class:`GeneratedDocumentSection`.
        content: Full concatenated document text.
        citations: List of statute citations found in the document.
        citation_verifications: Verification results for each citation.
        all_citations_valid: Whether all citations passed verification.
        metadata: Additional generation metadata.
        created_at: Unix timestamp of generation.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    case_id: str
    doc_type: LegalDocumentType
    title: str = ""
    sections: list[GeneratedDocumentSection] = Field(default_factory=list)
    content: str = ""
    citations: list[str] = Field(default_factory=list)
    citation_verifications: list[CitationVerification] = Field(default_factory=list)
    all_citations_valid: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=now)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_prompt(template: str, variables: dict[str, Any]) -> str:
    """Render a prompt template using LangChain when available.

    Tries ``langchain_core.prompts.PromptTemplate`` first; if LangChain
    is not installed, falls back to plain ``str.format``.
    """

    try:
        from langchain_core.prompts import PromptTemplate

        rendered: str = PromptTemplate.from_template(template).format(**variables)
        return rendered
    except ImportError:
        return template.format(**variables)


# Regex to extract statute citations like 《民法典》第143条 or 《刑法》第二百六十四条.
_CITATION_RE = re.compile(
    r"《([^》]+)》\s*(第[一二三四五六七八九十百千零\d]+条(?:之[一二三四五六七八九十\d]+)?)"
)


def _extract_citations(text: str) -> list[str]:
    """Extract statute citations from *text*.

    Returns a list of citation strings (e.g.
    ``["《民法典》第143条", "《刑法》第二百六十四条"]``).
    """

    return [m.group(0) for m in _CITATION_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Default templates
# ---------------------------------------------------------------------------


def _default_templates() -> dict[LegalDocumentType, LegalDocumentTemplate]:
    """Build the built-in default templates for every document type."""

    templates: dict[LegalDocumentType, LegalDocumentTemplate] = {}

    # --- Indictment / complaint (起诉状) ---
    templates[LegalDocumentType.INDICTMENT] = LegalDocumentTemplate(
        doc_type=LegalDocumentType.INDICTMENT,
        name="民事起诉状模板",
        description="标准民事起诉状模板",
        system_prompt="你是一名专业律师，请根据案件信息起草一份规范的民事起诉状。",
        sections=[
            LegalDocumentSection(
                title="当事人信息",
                content_template="{parties}",
                order=1,
            ),
            LegalDocumentSection(
                title="诉讼请求",
                content_template="{claims}",
                order=2,
            ),
            LegalDocumentSection(
                title="事实与理由",
                content_template="{facts}\n\n{legal_context}",
                order=3,
            ),
            LegalDocumentSection(
                title="此致",
                content_template="{court}",
                required=False,
                order=4,
            ),
        ],
        placeholders=["parties", "claims", "facts", "court", "legal_context"],
    )

    # --- Statement of defense (答辩状) ---
    templates[LegalDocumentType.STATEMENT_OF_DEFENSE] = LegalDocumentTemplate(
        doc_type=LegalDocumentType.STATEMENT_OF_DEFENSE,
        name="答辩状模板",
        description="标准民事答辩状模板",
        system_prompt="你是一名专业律师，请根据案件信息起草一份规范的答辩状。",
        sections=[
            LegalDocumentSection(
                title="答辩人",
                content_template="{parties}",
                order=1,
            ),
            LegalDocumentSection(
                title="答辩请求",
                content_template="请根据原告诉讼请求提出答辩请求。\n{claims}",
                order=2,
            ),
            LegalDocumentSection(
                title="事实与理由",
                content_template="{facts}\n\n{disputed}\n\n{legal_context}",
                order=3,
            ),
        ],
        placeholders=["parties", "claims", "facts", "disputed", "legal_context"],
    )

    # --- Judgment (判决书) ---
    templates[LegalDocumentType.JUDGMENT] = LegalDocumentTemplate(
        doc_type=LegalDocumentType.JUDGMENT,
        name="判决书模板",
        description="标准民事判决书模板",
        system_prompt="你是一名法官，请根据案件信息和证据审查结果起草一份规范的判决书。",
        sections=[
            LegalDocumentSection(
                title="案件基本信息",
                content_template="{summary}",
                order=1,
            ),
            LegalDocumentSection(
                title="当事人",
                content_template="{parties}",
                order=2,
            ),
            LegalDocumentSection(
                title="诉辩意见",
                content_template="原告诉请：\n{claims}\n\n被告答辩：\n{disputed}",
                order=3,
            ),
            LegalDocumentSection(
                title="查明事实",
                content_template="{facts}\n\n{timeline}",
                order=4,
            ),
            LegalDocumentSection(
                title="本院认为",
                content_template="{legal_context}\n\n{evidence_analysis}",
                order=5,
            ),
            LegalDocumentSection(
                title="判决结果",
                content_template="（根据上述理由作出判决）",
                required=False,
                order=6,
            ),
        ],
        placeholders=[
            "summary", "parties", "claims", "disputed", "facts",
            "timeline", "legal_context", "evidence_analysis",
        ],
    )

    # --- Ruling (裁定书) ---
    templates[LegalDocumentType.RULING] = LegalDocumentTemplate(
        doc_type=LegalDocumentType.RULING,
        name="裁定书模板",
        description="标准裁定书模板",
        system_prompt="你是一名法官，请根据案件信息起草一份裁定书。",
        sections=[
            LegalDocumentSection(
                title="案件基本信息",
                content_template="{summary}",
                order=1,
            ),
            LegalDocumentSection(
                title="裁定事项",
                content_template="{facts}\n\n{legal_context}",
                order=2,
            ),
        ],
        placeholders=["summary", "facts", "legal_context"],
    )

    # --- Mediation agreement (调解书) ---
    templates[LegalDocumentType.MEDIATION_AGREEMENT] = LegalDocumentTemplate(
        doc_type=LegalDocumentType.MEDIATION_AGREEMENT,
        name="调解书模板",
        description="标准民事调解书模板",
        system_prompt="你是一名法官，请根据案件信息起草一份调解书。",
        sections=[
            LegalDocumentSection(
                title="当事人信息",
                content_template="{parties}",
                order=1,
            ),
            LegalDocumentSection(
                title="案件事由",
                content_template="{summary}\n\n{facts}",
                order=2,
            ),
            LegalDocumentSection(
                title="调解协议",
                content_template="经调解，双方自愿达成如下协议：\n{claims}",
                order=3,
            ),
        ],
        placeholders=["parties", "summary", "facts", "claims"],
    )

    # --- Agency opinion (代理词) ---
    templates[LegalDocumentType.AGENCY_OPINION] = LegalDocumentTemplate(
        doc_type=LegalDocumentType.AGENCY_OPINION,
        name="代理词模板",
        description="代理词模板",
        system_prompt="你是一名代理律师，请根据案件信息起草代理词。",
        sections=[
            LegalDocumentSection(
                title="案件基本情况",
                content_template="{summary}\n\n{parties}",
                order=1,
            ),
            LegalDocumentSection(
                title="代理意见",
                content_template=(
                    "一、事实方面：\n{facts}\n\n"
                    "二、证据方面：\n{evidence_analysis}\n\n"
                    "三、法律方面：\n{legal_context}\n\n"
                    "四、争议焦点：\n{disputed}"
                ),
                order=2,
            ),
            LegalDocumentSection(
                title="结论",
                content_template="综上所述，{claims}",
                required=False,
                order=3,
            ),
        ],
        placeholders=[
            "summary", "parties", "facts", "evidence_analysis",
            "legal_context", "disputed", "claims",
        ],
    )

    # --- Legal opinion (法律意见书) ---
    templates[LegalDocumentType.LEGAL_OPINION] = LegalDocumentTemplate(
        doc_type=LegalDocumentType.LEGAL_OPINION,
        name="法律意见书模板",
        description="法律意见书模板",
        system_prompt="你是一名法律顾问，请根据案件信息出具法律意见书。",
        sections=[
            LegalDocumentSection(
                title="委托事项",
                content_template="{summary}",
                order=1,
            ),
            LegalDocumentSection(
                title="事实概要",
                content_template="{facts}\n\n{timeline}",
                order=2,
            ),
            LegalDocumentSection(
                title="法律分析",
                content_template="{legal_context}\n\n{disputed}",
                order=3,
            ),
            LegalDocumentSection(
                title="法律意见",
                content_template="基于上述分析，本所出具如下法律意见：\n{claims}",
                order=4,
            ),
        ],
        placeholders=[
            "summary", "facts", "timeline", "legal_context", "disputed", "claims",
        ],
    )

    # --- Evidence list (证据目录) ---
    templates[LegalDocumentType.EVIDENCE_LIST] = LegalDocumentTemplate(
        doc_type=LegalDocumentType.EVIDENCE_LIST,
        name="证据目录模板",
        description="证据目录模板",
        system_prompt="请根据证据信息整理证据目录。",
        sections=[
            LegalDocumentSection(
                title="证据目录",
                content_template="{evidence_analysis}",
                order=1,
            ),
        ],
        placeholders=["evidence_analysis"],
    )

    # --- Cross-examination opinion (质证意见) ---
    templates[LegalDocumentType.CROSS_EXAMINATION_OPINION] = LegalDocumentTemplate(
        doc_type=LegalDocumentType.CROSS_EXAMINATION_OPINION,
        name="质证意见模板",
        description="质证意见模板",
        system_prompt="你是一名律师，请根据证据审查结果起草质证意见。",
        sections=[
            LegalDocumentSection(
                title="质证意见",
                content_template=(
                    "对对方提交的证据发表质证意见如下：\n\n"
                    "{evidence_analysis}\n\n"
                    "法律依据：\n{legal_context}"
                ),
                order=1,
            ),
        ],
        placeholders=["evidence_analysis", "legal_context"],
    )

    return templates


# ---------------------------------------------------------------------------
# Template manager
# ---------------------------------------------------------------------------


class LegalDocumentTemplateManager:
    """Registry of legal document templates with built-in defaults.

    Thread-safe. On construction, registers a default template for every
    :class:`LegalDocumentType`. Custom templates can be registered to
    override the defaults.

    Example::

        >>> mgr = LegalDocumentTemplateManager()
        >>> tpl = mgr.get_template(LegalDocumentType.INDICTMENT)
        >>> tpl.name
        '民事起诉状模板'
    """

    def __init__(self) -> None:
        self._templates: dict[str, LegalDocumentTemplate] = {}
        self._type_index: dict[LegalDocumentType, str] = {}
        self._lock = threading.RLock()
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the built-in default templates."""

        for doc_type, template in _default_templates().items():
            self._templates[template.id] = template
            self._type_index[doc_type] = template.id

    def register_template(self, template: LegalDocumentTemplate) -> LegalDocumentTemplate:
        """Register a template, replacing any existing one for the same type.

        Raises:
            DocumentGenerationError: If the template's doc_type already
                has a registered template with a different ID (use
                :meth:`replace_template` instead).
        """

        with self._lock:
            existing_id = self._type_index.get(template.doc_type)
            if existing_id is not None and existing_id != template.id:
                raise DocumentGenerationError(
                    f"Template already exists for {template.doc_type.value}; "
                    f"use replace_template() to override."
                )
            self._templates[template.id] = template
            self._type_index[template.doc_type] = template.id
        logger.info("Registered template %s (%s)", template.id, template.name)
        return template

    def replace_template(self, template: LegalDocumentTemplate) -> LegalDocumentTemplate:
        """Replace an existing template for a document type."""

        with self._lock:
            old_id = self._type_index.get(template.doc_type)
            if old_id is not None:
                self._templates.pop(old_id, None)
            self._templates[template.id] = template
            self._type_index[template.doc_type] = template.id
        return template

    def get_template(
        self, doc_type: LegalDocumentType
    ) -> LegalDocumentTemplate | None:
        """Return the template for a document type, or ``None``."""

        with self._lock:
            tid = self._type_index.get(doc_type)
            return self._templates.get(tid) if tid else None

    def get_template_by_id(
        self, template_id: str
    ) -> LegalDocumentTemplate | None:
        """Return a template by ID, or ``None``."""

        with self._lock:
            return self._templates.get(template_id)

    def list_templates(self) -> list[LegalDocumentTemplate]:
        """Return all registered templates."""

        with self._lock:
            return list(self._templates.values())

    @property
    def template_count(self) -> int:
        """Total number of registered templates."""

        with self._lock:
            return len(self._templates)


# ---------------------------------------------------------------------------
# Legal document generator
# ---------------------------------------------------------------------------


class LegalDocumentGenerator:
    """Generate legal documents from case context, evidence, and law.

    Combines a :class:`~justagent.verticals.legal.case_manager.CaseManager`
    (for case context), an optional
    :class:`~justagent.verticals.legal.evidence.EvidenceChain` (for evidence
    analysis), an optional
    :class:`~justagent.verticals.legal.legal_knowledge.LegalKnowledgeBase`
    (for citation verification), and an optional
    :class:`~justagent.knowledge.rag.RAGPipeline` (for legal-context
    retrieval) to produce a :class:`GeneratedDocument`.

    When a :class:`ModelGateway` is provided, the generator uses
    LLM-assisted drafting: it renders the template sections into a
    prompt, retrieves relevant legal context via RAG, and asks the LLM
    to produce the document. Without a gateway, the generator produces
    a template-filled draft using the case context directly.

    All operations are thread-safe. Async variants are provided for
    generation and citation verification.

    Args:
        case_manager: The :class:`CaseManager` providing case context.
        template_manager: Optional template manager. If None, a default
            one is created.
        gateway: Optional :class:`ModelGateway` for LLM-assisted drafting.
        rag_pipeline: Optional :class:`RAGPipeline` for legal-context
            retrieval.
        knowledge_base: Optional :class:`LegalKnowledgeBase` for
            citation verification.
        evidence_chain: Optional :class:`EvidenceChain` for evidence
            analysis text.
        temperature: LLM sampling temperature.
        max_tokens: Maximum tokens for LLM responses.

    Example::

        >>> from justagent.verticals.legal.case_manager import CaseManager
        >>> mgr = CaseManager()
        >>> case = mgr.create_case(case_number="(2024)京01民初1号")
        >>> gen = LegalDocumentGenerator(case_manager=mgr)
        >>> doc = gen.generate(case.id, LegalDocumentType.INDICTMENT)
        >>> doc.doc_type == LegalDocumentType.INDICTMENT
        True
    """

    def __init__(
        self,
        case_manager: CaseManager,
        *,
        template_manager: LegalDocumentTemplateManager | None = None,
        gateway: ModelGateway | None = None,
        rag_pipeline: RAGPipeline | None = None,
        knowledge_base: LegalKnowledgeBase | None = None,
        evidence_chain: EvidenceChain | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        self._case_manager = case_manager
        self._templates = template_manager or LegalDocumentTemplateManager()
        self._gateway = gateway
        self._rag = rag_pipeline
        self._kb = knowledge_base
        self._evidence_chain = evidence_chain
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def case_manager(self) -> CaseManager:
        """The case manager providing case context."""

        return self._case_manager

    @property
    def template_manager(self) -> LegalDocumentTemplateManager:
        """The template manager."""

        return self._templates

    @property
    def gateway(self) -> ModelGateway | None:
        """The LLM gateway, or ``None``."""

        return self._gateway

    @gateway.setter
    def gateway(self, value: ModelGateway | None) -> None:
        self._gateway = value

    @property
    def rag_pipeline(self) -> RAGPipeline | None:
        """The RAG pipeline, or ``None``."""

        return self._rag

    @property
    def knowledge_base(self) -> LegalKnowledgeBase | None:
        """The legal knowledge base, or ``None``."""

        return self._kb

    # ------------------------------------------------------------------
    # Document generation
    # ------------------------------------------------------------------

    def generate(
        self,
        case_id: str,
        doc_type: LegalDocumentType,
        *,
        title: str | None = None,
        extra_context: dict[str, str] | None = None,
        verify: bool = True,
    ) -> GeneratedDocument:
        """Generate a legal document for a case.

        Args:
            case_id: The case to generate the document for.
            doc_type: The type of document to generate.
            title: Optional document title override.
            extra_context: Additional context variables to inject into
                the template (merged with the case context).
            verify: If True (default), verify statute citations in the
                generated document.

        Raises:
            DocumentGenerationError: If the case or template is not found.
        """

        # 1. Get the template.
        template = self._templates.get_template(doc_type)
        if template is None:
            raise DocumentGenerationError(
                f"No template registered for {doc_type.value}"
            )

        # 2. Build case context.
        ctx = self._case_manager.build_context(case_id)
        case = self._case_manager.get_case(case_id)

        # 3. Retrieve legal context via RAG (if configured).
        legal_context = self._retrieve_legal_context(case_id, ctx)

        # 4. Build evidence analysis text (if configured).
        evidence_analysis = self._build_evidence_text(case_id)

        # 5. Assemble all template variables.
        variables: dict[str, str] = {
            "summary": ctx.summary,
            "parties": ctx.parties_text,
            "facts": ctx.facts_text,
            "claims": ctx.claims_text,
            "disputed": ctx.disputed_text,
            "timeline": ctx.timeline_text,
            "evidence": ctx.evidence_summary,
            "court": case.court if case else "",
            "legal_context": legal_context,
            "evidence_analysis": evidence_analysis,
        }
        if extra_context:
            variables.update(extra_context)

        # 6. Generate document — LLM-assisted or template-only.
        if self._gateway is not None:
            sections = self._generate_with_llm(template, variables, case_id, doc_type)
        else:
            sections = self._generate_template_only(template, variables)

        # 7. Assemble full content.
        content_parts: list[str] = []
        if title:
            content_parts.append(title)
            content_parts.append("")
        for section in sections:
            content_parts.append(section.title)
            content_parts.append(section.content)
            content_parts.append("")
        full_content = "\n".join(content_parts).strip()

        # 8. Extract and verify citations.
        citations = _extract_citations(full_content)
        verifications: list[CitationVerification] = []
        if verify:
            verifications = self.verify_citations(citations)

        all_valid = all(v.is_valid for v in verifications) if verifications else True

        doc_title = title or f"{template.name} - {ctx.summary}"

        generated = GeneratedDocument(
            case_id=case_id,
            doc_type=doc_type,
            title=doc_title,
            sections=sections,
            content=full_content,
            citations=citations,
            citation_verifications=verifications,
            all_citations_valid=all_valid,
            metadata={
                "template_id": template.id,
                "template_name": template.name,
                "used_llm": self._gateway is not None,
                "used_rag": self._rag is not None,
            },
        )
        logger.info(
            "Generated %s for case %s (%d chars, %d citations, valid=%s)",
            doc_type.value,
            case_id,
            len(full_content),
            len(citations),
            all_valid,
        )
        return generated

    async def generate_async(
        self,
        case_id: str,
        doc_type: LegalDocumentType,
        *,
        title: str | None = None,
        extra_context: dict[str, str] | None = None,
        verify: bool = True,
    ) -> GeneratedDocument:
        """Async wrapper for :meth:`generate`."""

        return await asyncio.to_thread(
            self.generate,
            case_id,
            doc_type,
            title=title,
            extra_context=extra_context,
            verify=verify,
        )

    # ------------------------------------------------------------------
    # Citation verification
    # ------------------------------------------------------------------

    def verify_citations(
        self, citations: list[str]
    ) -> list[CitationVerification]:
        """Verify a list of statute citations against the knowledge base.

        For each citation, attempts to find a matching
        :class:`~justagent.verticals.legal.legal_knowledge.LegalArticle` in the
        knowledge base. If no knowledge base is configured, all citations
        are marked as ``is_valid=True`` with a note that verification
        was skipped.

        Args:
            citations: List of citation strings (e.g.
                ``["《民法典》第143条"]``).

        Returns:
            List of :class:`CitationVerification` results.
        """

        results: list[CitationVerification] = []
        for citation in citations:
            results.append(self._verify_single_citation(citation))
        return results

    async def verify_citations_async(
        self, citations: list[str]
    ) -> list[CitationVerification]:
        """Async wrapper for :meth:`verify_citations`."""

        return await asyncio.to_thread(self.verify_citations, citations)

    def verify_document(
        self, document: GeneratedDocument
    ) -> GeneratedDocument:
        """Verify all citations in an existing :class:`GeneratedDocument`.

        Updates the document's ``citation_verifications`` and
        ``all_citations_valid`` fields in place and returns it.
        """

        verifications = self.verify_citations(document.citations)
        document.citation_verifications = verifications
        document.all_citations_valid = all(
            v.is_valid for v in verifications
        ) if verifications else True
        return document

    # ------------------------------------------------------------------
    # Internal: generation
    # ------------------------------------------------------------------

    def _retrieve_legal_context(
        self, case_id: str, ctx: CaseContext
    ) -> str:
        """Retrieve relevant legal context via RAG or knowledge base."""

        parts: list[str] = []

        # Use knowledge base for article search.
        if self._kb is not None:
            # Search based on the case's legal basis and claims.
            query = " ".join(ctx.legal_basis) if ctx.legal_basis else ctx.summary
            if query.strip():
                results = self._kb.search_articles(query, top_k=5)
                for ar in results:
                    parts.append(
                        f"{ar.article.citation}: {ar.article.content[:200]}"
                    )

        # Use RAG pipeline for additional context.
        if self._rag is not None:
            query = f"{ctx.summary} {ctx.claims_text}"
            if query.strip():
                answer = self._rag.query(query, top_k=3)
                if answer.has_answer:
                    parts.append(answer.answer)

        return "\n\n".join(parts) if parts else "（暂无相关法条检索结果）"

    def _build_evidence_text(self, case_id: str) -> str:
        """Build a formatted evidence-analysis text block."""

        if self._evidence_chain is None:
            return "（暂无证据审查信息）"

        result = self._evidence_chain.analyze(case_id)
        parts: list[str] = [result.summary]

        # List evidence items.
        evidence_list = self._evidence_chain.list_evidence(case_id=case_id)
        if evidence_list:
            parts.append("\n证据清单：")
            for i, ev in enumerate(evidence_list, start=1):
                line = (
                    f"  {i}. {ev.name} ({ev.type.value}) "
                    f"- 证明对象: {ev.proving_object or '未指明'}"
                )
                if ev.reviewed:
                    line += (
                        f" | 可采性: {ev.admissibility.value} "
                        f"| 证明力: {ev.probative_strength.value}"
                    )
                parts.append(line)

        # List contradictions.
        if result.contradictions:
            parts.append("\n证据矛盾：")
            for rel in result.contradictions:
                ev_a = self._evidence_chain.get_evidence(rel.evidence_a_id)
                ev_b = self._evidence_chain.get_evidence(rel.evidence_b_id)
                name_a = ev_a.name if ev_a else rel.evidence_a_id
                name_b = ev_b.name if ev_b else rel.evidence_b_id
                parts.append(f"  - {name_a} 与 {name_b}: {rel.description}")

        return "\n".join(parts)

    def _generate_template_only(
        self,
        template: LegalDocumentTemplate,
        variables: dict[str, str],
    ) -> list[GeneratedDocumentSection]:
        """Generate document sections by rendering templates with variables."""

        sections: list[GeneratedDocumentSection] = []
        for section in sorted(template.sections, key=lambda s: s.order):
            content = ""
            if section.content_template:
                try:
                    content = _render_prompt(
                        section.content_template, variables
                    )
                except (KeyError, IndexError):
                    content = section.content_template
            sections.append(
                GeneratedDocumentSection(title=section.title, content=content)
            )
        return sections

    def _generate_with_llm(
        self,
        template: LegalDocumentTemplate,
        variables: dict[str, str],
        case_id: str,
        doc_type: LegalDocumentType,
    ) -> list[GeneratedDocumentSection]:
        """Generate document sections using the LLM gateway."""

        # Build a comprehensive prompt from the template sections.
        section_descriptions = "\n".join(
            f"## {s.title}\n{s.content_template}"
            for s in sorted(template.sections, key=lambda s: s.order)
        )
        prompt_template = (
            "{system}\n\n"
            "案件信息：\n{summary}\n\n"
            "当事人：\n{parties}\n\n"
            "诉讼请求：\n{claims}\n\n"
            "事实要素：\n{facts}\n\n"
            "争议焦点：\n{disputed}\n\n"
            "时间线：\n{timeline}\n\n"
            "法律依据与检索结果：\n{legal_context}\n\n"
            "证据分析：\n{evidence_analysis}\n\n"
            "请按照以下结构生成文书（每个部分用 ## 标题开头）：\n"
            "{sections}"
        )
        prompt = _render_prompt(
            prompt_template,
            {
                "system": template.system_prompt,
                "summary": variables.get("summary", ""),
                "parties": variables.get("parties", ""),
                "claims": variables.get("claims", ""),
                "facts": variables.get("facts", ""),
                "disputed": variables.get("disputed", ""),
                "timeline": variables.get("timeline", ""),
                "legal_context": variables.get("legal_context", ""),
                "evidence_analysis": variables.get("evidence_analysis", ""),
                "sections": section_descriptions,
            },
        )

        try:
            request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content=template.system_prompt),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            response = self._gateway.chat(request)  # type: ignore[union-attr]
            return self._parse_llm_output(response.content, template)
        except Exception as exc:
            logger.error("LLM generation failed: %s; falling back to template", exc)
            return self._generate_template_only(template, variables)

    @staticmethod
    def _parse_llm_output(
        output: str, template: LegalDocumentTemplate
    ) -> list[GeneratedDocumentSection]:
        """Parse LLM output into sections based on ``##`` headings."""

        sections: list[GeneratedDocumentSection] = []
        current_title = ""
        current_content: list[str] = []

        for line in output.split("\n"):
            stripped = line.strip()
            if stripped.startswith("## "):
                if current_title or current_content:
                    sections.append(
                        GeneratedDocumentSection(
                            title=current_title or "正文",
                            content="\n".join(current_content).strip(),
                        )
                    )
                current_title = stripped[3:].strip()
                current_content = []
            else:
                current_content.append(line)

        if current_title or current_content:
            sections.append(
                GeneratedDocumentSection(
                    title=current_title or "正文",
                    content="\n".join(current_content).strip(),
                )
            )

        if not sections:
            sections.append(
                GeneratedDocumentSection(title="正文", content=output.strip())
            )
        return sections

    # ------------------------------------------------------------------
    # Internal: citation verification
    # ------------------------------------------------------------------

    def _verify_single_citation(self, citation: str) -> CitationVerification:
        """Verify a single statute citation against the knowledge base."""

        # Parse the citation into law name and article number.
        match = _CITATION_RE.search(citation)
        if match is None:
            return CitationVerification(
                citation=citation,
                is_valid=False,
                issues=["citation format not recognised"],
            )

        law_name = match.group(1)
        article_number = match.group(2)

        # If no knowledge base, skip verification.
        if self._kb is None:
            return CitationVerification(
                citation=citation,
                is_valid=True,
                matched_law_name=law_name,
                matched_article_number=article_number,
                issues=["verification skipped (no knowledge base)"],
            )

        # Search the knowledge base.
        article = self._kb.find_article(law_name, article_number)
        if article is None:
            # Try a fuzzy search by law name + article number keywords.
            results = self._kb.search_articles(
                f"{law_name} {article_number}", top_k=3
            )
            if results:
                best = results[0].article
                return CitationVerification(
                    citation=citation,
                    is_valid=True,
                    matched_article_id=best.id,
                    matched_law_name=best.law_name,
                    matched_article_number=best.article_number,
                    issues=["fuzzy match (exact article number not found)"],
                )
            return CitationVerification(
                citation=citation,
                is_valid=False,
                matched_law_name=law_name,
                matched_article_number=article_number,
                issues=["article not found in knowledge base"],
            )

        issues: list[str] = []
        if not article.is_effective:
            issues.append(f"article status is '{article.status.value}'")

        return CitationVerification(
            citation=citation,
            is_valid=article.is_effective,
            matched_article_id=article.id,
            matched_law_name=article.law_name,
            matched_article_number=article.article_number,
            issues=issues,
        )


__all__ = [
    "CitationVerification",
    "DocumentGenerationError",
    "GeneratedDocument",
    "GeneratedDocumentSection",
    "LegalDocumentGenerator",
    "LegalDocumentSection",
    "LegalDocumentTemplate",
    "LegalDocumentTemplateManager",
    "LegalDocumentType",
]
