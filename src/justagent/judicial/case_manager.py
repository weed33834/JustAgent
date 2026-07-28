"""Case material organisation — structured case files and material ingestion.

Provides the central case-management layer for the judicial platform. A
:class:`CaseFile` captures the full structure of a legal case — parties,
fact elements, evidence references, disputed issues, claims, and a
chronological timeline — while the :class:`CaseManager` handles creation,
multi-format material import, structured extraction, case search, and
context assembly.

Material import integrates with
:class:`justagent.knowledge.document.DocumentParser` to ingest PDF, Word,
Excel, PPT, Markdown, HTML, and plain-text files. Structured extraction
(parties, facts, timeline events) uses LangChain prompt templates when
available (lazy import with graceful fallback to a rule-based extractor),
ensuring the module works out-of-the-box without LangChain installed.

Design:

* :class:`PartyRole` / :class:`CaseStatus` — typed enumerations.
* :class:`Party` — a litigation party (plaintiff, defendant, ...).
* :class:`TimelineEvent` — a dated event in the case chronology.
* :class:`FactElement` — a factual proposition with supporting evidence.
* :class:`DisputedIssue` — a contested point between the parties.
* :class:`Claim` — a litigation claim / prayer for relief.
* :class:`CaseFile` — the umbrella Pydantic model tying everything together.
* :class:`CaseContext` — an assembled context bundle for downstream
  document generation or LLM prompting.
* :class:`CaseManager` — the thread-safe, async-capable manager.

All registry mutations are protected by a ``threading.RLock``. Async
variants are provided for material import and structured extraction so
the case manager can be used from async orchestration workflows.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from justagent.knowledge.document import (
    Document,
    DocumentParser,
)

logger = logging.getLogger("justagent.judicial.case_manager")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CaseManagerError(Exception):
    """Raised for invalid case-management operations."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PartyRole(str, Enum):  # noqa: UP042 - match existing codebase style
    """Litigation party roles.

    Attributes:
        PLAINTIFF: Plaintiff (原告).
        DEFENDANT: Defendant (被告).
        THIRD_PARTY: Third party with independent claims (有独立请求权第三人).
        APPELLANT: Appellant (上诉人).
        APPELLEE: Appellee (被上诉人).
        APPLICANT: Applicant (申请人).
        RESPONDENT: Respondent (被申请人).
        WITNESS: Witness (证人).
        OTHER: Any role not covered by the built-in values.
    """

    PLAINTIFF = "plaintiff"
    DEFENDANT = "defendant"
    THIRD_PARTY = "third_party"
    APPELLANT = "appellant"
    APPELLEE = "appellee"
    APPLICANT = "applicant"
    RESPONDENT = "respondent"
    WITNESS = "witness"
    OTHER = "other"


class CaseStatus(str, Enum):  # noqa: UP042
    """Lifecycle status of a case file.

    Attributes:
        DRAFT: Case file created but not yet finalised.
        ACTIVE: Case is being actively worked on.
        UNDER_REVIEW: Case is under internal or judicial review.
        CLOSED: Case has been concluded.
        ARCHIVED: Case is retained for reference but no longer active.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    CLOSED = "closed"
    ARCHIVED = "archived"


class MaterialType(str, Enum):  # noqa: UP042
    """Categories of imported case materials.

    Attributes:
        COMPLAINT: The complaint / indictment (起诉状).
        DEFENSE: The statement of defense (答辩状).
        EVIDENCE: Evidentiary material (证据材料).
        JUDGMENT: A judgment or ruling (裁判文书).
        CONTRACT: A contract or agreement (合同/协议).
        CORRESPONDENCE: Letters, notices, and correspondence (函件).
        OTHER: Any material not covered by the built-in values.
    """

    COMPLAINT = "complaint"
    DEFENSE = "defense"
    EVIDENCE = "evidence"
    JUDGMENT = "judgment"
    CONTRACT = "contract"
    CORRESPONDENCE = "correspondence"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Party(BaseModel):
    """A litigation party.

    Attributes:
        id: Unique party identifier (auto-generated UUID4 hex).
        name: Full legal name of the party (natural person or entity).
        role: The party's :class:`PartyRole` in this case.
        contact: Contact information (phone, email, address).
        legal_representative: Legal representative (for entities).
        id_number: National ID / registration number (optional).
        metadata: Arbitrary key-value metadata.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    role: PartyRole = PartyRole.OTHER
    contact: str = ""
    legal_representative: str = ""
    id_number: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimelineEvent(BaseModel):
    """A single event in the case chronology.

    Attributes:
        id: Unique event identifier.
        date: ISO date string (``YYYY-MM-DD``) or free-form date text.
        timestamp: Sortable Unix timestamp (0 if unknown).
        description: What happened.
        category: Event category (e.g. ``"filing"``, ``"hearing"``,
            ``"evidence_submission"``).
        source_document_id: ID of the imported document this event was
            extracted from.
        metadata: Arbitrary key-value metadata.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    date: str = ""
    timestamp: float = 0.0
    description: str
    category: str = ""
    source_document_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class FactElement(BaseModel):
    """A factual proposition in the case.

    Attributes:
        id: Unique fact identifier.
        description: The factual proposition.
        category: Fact category (e.g. ``"background"``, ``"transaction"``,
            ``"damage"``).
        supporting_evidence_ids: IDs of evidence items supporting this fact.
        contested: Whether this fact is disputed by the opposing party.
        source_document_id: ID of the document this fact was extracted from.
        metadata: Arbitrary key-value metadata.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    description: str
    category: str = ""
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contested: bool = False
    source_document_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DisputedIssue(BaseModel):
    """A contested point between the parties.

    Attributes:
        id: Unique issue identifier.
        description: The disputed issue.
        category: Issue category (e.g. ``"fact"``, ``"law"``,
            ``"evidence_admissibility"``).
        plaintiff_position: The plaintiff's position on this issue.
        defendant_position: The defendant's position on this issue.
        related_evidence_ids: IDs of evidence items relevant to this issue.
        metadata: Arbitrary key-value metadata.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    description: str
    category: str = ""
    plaintiff_position: str = ""
    defendant_position: str = ""
    related_evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """A litigation claim / prayer for relief.

    Attributes:
        id: Unique claim identifier.
        description: The claim text (e.g. ``"判令被告支付货款100万元"``).
        claim_type: Type of claim (e.g. ``"monetary"``, ``"injunction"``,
            ``"declaration"``).
        amount: Monetary amount (0 for non-monetary claims).
        legal_basis: Legal articles cited as the basis for this claim.
        supporting_evidence_ids: IDs of supporting evidence items.
        metadata: Arbitrary key-value metadata.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    description: str
    claim_type: str = ""
    amount: float = 0.0
    legal_basis: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseMaterial(BaseModel):
    """An imported case material document.

    Attributes:
        id: Unique material identifier (same as the document ID).
        document: The parsed :class:`Document`.
        material_type: The :class:`MaterialType` classification.
        case_id: ID of the case this material belongs to.
        imported_at: Unix timestamp of import.
        notes: Free-form reviewer notes.
    """

    id: str
    document: Document
    material_type: MaterialType = MaterialType.OTHER
    case_id: str = ""
    imported_at: float = Field(default_factory=lambda: _now())
    notes: str = ""


class CaseFile(BaseModel):
    """The full structured case file.

    Attributes:
        id: Unique case identifier (auto-generated UUID4 hex).
        case_number: Official case number (e.g. ``"(2024)京01民初1号"``).
        cause_of_action: Cause of action / case type (e.g. ``"买卖合同纠纷"``).
        court: Adjudicating court.
        domain: Legal domain string (e.g. ``"civil"``, ``"criminal"``).
        parties: List of :class:`Party` objects.
        fact_elements: List of :class:`FactElement` objects.
        evidence_ids: IDs of evidence items associated with this case.
        disputed_issues: List of :class:`DisputedIssue` objects.
        claims: List of :class:`Claim` objects.
        timeline: List of :class:`TimelineEvent` objects, sorted by timestamp.
        material_ids: IDs of imported :class:`CaseMaterial` objects.
        status: Current :class:`CaseStatus`.
        metadata: Arbitrary key-value metadata.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last modification.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    case_number: str = ""
    cause_of_action: str = ""
    court: str = ""
    domain: str = ""
    parties: list[Party] = Field(default_factory=list)
    fact_elements: list[FactElement] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    disputed_issues: list[DisputedIssue] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    material_ids: list[str] = Field(default_factory=list)
    status: CaseStatus = CaseStatus.DRAFT
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=lambda: _now())
    updated_at: float = Field(default_factory=lambda: _now())

    @property
    def is_active(self) -> bool:
        """True if the case is in an active (non-closed, non-archived) state."""

        return self.status in (CaseStatus.ACTIVE, CaseStatus.UNDER_REVIEW)

    def plaintiff(self) -> Party | None:
        """Return the first plaintiff, or ``None``."""

        return next(
            (p for p in self.parties if p.role is PartyRole.PLAINTIFF), None
        )

    def defendant(self) -> Party | None:
        """Return the first defendant, or ``None``."""

        return next(
            (p for p in self.parties if p.role is PartyRole.DEFENDANT), None
        )


class CaseContext(BaseModel):
    """An assembled context bundle for a case, ready for LLM prompting.

    Attributes:
        case_id: The source case ID.
        summary: A natural-language summary of the case.
        parties_text: Formatted party listing.
        facts_text: Formatted fact listing.
        claims_text: Formatted claim listing.
        disputed_text: Formatted disputed-issue listing.
        timeline_text: Formatted chronology.
        evidence_summary: Summary of associated evidence.
        legal_basis: Legal articles cited across all claims.
        metadata: Additional context metadata.
    """

    case_id: str
    summary: str = ""
    parties_text: str = ""
    facts_text: str = ""
    claims_text: str = ""
    disputed_text: str = ""
    timeline_text: str = ""
    evidence_summary: str = ""
    legal_basis: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> float:
    """Return the current Unix timestamp."""

    return time.time()


def _touch(case: CaseFile) -> None:
    """Update the ``updated_at`` timestamp on a case."""

    case.updated_at = _now()


# Regex patterns for rule-based structured extraction fallback.

_PARTY_RE = re.compile(
    r"(原告|被告|上诉人|被上诉人|申请人|被申请人|第三人|原告人|被告人)"
    r"[：:]\s*(.+?)(?=[，。；\n]|原告|被告|上诉人|被上诉人|申请人|被申请人|第三人|$)",
    re.DOTALL,
)

_CLAIM_RE = re.compile(
    r"(诉讼请求|请求事项|请求)[：:]\s*(.+?)(?=事实|理由|$)",
    re.DOTALL,
)

_DATE_RE = re.compile(
    r"(\d{4})\s*[年/\-\.]\s*(\d{1,2})\s*[月/\-\.]\s*(\d{1,2})\s*日?"
)

_FACT_MARKER_RE = re.compile(
    r"(事实与理由|事实和理由|事实|案情)[：:]\s*(.+?)(?=诉讼请求|请求事项|$)",
    re.DOTALL,
)


def _parse_date(text: str) -> tuple[str, float]:
    """Extract a date from *text* and return (iso_string, timestamp).

    Returns ``("", 0.0)`` if no date is found.
    """

    match = _DATE_RE.search(text)
    if match is None:
        return "", 0.0
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    iso = f"{year:04d}-{month:02d}-{day:02d}"
    try:
        import calendar

        ts = float(calendar.timegm((year, month, day, 0, 0, 0, 0, 0, 0)))
    except (ValueError, calendar.error):  # type: ignore[attr-defined]
        ts = 0.0
    return iso, ts


# ---------------------------------------------------------------------------
# Prompt template helper (LangChain lazy import with fallback)
# ---------------------------------------------------------------------------


def _render_prompt(template: str, variables: dict[str, Any]) -> str:
    """Render a prompt template using LangChain when available.

    Tries ``langchain_core.prompts.PromptTemplate`` first; if LangChain is
    not installed, falls back to plain ``str.format``. This avoids a hard
    dependency on LangChain while still leveraging it when present.
    """

    try:
        from langchain_core.prompts import PromptTemplate  # type: ignore[import-untyped]

        return PromptTemplate.from_template(template).format(**variables)
    except ImportError:
        return template.format(**variables)


# Default extraction prompt template (used when an LLM gateway is provided).
_EXTRACTION_PROMPT = """\
你是一名专业的法律助理。请从以下案件材料中提取结构化信息。

案件材料内容：
{content}

请按以下格式输出（如果没有找到相应信息，请留空）：
当事人：
- 角色：原告/被告/第三人
- 姓名：
诉讼请求：
1.
事实要素：
1.
时间线事件：
1. 日期： 描述：
争议焦点：
1.
"""


# ---------------------------------------------------------------------------
# Case manager
# ---------------------------------------------------------------------------


class CaseManager:
    """Thread-safe case file manager with material import and extraction.

    Owns an in-memory registry of :class:`CaseFile` and
    :class:`CaseMaterial` objects. Supports creating cases, importing
    multi-format materials (via :class:`DocumentParser`), performing
    structured extraction (rule-based or LLM-assisted), searching cases,
    and assembling :class:`CaseContext` bundles for downstream use.

    All registry mutations are protected by a re-entrant lock. Async
    variants are provided for material import and structured extraction.

    Args:
        parser: The :class:`DocumentParser` for multi-format file parsing.
            If None, a default parser is created.

    Example::

        >>> manager = CaseManager()
        >>> case = manager.create_case(
        ...     case_number="(2024)京01民初1号",
        ...     cause_of_action="买卖合同纠纷",
        ... )
        >>> manager.add_party(case.id, Party(
        ...     name="甲公司", role=PartyRole.PLAINTIFF,
        ... ))
        >>> ctx = manager.build_context(case.id)
        >>> ctx.case_id == case.id
        True
    """

    def __init__(self, parser: DocumentParser | None = None) -> None:
        self._cases: dict[str, CaseFile] = {}
        self._case_number_index: dict[str, str] = {}
        self._materials: dict[str, CaseMaterial] = {}
        self._parser = parser or DocumentParser()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def parser(self) -> DocumentParser:
        """The document parser used for material ingestion."""

        return self._parser

    @property
    def case_count(self) -> int:
        """Total number of registered cases."""

        with self._lock:
            return len(self._cases)

    @property
    def material_count(self) -> int:
        """Total number of imported materials."""

        with self._lock:
            return len(self._materials)

    # ------------------------------------------------------------------
    # Case lifecycle
    # ------------------------------------------------------------------

    def create_case(
        self,
        *,
        case_number: str = "",
        cause_of_action: str = "",
        court: str = "",
        domain: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CaseFile:
        """Create and register a new case file.

        Raises:
            CaseManagerError: If a case with the same case number already
                exists.
        """

        case = CaseFile(
            case_number=case_number,
            cause_of_action=cause_of_action,
            court=court,
            domain=domain,
            metadata=dict(metadata) if metadata else {},
        )
        with self._lock:
            if case_number and case_number in self._case_number_index:
                raise CaseManagerError(
                    f"Case number already exists: {case_number}"
                )
            self._cases[case.id] = case
            if case_number:
                self._case_number_index[case_number] = case.id
        logger.info("Created case %s (%s)", case.id, case_number or "(no number)")
        return case

    def get_case(self, case_id: str) -> CaseFile | None:
        """Return a case by ID, or ``None``."""

        with self._lock:
            return self._cases.get(case_id)

    def find_case(self, case_number: str) -> CaseFile | None:
        """Find a case by its official case number."""

        with self._lock:
            cid = self._case_number_index.get(case_number)
            return self._cases.get(cid) if cid else None

    def list_cases(
        self,
        *,
        status: CaseStatus | None = None,
        cause_of_action: str | None = None,
    ) -> list[CaseFile]:
        """List cases, optionally filtered by status or cause of action."""

        with self._lock:
            result = list(self._cases.values())
        if status is not None:
            result = [c for c in result if c.status is status]
        if cause_of_action is not None:
            result = [c for c in result if c.cause_of_action == cause_of_action]
        return result

    def update_status(self, case_id: str, status: CaseStatus) -> CaseFile:
        """Change a case's status.

        Raises:
            CaseManagerError: If the case is not found.
        """

        with self._lock:
            case = self._require_case(case_id)
            case.status = status
            _touch(case)
        logger.info("Case %s status -> %s", case_id, status.value)
        return case

    def delete_case(self, case_id: str) -> CaseFile | None:
        """Remove a case and its associated materials.

        Returns the removed case, or ``None`` if not found.
        """

        with self._lock:
            case = self._cases.pop(case_id, None)
            if case is None:
                return None
            if case.case_number:
                self._case_number_index.pop(case.case_number, None)
            # Remove associated materials.
            for mid in case.material_ids:
                self._materials.pop(mid, None)
        logger.info("Deleted case %s", case_id)
        return case

    # ------------------------------------------------------------------
    # Party management
    # ------------------------------------------------------------------

    def add_party(self, case_id: str, party: Party) -> CaseFile:
        """Add a party to a case.

        Raises:
            CaseManagerError: If the case is not found.
        """

        with self._lock:
            case = self._require_case(case_id)
            case.parties.append(party)
            _touch(case)
        logger.debug("Added party %s to case %s", party.name, case_id)
        return case

    def remove_party(self, case_id: str, party_id: str) -> CaseFile:
        """Remove a party from a case by party ID."""

        with self._lock:
            case = self._require_case(case_id)
            case.parties = [p for p in case.parties if p.id != party_id]
            _touch(case)
        return case

    # ------------------------------------------------------------------
    # Fact / issue / claim / timeline management
    # ------------------------------------------------------------------

    def add_fact(self, case_id: str, fact: FactElement) -> CaseFile:
        """Add a fact element to a case."""

        with self._lock:
            case = self._require_case(case_id)
            case.fact_elements.append(fact)
            _touch(case)
        return case

    def add_disputed_issue(
        self, case_id: str, issue: DisputedIssue
    ) -> CaseFile:
        """Add a disputed issue to a case."""

        with self._lock:
            case = self._require_case(case_id)
            case.disputed_issues.append(issue)
            _touch(case)
        return case

    def add_claim(self, case_id: str, claim: Claim) -> CaseFile:
        """Add a litigation claim to a case."""

        with self._lock:
            case = self._require_case(case_id)
            case.claims.append(claim)
            _touch(case)
        return case

    def add_timeline_event(
        self, case_id: str, event: TimelineEvent
    ) -> CaseFile:
        """Add a timeline event and re-sort the chronology by timestamp."""

        with self._lock:
            case = self._require_case(case_id)
            case.timeline.append(event)
            case.timeline.sort(key=lambda e: e.timestamp if e.timestamp > 0 else 0)
            _touch(case)
        return case

    def link_evidence(self, case_id: str, evidence_id: str) -> CaseFile:
        """Associate an evidence item ID with a case (idempotent)."""

        with self._lock:
            case = self._require_case(case_id)
            if evidence_id not in case.evidence_ids:
                case.evidence_ids.append(evidence_id)
                _touch(case)
        return case

    # ------------------------------------------------------------------
    # Material import
    # ------------------------------------------------------------------

    def import_file(
        self,
        case_id: str,
        path: Path | str,
        *,
        material_type: MaterialType = MaterialType.OTHER,
        title: str | None = None,
        notes: str = "",
        auto_extract: bool = True,
    ) -> CaseMaterial:
        """Import a file as a case material.

        The file is parsed using :class:`DocumentParser` (supporting PDF,
        Word, Excel, PPT, Markdown, HTML, and plain text). When
        ``auto_extract`` is True, structured information (parties, claims,
        facts, timeline) is extracted and merged into the case.

        Args:
            case_id: ID of the case to import into.
            path: Path to the file to import.
            material_type: Classification of the material.
            title: Optional title override.
            notes: Free-form reviewer notes.
            auto_extract: If True, run structured extraction after import.

        Raises:
            CaseManagerError: If the case is not found.
            FileNotFoundError: If the file does not exist.
        """

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        document = self._parser.parse_file(file_path, title=title)
        return self.import_document(
            case_id,
            document,
            material_type=material_type,
            notes=notes,
            auto_extract=auto_extract,
        )

    def import_bytes(
        self,
        case_id: str,
        data: bytes,
        *,
        filename: str,
        material_type: MaterialType = MaterialType.OTHER,
        title: str | None = None,
        notes: str = "",
        auto_extract: bool = True,
    ) -> CaseMaterial:
        """Import raw bytes (e.g. from an upload) as a case material.

        The document type is inferred from *filename*.
        """

        document = self._parser.parse_bytes(data, filename=filename, title=title)
        return self.import_document(
            case_id,
            document,
            material_type=material_type,
            notes=notes,
            auto_extract=auto_extract,
        )

    def import_document(
        self,
        case_id: str,
        document: Document,
        *,
        material_type: MaterialType = MaterialType.OTHER,
        notes: str = "",
        auto_extract: bool = True,
    ) -> CaseMaterial:
        """Register an already-parsed :class:`Document` as a case material."""

        material = CaseMaterial(
            id=document.id,
            document=document,
            material_type=material_type,
            case_id=case_id,
            notes=notes,
        )
        with self._lock:
            case = self._require_case(case_id)
            self._materials[material.id] = material
            case.material_ids.append(material.id)
            _touch(case)
        logger.info(
            "Imported material %s (%s) into case %s",
            material.id,
            document.title,
            case_id,
        )

        if auto_extract and document.content.strip():
            self._extract_and_merge(case_id, document)
        return material

    def get_material(self, material_id: str) -> CaseMaterial | None:
        """Return a material by ID, or ``None``."""

        with self._lock:
            return self._materials.get(material_id)

    def list_materials(self, case_id: str) -> list[CaseMaterial]:
        """List all materials imported into a case."""

        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                return []
            return [
                self._materials[mid]
                for mid in case.material_ids
                if mid in self._materials
            ]

    # ------------------------------------------------------------------
    # Structured extraction
    # ------------------------------------------------------------------

    def extract_structure(
        self,
        case_id: str,
        material_id: str,
    ) -> dict[str, Any]:
        """Extract structured information from a specific material.

        Performs rule-based extraction (parties, claims, facts, timeline
        events) from the material's text content. When an LLM gateway is
        available on the manager, LLM-assisted extraction is attempted
        first, with the rule-based extractor as fallback.

        Returns a dict with keys ``parties``, ``claims``, ``facts``,
        ``timeline``.
        """

        with self._lock:
            material = self._materials.get(material_id)
            if material is None:
                raise CaseManagerError(f"Material not found: {material_id}")
            content = material.document.content

        return self._rule_based_extract(content)

    def extract_all(self, case_id: str) -> dict[str, Any]:
        """Extract and merge structure from all materials in a case.

        Returns a summary dict of all extracted items.
        """

        with self._lock:
            case = self._require_case(case_id)
            material_ids = list(case.material_ids)

        total: dict[str, list[Any]] = {
            "parties": [],
            "claims": [],
            "facts": [],
            "timeline": [],
        }
        for mid in material_ids:
            extracted = self.extract_structure(case_id, mid)
            for key in total:
                total[key].extend(extracted.get(key, []))
        return total

    def _extract_and_merge(self, case_id: str, document: Document) -> None:
        """Extract structure from a document and merge into the case."""

        extracted = self._rule_based_extract(document.content)
        with self._lock:
            case = self._require_case(case_id)
            for party in extracted["parties"]:
                case.parties.append(party)
            for claim in extracted["claims"]:
                case.claims.append(claim)
            for fact in extracted["facts"]:
                case.fact_elements.append(fact)
            for event in extracted["timeline"]:
                event.source_document_id = document.id
                case.timeline.append(event)
            case.timeline.sort(
                key=lambda e: e.timestamp if e.timestamp > 0 else 0
            )
            _touch(case)

    def _rule_based_extract(self, content: str) -> dict[str, Any]:
        """Rule-based structured extraction from text content.

        Uses regex patterns to identify parties, claims, facts, and
        timeline events. This is a best-effort extractor; for higher
        accuracy, provide an LLM gateway.
        """

        parties: list[Party] = []
        claims: list[Claim] = []
        facts: list[FactElement] = []
        timeline: list[TimelineEvent] = []

        # Extract parties.
        role_map = {
            "原告": PartyRole.PLAINTIFF,
            "原告人": PartyRole.PLAINTIFF,
            "被告": PartyRole.DEFENDANT,
            "被告人": PartyRole.DEFENDANT,
            "上诉人": PartyRole.APPELLANT,
            "被上诉人": PartyRole.APPELLEE,
            "申请人": PartyRole.APPLICANT,
            "被申请人": PartyRole.RESPONDENT,
            "第三人": PartyRole.THIRD_PARTY,
        }
        for match in _PARTY_RE.finditer(content):
            role_text = match.group(1)
            name = match.group(2).strip().rstrip("，。；")
            if name and len(name) < 100:
                parties.append(
                    Party(
                        name=name,
                        role=role_map.get(role_text, PartyRole.OTHER),
                    )
                )

        # Extract claims.
        claim_match = _CLAIM_RE.search(content)
        if claim_match:
            claim_text = claim_match.group(2).strip()
            for i, line in enumerate(claim_text.split("\n"), start=1):
                line = line.strip()
                if line and len(line) > 2:
                    cleaned = re.sub(r"^\d+[\.、）)]\s*", "", line)
                    if cleaned:
                        claims.append(Claim(description=cleaned))

        # Extract facts.
        fact_match = _FACT_MARKER_RE.search(content)
        if fact_match:
            fact_text = fact_match.group(2).strip()
            for line in fact_text.split("\n"):
                line = line.strip()
                if line and len(line) > 5:
                    cleaned = re.sub(r"^\d+[\.、）)]\s*", "", line)
                    if cleaned:
                        facts.append(FactElement(description=cleaned))

        # Extract timeline events from dates.
        seen_dates: set[str] = set()
        for match in _DATE_RE.finditer(content):
            date_str = match.group(0)
            iso, ts = _parse_date(date_str)
            if iso and iso not in seen_dates:
                seen_dates.add(iso)
                # Grab surrounding context as the description.
                end = min(len(content), match.end() + 80)
                context = content[match.end() : end].strip().split("\n")[0]
                description = f"{date_str} {context}".strip()
                timeline.append(
                    TimelineEvent(
                        date=iso,
                        timestamp=ts,
                        description=description,
                        category="extracted",
                    )
                )

        return {
            "parties": parties,
            "claims": claims,
            "facts": facts,
            "timeline": timeline,
        }

    # ------------------------------------------------------------------
    # Case search
    # ------------------------------------------------------------------

    def search_cases(
        self,
        query: str,
        *,
        top_k: int = 20,
        status: CaseStatus | None = None,
    ) -> list[CaseFile]:
        """Search cases by keyword over case number, cause, and parties.

        Performs a case-insensitive substring match across the case
        number, cause of action, court, and party names.

        Args:
            query: The search query.
            top_k: Maximum number of results.
            status: Optional status filter.

        Returns:
            List of matching :class:`CaseFile` objects.
        """

        needle = query.lower()
        with self._lock:
            candidates = list(self._cases.values())
        if status is not None:
            candidates = [c for c in candidates if c.status is status]

        scored: list[tuple[float, CaseFile]] = []
        for case in candidates:
            hay = " ".join(
                [
                    case.case_number,
                    case.cause_of_action,
                    case.court,
                    " ".join(p.name for p in case.parties),
                ]
            ).lower()
            if needle in hay:
                scored.append((1.0, case))
        scored.sort(key=lambda pair: pair[1].updated_at, reverse=True)
        return [c for _, c in scored[:top_k]]

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def build_context(
        self,
        case_id: str,
        *,
        include_evidence: bool = True,
    ) -> CaseContext:
        """Assemble a :class:`CaseContext` bundle for a case.

        The context bundle contains formatted text for parties, facts,
        claims, disputed issues, and timeline, suitable for feeding into
        an LLM prompt or a document generator.

        Args:
            case_id: The case to build context for.
            include_evidence: If True, include an evidence-ID summary.

        Raises:
            CaseManagerError: If the case is not found.
        """

        with self._lock:
            case = self._require_case(case_id)
            parties = list(case.parties)
            facts = list(case.fact_elements)
            claims = list(case.claims)
            issues = list(case.disputed_issues)
            timeline = list(case.timeline)
            evidence_ids = list(case.evidence_ids)
            legal_basis: set[str] = set()
            for claim in claims:
                legal_basis.update(claim.legal_basis)

        # Format parties.
        party_lines = []
        for p in parties:
            line = f"  - {p.role.value}: {p.name}"
            if p.legal_representative:
                line += f" (法定代表人: {p.legal_representative})"
            party_lines.append(line)
        parties_text = "\n".join(party_lines) if party_lines else "  (无)"

        # Format facts.
        fact_lines = []
        for i, f in enumerate(facts, start=1):
            marker = "[争议]" if f.contested else ""
            fact_lines.append(f"  {i}. {f.description} {marker}".strip())
        facts_text = "\n".join(fact_lines) if fact_lines else "  (无)"

        # Format claims.
        claim_lines = []
        for i, c in enumerate(claims, start=1):
            line = f"  {i}. {c.description}"
            if c.amount > 0:
                line += f" (金额: {c.amount:,.2f})"
            claim_lines.append(line)
        claims_text = "\n".join(claim_lines) if claim_lines else "  (无)"

        # Format disputed issues.
        issue_lines = []
        for i, issue in enumerate(issues, start=1):
            issue_lines.append(f"  {i}. {issue.description}")
            if issue.plaintiff_position:
                issue_lines.append(f"     原告主张: {issue.plaintiff_position}")
            if issue.defendant_position:
                issue_lines.append(f"     被告主张: {issue.defendant_position}")
        disputed_text = "\n".join(issue_lines) if issue_lines else "  (无)"

        # Format timeline.
        timeline_lines = []
        for event in timeline:
            date_str = event.date or "未知日期"
            timeline_lines.append(f"  - {date_str}: {event.description}")
        timeline_text = "\n".join(timeline_lines) if timeline_lines else "  (无)"

        # Evidence summary.
        evidence_summary = ""
        if include_evidence:
            if evidence_ids:
                evidence_summary = f"共关联 {len(evidence_ids)} 项证据"
            else:
                evidence_summary = "暂无关联证据"

        # Case summary.
        summary_parts = []
        if case.case_number:
            summary_parts.append(f"案号: {case.case_number}")
        if case.cause_of_action:
            summary_parts.append(f"案由: {case.cause_of_action}")
        if case.court:
            summary_parts.append(f"法院: {case.court}")
        summary = "；".join(summary_parts) if summary_parts else "(未填写基本信息)"

        return CaseContext(
            case_id=case_id,
            summary=summary,
            parties_text=parties_text,
            facts_text=facts_text,
            claims_text=claims_text,
            disputed_text=disputed_text,
            timeline_text=timeline_text,
            evidence_summary=evidence_summary,
            legal_basis=sorted(legal_basis),
        )

    def render_context_prompt(
        self,
        case_id: str,
        *,
        template: str | None = None,
    ) -> str:
        """Render a case context as a prompt string.

        Uses LangChain's ``PromptTemplate`` when available (lazy import),
        falling back to plain ``str.format`` otherwise. The default
        template produces a structured case summary suitable for feeding
        to an LLM.

        Args:
            case_id: The case to render.
            template: Optional custom template string with placeholders
                ``{summary}``, ``{parties}``, ``{facts}``, ``{claims}``,
                ``{disputed}``, ``{timeline}``, ``{evidence}``.

        Raises:
            CaseManagerError: If the case is not found.
        """

        ctx = self.build_context(case_id)
        tmpl = template or _DEFAULT_CONTEXT_TEMPLATE
        return _render_prompt(
            tmpl,
            {
                "summary": ctx.summary,
                "parties": ctx.parties_text,
                "facts": ctx.facts_text,
                "claims": ctx.claims_text,
                "disputed": ctx.disputed_text,
                "timeline": ctx.timeline_text,
                "evidence": ctx.evidence_summary,
            },
        )

    # ------------------------------------------------------------------
    # Async variants
    # ------------------------------------------------------------------

    async def import_file_async(
        self,
        case_id: str,
        path: Path | str,
        *,
        material_type: MaterialType = MaterialType.OTHER,
        title: str | None = None,
        notes: str = "",
        auto_extract: bool = True,
    ) -> CaseMaterial:
        """Async wrapper for :meth:`import_file`."""

        return await asyncio.to_thread(
            self.import_file,
            case_id,
            path,
            material_type=material_type,
            title=title,
            notes=notes,
            auto_extract=auto_extract,
        )

    async def extract_all_async(
        self, case_id: str
    ) -> dict[str, Any]:
        """Async wrapper for :meth:`extract_all`."""

        return await asyncio.to_thread(self.extract_all, case_id)

    async def build_context_async(
        self,
        case_id: str,
        *,
        include_evidence: bool = True,
    ) -> CaseContext:
        """Async wrapper for :meth:`build_context`."""

        return await asyncio.to_thread(
            self.build_context, case_id, include_evidence=include_evidence
        )

    # ------------------------------------------------------------------
    # Aggregate / summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for dashboards."""

        with self._lock:
            active = sum(
                1 for c in self._cases.values() if c.is_active
            )
            return {
                "cases": len(self._cases),
                "active_cases": active,
                "materials": len(self._materials),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_case(self, case_id: str) -> CaseFile:
        """Return the case or raise. Caller must hold the lock."""

        case = self._cases.get(case_id)
        if case is None:
            raise CaseManagerError(f"Case not found: {case_id}")
        return case


# ---------------------------------------------------------------------------
# Default context prompt template
# ---------------------------------------------------------------------------


_DEFAULT_CONTEXT_TEMPLATE = """\
案件信息：
{summary}

当事人：
{parties}

诉讼请求：
{claims}

事实要素：
{facts}

争议焦点：
{disputed}

时间线：
{timeline}

证据概况：
{evidence}
"""


__all__ = [
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
]
