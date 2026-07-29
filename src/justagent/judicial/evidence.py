"""Evidence review — admissibility, relevance, probative value, and chain analysis.

Provides structured evidence management for the judicial platform. Each
piece of evidence is modelled as an :class:`Evidence` object with its
type (documentary, physical, testimony, expert opinion, inspection
record, audio-visual, electronic data), source, proving object, and
probative-value assessment. The :class:`EvidenceReviewer` performs
legality (admissibility) checks, relevance assessment, and probative-
value rating, while the :class:`EvidenceChain` analyses the
completeness of the evidence chain, detects contradictions between
evidence items, and identifies evidentiary gaps.

Integration with :mod:`justagent.knowledge.graph` allows evidence items
and their proving objects to be represented as entities in a
:class:`KnowledgeGraph`, with ``supports`` / ``contradicts`` /
``corroborates`` relations forming the evidentiary graph. This enables
graph-based traversal of the evidence chain and visualisation of
support/contradiction relationships.

Design:

* :class:`EvidenceType` — the seven statutory evidence categories under
  PRC procedural law.
* :class:`Admissibility` / :class:`ProbativeStrength` — typed
  assessment results.
* :class:`Evidence` — the central Pydantic model.
* :class:`EvidenceRelation` — a typed edge between two evidence items.
* :class:`ChainAnalysisResult` — the output of evidence-chain analysis.
* :class:`ReviewResult` — the output of a single-evidence review.
* :class:`EvidenceChain` — completeness / contradiction / gap analysis.
* :class:`EvidenceReviewer` — legality / relevance / probative-value
  review (thread-safe, async-capable).
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
    Entity,
    EntityType,
    KnowledgeGraph,
)
from justagent.utils import now

logger = logging.getLogger("justagent.judicial.evidence")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EvidenceError(Exception):
    """Raised for invalid evidence operations."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvidenceType(str, Enum):  # noqa: UP042 - match existing codebase style
    """Statutory evidence categories under PRC procedural law.

    Attributes:
        DOCUMENTARY: Documentary evidence (书证) — written materials
            whose content proves a fact.
        PHYSICAL: Physical evidence (物证) — tangible objects whose
            external features or attributes prove a fact.
        TESTIMONY: Witness testimony (证人证言) — statements by
            witnesses regarding facts they have perceived.
        EXPERT_OPINION: Expert opinion (鉴定意见) — specialised
            analysis by qualified experts.
        INSPECTION_RECORD: Inspection record (勘验笔录) — records made
            by investigators at a scene or of an object.
        AUDIO_VISUAL: Audio-visual material (视听资料) — recordings,
            photographs, video.
        ELECTRONIC_DATA: Electronic data (电子数据) — digitally stored
            or transmitted information.
    """

    DOCUMENTARY = "documentary"
    PHYSICAL = "physical"
    TESTIMONY = "testimony"
    EXPERT_OPINION = "expert_opinion"
    INSPECTION_RECORD = "inspection_record"
    AUDIO_VISUAL = "audio_visual"
    ELECTRONIC_DATA = "electronic_data"


class Admissibility(str, Enum):  # noqa: UP042
    """Legality / admissibility assessment result.

    Attributes:
        ADMISSIBLE: The evidence is legally obtained and admissible.
        INADMISSIBLE: The evidence violates legality requirements and
            must be excluded.
        CONDITIONAL: Admissibility depends on supplementation or
            correction (e.g. missing signature, incomplete chain of
            custody).
    """

    ADMISSIBLE = "admissible"
    INADMISSIBLE = "inadmissible"
    CONDITIONAL = "conditional"


class ProbativeStrength(str, Enum):  # noqa: UP042
    """Probative-value strength rating.

    Attributes:
        HIGH: Strong probative value — directly and conclusively proves
            the target fact.
        MEDIUM: Moderate probative value — proves the target fact with
            some support needed from other evidence.
        LOW: Weak probative value — provides circumstantial or partial
            support.
        INSUFFICIENT: Insufficient probative value — does not meaningfully
            prove the target fact.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class EvidenceRelationType(str, Enum):  # noqa: UP042
    """Types of relations between evidence items.

    Attributes:
        SUPPORTS: Evidence A directly supports the same conclusion as
            evidence B.
        CORROBORATES: Evidence B independently confirms what evidence A
            establishes.
        CONTRADICTS: Evidence A and B lead to mutually exclusive
            conclusions.
        SUPPLEMENTS: Evidence B adds detail to what evidence A
            establishes.
    """

    SUPPORTS = "supports"
    CORROBORATES = "corroborates"
    CONTRADICTS = "contradicts"
    SUPPLEMENTS = "supplements"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A single piece of evidence in a case.

    Attributes:
        id: Unique evidence identifier (auto-generated UUID4 hex).
        name: Short descriptive name (e.g. ``"购销合同原件"``).
        type: The :class:`EvidenceType` category.
        description: Detailed description of the evidence.
        source: How / where the evidence was obtained.
        collector: Name of the person who collected the evidence.
        collection_date: Date of collection (ISO ``YYYY-MM-DD``).
        collection_method: Method of collection (e.g. ``"扣押"``,
            ``"调取"``, ``"当事人提供"``).
        proving_object: What fact the evidence is intended to prove
            (证明对象).
        proving_target: The specific proposition the evidence supports
            (证明目的).
        case_id: ID of the case this evidence belongs to.
        source_document_id: ID of the imported document (if any).
        admissibility: Legality assessment result.
        admissibility_issues: List of legality problems identified.
        relevance_score: Relevance assessment in ``[0, 1]``.
        probative_strength: Probative-value rating.
        probative_score: Numeric probative score in ``[0, 1]``.
        metadata: Arbitrary key-value metadata.
        created_at: Unix timestamp of creation.
        reviewed: Whether this evidence has been reviewed.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    type: EvidenceType = EvidenceType.DOCUMENTARY
    description: str = ""
    source: str = ""
    collector: str = ""
    collection_date: str = ""
    collection_method: str = ""
    proving_object: str = ""
    proving_target: str = ""
    case_id: str = ""
    source_document_id: str = ""
    admissibility: Admissibility = Admissibility.ADMISSIBLE
    admissibility_issues: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    probative_strength: ProbativeStrength = ProbativeStrength.MEDIUM
    probative_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=now)
    reviewed: bool = False

    @property
    def is_admissible(self) -> bool:
        """True if the evidence is admissible (not inadmissible)."""

        return self.admissibility is not Admissibility.INADMISSIBLE

    @property
    def is_excluded(self) -> bool:
        """True if the evidence has been excluded as inadmissible."""

        return self.admissibility is Admissibility.INADMISSIBLE


class EvidenceRelation(BaseModel):
    """A typed relation between two evidence items.

    Attributes:
        id: Unique relation identifier.
        evidence_a_id: ID of the first evidence item.
        evidence_b_id: ID of the second evidence item.
        relation_type: The :class:`EvidenceRelationType`.
        description: Human-readable explanation of the relation.
        weight: Confidence weight in ``[0, 1]``.
        metadata: Arbitrary key-value metadata.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    evidence_a_id: str
    evidence_b_id: str
    relation_type: EvidenceRelationType = EvidenceRelationType.SUPPORTS
    description: str = ""
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewResult(BaseModel):
    """The result of reviewing a single evidence item.

    Attributes:
        evidence_id: The reviewed evidence ID.
        is_legal: Whether the evidence passes the legality check.
        legality_issues: List of identified legality problems.
        admissibility: The assessed :class:`Admissibility`.
        relevance_score: Relevance score in ``[0, 1]``.
        relevance_reasoning: Explanation of the relevance assessment.
        probative_strength: The assessed :class:`ProbativeStrength`.
        probative_score: Numeric probative score in ``[0, 1]``.
        probative_reasoning: Explanation of the probative-value rating.
        recommendations: Actionable recommendations.
    """

    evidence_id: str
    is_legal: bool = True
    legality_issues: list[str] = Field(default_factory=list)
    admissibility: Admissibility = Admissibility.ADMISSIBLE
    relevance_score: float = 0.0
    relevance_reasoning: str = ""
    probative_strength: ProbativeStrength = ProbativeStrength.MEDIUM
    probative_score: float = 0.0
    probative_reasoning: str = ""
    recommendations: list[str] = Field(default_factory=list)


class ChainAnalysisResult(BaseModel):
    """The result of evidence-chain analysis.

    Attributes:
        case_id: The analysed case ID.
        completeness_score: Overall chain completeness in ``[0, 1]``.
        total_evidence: Total number of evidence items analysed.
        admissible_evidence: Number of admissible items.
        contradictions: List of contradicting evidence pairs.
        gaps: List of identified evidentiary gaps.
        supporting_relations: Count of support/corroborate relations.
        summary: Human-readable summary of the analysis.
    """

    case_id: str
    completeness_score: float = 0.0
    total_evidence: int = 0
    admissible_evidence: int = 0
    contradictions: list[EvidenceRelation] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    supporting_relations: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of terms using jieba for Chinese.

    Falls back to whitespace splitting when jieba is not installed.
    Filters out single-character tokens and common stop words to
    improve matching precision.
    """

    try:
        import jieba

        tokens = set(jieba.cut(text))
    except ImportError:
        tokens = set(text.split())

    # Remove empty strings, single punctuation, and common stop words.
    stop_words = {"的", "了", "和", "是", "在", "有", "与", "对", "为", "及", "或"}
    return {t for t in tokens if len(t) >= 2 and t not in stop_words}


def _probative_strength_from_score(score: float) -> ProbativeStrength:
    """Map a numeric probative score to a :class:`ProbativeStrength`."""

    if score >= 0.75:
        return ProbativeStrength.HIGH
    if score >= 0.5:
        return ProbativeStrength.MEDIUM
    if score >= 0.25:
        return ProbativeStrength.LOW
    return ProbativeStrength.INSUFFICIENT


# Legality check rules: (field_name, issue_description).
# Each rule checks a required field for admissibility.
_LEGALITY_RULES: list[tuple[str, str]] = [
    ("source", "证据来源不明"),
    ("collector", "收集人信息缺失"),
    ("collection_method", "收集方式未注明"),
]


# ---------------------------------------------------------------------------
# Evidence chain
# ---------------------------------------------------------------------------


class EvidenceChain:
    """Evidence-chain analyser — completeness, contradictions, gaps.

    Operates on a collection of :class:`Evidence` items and their
    :class:`EvidenceRelation` edges. When a :class:`KnowledgeGraph` is
    provided, evidence items and proving objects are also added as graph
    entities, enabling graph-based chain traversal and visualisation.

    Thread-safe via a re-entrant lock.

    Args:
        knowledge_graph: Optional :class:`KnowledgeGraph` for graph-based
            chain representation.

    Example::

        >>> chain = EvidenceChain()
        >>> chain.add_evidence(Evidence(name="合同", type=EvidenceType.DOCUMENTARY,
        ...     proving_object="合同关系成立"))
        >>> chain.add_evidence(Evidence(name="收据", type=EvidenceType.DOCUMENTARY,
        ...     proving_object="付款事实"))
        >>> chain.add_relation(evidence_a_id=chain.list_evidence()[0].id,
        ...     evidence_b_id=chain.list_evidence()[1].id,
        ...     relation_type=EvidenceRelationType.CORROBORATES)
        >>> result = chain.analyze("case_1")
        >>> result.completeness_score > 0
        True
    """

    def __init__(self, knowledge_graph: KnowledgeGraph | None = None) -> None:
        self._evidence: dict[str, Evidence] = {}
        self._relations: dict[str, EvidenceRelation] = {}
        self._graph = knowledge_graph
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def knowledge_graph(self) -> KnowledgeGraph | None:
        """The backing knowledge graph, or ``None``."""

        return self._graph

    @property
    def evidence_count(self) -> int:
        """Total number of evidence items."""

        with self._lock:
            return len(self._evidence)

    @property
    def relation_count(self) -> int:
        """Total number of evidence relations."""

        with self._lock:
            return len(self._relations)

    # ------------------------------------------------------------------
    # Evidence management
    # ------------------------------------------------------------------

    def add_evidence(self, evidence: Evidence) -> Evidence:
        """Register an evidence item.

        If a knowledge graph is configured, the evidence and its proving
        object are added as graph entities with a ``proves`` relation.
        """

        with self._lock:
            self._evidence[evidence.id] = evidence
            self._index_in_graph(evidence)
        logger.debug("Added evidence %s (%s)", evidence.id, evidence.name)
        return evidence

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Return an evidence item by ID, or ``None``."""

        with self._lock:
            return self._evidence.get(evidence_id)

    def list_evidence(
        self, *, case_id: str | None = None
    ) -> list[Evidence]:
        """List evidence items, optionally filtered by case."""

        with self._lock:
            result = list(self._evidence.values())
        if case_id is not None:
            result = [e for e in result if e.case_id == case_id]
        return result

    def remove_evidence(self, evidence_id: str) -> Evidence | None:
        """Remove an evidence item and its relations."""

        with self._lock:
            evidence = self._evidence.pop(evidence_id, None)
            if evidence is None:
                return None
            # Remove relations involving this evidence.
            to_remove = [
                rid
                for rid, rel in self._relations.items()
                if rel.evidence_a_id == evidence_id
                or rel.evidence_b_id == evidence_id
            ]
            for rid in to_remove:
                self._relations.pop(rid, None)
        return evidence

    # ------------------------------------------------------------------
    # Relation management
    # ------------------------------------------------------------------

    def add_relation(
        self,
        evidence_a_id: str,
        evidence_b_id: str,
        relation_type: EvidenceRelationType = EvidenceRelationType.SUPPORTS,
        *,
        description: str = "",
        weight: float = 1.0,
    ) -> EvidenceRelation:
        """Add a relation between two evidence items.

        Raises:
            EvidenceError: If either evidence ID is not registered.
        """

        with self._lock:
            if evidence_a_id not in self._evidence:
                raise EvidenceError(f"Evidence not found: {evidence_a_id}")
            if evidence_b_id not in self._evidence:
                raise EvidenceError(f"Evidence not found: {evidence_b_id}")
            relation = EvidenceRelation(
                evidence_a_id=evidence_a_id,
                evidence_b_id=evidence_b_id,
                relation_type=relation_type,
                description=description,
                weight=weight,
            )
            self._relations[relation.id] = relation

            # Also add to the knowledge graph if configured.
            if self._graph is not None:
                try:
                    self._graph.add_relation(
                        evidence_a_id,
                        evidence_b_id,
                        relation_type.value,
                        weight=weight,
                        metadata={"description": description},
                    )
                except KeyError as exc:
                    logger.debug("Entity not found in knowledge graph, skipping relation: %s", exc)
        return relation

    def list_relations(
        self,
        *,
        relation_type: EvidenceRelationType | None = None,
        evidence_id: str | None = None,
    ) -> list[EvidenceRelation]:
        """List relations, optionally filtered by type or evidence."""

        with self._lock:
            result = list(self._relations.values())
        if relation_type is not None:
            result = [r for r in result if r.relation_type is relation_type]
        if evidence_id is not None:
            result = [
                r
                for r in result
                if r.evidence_a_id == evidence_id
                or r.evidence_b_id == evidence_id
            ]
        return result

    # ------------------------------------------------------------------
    # Chain analysis
    # ------------------------------------------------------------------

    def analyze(self, case_id: str = "") -> ChainAnalysisResult:
        """Perform full evidence-chain analysis.

        Computes a completeness score, detects contradicting evidence
        pairs, and identifies evidentiary gaps (proving objects without
        supporting evidence).

        Args:
            case_id: The case to analyse. If empty, all evidence is used.

        Returns:
            A :class:`ChainAnalysisResult`.
        """

        with self._lock:
            if case_id:
                evidence_list = [
                    e for e in self._evidence.values() if e.case_id == case_id
                ]
            else:
                evidence_list = list(self._evidence.values())
            relations = list(self._relations.values())

        total = len(evidence_list)
        if total == 0:
            return ChainAnalysisResult(
                case_id=case_id,
                completeness_score=0.0,
                summary="无证据可供分析。",
            )

        admissible = sum(1 for e in evidence_list if e.is_admissible)

        # Detect contradictions.
        contradictions = [
            r for r in relations if r.relation_type is EvidenceRelationType.CONTRADICTS
        ]

        # Count supporting relations.
        supporting = sum(
            1
            for r in relations
            if r.relation_type
            in (EvidenceRelationType.SUPPORTS, EvidenceRelationType.CORROBORATES)
        )

        # Identify gaps: proving objects without any admissible evidence.
        proving_objects: dict[str, list[Evidence]] = {}
        for e in evidence_list:
            if e.proving_object:
                proving_objects.setdefault(e.proving_object, []).append(e)
        gaps: list[str] = []
        for obj, items in proving_objects.items():
            has_admissible = any(e.is_admissible for e in items)
            if not has_admissible:
                gaps.append(f"证明对象「{obj}」无可采信证据")

        # Completeness score: ratio of admissible evidence with non-empty
        # proving objects, penalised by contradictions and gaps.
        admissible_with_object = sum(
            1
            for e in evidence_list
            if e.is_admissible and e.proving_object
        )
        base_score = admissible_with_object / total if total > 0 else 0.0
        penalty = min(0.3, len(contradictions) * 0.1 + len(gaps) * 0.05)
        completeness = max(0.0, base_score - penalty)

        # Build summary.
        summary_parts = [
            f"共 {total} 项证据，其中 {admissible} 项可采信。",
            f"发现 {len(contradictions)} 处矛盾，{len(gaps)} 处缺口。",
            f"证据链完整度: {completeness:.1%}。",
        ]
        if gaps:
            summary_parts.append("需补充: " + "; ".join(gaps))

        return ChainAnalysisResult(
            case_id=case_id,
            completeness_score=round(completeness, 4),
            total_evidence=total,
            admissible_evidence=admissible,
            contradictions=contradictions,
            gaps=gaps,
            supporting_relations=supporting,
            summary=" ".join(summary_parts),
        )

    async def analyze_async(self, case_id: str = "") -> ChainAnalysisResult:
        """Async wrapper for :meth:`analyze`."""

        return await asyncio.to_thread(self.analyze, case_id)

    # ------------------------------------------------------------------
    # Graph integration
    # ------------------------------------------------------------------

    def get_evidence_graph_entities(self) -> list[Entity]:
        """Return all evidence-related entities from the knowledge graph.

        Returns an empty list if no knowledge graph is configured.
        """

        if self._graph is None:
            return []
        return [
            e
            for e in self._graph.list_entities()
            if e.entity_type == EntityType.CONCEPT.value
            and e.metadata.get("category") == "evidence"
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index_in_graph(self, evidence: Evidence) -> None:
        """Add the evidence and its proving object to the knowledge graph."""

        if self._graph is None:
            return
        # Add the evidence as an entity.
        ev_entity = self._graph.add_entity(
            name=evidence.name,
            entity_type=EntityType.CONCEPT.value,
            aliases=[],
            metadata={
                "category": "evidence",
                "evidence_type": evidence.type.value,
                "evidence_id": evidence.id,
                "case_id": evidence.case_id,
            },
            source_documents=[evidence.source_document_id]
            if evidence.source_document_id
            else [],
        )
        # Add the proving object as an entity (if specified).
        if evidence.proving_object:
            obj_entity = self._graph.add_entity(
                name=evidence.proving_object,
                entity_type=EntityType.CONCEPT.value,
                metadata={
                    "category": "proving_object",
                    "case_id": evidence.case_id,
                },
            )
            try:
                self._graph.add_relation(
                    ev_entity.id,
                    obj_entity.id,
                    "proves",
                    weight=evidence.probative_score,
                    source_documents=[evidence.source_document_id]
                    if evidence.source_document_id
                    else [],
                )
            except KeyError as exc:
                logger.debug("Entity not found in knowledge graph, skipping relation: %s", exc)


# ---------------------------------------------------------------------------
# Evidence reviewer
# ---------------------------------------------------------------------------


class EvidenceReviewer:
    """Evidence reviewer — legality, relevance, and probative-value assessment.

    Reviews individual :class:`Evidence` items against legality rules
    (source, collector, collection method), assesses relevance to the
    proving object, and assigns a probative-value rating. The reviewer
    is thread-safe and provides async variants for batch review.

    Args:
        evidence_chain: The :class:`EvidenceChain` that owns the evidence
            items to review. If None, a new chain is created.

    Example::

        >>> chain = EvidenceChain()
        >>> reviewer = EvidenceReviewer(chain)
        >>> ev = Evidence(name="合同", type=EvidenceType.DOCUMENTARY,
        ...     source="当事人提供", collector="张律师",
        ...     collection_method="当事人提供",
        ...     proving_object="合同关系成立")
        >>> chain.add_evidence(ev)
        >>> result = reviewer.review(ev.id)
        >>> result.is_legal
        True
    """

    def __init__(self, evidence_chain: EvidenceChain | None = None) -> None:
        self._chain = evidence_chain or EvidenceChain()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chain(self) -> EvidenceChain:
        """The underlying evidence chain."""

        return self._chain

    # ------------------------------------------------------------------
    # Legality review
    # ------------------------------------------------------------------

    def review_legality(self, evidence_id: str) -> tuple[bool, list[str], Admissibility]:
        """Check the legality (admissibility) of an evidence item.

        Verifies that required fields (source, collector, collection
        method) are present. Missing fields result in a ``CONDITIONAL``
        admissibility; explicitly marked illegal methods result in
        ``INADMISSIBLE``.

        Returns:
            A tuple of ``(is_legal, issues, admissibility)``.

        Raises:
            EvidenceError: If the evidence is not found.
        """

        evidence = self._chain.get_evidence(evidence_id)
        if evidence is None:
            raise EvidenceError(f"Evidence not found: {evidence_id}")

        issues: list[str] = []
        for field_name, issue_desc in _LEGALITY_RULES:
            value = getattr(evidence, field_name, "")
            if not value or not str(value).strip():
                issues.append(issue_desc)

        # Check for explicitly illegal collection methods.
        illegal_methods = {"刑讯逼供", "威胁", "引诱", "欺骗"}
        method_lower = evidence.collection_method.lower()
        if any(m in method_lower for m in illegal_methods):
            issues.append("收集方式涉嫌非法")
            return False, issues, Admissibility.INADMISSIBLE

        if issues:
            return True, issues, Admissibility.CONDITIONAL
        return True, [], Admissibility.ADMISSIBLE

    # ------------------------------------------------------------------
    # Relevance assessment
    # ------------------------------------------------------------------

    def assess_relevance(self, evidence_id: str) -> tuple[float, str]:
        """Assess the relevance of an evidence item to its proving object.

        Computes a relevance score in ``[0, 1]`` based on the textual
        overlap between the evidence description and the proving object.
        Uses :mod:`jieba` for proper Chinese word segmentation when
        available, falling back to character-level matching otherwise.
        A higher score means the evidence is more directly relevant to
        what it claims to prove.

        Returns:
            A tuple of ``(score, reasoning)``.

        Raises:
            EvidenceError: If the evidence is not found.
        """

        evidence = self._chain.get_evidence(evidence_id)
        if evidence is None:
            raise EvidenceError(f"Evidence not found: {evidence_id}")

        if not evidence.proving_object:
            return 0.0, "证据未指明证明对象，无法评估关联性。"

        if not evidence.description:
            return 0.3, "证据描述缺失，关联性难以判断。"

        # Use jieba for proper Chinese word segmentation.
        desc_terms = _tokenize(evidence.description.lower())
        obj_terms = _tokenize(evidence.proving_object.lower())
        if not obj_terms:
            return 0.3, "证明对象为空，关联性难以判断。"

        overlap = desc_terms & obj_terms
        score = len(overlap) / len(obj_terms) if obj_terms else 0.0

        # Boost score if the evidence type is inherently relevant.
        type_relevance_boost = {
            EvidenceType.DOCUMENTARY: 0.1,
            EvidenceType.PHYSICAL: 0.1,
            EvidenceType.ELECTRONIC_DATA: 0.05,
        }
        score = min(1.0, score + type_relevance_boost.get(evidence.type, 0.0))

        if score >= 0.7:
            reasoning = "证据与证明对象高度相关。"
        elif score >= 0.4:
            reasoning = "证据与证明对象存在一定关联。"
        else:
            reasoning = "证据与证明对象关联性较弱。"

        return round(score, 4), reasoning

    # ------------------------------------------------------------------
    # Probative-value rating
    # ------------------------------------------------------------------

    def rate_probative_value(
        self, evidence_id: str
    ) -> tuple[ProbativeStrength, float, str]:
        """Rate the probative value of an evidence item.

        Combines the relevance score, the evidence type's inherent
        probative weight, and the admissibility status to produce a
        numeric score and a qualitative :class:`ProbativeStrength`.

        Returns:
            A tuple of ``(strength, score, reasoning)``.

        Raises:
            EvidenceError: If the evidence is not found.
        """

        evidence = self._chain.get_evidence(evidence_id)
        if evidence is None:
            raise EvidenceError(f"Evidence not found: {evidence_id}")

        # Inadmissible evidence has zero probative value.
        if evidence.is_excluded:
            return (
                ProbativeStrength.INSUFFICIENT,
                0.0,
                "证据因合法性瑕疵被排除，无证明力。",
            )

        # Get relevance score.
        relevance, rel_reasoning = self.assess_relevance(evidence_id)

        # Type-based base probative weight.
        type_weights: dict[EvidenceType, float] = {
            EvidenceType.DOCUMENTARY: 0.7,
            EvidenceType.PHYSICAL: 0.8,
            EvidenceType.TESTIMONY: 0.5,
            EvidenceType.EXPERT_OPINION: 0.75,
            EvidenceType.INSPECTION_RECORD: 0.65,
            EvidenceType.AUDIO_VISUAL: 0.6,
            EvidenceType.ELECTRONIC_DATA: 0.55,
        }
        base_weight = type_weights.get(evidence.type, 0.5)

        # Combine: base weight (40%) + relevance (40%) + admissibility
        # bonus (20%).
        admissibility_bonus = 1.0 if evidence.admissibility is Admissibility.ADMISSIBLE else 0.5
        score = base_weight * 0.4 + relevance * 0.4 + admissibility_bonus * 0.2
        score = round(min(1.0, max(0.0, score)), 4)

        strength = _probative_strength_from_score(score)
        reasoning = (
            f"基础证明力权重 {base_weight:.2f}，关联性得分 {relevance:.2f}，"
            f"可采性系数 {admissibility_bonus:.1f}。{rel_reasoning}"
        )

        return strength, score, reasoning

    # ------------------------------------------------------------------
    # Full review
    # ------------------------------------------------------------------

    def review(self, evidence_id: str) -> ReviewResult:
        """Perform a full review of a single evidence item.

        Combines legality, relevance, and probative-value assessments
        into a single :class:`ReviewResult`, and updates the evidence
        item with the assessment results.

        Raises:
            EvidenceError: If the evidence is not found.
        """

        evidence = self._chain.get_evidence(evidence_id)
        if evidence is None:
            raise EvidenceError(f"Evidence not found: {evidence_id}")

        is_legal, issues, admissibility = self.review_legality(evidence_id)
        relevance, rel_reasoning = self.assess_relevance(evidence_id)
        strength, score, prob_reasoning = self.rate_probative_value(evidence_id)

        recommendations: list[str] = []
        if issues:
            recommendations.append("请补充缺失的合法性信息。")
        if relevance < 0.4:
            recommendations.append("证据与证明对象关联性较弱，建议补充说明。")
        if strength is ProbativeStrength.INSUFFICIENT:
            recommendations.append("证明力不足，建议补充其他证据。")

        result = ReviewResult(
            evidence_id=evidence_id,
            is_legal=is_legal,
            legality_issues=issues,
            admissibility=admissibility,
            relevance_score=relevance,
            relevance_reasoning=rel_reasoning,
            probative_strength=strength,
            probative_score=score,
            probative_reasoning=prob_reasoning,
            recommendations=recommendations,
        )

        # Update the evidence with assessment results.
        with self._lock:
            evidence.admissibility = admissibility
            evidence.admissibility_issues = issues
            evidence.relevance_score = relevance
            evidence.probative_strength = strength
            evidence.probative_score = score
            evidence.reviewed = True

        logger.info(
            "Reviewed evidence %s: admissibility=%s, strength=%s, score=%.2f",
            evidence_id,
            admissibility.value,
            strength.value,
            score,
        )
        return result

    def review_all(self, *, case_id: str | None = None) -> list[ReviewResult]:
        """Review all evidence items, optionally filtered by case.

        Returns a list of :class:`ReviewResult` objects.
        """

        evidence_list = self._chain.list_evidence(case_id=case_id)
        results: list[ReviewResult] = []
        for evidence in evidence_list:
            try:
                results.append(self.review(evidence.id))
            except EvidenceError as exc:
                logger.warning("Failed to review evidence %s: %s", evidence.id, exc)
        return results

    async def review_async(self, evidence_id: str) -> ReviewResult:
        """Async wrapper for :meth:`review`."""

        return await asyncio.to_thread(self.review, evidence_id)

    async def review_all_async(
        self, *, case_id: str | None = None
    ) -> list[ReviewResult]:
        """Async wrapper for :meth:`review_all`."""

        return await asyncio.to_thread(self.review_all, case_id=case_id)


__all__ = [
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
]
