"""Tests for the evidence review module (justagent.verticals.legal.evidence)."""

from __future__ import annotations

import threading

import pytest

from justagent.knowledge.graph import KnowledgeGraph
from justagent.verticals.legal.evidence import (
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chain() -> EvidenceChain:
    return EvidenceChain()


@pytest.fixture
def reviewer(chain: EvidenceChain) -> EvidenceReviewer:
    return EvidenceReviewer(chain)


def _make_valid_evidence(
    name: str = "合同",
    *,
    proving_object: str = "合同关系成立",
    case_id: str = "case-1",
) -> Evidence:
    """Return an evidence item with all legality fields filled."""
    return Evidence(
        name=name,
        type=EvidenceType.DOCUMENTARY,
        description="双方签订的购销合同原件，证明合同关系成立",
        source="当事人提供",
        collector="张律师",
        collection_method="当事人提供",
        proving_object=proving_object,
        case_id=case_id,
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_evidence_type_values(self) -> None:
        assert EvidenceType.DOCUMENTARY.value == "documentary"
        assert EvidenceType.PHYSICAL.value == "physical"
        assert EvidenceType.TESTIMONY.value == "testimony"
        assert EvidenceType.EXPERT_OPINION.value == "expert_opinion"
        assert EvidenceType.INSPECTION_RECORD.value == "inspection_record"
        assert EvidenceType.AUDIO_VISUAL.value == "audio_visual"
        assert EvidenceType.ELECTRONIC_DATA.value == "electronic_data"

    def test_admissibility_values(self) -> None:
        assert Admissibility.ADMISSIBLE.value == "admissible"
        assert Admissibility.INADMISSIBLE.value == "inadmissible"
        assert Admissibility.CONDITIONAL.value == "conditional"

    def test_probative_strength_values(self) -> None:
        assert ProbativeStrength.HIGH.value == "high"
        assert ProbativeStrength.MEDIUM.value == "medium"
        assert ProbativeStrength.LOW.value == "low"
        assert ProbativeStrength.INSUFFICIENT.value == "insufficient"

    def test_evidence_relation_type_values(self) -> None:
        assert EvidenceRelationType.SUPPORTS.value == "supports"
        assert EvidenceRelationType.CORROBORATES.value == "corroborates"
        assert EvidenceRelationType.CONTRADICTS.value == "contradicts"
        assert EvidenceRelationType.SUPPLEMENTS.value == "supplements"


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_evidence_defaults(self) -> None:
        ev = Evidence(name="测试证据")
        assert ev.name == "测试证据"
        assert ev.type is EvidenceType.DOCUMENTARY
        assert ev.admissibility is Admissibility.ADMISSIBLE
        assert ev.probative_strength is ProbativeStrength.MEDIUM
        assert ev.relevance_score == 0.0
        assert ev.reviewed is False
        assert ev.id

    def test_evidence_auto_id_unique(self) -> None:
        ev1 = Evidence(name="甲")
        ev2 = Evidence(name="乙")
        assert ev1.id != ev2.id

    def test_evidence_is_admissible(self) -> None:
        assert Evidence(name="x").is_admissible is True
        assert Evidence(
            name="x", admissibility=Admissibility.CONDITIONAL
        ).is_admissible is True
        assert Evidence(
            name="x", admissibility=Admissibility.INADMISSIBLE
        ).is_admissible is False

    def test_evidence_is_excluded(self) -> None:
        assert Evidence(
            name="x", admissibility=Admissibility.INADMISSIBLE
        ).is_excluded is True
        assert Evidence(name="x").is_excluded is False

    def test_evidence_relation_model(self) -> None:
        rel = EvidenceRelation(
            evidence_a_id="ev-1",
            evidence_b_id="ev-2",
            relation_type=EvidenceRelationType.CONTRADICTS,
            description="两证据矛盾",
            weight=0.8,
        )
        assert rel.evidence_a_id == "ev-1"
        assert rel.evidence_b_id == "ev-2"
        assert rel.relation_type is EvidenceRelationType.CONTRADICTS
        assert rel.weight == 0.8
        assert rel.id

    def test_review_result_model(self) -> None:
        result = ReviewResult(
            evidence_id="ev-1",
            is_legal=True,
            admissibility=Admissibility.ADMISSIBLE,
            relevance_score=0.8,
            probative_strength=ProbativeStrength.HIGH,
            probative_score=0.85,
        )
        assert result.evidence_id == "ev-1"
        assert result.is_legal is True
        assert result.relevance_score == 0.8
        assert result.recommendations == []

    def test_chain_analysis_result_model(self) -> None:
        result = ChainAnalysisResult(
            case_id="case-1",
            completeness_score=0.75,
            total_evidence=5,
            admissible_evidence=3,
        )
        assert result.case_id == "case-1"
        assert result.completeness_score == 0.75
        assert result.contradictions == []
        assert result.gaps == []


# ---------------------------------------------------------------------------
# EvidenceChain tests
# ---------------------------------------------------------------------------


class TestEvidenceChain:
    def test_add_and_get_evidence(self, chain: EvidenceChain) -> None:
        ev = _make_valid_evidence()
        chain.add_evidence(ev)
        assert chain.evidence_count == 1
        assert chain.get_evidence(ev.id) is ev
        assert chain.get_evidence("nonexistent") is None

    def test_list_evidence(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1", case_id="case-1")
        ev2 = _make_valid_evidence("证据2", case_id="case-2")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        assert len(chain.list_evidence()) == 2
        assert len(chain.list_evidence(case_id="case-1")) == 1

    def test_remove_evidence(self, chain: EvidenceChain) -> None:
        ev = _make_valid_evidence()
        chain.add_evidence(ev)
        removed = chain.remove_evidence(ev.id)
        assert removed is ev
        assert chain.evidence_count == 0
        assert chain.get_evidence(ev.id) is None

    def test_remove_evidence_not_found(self, chain: EvidenceChain) -> None:
        assert chain.remove_evidence("nonexistent") is None

    def test_remove_evidence_cleans_relations(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1")
        ev2 = _make_valid_evidence("证据2")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        chain.add_relation(ev1.id, ev2.id, EvidenceRelationType.SUPPORTS)
        assert chain.relation_count == 1
        chain.remove_evidence(ev1.id)
        assert chain.relation_count == 0

    def test_add_relation(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1")
        ev2 = _make_valid_evidence("证据2")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        rel = chain.add_relation(
            ev1.id, ev2.id, EvidenceRelationType.CORROBORATES, description="互相印证"
        )
        assert chain.relation_count == 1
        assert rel.relation_type is EvidenceRelationType.CORROBORATES

    def test_add_relation_missing_evidence(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1")
        chain.add_evidence(ev1)
        with pytest.raises(EvidenceError, match="Evidence not found"):
            chain.add_relation(ev1.id, "nonexistent")

    def test_list_relations_by_type(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1")
        ev2 = _make_valid_evidence("证据2")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        chain.add_relation(ev1.id, ev2.id, EvidenceRelationType.SUPPORTS)
        chain.add_relation(ev1.id, ev2.id, EvidenceRelationType.CONTRADICTS)
        supports = chain.list_relations(relation_type=EvidenceRelationType.SUPPORTS)
        contradictions = chain.list_relations(relation_type=EvidenceRelationType.CONTRADICTS)
        assert len(supports) == 1
        assert len(contradictions) == 1

    def test_list_relations_by_evidence(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1")
        ev2 = _make_valid_evidence("证据2")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        chain.add_relation(ev1.id, ev2.id, EvidenceRelationType.SUPPORTS)
        rels = chain.list_relations(evidence_id=ev1.id)
        assert len(rels) == 1

    def test_analyze_empty(self, chain: EvidenceChain) -> None:
        result = chain.analyze("case-1")
        assert result.completeness_score == 0.0
        assert result.total_evidence == 0
        assert "无证据" in result.summary

    def test_analyze_with_evidence(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("合同", proving_object="合同关系", case_id="case-1")
        ev2 = _make_valid_evidence("收据", proving_object="付款事实", case_id="case-1")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        result = chain.analyze("case-1")
        assert result.total_evidence == 2
        assert result.admissible_evidence == 2
        assert result.completeness_score > 0

    def test_analyze_filters_by_case(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1", case_id="case-1")
        ev2 = _make_valid_evidence("证据2", case_id="case-2")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        result = chain.analyze("case-1")
        assert result.total_evidence == 1

    def test_analyze_detects_contradictions(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1", case_id="case-1")
        ev2 = _make_valid_evidence("证据2", case_id="case-1")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        chain.add_relation(
            ev1.id, ev2.id, EvidenceRelationType.CONTRADICTS, description="矛盾"
        )
        result = chain.analyze("case-1")
        assert len(result.contradictions) == 1

    def test_analyze_detects_gaps(self, chain: EvidenceChain) -> None:
        # All evidence for a proving object is inadmissible.
        ev = _make_valid_evidence("非法证据", proving_object="关键事实", case_id="case-1")
        ev.admissibility = Admissibility.INADMISSIBLE
        chain.add_evidence(ev)
        result = chain.analyze("case-1")
        assert len(result.gaps) >= 1
        assert any("关键事实" in g for g in result.gaps)

    def test_analyze_supporting_relations_count(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1", case_id="case-1")
        ev2 = _make_valid_evidence("证据2", case_id="case-1")
        ev3 = _make_valid_evidence("证据3", case_id="case-1")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        chain.add_evidence(ev3)
        chain.add_relation(ev1.id, ev2.id, EvidenceRelationType.SUPPORTS)
        chain.add_relation(ev1.id, ev3.id, EvidenceRelationType.CORROBORATES)
        result = chain.analyze("case-1")
        assert result.supporting_relations == 2

    def test_analyze_all_evidence_when_no_case_id(self, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1", case_id="case-1")
        ev2 = _make_valid_evidence("证据2", case_id="case-2")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        result = chain.analyze("")
        assert result.total_evidence == 2


# ---------------------------------------------------------------------------
# EvidenceReviewer tests
# ---------------------------------------------------------------------------


class TestEvidenceReviewer:
    def test_review_legality_admissible(self, reviewer: EvidenceReviewer, chain: EvidenceChain) -> None:
        ev = _make_valid_evidence()
        chain.add_evidence(ev)
        is_legal, issues, admissibility = reviewer.review_legality(ev.id)
        assert is_legal is True
        assert issues == []
        assert admissibility is Admissibility.ADMISSIBLE

    def test_review_legality_conditional(self, reviewer: EvidenceReviewer, chain: EvidenceChain) -> None:
        ev = _make_valid_evidence()
        ev.source = ""  # missing source
        chain.add_evidence(ev)
        is_legal, issues, admissibility = reviewer.review_legality(ev.id)
        assert is_legal is True
        assert len(issues) >= 1
        assert admissibility is Admissibility.CONDITIONAL

    def test_review_legality_inadmissible(
        self, reviewer: EvidenceReviewer, chain: EvidenceChain
    ) -> None:
        ev = _make_valid_evidence()
        ev.collection_method = "刑讯逼供"
        chain.add_evidence(ev)
        is_legal, issues, admissibility = reviewer.review_legality(ev.id)
        assert is_legal is False
        assert admissibility is Admissibility.INADMISSIBLE

    def test_review_legality_not_found(self, reviewer: EvidenceReviewer) -> None:
        with pytest.raises(EvidenceError, match="Evidence not found"):
            reviewer.review_legality("nonexistent")

    def test_assess_relevance_no_proving_object(
        self, reviewer: EvidenceReviewer, chain: EvidenceChain
    ) -> None:
        ev = _make_valid_evidence()
        ev.proving_object = ""
        chain.add_evidence(ev)
        score, reasoning = reviewer.assess_relevance(ev.id)
        assert score == 0.0
        assert "证明对象" in reasoning

    def test_assess_relevance_no_description(
        self, reviewer: EvidenceReviewer, chain: EvidenceChain
    ) -> None:
        ev = _make_valid_evidence()
        ev.description = ""
        chain.add_evidence(ev)
        score, reasoning = reviewer.assess_relevance(ev.id)
        assert score == 0.3
        assert "描述缺失" in reasoning

    def test_assess_relevance_with_overlap(
        self, reviewer: EvidenceReviewer, chain: EvidenceChain
    ) -> None:
        ev = _make_valid_evidence()
        # Use English for word-overlap testing since assess_relevance splits on whitespace.
        ev.proving_object = "contract relationship"
        ev.description = "contract relationship evidence"
        chain.add_evidence(ev)
        score, _ = reviewer.assess_relevance(ev.id)
        assert score > 0.0

    def test_assess_relevance_not_found(self, reviewer: EvidenceReviewer) -> None:
        with pytest.raises(EvidenceError, match="Evidence not found"):
            reviewer.assess_relevance("nonexistent")

    def test_rate_probative_value_excluded(
        self, reviewer: EvidenceReviewer, chain: EvidenceChain
    ) -> None:
        ev = _make_valid_evidence()
        ev.admissibility = Admissibility.INADMISSIBLE
        chain.add_evidence(ev)
        strength, score, reasoning = reviewer.rate_probative_value(ev.id)
        assert strength is ProbativeStrength.INSUFFICIENT
        assert score == 0.0

    def test_rate_probative_value_valid(
        self, reviewer: EvidenceReviewer, chain: EvidenceChain
    ) -> None:
        ev = _make_valid_evidence()
        chain.add_evidence(ev)
        strength, score, reasoning = reviewer.rate_probative_value(ev.id)
        assert score > 0.0
        assert score <= 1.0
        assert strength in ProbativeStrength

    def test_rate_probative_value_not_found(self, reviewer: EvidenceReviewer) -> None:
        with pytest.raises(EvidenceError, match="Evidence not found"):
            reviewer.rate_probative_value("nonexistent")

    def test_review_full(self, reviewer: EvidenceReviewer, chain: EvidenceChain) -> None:
        ev = _make_valid_evidence()
        chain.add_evidence(ev)
        result = reviewer.review(ev.id)
        assert isinstance(result, ReviewResult)
        assert result.evidence_id == ev.id
        assert result.admissibility is Admissibility.ADMISSIBLE
        # Evidence should be marked as reviewed.
        assert ev.reviewed is True
        assert ev.relevance_score == result.relevance_score
        assert ev.probative_score == result.probative_score

    def test_review_with_issues_adds_recommendations(
        self, reviewer: EvidenceReviewer, chain: EvidenceChain
    ) -> None:
        ev = _make_valid_evidence()
        ev.source = ""  # causes issues
        ev.proving_object = ""  # causes low relevance
        chain.add_evidence(ev)
        result = reviewer.review(ev.id)
        assert len(result.recommendations) >= 1

    def test_review_not_found(self, reviewer: EvidenceReviewer) -> None:
        with pytest.raises(EvidenceError, match="Evidence not found"):
            reviewer.review("nonexistent")

    def test_review_all(self, reviewer: EvidenceReviewer, chain: EvidenceChain) -> None:
        ev1 = _make_valid_evidence("证据1")
        ev2 = _make_valid_evidence("证据2")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        results = reviewer.review_all()
        assert len(results) == 2
        assert all(r.evidence_id in (ev1.id, ev2.id) for r in results)

    def test_review_all_filtered_by_case(
        self, reviewer: EvidenceReviewer, chain: EvidenceChain
    ) -> None:
        ev1 = _make_valid_evidence("证据1", case_id="case-1")
        ev2 = _make_valid_evidence("证据2", case_id="case-2")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        results = reviewer.review_all(case_id="case-1")
        assert len(results) == 1
        assert results[0].evidence_id == ev1.id

    def test_chain_property(self, chain: EvidenceChain) -> None:
        reviewer = EvidenceReviewer(chain)
        assert reviewer.chain is chain

    def test_reviewer_creates_own_chain(self) -> None:
        reviewer = EvidenceReviewer()
        assert isinstance(reviewer.chain, EvidenceChain)


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestAsyncMethods:
    @pytest.mark.asyncio
    async def test_analyze_async(self, chain: EvidenceChain) -> None:
        ev = _make_valid_evidence(case_id="case-1")
        chain.add_evidence(ev)
        result = await chain.analyze_async("case-1")
        assert result.total_evidence == 1

    @pytest.mark.asyncio
    async def test_review_async(self, reviewer: EvidenceReviewer, chain: EvidenceChain) -> None:
        ev = _make_valid_evidence()
        chain.add_evidence(ev)
        result = await reviewer.review_async(ev.id)
        assert result.evidence_id == ev.id

    @pytest.mark.asyncio
    async def test_review_all_async(
        self, reviewer: EvidenceReviewer, chain: EvidenceChain
    ) -> None:
        ev1 = _make_valid_evidence("证据1")
        ev2 = _make_valid_evidence("证据2")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        results = await reviewer.review_all_async()
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Knowledge graph integration tests
# ---------------------------------------------------------------------------


class TestGraphIntegration:
    def test_evidence_indexed_in_graph(self) -> None:
        graph = KnowledgeGraph()
        chain = EvidenceChain(knowledge_graph=graph)
        ev = _make_valid_evidence()
        chain.add_evidence(ev)
        entities = chain.get_evidence_graph_entities()
        assert len(entities) >= 1
        assert any(e.metadata.get("category") == "evidence" for e in entities)

    def test_proving_object_entity_created(self) -> None:
        graph = KnowledgeGraph()
        chain = EvidenceChain(knowledge_graph=graph)
        ev = _make_valid_evidence(proving_object="合同关系成立")
        chain.add_evidence(ev)
        # The proving object should be an entity.
        entity = graph.find_entity("合同关系成立")
        assert entity is not None

    def test_relation_added_to_graph(self) -> None:
        graph = KnowledgeGraph()
        chain = EvidenceChain(knowledge_graph=graph)
        ev1 = _make_valid_evidence("证据1", proving_object="事实A")
        ev2 = _make_valid_evidence("证据2", proving_object="事实A")
        chain.add_evidence(ev1)
        chain.add_evidence(ev2)
        chain.add_relation(ev1.id, ev2.id, EvidenceRelationType.SUPPORTS)
        # Graph should have at least one relation.
        assert graph.relation_count >= 1

    def test_no_graph_returns_empty_entities(self, chain: EvidenceChain) -> None:
        assert chain.get_evidence_graph_entities() == []

    def test_knowledge_graph_property(self) -> None:
        graph = KnowledgeGraph()
        chain = EvidenceChain(knowledge_graph=graph)
        assert chain.knowledge_graph is graph

    def test_knowledge_graph_none(self, chain: EvidenceChain) -> None:
        assert chain.knowledge_graph is None


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_add_evidence(self) -> None:
        chain = EvidenceChain()
        errors: list[Exception] = []

        def add(i: int) -> None:
            try:
                chain.add_evidence(
                    Evidence(name=f"证据{i}", case_id="case-1")
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=add, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert chain.evidence_count == 20

    def test_concurrent_review(self) -> None:
        chain = EvidenceChain()
        reviewer = EvidenceReviewer(chain)
        evidence_ids: list[str] = []
        for i in range(10):
            ev = _make_valid_evidence(f"证据{i}")
            chain.add_evidence(ev)
            evidence_ids.append(ev.id)

        errors: list[Exception] = []

        def review(eid: str) -> None:
            try:
                reviewer.review(eid)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=review, args=(eid,)) for eid in evidence_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
