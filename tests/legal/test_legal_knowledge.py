"""Tests for the legal knowledge base module
(justagent.verticals.legal.legal_knowledge)."""

from __future__ import annotations

import threading

import pytest

from justagent.knowledge.graph import EntityType, KnowledgeGraph
from justagent.knowledge.vector import (
    HashingEmbedder,
    InMemoryVectorStore,
)
from justagent.verticals.legal.legal_knowledge import (
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_article(
    law_name: str = "民法典",
    article_number: str = "第143条",
    content: str = "具备下列条件的民事法律行为有效：行为人具有相应的民事行为能力；"
    "意思表示真实；不违反法律、行政法规的强制性规定，不违背公序良俗。",
    domain: LegalDomain = LegalDomain.CIVIL,
    keywords: list[str] | None = None,
) -> LegalArticle:
    return LegalArticle(
        law_name=law_name,
        article_number=article_number,
        chapter="第三章",
        content=content,
        domain=domain,
        effective_date="2021-01-01",
        keywords=keywords or ["民事法律行为", "有效"],
    )


def _make_case(
    case_number: str = "(2023)京01民终123号",
    cause_of_action: str = "买卖合同纠纷",
    domain: LegalDomain = LegalDomain.CIVIL,
    level: CaseLevel = CaseLevel.ORDINARY,
    ruling_essence: str = "当事人应当按照约定全面履行自己的义务",
) -> LegalCase:
    return LegalCase(
        case_number=case_number,
        cause_of_action=cause_of_action,
        court="北京市第一中级人民法院",
        judgment_date="2023-06-15",
        ruling_essence=ruling_essence,
        ruling_result="驳回上诉，维持原判",
        domain=domain,
        level=level,
        keywords=["合同", "违约"],
        summary="上诉人与被上诉人因买卖合同货款支付问题产生纠纷",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kb() -> LegalKnowledgeBase:
    return LegalKnowledgeBase()


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_legal_domain_values(self) -> None:
        assert LegalDomain.CRIMINAL.value == "criminal"
        assert LegalDomain.CIVIL.value == "civil"
        assert LegalDomain.ADMINISTRATIVE.value == "administrative"
        assert LegalDomain.PROCEDURAL.value == "procedural"
        assert LegalDomain.COMMERCIAL.value == "commercial"
        assert LegalDomain.LABOR.value == "labor"
        assert LegalDomain.CONSTITUTIONAL.value == "constitutional"
        assert LegalDomain.CUSTOM.value == "custom"

    def test_article_status_values(self) -> None:
        assert ArticleStatus.EFFECTIVE.value == "effective"
        assert ArticleStatus.AMENDED.value == "amended"
        assert ArticleStatus.REPEALED.value == "repealed"

    def test_case_level_values(self) -> None:
        assert CaseLevel.GUIDING.value == "guiding"
        assert CaseLevel.TYPICAL.value == "typical"
        assert CaseLevel.ORDINARY.value == "ordinary"


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestLegalArticle:
    def test_defaults(self) -> None:
        article = LegalArticle(
            law_name="民法典",
            article_number="第143条",
            content="测试内容",
        )
        assert article.law_name == "民法典"
        assert article.article_number == "第143条"
        assert article.domain is LegalDomain.CIVIL
        assert article.status is ArticleStatus.EFFECTIVE
        assert article.keywords == []
        assert article.metadata == {}
        assert article.id
        assert article.created_at > 0

    def test_auto_id_unique(self) -> None:
        a1 = LegalArticle(law_name="法", article_number="第1条", content="甲")
        a2 = LegalArticle(law_name="法", article_number="第2条", content="乙")
        assert a1.id != a2.id

    def test_is_effective(self) -> None:
        assert _make_article().is_effective is True
        repealed = _make_article()
        repealed.status = ArticleStatus.REPEALED
        assert repealed.is_effective is False

    def test_citation(self) -> None:
        article = _make_article()
        assert article.citation == "《民法典》第143条"

    def test_searchable_text_includes_all_fields(self) -> None:
        article = _make_article(keywords=[" keyword1 ", "keyword2"])
        text = article.searchable_text()
        assert "民法典" in text
        assert "第143条" in text
        assert article.content in text
        assert "第三章" in text
        assert "keyword1" in text
        assert "keyword2" in text

    def test_searchable_text_without_optional_fields(self) -> None:
        article = LegalArticle(
            law_name="法",
            article_number="第1条",
            content="内容",
        )
        text = article.searchable_text()
        assert "法" in text
        assert "第1条" in text
        assert "内容" in text


class TestLegalCase:
    def test_defaults(self) -> None:
        case = _make_case()
        assert case.case_number == "(2023)京01民终123号"
        assert case.domain is LegalDomain.CIVIL
        assert case.level is CaseLevel.ORDINARY
        assert case.applied_article_ids == []
        assert case.applied_articles == []
        assert case.id
        assert case.created_at > 0

    def test_auto_id_unique(self) -> None:
        c1 = _make_case("(2023)甲001号")
        c2 = _make_case("(2023)甲002号")
        assert c1.id != c2.id

    def test_is_guiding(self) -> None:
        assert _make_case(level=CaseLevel.GUIDING).is_guiding is True
        assert _make_case(level=CaseLevel.ORDINARY).is_guiding is False

    def test_searchable_text_includes_all_fields(self) -> None:
        case = _make_case()
        text = case.searchable_text()
        assert case.case_number in text
        assert case.cause_of_action in text
        assert case.ruling_essence in text
        assert case.ruling_result in text
        assert case.summary in text
        assert "合同" in text


class TestSearchResultModels:
    def test_article_search_result(self) -> None:
        article = _make_article()
        result = ArticleSearchResult(
            article=article,
            score=0.85,
            match_type="semantic",
        )
        assert result.article is article
        assert result.score == 0.85
        assert result.match_type == "semantic"

    def test_article_search_result_defaults(self) -> None:
        result = ArticleSearchResult(article=_make_article())
        assert result.score == 0.0
        assert result.match_type == "semantic"

    def test_case_search_result(self) -> None:
        case = _make_case()
        result = CaseSearchResult(
            case=case,
            score=0.72,
            match_type="keyword",
        )
        assert result.case is case
        assert result.score == 0.72
        assert result.match_type == "keyword"

    def test_concept_explanation(self) -> None:
        explanation = ConceptExplanation(
            concept="民事法律行为",
            definition="民事主体通过意思表示设立、变更、终止民事法律关系的行为",
            related_concepts=["意思表示", "法律行为"],
            defining_articles=["《民法典》第133条"],
            source="graph",
        )
        assert explanation.concept == "民事法律行为"
        assert "意思表示" in explanation.related_concepts
        assert "《民法典》第133条" in explanation.defining_articles
        assert explanation.source == "graph"

    def test_concept_explanation_defaults(self) -> None:
        explanation = ConceptExplanation(concept="测试概念")
        assert explanation.definition == ""
        assert explanation.related_concepts == []
        assert explanation.defining_articles == []
        assert explanation.source == "articles"


# ---------------------------------------------------------------------------
# Article management tests
# ---------------------------------------------------------------------------


class TestArticleManagement:
    def test_add_and_get_article(self, kb: LegalKnowledgeBase) -> None:
        article = _make_article()
        kb.add_article(article)
        assert kb.article_count == 1
        assert kb.get_article(article.id) is article
        assert kb.get_article("nonexistent") is None

    def test_add_article_returns_same(self, kb: LegalKnowledgeBase) -> None:
        article = _make_article()
        returned = kb.add_article(article)
        assert returned is article

    def test_add_duplicate_raises(self, kb: LegalKnowledgeBase) -> None:
        article = _make_article()
        kb.add_article(article)
        duplicate = LegalArticle(
            law_name=article.law_name,
            article_number=article.article_number,
            content="不同内容",
        )
        with pytest.raises(LegalKnowledgeError, match="Article already exists"):
            kb.add_article(duplicate)

    def test_update_article_same_id(self, kb: LegalKnowledgeBase) -> None:
        article = _make_article()
        kb.add_article(article)
        article.content = "修改后的内容"
        kb.add_article(article)
        assert kb.article_count == 1
        assert kb.get_article(article.id).content == "修改后的内容"

    def test_add_articles_bulk(self, kb: LegalKnowledgeBase) -> None:
        articles = [
            _make_article("民法典", f"第{i}条") for i in range(100, 103)
        ]
        count = kb.add_articles(articles)
        assert count == 3
        assert kb.article_count == 3

    def test_add_articles_skips_duplicates(self, kb: LegalKnowledgeBase) -> None:
        a1 = _make_article("民法典", "第143条")
        a2 = _make_article("民法典", "第143条")  # same law + number
        a3 = _make_article("民法典", "第144条")
        count = kb.add_articles([a1, a2, a3])
        assert count == 2
        assert kb.article_count == 2

    def test_find_article(self, kb: LegalKnowledgeBase) -> None:
        article = _make_article()
        kb.add_article(article)
        found = kb.find_article("民法典", "第143条")
        assert found is article
        assert kb.find_article("刑法", "第1条") is None

    def test_list_articles_all(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article("民法典", "第143条", domain=LegalDomain.CIVIL))
        kb.add_article(_make_article("刑法", "第1条", domain=LegalDomain.CRIMINAL))
        assert len(kb.list_articles()) == 2

    def test_list_articles_filter_by_domain(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article("民法典", "第143条", domain=LegalDomain.CIVIL))
        kb.add_article(_make_article("刑法", "第1条", domain=LegalDomain.CRIMINAL))
        civil = kb.list_articles(domain=LegalDomain.CIVIL)
        assert len(civil) == 1
        assert civil[0].law_name == "民法典"

    def test_list_articles_filter_by_law_name(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article("民法典", "第143条"))
        kb.add_article(_make_article("民法典", "第144条"))
        kb.add_article(_make_article("刑法", "第1条"))
        results = kb.list_articles(law_name="民法典")
        assert len(results) == 2

    def test_list_articles_filter_by_status(self, kb: LegalKnowledgeBase) -> None:
        effective = _make_article("民法典", "第143条")
        repealed = _make_article("民法典", "第144条")
        repealed.status = ArticleStatus.REPEALED
        kb.add_article(effective)
        kb.add_article(repealed)
        results = kb.list_articles(status=ArticleStatus.EFFECTIVE)
        assert len(results) == 1
        assert results[0].article_number == "第143条"

    def test_remove_article(self, kb: LegalKnowledgeBase) -> None:
        article = _make_article()
        kb.add_article(article)
        removed = kb.remove_article(article.id)
        assert removed is article
        assert kb.article_count == 0
        assert kb.get_article(article.id) is None

    def test_remove_article_not_found(self, kb: LegalKnowledgeBase) -> None:
        assert kb.remove_article("nonexistent") is None

    def test_remove_article_cleans_index(self, kb: LegalKnowledgeBase) -> None:
        article = _make_article()
        kb.add_article(article)
        kb.remove_article(article.id)
        assert kb.find_article("民法典", "第143条") is None

    def test_remove_article_with_vector_store(self) -> None:
        kb = LegalKnowledgeBase(vector_store=InMemoryVectorStore())
        article = _make_article()
        kb.add_article(article)
        assert kb.vector_store is not None
        kb.remove_article(article.id)
        assert kb.article_count == 0


# ---------------------------------------------------------------------------
# Case management tests
# ---------------------------------------------------------------------------


class TestCaseManagement:
    def test_add_and_get_case(self, kb: LegalKnowledgeBase) -> None:
        case = _make_case()
        kb.add_case(case)
        assert kb.case_count == 1
        assert kb.get_case(case.id) is case
        assert kb.get_case("nonexistent") is None

    def test_add_case_returns_same(self, kb: LegalKnowledgeBase) -> None:
        case = _make_case()
        returned = kb.add_case(case)
        assert returned is case

    def test_add_duplicate_case_raises(self, kb: LegalKnowledgeBase) -> None:
        case = _make_case()
        kb.add_case(case)
        duplicate = _make_case(case.case_number)
        with pytest.raises(LegalKnowledgeError, match="Case already exists"):
            kb.add_case(duplicate)

    def test_update_case_same_id(self, kb: LegalKnowledgeBase) -> None:
        case = _make_case()
        kb.add_case(case)
        case.ruling_result = "改判"
        kb.add_case(case)
        assert kb.case_count == 1
        assert kb.get_case(case.id).ruling_result == "改判"

    def test_add_cases_bulk(self, kb: LegalKnowledgeBase) -> None:
        cases = [
            _make_case(f"(2023)京01民终{i}号") for i in range(100, 103)
        ]
        count = kb.add_cases(cases)
        assert count == 3
        assert kb.case_count == 3

    def test_add_cases_skips_duplicates(self, kb: LegalKnowledgeBase) -> None:
        c1 = _make_case("(2023)甲001号")
        c2 = _make_case("(2023)甲001号")  # same case number
        c3 = _make_case("(2023)甲002号")
        count = kb.add_cases([c1, c2, c3])
        assert count == 2

    def test_find_case(self, kb: LegalKnowledgeBase) -> None:
        case = _make_case()
        kb.add_case(case)
        found = kb.find_case("(2023)京01民终123号")
        assert found is case
        assert kb.find_case("nonexistent") is None

    def test_list_cases_all(self, kb: LegalKnowledgeBase) -> None:
        kb.add_case(_make_case("(2023)甲001号"))
        kb.add_case(_make_case("(2023)甲002号"))
        assert len(kb.list_cases()) == 2

    def test_list_cases_filter_by_domain(self, kb: LegalKnowledgeBase) -> None:
        kb.add_case(_make_case("(2023)甲001号", domain=LegalDomain.CIVIL))
        kb.add_case(_make_case("(2023)甲002号", domain=LegalDomain.CRIMINAL))
        results = kb.list_cases(domain=LegalDomain.CIVIL)
        assert len(results) == 1

    def test_list_cases_filter_by_cause(self, kb: LegalKnowledgeBase) -> None:
        kb.add_case(_make_case("(2023)甲001号", cause_of_action="买卖合同纠纷"))
        kb.add_case(_make_case("(2023)甲002号", cause_of_action="侵权纠纷"))
        results = kb.list_cases(cause_of_action="买卖合同纠纷")
        assert len(results) == 1

    def test_list_cases_filter_by_level(self, kb: LegalKnowledgeBase) -> None:
        kb.add_case(_make_case("(2023)甲001号", level=CaseLevel.GUIDING))
        kb.add_case(_make_case("(2023)甲002号", level=CaseLevel.ORDINARY))
        results = kb.list_cases(level=CaseLevel.GUIDING)
        assert len(results) == 1

    def test_remove_case(self, kb: LegalKnowledgeBase) -> None:
        case = _make_case()
        kb.add_case(case)
        removed = kb.remove_case(case.id)
        assert removed is case
        assert kb.case_count == 0

    def test_remove_case_not_found(self, kb: LegalKnowledgeBase) -> None:
        assert kb.remove_case("nonexistent") is None

    def test_remove_case_cleans_index(self, kb: LegalKnowledgeBase) -> None:
        case = _make_case()
        kb.add_case(case)
        kb.remove_case(case.id)
        assert kb.find_case(case.case_number) is None


# ---------------------------------------------------------------------------
# Article search tests
# ---------------------------------------------------------------------------


class TestArticleSearch:
    def test_search_keyword_match(self, kb: LegalKnowledgeBase) -> None:
        article = _make_article(
            content="行为人具有相应的民事行为能力意思表示真实",
        )
        kb.add_article(article)
        results = kb.search_articles("民事 行为")
        assert len(results) >= 1
        assert results[0].article.id == article.id
        assert results[0].match_type == "keyword"
        assert results[0].score > 0

    def test_search_no_match_returns_empty(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article())
        results = kb.search_articles("nonexistentterm", min_score=0.01)
        assert len(results) == 0

    def test_search_min_score_filter(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article(content="内容甲"))
        kb.add_article(
            _make_article(
                law_name="刑法",
                article_number="第1条",
                content="completely different content here",
                domain=LegalDomain.CRIMINAL,
            )
        )
        results = kb.search_articles("内容甲", min_score=0.5)
        assert len(results) == 1
        assert results[0].article.content == "内容甲"

    def test_search_top_k_limit(self, kb: LegalKnowledgeBase) -> None:
        for i in range(5):
            kb.add_article(
                _make_article("民法典", f"第{i}条", content=f"民事法律行为第{i}条")
            )
        results = kb.search_articles("民事", top_k=2)
        assert len(results) <= 2

    def test_search_filter_by_domain(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article("民法典", "第143条", domain=LegalDomain.CIVIL))
        kb.add_article(_make_article("刑法", "第1条", domain=LegalDomain.CRIMINAL))
        results = kb.search_articles("第", domain=LegalDomain.CIVIL)
        assert all(r.article.domain is LegalDomain.CIVIL for r in results)

    def test_search_filter_by_law_name(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article("民法典", "第143条"))
        kb.add_article(_make_article("刑法", "第1条"))
        results = kb.search_articles("第", law_name="民法典")
        assert all(r.article.law_name == "民法典" for r in results)

    def test_search_empty_kb(self, kb: LegalKnowledgeBase) -> None:
        assert kb.search_articles("anything") == []

    def test_search_results_sorted_by_score(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article("民法典", "第143条", content="民事 法律 行为"))
        kb.add_article(
            _make_article("民法典", "第144条", content="无民事行为能力人")
        )
        results = kb.search_articles("民事 法律 行为")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_vector_store(self) -> None:
        kb = LegalKnowledgeBase(vector_store=InMemoryVectorStore())
        kb.add_article(
            _make_article(
                content="civil legal act validity capacity consent",
            )
        )
        results = kb.search_articles("civil legal act")
        assert len(results) >= 1
        assert results[0].score > 0

    def test_search_semantic_match_type(self) -> None:
        kb = LegalKnowledgeBase(vector_store=InMemoryVectorStore())
        article = _make_article(
            content="contract breach damages compensation",
        )
        kb.add_article(article)
        results = kb.search_articles("contract breach damages")
        assert len(results) >= 1
        # With overlapping tokens, semantic score should be high.
        assert results[0].match_type in ("semantic", "keyword")


# ---------------------------------------------------------------------------
# Case search tests
# ---------------------------------------------------------------------------


class TestCaseSearch:
    def test_search_keyword_match(self, kb: LegalKnowledgeBase) -> None:
        case = _make_case()
        kb.add_case(case)
        results = kb.search_cases("买卖合同")
        assert len(results) >= 1
        assert results[0].case.id == case.id

    def test_search_no_match(self, kb: LegalKnowledgeBase) -> None:
        kb.add_case(_make_case())
        results = kb.search_cases("nonexistentterm", min_score=0.01)
        assert len(results) == 0

    def test_search_top_k(self, kb: LegalKnowledgeBase) -> None:
        for i in range(5):
            kb.add_case(_make_case(f"(2023)甲{i}号", cause_of_action="合同纠纷"))
        results = kb.search_cases("合同", top_k=2)
        assert len(results) <= 2

    def test_search_filter_by_domain(self, kb: LegalKnowledgeBase) -> None:
        kb.add_case(_make_case("(2023)甲001号", domain=LegalDomain.CIVIL))
        kb.add_case(_make_case("(2023)甲002号", domain=LegalDomain.CRIMINAL))
        results = kb.search_cases("纠纷", domain=LegalDomain.CIVIL)
        assert all(r.case.domain is LegalDomain.CIVIL for r in results)

    def test_search_filter_by_cause(self, kb: LegalKnowledgeBase) -> None:
        kb.add_case(_make_case("(2023)甲001号", cause_of_action="买卖合同纠纷"))
        kb.add_case(_make_case("(2023)甲002号", cause_of_action="侵权纠纷"))
        results = kb.search_cases("纠纷", cause_of_action="买卖合同纠纷")
        assert all(r.case.cause_of_action == "买卖合同纠纷" for r in results)

    def test_search_empty_kb(self, kb: LegalKnowledgeBase) -> None:
        assert kb.search_cases("anything") == []

    def test_search_with_vector_store(self) -> None:
        kb = LegalKnowledgeBase(vector_store=InMemoryVectorStore())
        kb.add_case(
            _make_case(
                case_number="(2023)EN001",
                cause_of_action="contract dispute",
                ruling_essence="party shall perform obligation contract",
            )
        )
        results = kb.search_cases("contract dispute")
        assert len(results) >= 1
        assert results[0].score > 0


# ---------------------------------------------------------------------------
# Concept explanation tests
# ---------------------------------------------------------------------------


class TestConceptExplanation:
    def test_explain_concept_from_articles(self, kb: LegalKnowledgeBase) -> None:
        article = _make_article(
            content="民事法律行为是民事主体通过意思表示设立、变更、终止民事法律关系的行为",
            keywords=["民事法律行为"],
        )
        kb.add_article(article)
        explanation = kb.explain_concept("民事法律行为")
        assert explanation.concept == "民事法律行为"
        assert explanation.definition  # should be non-empty
        assert explanation.source == "articles"

    def test_explain_concept_with_graph_definition(self) -> None:
        graph = KnowledgeGraph()
        # Pre-populate the graph with a concept entity that has a definition.
        graph.add_entity(
            name="不可抗力",
            entity_type=EntityType.CONCEPT.value,
            metadata={"definition": "不能预见、不能避免且不能克服的客观情况"},
        )
        kb = LegalKnowledgeBase(knowledge_graph=graph)
        explanation = kb.explain_concept("不可抗力")
        assert explanation.definition == "不能预见、不能避免且不能克服的客观情况"
        assert explanation.source == "graph"

    def test_explain_concept_with_graph_related(self) -> None:
        graph = KnowledgeGraph()
        entity_a = graph.add_entity(name="违约责任", entity_type=EntityType.CONCEPT.value)
        graph.add_entity(name="继续履行", entity_type=EntityType.CONCEPT.value)
        graph.add_relation(entity_a.id, graph.find_entity("继续履行").id, "related_to")
        kb = LegalKnowledgeBase(knowledge_graph=graph)
        explanation = kb.explain_concept("违约责任")
        assert "继续履行" in explanation.related_concepts

    def test_explain_concept_no_results(self, kb: LegalKnowledgeBase) -> None:
        explanation = kb.explain_concept("不存在的概念")
        assert explanation.concept == "不存在的概念"
        assert explanation.definition == ""

    def test_explain_concept_defining_articles(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article("民法典", "第143条"))
        kb.add_article(_make_article("民法典", "第144条"))
        explanation = kb.explain_concept("民法典")
        assert len(explanation.defining_articles) >= 1

    def test_explain_concept_with_domain_filter(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(
            _make_article("民法典", "第143条", domain=LegalDomain.CIVIL)
        )
        kb.add_article(
            _make_article(
                "刑法",
                "第1条",
                content="民法典 民事法律行为",
                domain=LegalDomain.CRIMINAL,
            )
        )
        explanation = kb.explain_concept("民法典", domain=LegalDomain.CIVIL)
        defining = explanation.defining_articles
        assert all("民法典" in c or "刑法" not in c for c in defining)


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestAsyncMethods:
    @pytest.mark.asyncio
    async def test_search_articles_async(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article(content="民事 法律 行为"))
        results = await kb.search_articles_async("民事")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_articles_async_with_filters(
        self, kb: LegalKnowledgeBase
    ) -> None:
        kb.add_article(_make_article("民法典", "第143条", domain=LegalDomain.CIVIL))
        kb.add_article(_make_article("刑法", "第1条", domain=LegalDomain.CRIMINAL))
        results = await kb.search_articles_async(
            "第", domain=LegalDomain.CIVIL
        )
        assert all(r.article.domain is LegalDomain.CIVIL for r in results)

    @pytest.mark.asyncio
    async def test_search_cases_async(self, kb: LegalKnowledgeBase) -> None:
        kb.add_case(_make_case())
        results = await kb.search_cases_async("合同")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_cases_async_with_filters(
        self, kb: LegalKnowledgeBase
    ) -> None:
        kb.add_case(_make_case("(2023)甲001号", domain=LegalDomain.CIVIL))
        kb.add_case(_make_case("(2023)甲002号", domain=LegalDomain.CRIMINAL))
        results = await kb.search_cases_async(
            "纠纷", domain=LegalDomain.CIVIL
        )
        assert all(r.case.domain is LegalDomain.CIVIL for r in results)

    @pytest.mark.asyncio
    async def test_explain_concept_async(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(
            _make_article(content="民事法律行为的概念定义", keywords=["民事法律行为"])
        )
        explanation = await kb.explain_concept_async("民事法律行为")
        assert explanation.concept == "民事法律行为"


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_empty(self, kb: LegalKnowledgeBase) -> None:
        s = kb.summary()
        assert s["articles"] == 0
        assert s["cases"] == 0
        assert s["domains"] == {}
        assert s["has_vector_store"] is False
        assert s["has_knowledge_graph"] is False

    def test_summary_with_data(self, kb: LegalKnowledgeBase) -> None:
        kb.add_article(_make_article("民法典", "第143条", domain=LegalDomain.CIVIL))
        kb.add_article(_make_article("刑法", "第1条", domain=LegalDomain.CRIMINAL))
        kb.add_case(_make_case())
        s = kb.summary()
        assert s["articles"] == 2
        assert s["cases"] == 1
        assert s["domains"]["civil"] == 1
        assert s["domains"]["criminal"] == 1

    def test_summary_with_vector_store(self) -> None:
        kb = LegalKnowledgeBase(vector_store=InMemoryVectorStore())
        s = kb.summary()
        assert s["has_vector_store"] is True

    def test_summary_with_graph(self) -> None:
        graph = KnowledgeGraph()
        kb = LegalKnowledgeBase(knowledge_graph=graph)
        s = kb.summary()
        assert s["has_knowledge_graph"] is True


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestProperties:
    def test_vector_store_none(self, kb: LegalKnowledgeBase) -> None:
        assert kb.vector_store is None

    def test_vector_store_set(self) -> None:
        store = InMemoryVectorStore()
        kb = LegalKnowledgeBase(vector_store=store)
        assert kb.vector_store is store

    def test_embedder_none(self, kb: LegalKnowledgeBase) -> None:
        assert kb.embedder is None

    def test_embedder_auto_created(self) -> None:
        kb = LegalKnowledgeBase(vector_store=InMemoryVectorStore())
        assert kb.embedder is not None

    def test_embedder_explicit(self) -> None:
        embedder = HashingEmbedder(dim=128)
        kb = LegalKnowledgeBase(embedder=embedder)
        assert kb.embedder is embedder

    def test_knowledge_graph_none(self, kb: LegalKnowledgeBase) -> None:
        assert kb.knowledge_graph is None

    def test_knowledge_graph_set(self) -> None:
        graph = KnowledgeGraph()
        kb = LegalKnowledgeBase(knowledge_graph=graph)
        assert kb.knowledge_graph is graph

    def test_article_count(self, kb: LegalKnowledgeBase) -> None:
        assert kb.article_count == 0
        kb.add_article(_make_article())
        assert kb.article_count == 1

    def test_case_count(self, kb: LegalKnowledgeBase) -> None:
        assert kb.case_count == 0
        kb.add_case(_make_case())
        assert kb.case_count == 1


# ---------------------------------------------------------------------------
# Knowledge graph integration tests
# ---------------------------------------------------------------------------


class TestGraphIntegration:
    def test_article_keywords_added_to_graph(self) -> None:
        graph = KnowledgeGraph()
        kb = LegalKnowledgeBase(knowledge_graph=graph)
        kb.add_article(_make_article(keywords=["违约责任", "损害赔偿"]))
        entity = graph.find_entity("违约责任")
        assert entity is not None
        assert entity.entity_type == EntityType.CONCEPT.value

    def test_article_law_name_entity_created(self) -> None:
        graph = KnowledgeGraph()
        kb = LegalKnowledgeBase(knowledge_graph=graph)
        kb.add_article(_make_article(law_name="民法典"))
        entity = graph.find_entity("民法典")
        assert entity is not None

    def test_article_concept_relation_added(self) -> None:
        graph = KnowledgeGraph()
        kb = LegalKnowledgeBase(knowledge_graph=graph)
        kb.add_article(_make_article(keywords=["违约责任"]))
        assert graph.relation_count >= 1

    def test_no_graph_no_error(self, kb: LegalKnowledgeBase) -> None:
        # Adding an article without a graph should not raise.
        kb.add_article(_make_article(keywords=["测试"]))
        assert kb.article_count == 1

    def test_graph_explain_concept_uses_graph(self) -> None:
        graph = KnowledgeGraph()
        graph.add_entity(
            name="善意取得",
            entity_type=EntityType.CONCEPT.value,
            metadata={"definition": "善意第三人取得动产或不动产所有权"},
        )
        kb = LegalKnowledgeBase(knowledge_graph=graph)
        explanation = kb.explain_concept("善意取得")
        assert explanation.source == "graph"
        assert "善意第三人" in explanation.definition


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_add_articles(self) -> None:
        kb = LegalKnowledgeBase()
        errors: list[Exception] = []

        def add(i: int) -> None:
            try:
                kb.add_article(_make_article("民法典", f"第{i}条"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=add, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert kb.article_count == 20

    def test_concurrent_add_cases(self) -> None:
        kb = LegalKnowledgeBase()
        errors: list[Exception] = []

        def add(i: int) -> None:
            try:
                kb.add_case(_make_case(f"(2023)京{i:04d}号"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=add, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert kb.case_count == 20

    def test_concurrent_search(self) -> None:
        kb = LegalKnowledgeBase()
        for i in range(10):
            kb.add_article(_make_article("民法典", f"第{i}条", content=f"民事 法律 行为{i}"))

        errors: list[Exception] = []

        def search() -> None:
            try:
                results = kb.search_articles("民事")
                assert len(results) >= 1
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=search) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_add_and_search(self) -> None:
        kb = LegalKnowledgeBase()
        errors: list[Exception] = []

        def add_and_search(i: int) -> None:
            try:
                kb.add_article(_make_article("民法典", f"第{i}条"))
                kb.search_articles("民事")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=add_and_search, args=(i,)) for i in range(15)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert kb.article_count == 15

    def test_concurrent_add_remove(self) -> None:
        kb = LegalKnowledgeBase()
        # Pre-populate.
        for i in range(10):
            kb.add_article(_make_article("民法典", f"第{i}条"))

        errors: list[Exception] = []

        def remove(i: int) -> None:
            try:
                article = kb.find_article("民法典", f"第{i}条")
                if article:
                    kb.remove_article(article.id)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=remove, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert kb.article_count == 0
