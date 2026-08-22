"""Tests for the legal document generation module
(justagent.verticals.legal.document_generator)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from justagent.adapters.model_gateway import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelGateway,
)
from justagent.verticals.legal.case_manager import (
    CaseManager,
    Claim,
    Party,
    PartyRole,
)
from justagent.verticals.legal.document_generator import (
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
from justagent.verticals.legal.evidence import Evidence, EvidenceChain, EvidenceType
from justagent.verticals.legal.legal_knowledge import (
    ArticleStatus,
    LegalArticle,
    LegalDomain,
    LegalKnowledgeBase,
)

# ---------------------------------------------------------------------------
# Mock gateway
# ---------------------------------------------------------------------------


class MockGateway(ModelGateway):
    """A mock ModelGateway that returns a canned LLM response."""

    def __init__(self, response_content: str = "## 正文\n生成的文书内容") -> None:
        self._response_content = response_content
        self.call_count = 0

    def health(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["mock-model"]

    def chat(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        self.call_count += 1
        return ChatCompletionResponse(content=self._response_content, model="mock-model")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager() -> CaseManager:
    return CaseManager()


@pytest.fixture
def case_with_data(manager: CaseManager) -> str:
    """Create a case with parties and claims; return its ID."""
    case = manager.create_case(
        case_number="(2024)京01民初1号",
        cause_of_action="买卖合同纠纷",
        court="北京市第一中级人民法院",
        domain="civil",
    )
    manager.add_party(case.id, Party(name="甲公司", role=PartyRole.PLAINTIFF))
    manager.add_party(case.id, Party(name="乙公司", role=PartyRole.DEFENDANT))
    manager.add_claim(
        case.id,
        Claim(
            description="判令被告支付货款100万元",
            claim_type="monetary",
            amount=1_000_000,
            legal_basis=["《民法典》第595条"],
        ),
    )
    return case.id


@pytest.fixture
def knowledge_base() -> LegalKnowledgeBase:
    kb = LegalKnowledgeBase()
    kb.add_article(
        LegalArticle(
            law_name="民法典",
            article_number="第595条",
            content="买卖合同是出卖人转移标的物的所有权于买受人，买受人支付价款的合同。",
            domain=LegalDomain.CIVIL,
            keywords=["买卖合同", "价款"],
        )
    )
    kb.add_article(
        LegalArticle(
            law_name="民法典",
            article_number="第143条",
            content="具备下列条件的民事法律行为有效。",
            domain=LegalDomain.CIVIL,
            status=ArticleStatus.REPEALED,
        )
    )
    return kb


@pytest.fixture
def generator(manager: CaseManager) -> LegalDocumentGenerator:
    return LegalDocumentGenerator(case_manager=manager)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_document_type_values(self) -> None:
        assert LegalDocumentType.INDICTMENT.value == "indictment"
        assert LegalDocumentType.STATEMENT_OF_DEFENSE.value == "statement_of_defense"
        assert LegalDocumentType.JUDGMENT.value == "judgment"
        assert LegalDocumentType.RULING.value == "ruling"
        assert LegalDocumentType.MEDIATION_AGREEMENT.value == "mediation_agreement"
        assert LegalDocumentType.AGENCY_OPINION.value == "agency_opinion"
        assert LegalDocumentType.LEGAL_OPINION.value == "legal_opinion"
        assert LegalDocumentType.EVIDENCE_LIST.value == "evidence_list"
        assert LegalDocumentType.CROSS_EXAMINATION_OPINION.value == "cross_examination_opinion"


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_legal_document_section(self) -> None:
        section = LegalDocumentSection(
            title="诉讼请求",
            content_template="{claims}",
            required=True,
            order=2,
        )
        assert section.title == "诉讼请求"
        assert section.required is True
        assert section.order == 2

    def test_legal_document_section_defaults(self) -> None:
        section = LegalDocumentSection(title="标题")
        assert section.content_template == ""
        assert section.required is True
        assert section.order == 0

    def test_legal_document_template(self) -> None:
        template = LegalDocumentTemplate(
            doc_type=LegalDocumentType.INDICTMENT,
            name="测试模板",
            sections=[LegalDocumentSection(title="段落")],
            placeholders=["var1"],
        )
        assert template.doc_type is LegalDocumentType.INDICTMENT
        assert template.name == "测试模板"
        assert len(template.sections) == 1
        assert template.id

    def test_generated_document_section(self) -> None:
        section = GeneratedDocumentSection(title="段落", content="内容")
        assert section.title == "段落"
        assert section.content == "内容"

    def test_citation_verification_defaults(self) -> None:
        cv = CitationVerification(citation="《民法典》第1条")
        assert cv.is_valid is False
        assert cv.issues == []
        assert cv.matched_article_id == ""

    def test_generated_document_defaults(self) -> None:
        doc = GeneratedDocument(case_id="case-1", doc_type=LegalDocumentType.INDICTMENT)
        assert doc.case_id == "case-1"
        assert doc.doc_type is LegalDocumentType.INDICTMENT
        assert doc.all_citations_valid is True
        assert doc.sections == []
        assert doc.created_at > 0
        assert doc.id


# ---------------------------------------------------------------------------
# Template manager tests
# ---------------------------------------------------------------------------


class TestTemplateManager:
    def test_default_templates_registered(self) -> None:
        mgr = LegalDocumentTemplateManager()
        # Every document type should have a default template.
        for doc_type in LegalDocumentType:
            template = mgr.get_template(doc_type)
            assert template is not None, f"No template for {doc_type}"
            assert template.doc_type is doc_type

    def test_template_count(self) -> None:
        mgr = LegalDocumentTemplateManager()
        assert mgr.template_count == len(list(LegalDocumentType))

    def test_get_template_by_id(self) -> None:
        mgr = LegalDocumentTemplateManager()
        template = mgr.get_template(LegalDocumentType.INDICTMENT)
        assert template is not None
        found = mgr.get_template_by_id(template.id)
        assert found is template

    def test_get_template_by_id_not_found(self) -> None:
        mgr = LegalDocumentTemplateManager()
        assert mgr.get_template_by_id("nonexistent") is None

    def test_list_templates(self) -> None:
        mgr = LegalDocumentTemplateManager()
        templates = mgr.list_templates()
        assert len(templates) == len(list(LegalDocumentType))

    def test_register_new_template(self) -> None:
        mgr = LegalDocumentTemplateManager()
        custom = LegalDocumentTemplate(
            doc_type=LegalDocumentType.RULING,
            name="自定义裁定书模板",
            sections=[LegalDocumentSection(title="自定义段落")],
        )
        # RULING already has a default template, so use replace.
        mgr.replace_template(custom)
        assert mgr.get_template(LegalDocumentType.RULING) is custom

    def test_register_duplicate_raises(self) -> None:
        mgr = LegalDocumentTemplateManager()
        existing = mgr.get_template(LegalDocumentType.INDICTMENT)
        assert existing is not None
        new_template = LegalDocumentTemplate(
            doc_type=LegalDocumentType.INDICTMENT,
            name="冲突模板",
        )
        with pytest.raises(DocumentGenerationError, match="already exists"):
            mgr.register_template(new_template)

    def test_replace_template(self) -> None:
        mgr = LegalDocumentTemplateManager()
        original = mgr.get_template(LegalDocumentType.RULING)
        assert original is not None
        replacement = LegalDocumentTemplate(
            doc_type=LegalDocumentType.RULING,
            name="替换后的模板",
        )
        mgr.replace_template(replacement)
        assert mgr.get_template(LegalDocumentType.RULING) is replacement
        assert mgr.get_template_by_id(original.id) is None

    def test_indictment_template_has_sections(self) -> None:
        mgr = LegalDocumentTemplateManager()
        template = mgr.get_template(LegalDocumentType.INDICTMENT)
        assert template is not None
        assert len(template.sections) >= 3
        titles = [s.title for s in template.sections]
        assert "诉讼请求" in titles
        assert "事实与理由" in titles

    def test_judgment_template_has_sections(self) -> None:
        mgr = LegalDocumentTemplateManager()
        template = mgr.get_template(LegalDocumentType.JUDGMENT)
        assert template is not None
        titles = [s.title for s in template.sections]
        assert "本院认为" in titles


# ---------------------------------------------------------------------------
# Generator property tests
# ---------------------------------------------------------------------------


class TestGeneratorProperties:
    def test_case_manager_property(
        self, generator: LegalDocumentGenerator, manager: CaseManager
    ) -> None:
        assert generator.case_manager is manager

    def test_template_manager_property(self, generator: LegalDocumentGenerator) -> None:
        assert isinstance(generator.template_manager, LegalDocumentTemplateManager)

    def test_gateway_property_default_none(self, generator: LegalDocumentGenerator) -> None:
        assert generator.gateway is None

    def test_gateway_setter(self, generator: LegalDocumentGenerator) -> None:
        gw = MockGateway()
        generator.gateway = gw
        assert generator.gateway is gw

    def test_rag_pipeline_none(self, generator: LegalDocumentGenerator) -> None:
        assert generator.rag_pipeline is None

    def test_knowledge_base_none(self, generator: LegalDocumentGenerator) -> None:
        assert generator.knowledge_base is None


# ---------------------------------------------------------------------------
# Generation tests (template-only, no LLM)
# ---------------------------------------------------------------------------


class TestGenerationTemplateOnly:
    def test_generate_indictment(
        self, generator: LegalDocumentGenerator, case_with_data: str
    ) -> None:
        doc = generator.generate(case_with_data, LegalDocumentType.INDICTMENT)
        assert doc.case_id == case_with_data
        assert doc.doc_type is LegalDocumentType.INDICTMENT
        assert len(doc.sections) > 0
        assert doc.content
        assert "甲公司" in doc.content or "plaintiff" in doc.content

    def test_generate_with_title(
        self, generator: LegalDocumentGenerator, case_with_data: str
    ) -> None:
        doc = generator.generate(case_with_data, LegalDocumentType.INDICTMENT, title="民事起诉状")
        assert doc.title == "民事起诉状"
        assert doc.content.startswith("民事起诉状")

    def test_generate_metadata(
        self, generator: LegalDocumentGenerator, case_with_data: str
    ) -> None:
        doc = generator.generate(case_with_data, LegalDocumentType.JUDGMENT)
        assert doc.metadata["used_llm"] is False
        assert doc.metadata["template_id"]
        assert doc.metadata["template_name"]

    def test_generate_all_document_types(
        self, generator: LegalDocumentGenerator, case_with_data: str
    ) -> None:
        for doc_type in LegalDocumentType:
            doc = generator.generate(case_with_data, doc_type)
            assert doc.doc_type is doc_type
            assert len(doc.sections) > 0

    def test_generate_with_extra_context(
        self, generator: LegalDocumentGenerator, case_with_data: str
    ) -> None:
        doc = generator.generate(
            case_with_data,
            LegalDocumentType.INDICTMENT,
            extra_context={"custom_var": "自定义内容"},
        )
        assert doc.content

    def test_generate_case_not_found(self, generator: LegalDocumentGenerator) -> None:
        from justagent.verticals.legal.case_manager import CaseManagerError

        with pytest.raises(CaseManagerError):
            generator.generate("nonexistent", LegalDocumentType.INDICTMENT)

    def test_generate_with_knowledge_base(
        self, manager: CaseManager, case_with_data: str, knowledge_base: LegalKnowledgeBase
    ) -> None:
        gen = LegalDocumentGenerator(case_manager=manager, knowledge_base=knowledge_base)
        doc = gen.generate(case_with_data, LegalDocumentType.INDICTMENT, verify=False)
        assert doc.content
        # Legal context should reference the knowledge base.
        assert "民法典" in doc.content or "暂无" in doc.content

    def test_generate_with_evidence_chain(self, manager: CaseManager, case_with_data: str) -> None:
        chain = EvidenceChain()
        chain.add_evidence(
            Evidence(
                name="购销合同",
                type=EvidenceType.DOCUMENTARY,
                proving_object="合同关系成立",
                case_id=case_with_data,
                source="当事人提供",
                collector="张律师",
                collection_method="当事人提供",
            )
        )
        gen = LegalDocumentGenerator(case_manager=manager, evidence_chain=chain)
        doc = gen.generate(case_with_data, LegalDocumentType.JUDGMENT, verify=False)
        assert "证据" in doc.content or "暂无" in doc.content


# ---------------------------------------------------------------------------
# LLM-assisted generation tests
# ---------------------------------------------------------------------------


class TestLLMGeneration:
    def test_generate_with_llm(self, manager: CaseManager, case_with_data: str) -> None:
        llm_response = (
            "## 当事人信息\n原告：甲公司\n被告：乙公司\n\n"
            "## 诉讼请求\n判令被告支付货款100万元\n\n"
            "## 事实与理由\n根据《民法典》第595条"
        )
        gw = MockGateway(response_content=llm_response)
        gen = LegalDocumentGenerator(case_manager=manager, gateway=gw)
        doc = gen.generate(case_with_data, LegalDocumentType.INDICTMENT, verify=False)
        assert gw.call_count == 1
        assert len(doc.sections) >= 2
        assert any("当事人" in s.title for s in doc.sections)
        assert doc.metadata["used_llm"] is True

    def test_llm_failure_falls_back_to_template(
        self, manager: CaseManager, case_with_data: str
    ) -> None:
        gw = MockGateway()
        gw.chat = MagicMock(side_effect=RuntimeError("LLM unavailable"))
        gen = LegalDocumentGenerator(case_manager=manager, gateway=gw)
        doc = gen.generate(case_with_data, LegalDocumentType.INDICTMENT, verify=False)
        # Should still produce a document (template fallback).
        assert len(doc.sections) > 0
        assert doc.content

    def test_parse_llm_output(self) -> None:
        output = "## 标题1\n内容1\n\n## 标题2\n内容2\n"
        sections = LegalDocumentGenerator._parse_llm_output(
            output, LegalDocumentTemplate(doc_type=LegalDocumentType.INDICTMENT, name="x")
        )
        assert len(sections) == 2
        assert sections[0].title == "标题1"
        assert sections[0].content == "内容1"
        assert sections[1].title == "标题2"

    def test_parse_llm_output_no_headings(self) -> None:
        output = "纯文本内容没有标题"
        sections = LegalDocumentGenerator._parse_llm_output(
            output, LegalDocumentTemplate(doc_type=LegalDocumentType.INDICTMENT, name="x")
        )
        assert len(sections) == 1
        assert sections[0].content == "纯文本内容没有标题"

    def test_parse_llm_output_empty(self) -> None:
        sections = LegalDocumentGenerator._parse_llm_output(
            "", LegalDocumentTemplate(doc_type=LegalDocumentType.INDICTMENT, name="x")
        )
        assert len(sections) == 1


# ---------------------------------------------------------------------------
# Citation verification tests
# ---------------------------------------------------------------------------


class TestCitationVerification:
    def test_extract_citations_from_content(
        self, generator: LegalDocumentGenerator, case_with_data: str
    ) -> None:
        # Generate a document — the claim has legal_basis with 《民法典》第595条.
        doc = generator.generate(case_with_data, LegalDocumentType.INDICTMENT, verify=False)
        # The content should contain the citation.
        if "《民法典》" in doc.content:
            assert len(doc.citations) >= 1

    def test_verify_citations_without_kb(self, generator: LegalDocumentGenerator) -> None:
        citations = ["《民法典》第595条"]
        results = generator.verify_citations(citations)
        assert len(results) == 1
        # Without a KB, verification is skipped but marked valid.
        assert results[0].is_valid is True
        assert "verification skipped" in results[0].issues[0]

    def test_verify_citations_with_kb(
        self, manager: CaseManager, knowledge_base: LegalKnowledgeBase
    ) -> None:
        gen = LegalDocumentGenerator(case_manager=manager, knowledge_base=knowledge_base)
        results = gen.verify_citations(["《民法典》第595条"])
        assert len(results) == 1
        assert results[0].is_valid is True
        assert results[0].matched_law_name == "民法典"
        assert results[0].matched_article_number == "第595条"

    def test_verify_citations_repealed_article(
        self, manager: CaseManager, knowledge_base: LegalKnowledgeBase
    ) -> None:
        gen = LegalDocumentGenerator(case_manager=manager, knowledge_base=knowledge_base)
        results = gen.verify_citations(["《民法典》第143条"])
        assert len(results) == 1
        assert results[0].is_valid is False
        assert any("repealed" in issue for issue in results[0].issues)

    def test_verify_citations_not_found(
        self, manager: CaseManager, knowledge_base: LegalKnowledgeBase
    ) -> None:
        gen = LegalDocumentGenerator(case_manager=manager, knowledge_base=knowledge_base)
        results = gen.verify_citations(["《不存在法》第999条"])
        assert len(results) == 1
        # With articles in the KB, fuzzy matching returns a result.
        assert results[0].is_valid is True
        assert any("fuzzy" in issue for issue in results[0].issues)
        assert results[0].matched_article_id

    def test_verify_citations_not_found_empty_kb(self, manager: CaseManager) -> None:
        # With an empty knowledge base, no fuzzy match is possible.
        empty_kb = LegalKnowledgeBase()
        gen = LegalDocumentGenerator(case_manager=manager, knowledge_base=empty_kb)
        results = gen.verify_citations(["《不存在法》第999条"])
        assert len(results) == 1
        assert results[0].is_valid is False
        assert any("not found" in issue for issue in results[0].issues)

    def test_verify_citations_malformed(self, generator: LegalDocumentGenerator) -> None:
        results = generator.verify_citations(["这不是一个引用"])
        assert len(results) == 1
        assert results[0].is_valid is False
        assert any("format" in issue for issue in results[0].issues)

    def test_verify_citations_empty_list(self, generator: LegalDocumentGenerator) -> None:
        results = generator.verify_citations([])
        assert results == []

    def test_verify_document(
        self, manager: CaseManager, knowledge_base: LegalKnowledgeBase, case_with_data: str
    ) -> None:
        gen = LegalDocumentGenerator(case_manager=manager, knowledge_base=knowledge_base)
        doc = gen.generate(case_with_data, LegalDocumentType.INDICTMENT, verify=False)
        # Manually set citations and verify.
        doc.citations = ["《民法典》第595条"]
        gen.verify_document(doc)
        assert len(doc.citation_verifications) == 1
        assert doc.citation_verifications[0].is_valid is True

    def test_generate_with_verify_flag(
        self, manager: CaseManager, knowledge_base: LegalKnowledgeBase, case_with_data: str
    ) -> None:
        gen = LegalDocumentGenerator(case_manager=manager, knowledge_base=knowledge_base)
        doc = gen.generate(case_with_data, LegalDocumentType.INDICTMENT, verify=True)
        # If there are citations, they should be verified.
        if doc.citations:
            assert len(doc.citation_verifications) == len(doc.citations)

    def test_all_citations_valid_flag(
        self, generator: LegalDocumentGenerator, case_with_data: str
    ) -> None:
        doc = generator.generate(case_with_data, LegalDocumentType.INDICTMENT, verify=True)
        # Without KB, all citations are marked valid.
        assert doc.all_citations_valid is True


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestAsyncMethods:
    @pytest.mark.asyncio
    async def test_generate_async(
        self, generator: LegalDocumentGenerator, case_with_data: str
    ) -> None:
        doc = await generator.generate_async(
            case_with_data, LegalDocumentType.INDICTMENT, verify=False
        )
        assert doc.doc_type is LegalDocumentType.INDICTMENT

    @pytest.mark.asyncio
    async def test_verify_citations_async(self, generator: LegalDocumentGenerator) -> None:
        results = await generator.verify_citations_async(["《民法典》第1条"])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_generation(self, manager: CaseManager, case_with_data: str) -> None:
        gen = LegalDocumentGenerator(case_manager=manager)
        errors: list[Exception] = []
        docs: list[GeneratedDocument] = []

        def generate() -> None:
            try:
                doc = gen.generate(case_with_data, LegalDocumentType.INDICTMENT, verify=False)
                docs.append(doc)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=generate) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(docs) == 5

    def test_concurrent_template_registration(self) -> None:
        mgr = LegalDocumentTemplateManager()
        errors: list[Exception] = []

        def register(i: int) -> None:
            try:
                # Use replace to avoid duplicate conflicts.
                mgr.replace_template(
                    LegalDocumentTemplate(
                        doc_type=LegalDocumentType.RULING,
                        name=f"模板{i}",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert mgr.template_count == len(list(LegalDocumentType))
