"""Tests for the case management module (justagent.verticals.legal.case_manager)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from justagent.knowledge.document import Document, DocumentParser, DocumentType
from justagent.verticals.legal.case_manager import (
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager() -> CaseManager:
    """Return a fresh CaseManager."""
    return CaseManager()


@pytest.fixture
def sample_party() -> Party:
    return Party(
        name="甲公司",
        role=PartyRole.PLAINTIFF,
        contact="12345678901",
        legal_representative="张三",
        id_number="91110000XXXXXXX",
    )


@pytest.fixture
def populated_case(manager: CaseManager) -> CaseFile:
    """Create a case with parties, facts, claims, and timeline."""
    case = manager.create_case(
        case_number="(2024)京01民初1号",
        cause_of_action="买卖合同纠纷",
        court="北京市第一中级人民法院",
        domain="civil",
    )
    manager.add_party(case.id, Party(name="甲公司", role=PartyRole.PLAINTIFF))
    manager.add_party(case.id, Party(name="乙公司", role=PartyRole.DEFENDANT))
    manager.add_fact(case.id, FactElement(description="双方签订买卖合同", category="transaction"))
    manager.add_claim(
        case.id,
        Claim(
            description="判令被告支付货款100万元",
            claim_type="monetary",
            amount=1_000_000,
            legal_basis=["《民法典》第595条"],
        ),
    )
    manager.add_timeline_event(
        case.id,
        TimelineEvent(date="2024-01-15", timestamp=1705276800, description="签订合同"),
    )
    manager.add_timeline_event(
        case.id,
        TimelineEvent(date="2024-03-01", timestamp=1709251200, description="交付货物"),
    )
    return case


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_party_role_values(self) -> None:
        assert PartyRole.PLAINTIFF.value == "plaintiff"
        assert PartyRole.DEFENDANT.value == "defendant"
        assert PartyRole.THIRD_PARTY.value == "third_party"
        assert PartyRole.OTHER.value == "other"

    def test_case_status_values(self) -> None:
        assert CaseStatus.DRAFT.value == "draft"
        assert CaseStatus.ACTIVE.value == "active"
        assert CaseStatus.CLOSED.value == "closed"
        assert CaseStatus.ARCHIVED.value == "archived"

    def test_material_type_values(self) -> None:
        assert MaterialType.COMPLAINT.value == "complaint"
        assert MaterialType.EVIDENCE.value == "evidence"
        assert MaterialType.JUDGMENT.value == "judgment"


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_party_defaults(self) -> None:
        party = Party(name="测试")
        assert party.name == "测试"
        assert party.role is PartyRole.OTHER
        assert party.contact == ""
        assert party.id  # auto-generated
        assert party.metadata == {}

    def test_party_auto_id_unique(self) -> None:
        p1 = Party(name="甲")
        p2 = Party(name="乙")
        assert p1.id != p2.id

    def test_timeline_event_defaults(self) -> None:
        event = TimelineEvent(description="事件")
        assert event.description == "事件"
        assert event.timestamp == 0.0
        assert event.date == ""
        assert event.id

    def test_fact_element_defaults(self) -> None:
        fact = FactElement(description="事实")
        assert fact.description == "事实"
        assert fact.contested is False
        assert fact.supporting_evidence_ids == []

    def test_disputed_issue_defaults(self) -> None:
        issue = DisputedIssue(description="争议")
        assert issue.description == "争议"
        assert issue.plaintiff_position == ""
        assert issue.related_evidence_ids == []

    def test_claim_defaults(self) -> None:
        claim = Claim(description="请求")
        assert claim.amount == 0.0
        assert claim.legal_basis == []
        assert claim.claim_type == ""

    def test_claim_with_amount(self) -> None:
        claim = Claim(
            description="支付货款",
            claim_type="monetary",
            amount=500_000,
            legal_basis=["《民法典》第595条"],
        )
        assert claim.amount == 500_000
        assert len(claim.legal_basis) == 1

    def test_case_file_defaults(self) -> None:
        case = CaseFile()
        assert case.status is CaseStatus.DRAFT
        assert case.parties == []
        assert case.fact_elements == []
        assert case.claims == []
        assert case.timeline == []
        assert case.evidence_ids == []
        assert case.material_ids == []
        assert case.created_at > 0
        assert case.updated_at > 0

    def test_case_file_is_active(self) -> None:
        assert CaseFile(status=CaseStatus.ACTIVE).is_active is True
        assert CaseFile(status=CaseStatus.UNDER_REVIEW).is_active is True
        assert CaseFile(status=CaseStatus.DRAFT).is_active is False
        assert CaseFile(status=CaseStatus.CLOSED).is_active is False
        assert CaseFile(status=CaseStatus.ARCHIVED).is_active is False

    def test_case_file_plaintiff_defendant(self) -> None:
        plaintiff = Party(name="原告甲", role=PartyRole.PLAINTIFF)
        defendant = Party(name="被告乙", role=PartyRole.DEFENDANT)
        witness = Party(name="证人丙", role=PartyRole.WITNESS)
        case = CaseFile(parties=[plaintiff, defendant, witness])
        assert case.plaintiff() is plaintiff
        assert case.defendant() is defendant

    def test_case_file_no_plaintiff(self) -> None:
        case = CaseFile(parties=[Party(name="证人", role=PartyRole.WITNESS)])
        assert case.plaintiff() is None
        assert case.defendant() is None

    def test_case_context_model(self) -> None:
        ctx = CaseContext(
            case_id="abc",
            summary="案号: 001",
            parties_text="原告: 甲",
            legal_basis=["第1条", "第2条"],
        )
        assert ctx.case_id == "abc"
        assert ctx.summary == "案号: 001"
        assert ctx.legal_basis == ["第1条", "第2条"]
        assert ctx.metadata == {}

    def test_case_material_model(self) -> None:
        doc = Document(title="test.txt", content="内容", type=DocumentType.PLAIN_TEXT)
        material = CaseMaterial(
            id=doc.id,
            document=doc,
            material_type=MaterialType.EVIDENCE,
            case_id="case-1",
        )
        assert material.material_type is MaterialType.EVIDENCE
        assert material.case_id == "case-1"
        assert material.imported_at > 0


# ---------------------------------------------------------------------------
# CaseManager lifecycle tests
# ---------------------------------------------------------------------------


class TestCaseManagerLifecycle:
    def test_create_case(self, manager: CaseManager) -> None:
        case = manager.create_case(
            case_number="(2024)京01民初1号",
            cause_of_action="买卖合同纠纷",
            court="北京一中院",
            domain="civil",
        )
        assert case.case_number == "(2024)京01民初1号"
        assert case.cause_of_action == "买卖合同纠纷"
        assert case.status is CaseStatus.DRAFT
        assert manager.case_count == 1

    def test_create_case_without_number(self, manager: CaseManager) -> None:
        case = manager.create_case(cause_of_action="纠纷")
        assert case.case_number == ""
        assert manager.case_count == 1

    def test_create_duplicate_case_number_raises(self, manager: CaseManager) -> None:
        manager.create_case(case_number="(2024)京01民初1号")
        with pytest.raises(CaseManagerError, match="already exists"):
            manager.create_case(case_number="(2024)京01民初1号")

    def test_get_case(self, manager: CaseManager) -> None:
        case = manager.create_case(case_number="001")
        assert manager.get_case(case.id) is case
        assert manager.get_case("nonexistent") is None

    def test_find_case_by_number(self, manager: CaseManager) -> None:
        case = manager.create_case(case_number="(2024)京01民初1号")
        found = manager.find_case("(2024)京01民初1号")
        assert found is case
        assert manager.find_case("不存在") is None

    def test_list_cases(self, manager: CaseManager) -> None:
        c1 = manager.create_case(case_number="001", cause_of_action="合同纠纷")
        c2 = manager.create_case(case_number="002", cause_of_action="侵权纠纷")
        manager.update_status(c2.id, CaseStatus.ACTIVE)

        all_cases = manager.list_cases()
        assert len(all_cases) == 2

        active_cases = manager.list_cases(status=CaseStatus.ACTIVE)
        assert len(active_cases) == 1
        assert active_cases[0] is c2

        contract_cases = manager.list_cases(cause_of_action="合同纠纷")
        assert len(contract_cases) == 1
        assert contract_cases[0] is c1

    def test_update_status(self, manager: CaseManager) -> None:
        case = manager.create_case()
        updated = manager.update_status(case.id, CaseStatus.ACTIVE)
        assert updated.status is CaseStatus.ACTIVE
        assert updated.updated_at >= case.created_at

    def test_update_status_not_found(self, manager: CaseManager) -> None:
        with pytest.raises(CaseManagerError, match="Case not found"):
            manager.update_status("nonexistent", CaseStatus.ACTIVE)

    def test_delete_case(self, manager: CaseManager) -> None:
        case = manager.create_case(case_number="001")
        removed = manager.delete_case(case.id)
        assert removed is case
        assert manager.case_count == 0
        assert manager.find_case("001") is None

    def test_delete_case_not_found(self, manager: CaseManager) -> None:
        assert manager.delete_case("nonexistent") is None

    def test_delete_case_removes_materials(self, manager: CaseManager) -> None:
        case = manager.create_case()
        doc = Document(title="test", content="content")
        material = manager.import_document(case.id, doc)
        assert manager.material_count == 1
        manager.delete_case(case.id)
        assert manager.material_count == 0
        assert manager.get_material(material.id) is None


# ---------------------------------------------------------------------------
# Party / fact / claim / timeline management
# ---------------------------------------------------------------------------


class TestCaseContentManagement:
    def test_add_party(self, manager: CaseManager, sample_party: Party) -> None:
        case = manager.create_case()
        result = manager.add_party(case.id, sample_party)
        assert len(result.parties) == 1
        assert result.parties[0].name == "甲公司"
        assert result.parties[0].role is PartyRole.PLAINTIFF

    def test_add_party_not_found(self, manager: CaseManager) -> None:
        with pytest.raises(CaseManagerError):
            manager.add_party("nonexistent", Party(name="x"))

    def test_remove_party(self, manager: CaseManager) -> None:
        case = manager.create_case()
        party = Party(name="甲", role=PartyRole.PLAINTIFF)
        manager.add_party(case.id, party)
        result = manager.remove_party(case.id, party.id)
        assert len(result.parties) == 0

    def test_add_fact(self, manager: CaseManager) -> None:
        case = manager.create_case()
        fact = FactElement(description="事实描述", category="background", contested=True)
        result = manager.add_fact(case.id, fact)
        assert len(result.fact_elements) == 1
        assert result.fact_elements[0].contested is True

    def test_add_disputed_issue(self, manager: CaseManager) -> None:
        case = manager.create_case()
        issue = DisputedIssue(
            description="合同效力争议",
            category="law",
            plaintiff_position="合同有效",
            defendant_position="合同无效",
        )
        result = manager.add_disputed_issue(case.id, issue)
        assert len(result.disputed_issues) == 1
        assert result.disputed_issues[0].plaintiff_position == "合同有效"

    def test_add_claim(self, manager: CaseManager) -> None:
        case = manager.create_case()
        claim = Claim(description="支付货款", amount=100_000, claim_type="monetary")
        result = manager.add_claim(case.id, claim)
        assert len(result.claims) == 1
        assert result.claims[0].amount == 100_000

    def test_add_timeline_event_sorts_by_timestamp(self, manager: CaseManager) -> None:
        case = manager.create_case()
        # Add events out of order.
        manager.add_timeline_event(
            case.id,
            TimelineEvent(date="2024-03-01", timestamp=1709251200, description="后期"),
        )
        manager.add_timeline_event(
            case.id,
            TimelineEvent(date="2024-01-15", timestamp=1705276800, description="早期"),
        )
        manager.add_timeline_event(
            case.id,
            TimelineEvent(date="2024-02-01", timestamp=1706745600, description="中期"),
        )
        updated = manager.get_case(case.id)
        assert updated is not None
        timestamps = [e.timestamp for e in updated.timeline]
        assert timestamps == sorted(timestamps)
        assert updated.timeline[0].description == "早期"

    def test_link_evidence_idempotent(self, manager: CaseManager) -> None:
        case = manager.create_case()
        manager.link_evidence(case.id, "ev-1")
        manager.link_evidence(case.id, "ev-1")  # duplicate
        updated = manager.get_case(case.id)
        assert updated is not None
        assert updated.evidence_ids == ["ev-1"]

    def test_link_evidence_multiple(self, manager: CaseManager) -> None:
        case = manager.create_case()
        manager.link_evidence(case.id, "ev-1")
        manager.link_evidence(case.id, "ev-2")
        updated = manager.get_case(case.id)
        assert updated is not None
        assert set(updated.evidence_ids) == {"ev-1", "ev-2"}


# ---------------------------------------------------------------------------
# Material import tests
# ---------------------------------------------------------------------------


class TestMaterialImport:
    def test_import_document(self, manager: CaseManager) -> None:
        case = manager.create_case()
        doc = Document(title="起诉状", content="原告：张三", type=DocumentType.PLAIN_TEXT)
        material = manager.import_document(
            case.id, doc, material_type=MaterialType.COMPLAINT
        )
        assert material.material_type is MaterialType.COMPLAINT
        assert material.case_id == case.id
        assert manager.material_count == 1
        assert case.material_ids == [doc.id]

    def test_import_document_auto_extract(self, manager: CaseManager) -> None:
        case = manager.create_case()
        content = "原告：张三\n被告：李四\n诉讼请求：判令被告支付10万元\n事实与理由：双方存在合同关系"
        doc = Document(title="起诉状", content=content, type=DocumentType.PLAIN_TEXT)
        manager.import_document(case.id, doc, auto_extract=True)
        updated = manager.get_case(case.id)
        assert updated is not None
        # Should have extracted parties.
        assert len(updated.parties) >= 2
        # Should have extracted claims.
        assert len(updated.claims) >= 1

    def test_import_document_no_extract_on_empty(self, manager: CaseManager) -> None:
        case = manager.create_case()
        doc = Document(title="空", content="", type=DocumentType.PLAIN_TEXT)
        manager.import_document(case.id, doc, auto_extract=True)
        updated = manager.get_case(case.id)
        assert updated is not None
        assert updated.parties == []

    def test_import_file(self, manager: CaseManager, tmp_path: Path) -> None:
        case = manager.create_case()
        file_path = tmp_path / "complaint.txt"
        file_path.write_text("原告：张三\n被告：李四", encoding="utf-8")
        material = manager.import_file(
            case.id, file_path, material_type=MaterialType.COMPLAINT
        )
        assert material.document.title == "complaint.txt"
        assert material.document.content == "原告：张三\n被告：李四"

    def test_import_file_not_found(self, manager: CaseManager) -> None:
        case = manager.create_case()
        with pytest.raises(FileNotFoundError):
            manager.import_file(case.id, "/nonexistent/path.txt")

    def test_import_bytes(self, manager: CaseManager) -> None:
        case = manager.create_case()
        data = "原告：张三".encode()
        material = manager.import_bytes(
            case.id, data, filename="complaint.txt", material_type=MaterialType.COMPLAINT
        )
        assert "原告" in material.document.content

    def test_get_material(self, manager: CaseManager) -> None:
        case = manager.create_case()
        doc = Document(title="doc", content="content")
        material = manager.import_document(case.id, doc)
        assert manager.get_material(material.id) is material
        assert manager.get_material("nonexistent") is None

    def test_list_materials(self, manager: CaseManager) -> None:
        case = manager.create_case()
        doc1 = Document(title="d1", content="content1")
        doc2 = Document(title="d2", content="content2")
        manager.import_document(case.id, doc1)
        manager.import_document(case.id, doc2)
        materials = manager.list_materials(case.id)
        assert len(materials) == 2

    def test_list_materials_empty_for_unknown_case(self, manager: CaseManager) -> None:
        assert manager.list_materials("nonexistent") == []


# ---------------------------------------------------------------------------
# Structured extraction tests
# ---------------------------------------------------------------------------


class TestStructuredExtraction:
    def test_extract_structure_parties(self, manager: CaseManager) -> None:
        case = manager.create_case()
        content = "原告：甲公司\n被告：乙公司\n上诉人：丙"
        doc = Document(title="doc", content=content)
        material = manager.import_document(case.id, doc, auto_extract=False)
        extracted = manager.extract_structure(case.id, material.id)
        roles = {p.role for p in extracted["parties"]}
        assert PartyRole.PLAINTIFF in roles
        assert PartyRole.DEFENDANT in roles
        assert PartyRole.APPELLANT in roles

    def test_extract_structure_claims(self, manager: CaseManager) -> None:
        case = manager.create_case()
        content = "诉讼请求：判令被告支付货款100万元\n事实与理由：合同关系成立"
        doc = Document(title="doc", content=content)
        material = manager.import_document(case.id, doc, auto_extract=False)
        extracted = manager.extract_structure(case.id, material.id)
        assert len(extracted["claims"]) >= 1

    def test_extract_structure_facts(self, manager: CaseManager) -> None:
        case = manager.create_case()
        content = "事实与理由：双方于2024年1月签订合同并履行了交货义务\n诉讼请求：支付货款"
        doc = Document(title="doc", content=content)
        material = manager.import_document(case.id, doc, auto_extract=False)
        extracted = manager.extract_structure(case.id, material.id)
        assert len(extracted["facts"]) >= 1

    def test_extract_structure_timeline(self, manager: CaseManager) -> None:
        case = manager.create_case()
        content = "2024年1月15日签订合同，2024年3月1日交付货物"
        doc = Document(title="doc", content=content)
        material = manager.import_document(case.id, doc, auto_extract=False)
        extracted = manager.extract_structure(case.id, material.id)
        assert len(extracted["timeline"]) >= 1
        for event in extracted["timeline"]:
            assert event.date  # ISO date should be populated
            assert event.timestamp > 0

    def test_extract_structure_not_found(self, manager: CaseManager) -> None:
        case = manager.create_case()
        with pytest.raises(CaseManagerError, match="Material not found"):
            manager.extract_structure(case.id, "nonexistent")

    def test_extract_all(self, manager: CaseManager) -> None:
        case = manager.create_case()
        content1 = "原告：甲\n被告：乙"
        content2 = "诉讼请求：支付货款\n事实与理由：合同成立"
        manager.import_document(
            case.id, Document(title="d1", content=content1), auto_extract=False
        )
        manager.import_document(
            case.id, Document(title="d2", content=content2), auto_extract=False
        )
        total = manager.extract_all(case.id)
        assert "parties" in total
        assert "claims" in total
        assert "facts" in total
        assert "timeline" in total
        assert len(total["parties"]) >= 2


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


class TestCaseSearch:
    def test_search_by_case_number(self, manager: CaseManager) -> None:
        manager.create_case(case_number="(2024)京01民初1号", cause_of_action="合同纠纷")
        results = manager.search_cases("京01民初1号")
        assert len(results) == 1

    def test_search_by_party_name(self, manager: CaseManager) -> None:
        case = manager.create_case(case_number="001", cause_of_action="纠纷")
        manager.add_party(case.id, Party(name="甲公司", role=PartyRole.PLAINTIFF))
        results = manager.search_cases("甲公司")
        assert len(results) == 1

    def test_search_by_cause_of_action(self, manager: CaseManager) -> None:
        manager.create_case(case_number="001", cause_of_action="买卖合同纠纷")
        manager.create_case(case_number="002", cause_of_action="侵权纠纷")
        results = manager.search_cases("买卖合同")
        assert len(results) == 1

    def test_search_no_match(self, manager: CaseManager) -> None:
        manager.create_case(case_number="001", cause_of_action="纠纷")
        results = manager.search_cases("不存在的关键词")
        assert len(results) == 0

    def test_search_with_status_filter(self, manager: CaseManager) -> None:
        manager.create_case(case_number="001", cause_of_action="纠纷")
        c2 = manager.create_case(case_number="002", cause_of_action="纠纷")
        manager.update_status(c2.id, CaseStatus.ACTIVE)
        results = manager.search_cases("纠纷", status=CaseStatus.ACTIVE)
        assert len(results) == 1
        assert results[0].id == c2.id

    def test_search_top_k(self, manager: CaseManager) -> None:
        for i in range(5):
            manager.create_case(case_number=f"00{i}", cause_of_action="纠纷")
        results = manager.search_cases("纠纷", top_k=2)
        assert len(results) == 2

    def test_search_case_insensitive(self, manager: CaseManager) -> None:
        manager.create_case(case_number="ABC123")
        results = manager.search_cases("abc")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Context building tests
# ---------------------------------------------------------------------------


class TestContextBuilding:
    def test_build_context(self, manager: CaseManager, populated_case: CaseFile) -> None:
        ctx = manager.build_context(populated_case.id)
        assert ctx.case_id == populated_case.id
        assert "买卖合同纠纷" in ctx.summary
        assert "甲公司" in ctx.parties_text
        assert "plaintiff" in ctx.parties_text
        assert "支付货款" in ctx.claims_text
        assert "1,000,000.00" in ctx.claims_text

    def test_build_context_with_evidence(self, manager: CaseManager) -> None:
        case = manager.create_case()
        manager.link_evidence(case.id, "ev-1")
        manager.link_evidence(case.id, "ev-2")
        ctx = manager.build_context(case.id, include_evidence=True)
        assert "2" in ctx.evidence_summary

    def test_build_context_without_evidence(self, manager: CaseManager) -> None:
        case = manager.create_case()
        ctx = manager.build_context(case.id, include_evidence=False)
        assert ctx.evidence_summary == ""

    def test_build_context_legal_basis(self, manager: CaseManager) -> None:
        case = manager.create_case()
        manager.add_claim(
            case.id,
            Claim(description="请求", legal_basis=["《民法典》第1条", "《民法典》第2条"]),
        )
        ctx = manager.build_context(case.id)
        assert "《民法典》第1条" in ctx.legal_basis
        assert "《民法典》第2条" in ctx.legal_basis

    def test_build_context_not_found(self, manager: CaseManager) -> None:
        with pytest.raises(CaseManagerError):
            manager.build_context("nonexistent")

    def test_render_context_prompt(self, manager: CaseManager, populated_case: CaseFile) -> None:
        prompt = manager.render_context_prompt(populated_case.id)
        assert "案件信息" in prompt
        assert "当事人" in prompt
        assert "甲公司" in prompt

    def test_render_context_prompt_custom_template(
        self, manager: CaseManager, populated_case: CaseFile
    ) -> None:
        prompt = manager.render_context_prompt(
            populated_case.id, template="摘要: {summary}\n当事人: {parties}"
        )
        assert prompt.startswith("摘要:")
        assert "甲公司" in prompt

    def test_build_context_empty_case(self, manager: CaseManager) -> None:
        case = manager.create_case()
        ctx = manager.build_context(case.id)
        assert ctx.parties_text == "  (无)"
        assert ctx.facts_text == "  (无)"
        assert ctx.claims_text == "  (无)"


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestAsyncMethods:
    @pytest.mark.asyncio
    async def test_build_context_async(self, manager: CaseManager) -> None:
        case = manager.create_case(case_number="001")
        manager.add_party(case.id, Party(name="甲", role=PartyRole.PLAINTIFF))
        ctx = await manager.build_context_async(case.id)
        assert ctx.case_id == case.id
        assert "甲" in ctx.parties_text

    @pytest.mark.asyncio
    async def test_extract_all_async(self, manager: CaseManager) -> None:
        case = manager.create_case()
        content = "原告：甲\n被告：乙"
        manager.import_document(
            case.id, Document(title="d", content=content), auto_extract=False
        )
        total = await manager.extract_all_async(case.id)
        assert len(total["parties"]) >= 2

    @pytest.mark.asyncio
    async def test_import_file_async(self, manager: CaseManager, tmp_path: Path) -> None:
        case = manager.create_case()
        file_path = tmp_path / "doc.txt"
        file_path.write_text("原告：张三", encoding="utf-8")
        material = await manager.import_file_async(case.id, file_path)
        assert material.document.title == "doc.txt"


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_case_creation(self) -> None:
        manager = CaseManager()
        errors: list[Exception] = []

        def create(i: int) -> None:
            try:
                manager.create_case(case_number=f"case-{i}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert manager.case_count == 20

    def test_concurrent_add_party(self, manager: CaseManager) -> None:
        case = manager.create_case()
        errors: list[Exception] = []

        def add_party(i: int) -> None:
            try:
                manager.add_party(
                    case.id, Party(name=f"当事人{i}", role=PartyRole.OTHER)
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=add_party, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        updated = manager.get_case(case.id)
        assert updated is not None
        assert len(updated.parties) == 10


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_empty(self, manager: CaseManager) -> None:
        s = manager.summary()
        assert s["cases"] == 0
        assert s["active_cases"] == 0
        assert s["materials"] == 0

    def test_summary_with_data(self, manager: CaseManager) -> None:
        c1 = manager.create_case()
        c2 = manager.create_case()
        manager.update_status(c1.id, CaseStatus.ACTIVE)
        manager.update_status(c2.id, CaseStatus.CLOSED)
        manager.import_document(c1.id, Document(title="d", content="c"))
        s = manager.summary()
        assert s["cases"] == 2
        assert s["active_cases"] == 1  # only ACTIVE counts
        assert s["materials"] == 1

    def test_parser_property(self, manager: CaseManager) -> None:
        assert isinstance(manager.parser, DocumentParser)


# ---------------------------------------------------------------------------
# Rule-based extraction detail tests
# ---------------------------------------------------------------------------


class TestRuleBasedExtraction:
    def test_extract_with_chinese_numerals_date(self, manager: CaseManager) -> None:
        case = manager.create_case()
        content = "2024年1月15日签订合同"
        doc = Document(title="d", content=content)
        material = manager.import_document(case.id, doc, auto_extract=False)
        extracted = manager.extract_structure(case.id, material.id)
        assert len(extracted["timeline"]) >= 1
        assert extracted["timeline"][0].date == "2024-01-15"

    def test_extract_dashed_date(self, manager: CaseManager) -> None:
        case = manager.create_case()
        content = "2024-03-01交付货物"
        doc = Document(title="d", content=content)
        material = manager.import_document(case.id, doc, auto_extract=False)
        extracted = manager.extract_structure(case.id, material.id)
        assert len(extracted["timeline"]) >= 1

    def test_extract_third_party(self, manager: CaseManager) -> None:
        case = manager.create_case()
        content = "原告：甲\n被告：乙\n第三人：丙"
        doc = Document(title="d", content=content)
        material = manager.import_document(case.id, doc, auto_extract=False)
        extracted = manager.extract_structure(case.id, material.id)
        roles = {p.role for p in extracted["parties"]}
        assert PartyRole.THIRD_PARTY in roles

    def test_extract_empty_content(self, manager: CaseManager) -> None:
        case = manager.create_case()
        doc = Document(title="d", content="")
        material = manager.import_document(case.id, doc, auto_extract=False)
        extracted = manager.extract_structure(case.id, material.id)
        assert extracted["parties"] == []
        assert extracted["claims"] == []
        assert extracted["facts"] == []
        assert extracted["timeline"] == []
