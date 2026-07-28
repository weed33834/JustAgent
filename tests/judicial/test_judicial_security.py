"""Tests for the judicial security module
(justagent.security.judicial_security)."""

from __future__ import annotations

import hashlib
import threading

import pytest

from justagent.security.compliance import AuditResult, AuditTrail
from justagent.security.judicial_security import (
    CaseClassification,
    CaseSecurity,
    CustodyRecord,
    EvidenceChainSecurity,
    EvidenceItem,
    JudicialAuditLogger,
    JudicialSecurityError,
    JudicialSecurityLevel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def case_security() -> CaseSecurity:
    return CaseSecurity()


@pytest.fixture
def audit_logger() -> JudicialAuditLogger:
    return JudicialAuditLogger()


@pytest.fixture
def evidence_chain() -> EvidenceChainSecurity:
    return EvidenceChainSecurity()


# ---------------------------------------------------------------------------
# JudicialSecurityLevel tests
# ---------------------------------------------------------------------------


class TestJudicialSecurityLevel:
    def test_enum_values(self) -> None:
        assert JudicialSecurityLevel.PUBLIC.value == "public"
        assert JudicialSecurityLevel.INTERNAL.value == "internal"
        assert JudicialSecurityLevel.SECRET.value == "secret"
        assert JudicialSecurityLevel.CONFIDENTIAL.value == "confidential"

    def test_from_value_valid(self) -> None:
        assert JudicialSecurityLevel.from_value("public") is JudicialSecurityLevel.PUBLIC
        assert JudicialSecurityLevel.from_value("INTERNAL") is JudicialSecurityLevel.INTERNAL
        assert JudicialSecurityLevel.from_value("Secret") is JudicialSecurityLevel.SECRET
        assert (
            JudicialSecurityLevel.from_value("CONFIDENTIAL")
            is JudicialSecurityLevel.CONFIDENTIAL
        )

    def test_from_value_invalid(self) -> None:
        with pytest.raises(JudicialSecurityError, match="Unknown judicial security level"):
            JudicialSecurityLevel.from_value("top_secret")

    def test_from_value_empty(self) -> None:
        with pytest.raises(JudicialSecurityError, match="Unknown judicial security level"):
            JudicialSecurityLevel.from_value("")

    def test_ge_operator(self) -> None:
        assert JudicialSecurityLevel.SECRET >= JudicialSecurityLevel.PUBLIC
        assert JudicialSecurityLevel.SECRET >= JudicialSecurityLevel.SECRET
        assert not (JudicialSecurityLevel.PUBLIC >= JudicialSecurityLevel.SECRET)

    def test_gt_operator(self) -> None:
        assert JudicialSecurityLevel.CONFIDENTIAL > JudicialSecurityLevel.SECRET
        assert not (JudicialSecurityLevel.INTERNAL > JudicialSecurityLevel.INTERNAL)

    def test_le_operator(self) -> None:
        assert JudicialSecurityLevel.PUBLIC <= JudicialSecurityLevel.INTERNAL
        assert JudicialSecurityLevel.SECRET <= JudicialSecurityLevel.SECRET
        assert not (JudicialSecurityLevel.CONFIDENTIAL <= JudicialSecurityLevel.PUBLIC)

    def test_lt_operator(self) -> None:
        assert JudicialSecurityLevel.PUBLIC < JudicialSecurityLevel.INTERNAL
        assert not (JudicialSecurityLevel.SECRET < JudicialSecurityLevel.SECRET)

    def test_comparison_with_non_level(self) -> None:
        assert JudicialSecurityLevel.PUBLIC.__ge__("not_a_level") is NotImplemented


# ---------------------------------------------------------------------------
# CaseClassification tests
# ---------------------------------------------------------------------------


class TestCaseClassification:
    def test_defaults(self) -> None:
        cc = CaseClassification(case_id="case-1")
        assert cc.case_id == "case-1"
        assert cc.security_level is JudicialSecurityLevel.INTERNAL
        assert cc.classified_by == ""
        assert cc.reason == ""
        assert cc.metadata == {}
        assert cc.classified_at > 0

    def test_with_all_fields(self) -> None:
        cc = CaseClassification(
            case_id="case-1",
            security_level=JudicialSecurityLevel.SECRET,
            classified_by="judge_li",
            reason="涉及国家秘密",
            metadata={"department": "刑事庭"},
        )
        assert cc.security_level is JudicialSecurityLevel.SECRET
        assert cc.classified_by == "judge_li"
        assert cc.reason == "涉及国家秘密"
        assert cc.metadata["department"] == "刑事庭"


# ---------------------------------------------------------------------------
# CaseSecurity tests
# ---------------------------------------------------------------------------


class TestCaseSecurityClassify:
    def test_classify_case(self, case_security: CaseSecurity) -> None:
        cc = case_security.classify_case(
            "case-1",
            JudicialSecurityLevel.SECRET,
            classified_by="judge_li",
            reason="涉及国家秘密",
        )
        assert cc.case_id == "case-1"
        assert cc.security_level is JudicialSecurityLevel.SECRET
        assert cc.classified_by == "judge_li"
        assert cc.reason == "涉及国家秘密"
        assert case_security.case_count == 1

    def test_classify_case_with_metadata(self, case_security: CaseSecurity) -> None:
        cc = case_security.classify_case(
            "case-1",
            JudicialSecurityLevel.CONFIDENTIAL,
            metadata={"department": "国家安全"},
        )
        assert cc.metadata["department"] == "国家安全"

    def test_classify_case_overwrites(self, case_security: CaseSecurity) -> None:
        case_security.classify_case("case-1", JudicialSecurityLevel.PUBLIC)
        case_security.classify_case("case-1", JudicialSecurityLevel.SECRET)
        cc = case_security.get_classification("case-1")
        assert cc is not None
        assert cc.security_level is JudicialSecurityLevel.SECRET
        assert case_security.case_count == 1

    def test_get_classification(self, case_security: CaseSecurity) -> None:
        case_security.classify_case("case-1", JudicialSecurityLevel.SECRET)
        cc = case_security.get_classification("case-1")
        assert cc is not None
        assert cc.security_level is JudicialSecurityLevel.SECRET

    def test_get_classification_not_found(self, case_security: CaseSecurity) -> None:
        assert case_security.get_classification("nonexistent") is None

    def test_update_classification(self, case_security: CaseSecurity) -> None:
        case_security.classify_case("case-1", JudicialSecurityLevel.INTERNAL)
        updated = case_security.update_classification(
            "case-1", JudicialSecurityLevel.SECRET, classified_by="judge_wang"
        )
        assert updated.security_level is JudicialSecurityLevel.SECRET
        assert updated.classified_by == "judge_wang"

    def test_update_classification_not_found(self, case_security: CaseSecurity) -> None:
        with pytest.raises(JudicialSecurityError, match="Case not classified"):
            case_security.update_classification(
                "nonexistent", JudicialSecurityLevel.SECRET
            )

    def test_remove_case(self, case_security: CaseSecurity) -> None:
        case_security.classify_case("case-1", JudicialSecurityLevel.SECRET)
        removed = case_security.remove_case("case-1")
        assert removed is not None
        assert removed.security_level is JudicialSecurityLevel.SECRET
        assert case_security.case_count == 0

    def test_remove_case_not_found(self, case_security: CaseSecurity) -> None:
        assert case_security.remove_case("nonexistent") is None

    def test_list_cases_all(self, case_security: CaseSecurity) -> None:
        case_security.classify_case("case-1", JudicialSecurityLevel.PUBLIC)
        case_security.classify_case("case-2", JudicialSecurityLevel.SECRET)
        assert len(case_security.list_cases()) == 2

    def test_list_cases_filter_by_level(self, case_security: CaseSecurity) -> None:
        case_security.classify_case("case-1", JudicialSecurityLevel.PUBLIC)
        case_security.classify_case("case-2", JudicialSecurityLevel.SECRET)
        case_security.classify_case("case-3", JudicialSecurityLevel.SECRET)
        secret_cases = case_security.list_cases(JudicialSecurityLevel.SECRET)
        assert len(secret_cases) == 2
        assert all(c.security_level is JudicialSecurityLevel.SECRET for c in secret_cases)


class TestCaseSecurityClearance:
    def test_set_and_get_clearance(self, case_security: CaseSecurity) -> None:
        case_security.set_user_clearance("clerk_wang", JudicialSecurityLevel.SECRET)
        assert (
            case_security.get_user_clearance("clerk_wang")
            is JudicialSecurityLevel.SECRET
        )
        assert case_security.user_count == 1

    def test_get_clearance_default(self, case_security: CaseSecurity) -> None:
        assert (
            case_security.get_user_clearance("unknown_user")
            is JudicialSecurityLevel.PUBLIC
        )

    def test_revoke_clearance(self, case_security: CaseSecurity) -> None:
        case_security.set_user_clearance("clerk_wang", JudicialSecurityLevel.SECRET)
        case_security.revoke_clearance("clerk_wang")
        assert (
            case_security.get_user_clearance("clerk_wang")
            is JudicialSecurityLevel.PUBLIC
        )
        assert case_security.user_count == 0

    def test_revoke_clearance_not_set(self, case_security: CaseSecurity) -> None:
        # Should not raise even if user has no clearance.
        case_security.revoke_clearance("unknown_user")
        assert case_security.user_count == 0


class TestCaseSecurityAccess:
    def test_can_access_sufficient_clearance(self, case_security: CaseSecurity) -> None:
        case_security.classify_case("case-1", JudicialSecurityLevel.SECRET)
        case_security.set_user_clearance("clerk_wang", JudicialSecurityLevel.SECRET)
        assert case_security.can_access("clerk_wang", "case-1") is True

    def test_can_access_insufficient_clearance(
        self, case_security: CaseSecurity
    ) -> None:
        case_security.classify_case("case-1", JudicialSecurityLevel.CONFIDENTIAL)
        case_security.set_user_clearance("clerk_wang", JudicialSecurityLevel.SECRET)
        assert case_security.can_access("clerk_wang", "case-1") is False

    def test_can_access_unclassified_case(self, case_security: CaseSecurity) -> None:
        # Unclassified cases default to INTERNAL.
        case_security.set_user_clearance("clerk_wang", JudicialSecurityLevel.INTERNAL)
        assert case_security.can_access("clerk_wang", "unclassified-case") is True

    def test_can_access_unclassified_case_low_clearance(
        self, case_security: CaseSecurity
    ) -> None:
        # Default clearance is PUBLIC, which is below INTERNAL.
        assert case_security.can_access("default_user", "unclassified-case") is False

    def test_can_access_public_case(self, case_security: CaseSecurity) -> None:
        case_security.classify_case("case-1", JudicialSecurityLevel.PUBLIC)
        # Even a user with only default (PUBLIC) clearance can access.
        assert case_security.can_access("anyone", "case-1") is True


class TestCaseSecuritySummary:
    def test_summary_empty(self, case_security: CaseSecurity) -> None:
        s = case_security.summary()
        assert s["total_cases"] == 0
        assert s["total_users_with_clearance"] == 0
        assert s["by_level"] == {}

    def test_summary_with_data(self, case_security: CaseSecurity) -> None:
        case_security.classify_case("case-1", JudicialSecurityLevel.PUBLIC)
        case_security.classify_case("case-2", JudicialSecurityLevel.SECRET)
        case_security.classify_case("case-3", JudicialSecurityLevel.SECRET)
        case_security.set_user_clearance("user_a", JudicialSecurityLevel.SECRET)
        s = case_security.summary()
        assert s["total_cases"] == 3
        assert s["total_users_with_clearance"] == 1
        assert s["by_level"]["public"] == 1
        assert s["by_level"]["secret"] == 2


# ---------------------------------------------------------------------------
# JudicialAuditLogger tests
# ---------------------------------------------------------------------------


class TestJudicialAuditLogger:
    def test_log_review(self, audit_logger: JudicialAuditLogger) -> None:
        entry = audit_logger.log_review(
            "judge_li", case_id="case-1", resource="verdict.pdf"
        )
        assert entry.actor == "judge_li"
        assert entry.action == "judicial.review"
        assert entry.resource == "verdict.pdf"
        assert entry.metadata["case_id"] == "case-1"
        assert entry.entry_hash  # should be sealed
        assert audit_logger.count == 1

    def test_log_seal(self, audit_logger: JudicialAuditLogger) -> None:
        entry = audit_logger.log_seal(
            "judge_wang", case_id="case-2", resource="judgment.pdf"
        )
        assert entry.action == "judicial.seal"
        assert entry.metadata["case_id"] == "case-2"

    def test_log_archive(self, audit_logger: JudicialAuditLogger) -> None:
        entry = audit_logger.log_archive(
            "clerk_zhang", case_id="case-3", resource="case-file-001"
        )
        assert entry.action == "judicial.archive"
        assert entry.metadata["case_id"] == "case-3"

    def test_log_serve(self, audit_logger: JudicialAuditLogger) -> None:
        entry = audit_logger.log_serve(
            "clerk_li",
            case_id="case-4",
            target="被告甲公司",
            resource="summons.pdf",
        )
        assert entry.action == "judicial.serve"
        assert entry.metadata["case_id"] == "case-4"
        assert entry.metadata["target"] == "被告甲公司"

    def test_log_serve_without_target(self, audit_logger: JudicialAuditLogger) -> None:
        entry = audit_logger.log_serve("clerk_li", case_id="case-4")
        assert entry.action == "judicial.serve"
        assert "target" not in entry.metadata

    def test_log_evidence_access(self, audit_logger: JudicialAuditLogger) -> None:
        entry = audit_logger.log_evidence_access(
            "officer_zhang",
            case_id="case-5",
            evidence_id="ev-1",
            action="collect",
        )
        assert entry.action == "judicial.evidence"
        assert entry.metadata["case_id"] == "case-5"
        assert entry.metadata["evidence_id"] == "ev-1"
        assert entry.metadata["evidence_action"] == "collect"

    def test_log_with_failure_result(self, audit_logger: JudicialAuditLogger) -> None:
        entry = audit_logger.log_review(
            "judge_li",
            case_id="case-1",
            result=AuditResult.FAILURE,
        )
        assert entry.result is AuditResult.FAILURE

    def test_log_with_ip_and_user_agent(
        self, audit_logger: JudicialAuditLogger
    ) -> None:
        entry = audit_logger.log_review(
            "judge_li",
            case_id="case-1",
            ip_address="10.0.0.1",
            user_agent="Mozilla/5.0",
        )
        assert entry.ip_address == "10.0.0.1"
        assert entry.user_agent == "Mozilla/5.0"

    def test_log_with_extra_metadata(self, audit_logger: JudicialAuditLogger) -> None:
        entry = audit_logger.log_review(
            "judge_li",
            case_id="case-1",
            metadata={"custom_field": "custom_value"},
        )
        assert entry.metadata["custom_field"] == "custom_value"
        assert entry.metadata["case_id"] == "case-1"

    def test_query_by_case(self, audit_logger: JudicialAuditLogger) -> None:
        audit_logger.log_review("judge_li", case_id="case-1")
        audit_logger.log_seal("judge_wang", case_id="case-1")
        audit_logger.log_review("judge_li", case_id="case-2")
        results = audit_logger.query_by_case("case-1")
        assert len(results) == 2
        assert all(r.metadata["case_id"] == "case-1" for r in results)

    def test_query_by_case_empty(self, audit_logger: JudicialAuditLogger) -> None:
        audit_logger.log_review("judge_li", case_id="case-1")
        assert audit_logger.query_by_case("nonexistent") == []

    def test_query_by_evidence(self, audit_logger: JudicialAuditLogger) -> None:
        audit_logger.log_evidence_access(
            "officer_a", case_id="case-1", evidence_id="ev-1", action="collect"
        )
        audit_logger.log_evidence_access(
            "officer_b", case_id="case-1", evidence_id="ev-1", action="inspect"
        )
        audit_logger.log_evidence_access(
            "officer_a", case_id="case-1", evidence_id="ev-2", action="collect"
        )
        results = audit_logger.query_by_evidence("ev-1")
        assert len(results) == 2
        assert all(r.metadata["evidence_id"] == "ev-1" for r in results)

    def test_query_by_evidence_empty(self, audit_logger: JudicialAuditLogger) -> None:
        assert audit_logger.query_by_evidence("nonexistent") == []

    def test_chain_integrity(self, audit_logger: JudicialAuditLogger) -> None:
        for i in range(5):
            audit_logger.log_review("judge_li", case_id=f"case-{i}")
        assert audit_logger.verify_chain() is True

    def test_inherited_query_method(self, audit_logger: JudicialAuditLogger) -> None:
        audit_logger.log_review("alice", case_id="case-1")
        audit_logger.log_review("alice", case_id="case-2")
        audit_logger.log_seal("bob", case_id="case-3")
        alice_entries = audit_logger.query({"actor": "alice"})
        assert len(alice_entries) == 2
        bob_entries = audit_logger.query({"actor": "bob"})
        assert len(bob_entries) == 1

    def test_inherited_count(self, audit_logger: JudicialAuditLogger) -> None:
        audit_logger.log_review("judge_li", case_id="case-1")
        audit_logger.log_seal("judge_li", case_id="case-1")
        assert audit_logger.count == 2
        assert len(audit_logger) == 2

    def test_inherited_summary(self, audit_logger: JudicialAuditLogger) -> None:
        audit_logger.log_review("judge_li", case_id="case-1")
        audit_logger.log_seal("judge_li", case_id="case-1", result=AuditResult.FAILURE)
        s = audit_logger.summary()
        assert s["total_entries"] == 2
        assert s["by_action"]["judicial.review"] == 1
        assert s["by_action"]["judicial.seal"] == 1
        assert s["by_result"]["success"] == 1
        assert s["by_result"]["failure"] == 1

    def test_genesis_entry_prev_hash(self, audit_logger: JudicialAuditLogger) -> None:
        entry = audit_logger.log_review("judge_li", case_id="case-1")
        assert entry.prev_hash == "0" * 64

    def test_chained_prev_hash(self, audit_logger: JudicialAuditLogger) -> None:
        e1 = audit_logger.log_review("judge_li", case_id="case-1")
        e2 = audit_logger.log_seal("judge_li", case_id="case-1")
        assert e2.prev_hash == e1.entry_hash


# ---------------------------------------------------------------------------
# EvidenceItem tests
# ---------------------------------------------------------------------------


class TestEvidenceItem:
    def test_defaults(self) -> None:
        item = EvidenceItem(
            case_id="case-1",
            content_hash="abc123",
        )
        assert item.case_id == "case-1"
        assert item.content_hash == "abc123"
        assert item.registered_by == ""
        assert item.description == ""
        assert item.metadata == {}
        assert item.evidence_id
        assert item.registered_at > 0

    def test_auto_id_unique(self) -> None:
        item1 = EvidenceItem(case_id="case-1", content_hash="a")
        item2 = EvidenceItem(case_id="case-1", content_hash="b")
        assert item1.evidence_id != item2.evidence_id


# ---------------------------------------------------------------------------
# CustodyRecord tests
# ---------------------------------------------------------------------------


class TestCustodyRecord:
    def test_defaults(self) -> None:
        record = CustodyRecord(
            evidence_id="ev-1",
            handler="officer_zhang",
            action="collect",
        )
        assert record.evidence_id == "ev-1"
        assert record.handler == "officer_zhang"
        assert record.action == "collect"
        assert record.notes == ""
        assert record.prev_hash == "0" * 64
        assert record.record_hash == ""
        assert record.record_id
        assert record.timestamp > 0

    def test_auto_id_unique(self) -> None:
        r1 = CustodyRecord(evidence_id="ev-1", handler="a", action="collect")
        r2 = CustodyRecord(evidence_id="ev-1", handler="b", action="transfer")
        assert r1.record_id != r2.record_id

    def test_compute_hash(self) -> None:
        record = CustodyRecord(
            evidence_id="ev-1",
            handler="officer_zhang",
            action="collect",
            notes="from crime scene",
        )
        h = record.compute_hash()
        assert len(h) == 64  # SHA-256 hex digest

    def test_compute_hash_deterministic(self) -> None:
        record = CustodyRecord(
            evidence_id="ev-1",
            handler="officer_zhang",
            action="collect",
            notes="test",
        )
        assert record.compute_hash() == record.compute_hash()

    def test_seal(self) -> None:
        record = CustodyRecord(
            evidence_id="ev-1",
            handler="officer_zhang",
            action="collect",
        )
        sealed = record.seal()
        assert sealed.record_hash != ""
        assert sealed.record_hash == record.compute_hash()

    def test_seal_idempotent(self) -> None:
        record = CustodyRecord(
            evidence_id="ev-1",
            handler="officer_zhang",
            action="collect",
        )
        sealed1 = record.seal()
        sealed2 = sealed1.seal()
        assert sealed2 is sealed1

    def test_verify_unsealed(self) -> None:
        record = CustodyRecord(
            evidence_id="ev-1",
            handler="officer_zhang",
            action="collect",
        )
        assert record.verify() is False

    def test_verify_sealed(self) -> None:
        record = CustodyRecord(
            evidence_id="ev-1",
            handler="officer_zhang",
            action="collect",
        )
        sealed = record.seal()
        assert sealed.verify() is True

    def test_verify_tampered(self) -> None:
        record = CustodyRecord(
            evidence_id="ev-1",
            handler="officer_zhang",
            action="collect",
        )
        sealed = record.seal()
        # Tamper with the handler after sealing.
        sealed.handler = "hacker"
        assert sealed.verify() is False

    def test_signing_payload_excludes_record_hash(self) -> None:
        record = CustodyRecord(
            evidence_id="ev-1",
            handler="officer_zhang",
            action="collect",
        )
        sealed = record.seal()
        # The signing payload should not include record_hash.
        payload = sealed._signing_payload()
        assert "record_hash" not in payload


# ---------------------------------------------------------------------------
# EvidenceChainSecurity tests
# ---------------------------------------------------------------------------


class TestEvidenceRegistration:
    def test_register_evidence(self, evidence_chain: EvidenceChainSecurity) -> None:
        item = evidence_chain.register_evidence(
            "ev-1",
            "case-1",
            b"evidence content",
            registered_by="officer_zhang",
            description="现场提取的物证",
        )
        assert item.evidence_id == "ev-1"
        assert item.case_id == "case-1"
        assert item.registered_by == "officer_zhang"
        assert item.description == "现场提取的物证"
        assert item.content_hash == hashlib.sha256(b"evidence content").hexdigest()
        assert evidence_chain.evidence_count == 1

    def test_register_evidence_auto_id(self, evidence_chain: EvidenceChainSecurity) -> None:
        item = evidence_chain.register_evidence(
            None,
            "case-1",
            b"data",
        )
        assert item.evidence_id  # non-empty auto-generated ID

    def test_register_evidence_string_content(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        item = evidence_chain.register_evidence(
            "ev-1",
            "case-1",
            "text evidence",
        )
        assert item.content_hash == hashlib.sha256("text evidence".encode("utf-8")).hexdigest()

    def test_register_evidence_with_metadata(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        item = evidence_chain.register_evidence(
            "ev-1",
            "case-1",
            b"data",
            metadata={"location": "scene_1", "time": "2024-01-01"},
        )
        assert item.metadata["location"] == "scene_1"
        assert item.metadata["time"] == "2024-01-01"

    def test_register_duplicate_raises(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"data")
        with pytest.raises(JudicialSecurityError, match="Evidence already registered"):
            evidence_chain.register_evidence("ev-1", "case-1", b"different data")

    def test_get_evidence(self, evidence_chain: EvidenceChainSecurity) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"data")
        item = evidence_chain.get_evidence("ev-1")
        assert item is not None
        assert item.evidence_id == "ev-1"

    def test_get_evidence_not_found(self, evidence_chain: EvidenceChainSecurity) -> None:
        assert evidence_chain.get_evidence("nonexistent") is None

    def test_list_evidence_all(self, evidence_chain: EvidenceChainSecurity) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"a")
        evidence_chain.register_evidence("ev-2", "case-1", b"b")
        evidence_chain.register_evidence("ev-3", "case-2", b"c")
        assert len(evidence_chain.list_evidence()) == 3

    def test_list_evidence_by_case(self, evidence_chain: EvidenceChainSecurity) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"a")
        evidence_chain.register_evidence("ev-2", "case-1", b"b")
        evidence_chain.register_evidence("ev-3", "case-2", b"c")
        case1_items = evidence_chain.list_evidence("case-1")
        assert len(case1_items) == 2
        assert all(i.case_id == "case-1" for i in case1_items)


class TestCustodyChain:
    def test_add_custody(self, evidence_chain: EvidenceChainSecurity) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"data")
        record = evidence_chain.add_custody(
            "ev-1", "lab_tech", "transfer", notes="送检"
        )
        assert record.evidence_id == "ev-1"
        assert record.handler == "lab_tech"
        assert record.action == "transfer"
        assert record.notes == "送检"
        assert record.record_hash  # sealed
        assert record.prev_hash == "0" * 64  # genesis

    def test_add_custody_not_registered(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        with pytest.raises(JudicialSecurityError, match="Evidence not registered"):
            evidence_chain.add_custody("nonexistent", "handler", "action")

    def test_add_multiple_custody_chains(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"data")
        r1 = evidence_chain.add_custody("ev-1", "officer_a", "collect")
        r2 = evidence_chain.add_custody("ev-1", "officer_b", "transfer")
        r3 = evidence_chain.add_custody("ev-1", "lab_tech", "inspect")
        chain = evidence_chain.get_custody_chain("ev-1")
        assert len(chain) == 3
        # Each record should link to the previous.
        assert r1.prev_hash == "0" * 64
        assert r2.prev_hash == r1.record_hash
        assert r3.prev_hash == r2.record_hash

    def test_get_custody_chain_empty(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"data")
        assert evidence_chain.get_custody_chain("ev-1") == []

    def test_get_custody_chain_not_registered(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        assert evidence_chain.get_custody_chain("nonexistent") == []


class TestEvidenceVerification:
    def test_verify_evidence_match(self, evidence_chain: EvidenceChainSecurity) -> None:
        content = b"original evidence data"
        evidence_chain.register_evidence("ev-1", "case-1", content)
        assert evidence_chain.verify_evidence("ev-1", content) is True

    def test_verify_evidence_mismatch(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"original")
        assert evidence_chain.verify_evidence("ev-1", b"tampered") is False

    def test_verify_evidence_not_found(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        assert evidence_chain.verify_evidence("nonexistent", b"data") is False

    def test_verify_evidence_string_content(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", "text data")
        assert evidence_chain.verify_evidence("ev-1", "text data") is True
        assert evidence_chain.verify_evidence("ev-1", "wrong text") is False


class TestChainVerification:
    def test_verify_chain_single_evidence(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"data")
        evidence_chain.add_custody("ev-1", "officer_a", "collect")
        evidence_chain.add_custody("ev-1", "officer_b", "transfer")
        assert evidence_chain.verify_chain("ev-1") is True

    def test_verify_chain_all(self, evidence_chain: EvidenceChainSecurity) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"data1")
        evidence_chain.register_evidence("ev-2", "case-2", b"data2")
        evidence_chain.add_custody("ev-1", "officer_a", "collect")
        evidence_chain.add_custody("ev-2", "officer_b", "collect")
        assert evidence_chain.verify_chain() is True

    def test_verify_chain_empty(self, evidence_chain: EvidenceChainSecurity) -> None:
        assert evidence_chain.verify_chain() is True

    def test_verify_chain_no_custody(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"data")
        assert evidence_chain.verify_chain("ev-1") is True

    def test_verify_chain_not_registered(
        self, evidence_chain: EvidenceChainSecurity
    ) -> None:
        # Verifying a non-registered evidence ID verifies an empty chain.
        assert evidence_chain.verify_chain("nonexistent") is True


class TestEvidenceChainSummary:
    def test_summary_empty(self, evidence_chain: EvidenceChainSecurity) -> None:
        s = evidence_chain.summary()
        assert s["total_evidence"] == 0
        assert s["total_custody_records"] == 0
        assert s["total_cases"] == 0
        assert s["all_chains_valid"] is True

    def test_summary_with_data(self, evidence_chain: EvidenceChainSecurity) -> None:
        evidence_chain.register_evidence("ev-1", "case-1", b"data1")
        evidence_chain.register_evidence("ev-2", "case-1", b"data2")
        evidence_chain.register_evidence("ev-3", "case-2", b"data3")
        evidence_chain.add_custody("ev-1", "officer_a", "collect")
        evidence_chain.add_custody("ev-1", "officer_b", "transfer")
        evidence_chain.add_custody("ev-2", "officer_c", "collect")
        s = evidence_chain.summary()
        assert s["total_evidence"] == 3
        assert s["total_custody_records"] == 3
        assert s["total_cases"] == 2
        assert s["all_chains_valid"] is True

    def test_evidence_count(self, evidence_chain: EvidenceChainSecurity) -> None:
        assert evidence_chain.evidence_count == 0
        evidence_chain.register_evidence("ev-1", "case-1", b"data")
        assert evidence_chain.evidence_count == 1


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestCaseSecurityThreadSafety:
    def test_concurrent_classify(self) -> None:
        cs = CaseSecurity()
        errors: list[Exception] = []

        def classify(i: int) -> None:
            try:
                level = list(JudicialSecurityLevel)[i % 4]
                cs.classify_case(f"case-{i}", level)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=classify, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cs.case_count == 20

    def test_concurrent_clearance_and_access(self) -> None:
        cs = CaseSecurity()
        cs.classify_case("case-1", JudicialSecurityLevel.SECRET)
        errors: list[Exception] = []

        def access(user: str) -> None:
            try:
                cs.set_user_clearance(user, JudicialSecurityLevel.SECRET)
                cs.can_access(user, "case-1")
                cs.get_user_clearance(user)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=access, args=(f"user-{i}",)) for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cs.user_count == 20


class TestAuditLoggerThreadSafety:
    def test_concurrent_logging(self) -> None:
        logger = JudicialAuditLogger()
        errors: list[Exception] = []

        def log(i: int) -> None:
            try:
                logger.log_review(
                    f"judge_{i}", case_id=f"case-{i}", resource=f"doc-{i}.pdf"
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=log, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert logger.count == 20
        # The hash chain should remain intact.
        assert logger.verify_chain() is True


class TestEvidenceChainThreadSafety:
    def test_concurrent_register(self) -> None:
        ecs = EvidenceChainSecurity()
        errors: list[Exception] = []

        def register(i: int) -> None:
            try:
                ecs.register_evidence(
                    f"ev-{i}", "case-1", f"data-{i}".encode()
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=register, args=(i,)) for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert ecs.evidence_count == 20
        assert ecs.verify_chain() is True

    def test_concurrent_add_custody(self) -> None:
        ecs = EvidenceChainSecurity()
        ecs.register_evidence("ev-1", "case-1", b"data")
        errors: list[Exception] = []

        def add_custody(i: int) -> None:
            try:
                ecs.add_custody("ev-1", f"handler-{i}", "transfer")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=add_custody, args=(i,)) for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        chain = ecs.get_custody_chain("ev-1")
        assert len(chain) == 20
        assert ecs.verify_chain("ev-1") is True

    def test_concurrent_register_and_verify(self) -> None:
        ecs = EvidenceChainSecurity()
        errors: list[Exception] = []

        def register_and_verify(i: int) -> None:
            try:
                ecs.register_evidence(f"ev-{i}", "case-1", f"data-{i}".encode())
                ecs.verify_chain()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=register_and_verify, args=(i,)) for i in range(15)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert ecs.evidence_count == 15
