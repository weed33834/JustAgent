"""Tests for the compliance and audit trail module."""

from __future__ import annotations

import json
import time

import pytest

from myagent.security.compliance import (
    AuditResult,
    AuditTrail,
    AuditTrailManager,
    ComplianceChecker,
    ComplianceError,
    ComplianceFramework,
    ComplianceRule,
    PolicyDecision,
    Severity,
)


class TestComplianceChecker:
    def test_default_rules_loaded(self) -> None:
        checker = ComplianceChecker()
        assert checker.rule_count > 0
        gdpr_rules = checker.get_rules(ComplianceFramework.GDPR)
        assert len(gdpr_rules) >= 2
        hipaa_rules = checker.get_rules(ComplianceFramework.HIPAA)
        assert len(hipaa_rules) >= 1

    def test_check_data_access_phi_export_blocked(self) -> None:
        checker = ComplianceChecker()
        decision = checker.check_data_access("alice", "phi", "export")
        assert not decision.allowed
        assert len(decision.violated_rules) > 0
        assert any(r.severity is Severity.CRITICAL for r in decision.violated_rules)

    def test_check_data_access_pii_export_blocked(self) -> None:
        checker = ComplianceChecker()
        decision = checker.check_data_access("alice", "pii", "export")
        assert not decision.allowed

    def test_check_data_access_clean_data_allowed(self) -> None:
        checker = ComplianceChecker()
        decision = checker.check_data_access("alice", "public_data", "read")
        assert decision.allowed

    def test_check_data_access_credit_card_share_blocked(self) -> None:
        checker = ComplianceChecker()
        decision = checker.check_data_access("alice", "credit_card", "share")
        assert not decision.allowed

    def test_add_custom_rule(self) -> None:
        checker = ComplianceChecker()
        rule = ComplianceRule(
            framework=ComplianceFramework.CCPA,
            requirement="right_to_delete",
            description="Users can request data deletion.",
            severity=Severity.HIGH,
            data_types=["pii"],
            actions=["delete"],
        )
        checker.add_rule(rule)
        retrieved = checker.get_rule(rule.id)
        assert retrieved is not None
        assert retrieved.framework is ComplianceFramework.CCPA

    def test_remove_rule(self) -> None:
        checker = ComplianceChecker()
        rule = ComplianceRule(
            framework=ComplianceFramework.NONE,
            requirement="test",
            description="Test rule",
        )
        checker.add_rule(rule)
        removed = checker.remove_rule(rule.id)
        assert removed is not None
        assert checker.get_rule(rule.id) is None

    def test_get_rules_by_framework(self) -> None:
        checker = ComplianceChecker()
        gdpr = checker.get_rules(ComplianceFramework.GDPR)
        for rule in gdpr:
            assert rule.framework is ComplianceFramework.GDPR

    def test_check_data_retention_within_limit(self) -> None:
        checker = ComplianceChecker()
        assert checker.check_data_retention(100, ComplianceFramework.GDPR)

    def test_check_data_retention_exceeds_limit(self) -> None:
        checker = ComplianceChecker()
        assert not checker.check_data_retention(10000, ComplianceFramework.GDPR)

    def test_check_data_retention_no_limit(self) -> None:
        checker = ComplianceChecker()
        assert checker.check_data_retention(999999, ComplianceFramework.NONE)

    def test_export_report_json(self) -> None:
        checker = ComplianceChecker()
        report = checker.export_report("json")
        data = json.loads(report)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "framework" in data[0]

    def test_export_report_csv(self) -> None:
        checker = ComplianceChecker()
        report = checker.export_report("csv")
        assert "rule_id" in report
        assert "framework" in report

    def test_policy_decision_bool(self) -> None:
        d_true = PolicyDecision(allowed=True)
        d_false = PolicyDecision(allowed=False)
        assert bool(d_true) is True
        assert bool(d_false) is False

    def test_summary(self) -> None:
        checker = ComplianceChecker()
        summary = checker.summary()
        assert "total_rules" in summary
        assert "by_framework" in summary
        assert "by_severity" in summary
        assert summary["total_rules"] > 0

    def test_disabled_rule_not_checked(self) -> None:
        checker = ComplianceChecker()
        # Find a blocking rule and disable it.
        rule = ComplianceRule(
            id="test_disable",
            framework=ComplianceFramework.GDPR,
            requirement="test_req",
            description="Test",
            severity=Severity.CRITICAL,
            data_types=["pii"],
            actions=["export"],
        )
        checker.add_rule(rule)
        # Initially blocked.
        d1 = checker.check_data_access("alice", "pii", "export")
        assert not d1.allowed

        # Disable the rule.
        rule.enabled = False
        d2 = checker.check_data_access("alice", "pii", "export")
        # Still blocked by other rules (gdpr_consent).
        assert not d2.allowed


class TestAuditTrail:
    def test_record_and_query(self) -> None:
        manager = AuditTrailManager()
        event = AuditTrail(
            actor="alice",
            action="data.export",
            resource="customer_records",
            result=AuditResult.SUCCESS,
        )
        sealed = manager.record(event)
        assert sealed.entry_hash
        assert sealed.verify()

        entries = manager.query()
        assert len(entries) == 1
        assert entries[0].actor == "alice"

    def test_query_by_actor(self) -> None:
        manager = AuditTrailManager()
        manager.record(AuditTrail(actor="alice", action="read", resource="doc1"))
        manager.record(AuditTrail(actor="bob", action="write", resource="doc2"))
        alice_entries = manager.query({"actor": "alice"})
        assert len(alice_entries) == 1
        assert alice_entries[0].actor == "alice"

    def test_query_by_action(self) -> None:
        manager = AuditTrailManager()
        manager.record(AuditTrail(actor="alice", action="read", resource="doc1"))
        manager.record(AuditTrail(actor="bob", action="write", resource="doc2"))
        write_entries = manager.query({"action": "write"})
        assert len(write_entries) == 1
        assert write_entries[0].action == "write"

    def test_query_by_resource(self) -> None:
        manager = AuditTrailManager()
        manager.record(AuditTrail(actor="alice", action="read", resource="doc1"))
        manager.record(AuditTrail(actor="bob", action="read", resource="doc2"))
        doc1_entries = manager.query({"resource": "doc1"})
        assert len(doc1_entries) == 1

    def test_query_by_result(self) -> None:
        manager = AuditTrailManager()
        manager.record(AuditTrail(actor="alice", action="op1", result=AuditResult.SUCCESS))
        manager.record(AuditTrail(actor="bob", action="op2", result=AuditResult.DENIED))
        denied = manager.query({"result": AuditResult.DENIED})
        assert len(denied) == 1

    def test_hash_chain_integrity(self) -> None:
        manager = AuditTrailManager()
        e1 = manager.record(AuditTrail(actor="alice", action="action1"))
        e2 = manager.record(AuditTrail(actor="bob", action="action2"))
        e3 = manager.record(AuditTrail(actor="carol", action="action3"))

        assert e1.prev_hash == "0" * 64
        assert e2.prev_hash == e1.entry_hash
        assert e3.prev_hash == e2.entry_hash

        # All entries verify.
        assert e1.verify()
        assert e2.verify()
        assert e3.verify()

    def test_record_sealed_event_raises(self) -> None:
        manager = AuditTrailManager()
        event = AuditTrail(actor="alice", action="read")
        sealed = manager.record(event)
        with pytest.raises(ComplianceError, match="already-sealed"):
            manager.record(sealed)

    def test_verify_chain(self) -> None:
        manager = AuditTrailManager()
        for i in range(5):
            manager.record(AuditTrail(actor=f"user{i}", action=f"action{i}"))
        entries = manager.query()
        # All entries should verify.
        for entry in entries:
            assert entry.verify()
        # Chain should be consistent.
        for i in range(1, len(entries)):
            assert entries[i].prev_hash == entries[i - 1].entry_hash

    def test_persistence_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "audit.jsonl"
        manager1 = AuditTrailManager(persistence_path=path)
        manager1.record(AuditTrail(actor="alice", action="action1"))
        manager1.record(AuditTrail(actor="bob", action="action2"))
        assert path.exists()

        manager2 = AuditTrailManager(persistence_path=path)
        entries = manager2.query()
        assert len(entries) == 2
        assert entries[0].actor == "alice"
        assert entries[1].actor == "bob"
        # Chain should be intact after reload.
        assert entries[1].prev_hash == entries[0].entry_hash

    def test_export_json(self) -> None:
        manager = AuditTrailManager()
        manager.record(AuditTrail(actor="alice", action="read", resource="doc1"))
        exported = manager.export("json")
        data = json.loads(exported)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["actor"] == "alice"

    def test_audit_result_values(self) -> None:
        assert AuditResult.SUCCESS.value == "success"
        assert AuditResult.FAILURE.value == "failure"
        assert AuditResult.DENIED.value == "denied"
        assert AuditResult.ERROR.value == "error"

    def test_compliance_framework_values(self) -> None:
        assert ComplianceFramework.GDPR.value == "gdpr"
        assert ComplianceFramework.HIPAA.value == "hipaa"
        assert ComplianceFramework.PCI_DSS.value == "pci_dss"
        assert ComplianceFramework.NONE.value == "none"

    def test_severity_values(self) -> None:
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
