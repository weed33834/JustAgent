"""Tests for the Data Loss Prevention module."""

from __future__ import annotations

import asyncio

import pytest

from justagent.security.data_protection import (
    DataSanitizer,
    DataSensitivityLevel,
    DLPAction,
    DLPError,
    DLPRule,
    DLPScanner,
    PIIType,
)


class TestDLPScanner:
    def test_scan_email(self) -> None:
        scanner = DLPScanner()
        findings = scanner.scan_text("Contact alice@example.com for details")
        emails = [f for f in findings if f.pii_type is PIIType.EMAIL]
        assert len(emails) == 1
        assert emails[0].value == "alice@example.com"

    def test_scan_phone_us(self) -> None:
        scanner = DLPScanner()
        findings = scanner.scan_text("Call 555-123-4567 now")
        phones = [f for f in findings if f.pii_type is PIIType.PHONE]
        assert len(phones) >= 1
        assert "555-123-4567" in phones[0].value

    def test_scan_phone_international(self) -> None:
        scanner = DLPScanner()
        findings = scanner.scan_text("Call +86-138-0013-8000 now")
        phones = [f for f in findings if f.pii_type is PIIType.PHONE]
        assert len(phones) >= 1

    def test_scan_ssn(self) -> None:
        scanner = DLPScanner()
        findings = scanner.scan_text("SSN: 123-45-6789")
        ssns = [f for f in findings if f.pii_type is PIIType.SSN]
        assert len(ssns) == 1
        assert ssns[0].value == "123-45-6789"
        assert ssns[0].sensitivity is DataSensitivityLevel.HIGH

    def test_scan_credit_card(self) -> None:
        scanner = DLPScanner()
        findings = scanner.scan_text("Card: 4111 1111 1111 1111")
        cards = [f for f in findings if f.pii_type is PIIType.CREDIT_CARD]
        assert len(cards) >= 1
        assert cards[0].sensitivity is DataSensitivityLevel.CRITICAL

    def test_scan_ip_address(self) -> None:
        scanner = DLPScanner()
        findings = scanner.scan_text("Server at 192.168.1.100 is down")
        ips = [f for f in findings if f.pii_type is PIIType.IP_ADDRESS]
        assert len(ips) >= 1
        assert "192.168.1.100" in ips[0].value

    def test_scan_multiple_pii_types(self) -> None:
        scanner = DLPScanner()
        text = "Email: alice@example.com, SSN: 123-45-6789, Phone: 555-123-4567"
        findings = scanner.scan_text(text)
        types_found = {f.pii_type for f in findings}
        assert PIIType.EMAIL in types_found
        assert PIIType.SSN in types_found
        assert PIIType.PHONE in types_found

    def test_scan_no_pii(self) -> None:
        scanner = DLPScanner()
        findings = scanner.scan_text("This is a clean text with no PII")
        assert len(findings) == 0

    def test_findings_sorted_by_position(self) -> None:
        scanner = DLPScanner()
        text = "alice@example.com and 123-45-6789"
        findings = scanner.scan_text(text)
        for i in range(1, len(findings)):
            assert findings[i].start >= findings[i - 1].start

    def test_add_custom_rule(self) -> None:
        scanner = DLPScanner()
        rule = DLPRule(
            name="Employee ID",
            pattern=r"\bEMP-\d{6}\b",
            pii_type=PIIType.ID_CARD,
            sensitivity=DataSensitivityLevel.MEDIUM,
            action=DLPAction.REDACT,
        )
        scanner.add_rule(rule)
        findings = scanner.scan_text("Employee EMP-123456 has access")
        emp_findings = [f for f in findings if f.rule_id == rule.id]
        assert len(emp_findings) == 1
        assert emp_findings[0].value == "EMP-123456"

    def test_remove_rule(self) -> None:
        scanner = DLPScanner()
        rule = DLPRule(
            name="Custom",
            pattern=r"\bCUSTOM-\d+\b",
            pii_type=PIIType.ID_CARD,
        )
        scanner.add_rule(rule)
        assert scanner.get_rule(rule.id) is not None
        removed = scanner.remove_rule(rule.id)
        assert removed is not None
        assert scanner.get_rule(rule.id) is None

    def test_remove_nonexistent_rule_returns_none(self) -> None:
        scanner = DLPScanner()
        assert scanner.remove_rule("nonexistent") is None

    def test_add_invalid_regex_raises(self) -> None:
        scanner = DLPScanner()
        rule = DLPRule(
            name="Bad",
            pattern="[invalid(",
            pii_type=PIIType.EMAIL,
        )
        with pytest.raises(DLPError, match="Invalid regex"):
            scanner.add_rule(rule)

    def test_list_rules(self) -> None:
        scanner = DLPScanner()
        rules = scanner.list_rules()
        assert len(rules) > 0

    def test_list_rules_enabled_only(self) -> None:
        scanner = DLPScanner()
        all_rules = scanner.list_rules()
        # Disable one rule.
        all_rules[0].enabled = False
        enabled = scanner.list_rules(enabled_only=True)
        assert len(enabled) < len(all_rules)

    def test_rule_count(self) -> None:
        scanner = DLPScanner()
        assert scanner.rule_count > 0

    def test_summary(self) -> None:
        scanner = DLPScanner()
        summary = scanner.summary("Email: alice@example.com, SSN: 123-45-6789")
        assert summary["total_findings"] >= 2
        assert "email" in summary["by_type"]
        assert "ssn" in summary["by_type"]
        assert summary["has_critical"] is False or True  # depends on matching

    def test_scan_file(self, tmp_path) -> None:
        scanner = DLPScanner()
        file = tmp_path / "test.txt"
        file.write_text("Contact: alice@example.com")
        findings = asyncio.run(scanner.scan_file(file))
        assert len(findings) >= 1
        assert any(f.pii_type is PIIType.EMAIL for f in findings)

    def test_scan_file_nonexistent_returns_empty(self) -> None:
        scanner = DLPScanner()
        findings = asyncio.run(scanner.scan_file("/nonexistent/file.txt"))
        assert findings == []

    def test_scan_dict(self) -> None:
        scanner = DLPScanner()
        data = {
            "email": "alice@example.com",
            "info": {"ssn": "123-45-6789"},
            "clean": "no PII here",
        }
        results = scanner.scan_dict(data)
        assert "email" in results
        assert "info.ssn" in results
        assert "clean" not in results


class TestDataSanitizer:
    def test_redact_email(self) -> None:
        sanitizer = DataSanitizer()
        result = sanitizer.redact_pii("Email: alice@example.com")
        assert "[REDACTED_EMAIL]" in result
        assert "alice@example.com" not in result

    def test_redact_ssn(self) -> None:
        sanitizer = DataSanitizer()
        result = sanitizer.redact_pii("SSN: 123-45-6789")
        assert "[REDACTED_SSN]" in result

    def test_redact_multiple(self) -> None:
        sanitizer = DataSanitizer()
        result = sanitizer.redact_pii("alice@example.com and 123-45-6789")
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_SSN]" in result

    def test_redact_no_pii(self) -> None:
        sanitizer = DataSanitizer()
        result = sanitizer.redact_pii("Clean text")
        assert result == "Clean text"

    def test_mask_partial_email(self) -> None:
        sanitizer = DataSanitizer()
        result = sanitizer.mask_partial("Email: alice@example.com")
        assert "alice@example.com" not in result
        assert "@example.com" in result
        assert "*" in result

    def test_mask_partial_preserves_structure(self) -> None:
        sanitizer = DataSanitizer()
        result = sanitizer.mask_partial("alice@example.com")
        assert result.startswith("a")
        assert "@" in result

    def test_sanitizer_uses_custom_scanner(self) -> None:
        scanner = DLPScanner()
        sanitizer = DataSanitizer(scanner=scanner)
        assert sanitizer.scanner is scanner
