"""Data Loss Prevention — PII scanning, classification and sanitisation.

Provides a DLP engine that detects personally identifiable information
(PII) in text and files, classifies findings by sensitivity, and
sanitises content by redacting or masking sensitive values.

Design:

* :class:`PIIType` — categories of detectable PII.
* :class:`DataSensitivityLevel` — severity tiers for findings.
* :class:`DLPAction` — actions a DLP rule can prescribe.
* :class:`DLPRule` — a named regex-based detection rule.
* :class:`PIIFinding` — a single PII match with location and metadata.
* :class:`DLPScanner` — thread-safe scanner with default PII patterns.
* :class:`DataSanitizer` — redaction and partial-masking utilities.

The scanner ships with sensible default regex patterns for common PII
types (email, phone, SSN, credit card, IP address, etc.). Custom rules
can be added or removed at runtime.

Example::

    scanner = DLPScanner()
    findings = scanner.scan_text("Contact me at alice@example.com or 555-123-4567")
    # -> [PIIFinding(pii_type=EMAIL, ...), PIIFinding(pii_type=PHONE, ...)]

    sanitizer = DataSanitizer()
    clean = sanitizer.redact_pii("My SSN is 123-45-6789")
    # -> "My SSN is [REDACTED_SSN]"
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("justagent.security.data_protection")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DLPError(Exception):
    """Raised for DLP scanning or sanitisation errors."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PIIType(str, Enum):  # noqa: UP042 - match existing codebase style
    """Categories of personally identifiable information.

    Attributes:
        EMAIL: Email addresses.
        PHONE: Phone numbers (international and US formats).
        SSN: US Social Security Numbers.
        CREDIT_CARD: Credit / debit card numbers.
        IP_ADDRESS: IPv4 and IPv6 addresses.
        PASSPORT: Passport numbers (common formats).
        BANK_ACCOUNT: Bank account numbers.
        ID_CARD: National ID / driver's license numbers.
        ADDRESS: Street addresses.
        NAME: Person names (conservative heuristic).
        CHINESE_ID_CARD: PRC resident identity card numbers (18-digit).
        UNIFIED_SOCIAL_CREDIT_CODE: PRC unified social credit codes.
        CASE_NUMBER: PRC judicial case numbers (案号).
        BUSINESS_LICENSE: PRC business licence numbers (营业执照号).
    """

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    ID_CARD = "id_card"
    ADDRESS = "address"
    NAME = "name"
    CHINESE_ID_CARD = "chinese_id_card"
    UNIFIED_SOCIAL_CREDIT_CODE = "unified_social_credit_code"
    CASE_NUMBER = "case_number"
    BUSINESS_LICENSE = "business_license"


class DataSensitivityLevel(str, Enum):  # noqa: UP042
    """Sensitivity tier for a PII finding.

    Attributes:
        LOW: Minimal risk if exposed (e.g. first name).
        MEDIUM: Moderate risk (e.g. email, phone).
        HIGH: Significant risk (e.g. SSN, passport).
        CRITICAL: Severe risk (e.g. full credit card, bank account).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DLPAction(str, Enum):  # noqa: UP042
    """Action prescribed by a DLP rule when a match is found.

    Attributes:
        ALLOW: Permit the data (log only).
        BLOCK: Prevent the operation entirely.
        REDACT: Replace the matched value with a placeholder.
        QUARANTINE: Isolate the content for review.
        ENCRYPT: Encrypt the matched value before storage.
        AUDIT: Record the finding in the audit trail.
    """

    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    QUARANTINE = "quarantine"
    ENCRYPT = "encrypt"
    AUDIT = "audit"


# ---------------------------------------------------------------------------
# Default PII patterns
# ---------------------------------------------------------------------------

#: Default sensitivity mapping per PII type.
_DEFAULT_SENSITIVITY: dict[PIIType, DataSensitivityLevel] = {
    PIIType.EMAIL: DataSensitivityLevel.MEDIUM,
    PIIType.PHONE: DataSensitivityLevel.MEDIUM,
    PIIType.SSN: DataSensitivityLevel.HIGH,
    PIIType.CREDIT_CARD: DataSensitivityLevel.CRITICAL,
    PIIType.IP_ADDRESS: DataSensitivityLevel.LOW,
    PIIType.PASSPORT: DataSensitivityLevel.HIGH,
    PIIType.BANK_ACCOUNT: DataSensitivityLevel.CRITICAL,
    PIIType.ID_CARD: DataSensitivityLevel.HIGH,
    PIIType.ADDRESS: DataSensitivityLevel.MEDIUM,
    PIIType.NAME: DataSensitivityLevel.LOW,
    # PRC (China) specific PII types
    PIIType.CHINESE_ID_CARD: DataSensitivityLevel.CRITICAL,
    PIIType.UNIFIED_SOCIAL_CREDIT_CODE: DataSensitivityLevel.HIGH,
    PIIType.CASE_NUMBER: DataSensitivityLevel.MEDIUM,
    PIIType.BUSINESS_LICENSE: DataSensitivityLevel.MEDIUM,
}


def _build_default_rules() -> list[DLPRule]:
    """Return the built-in default DLP rules."""

    return [
        DLPRule(
            id="default_email",
            name="Email Address",
            pattern=r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            pii_type=PIIType.EMAIL,
            sensitivity=DataSensitivityLevel.MEDIUM,
            action=DLPAction.REDACT,
            description="Detects standard email addresses.",
        ),
        DLPRule(
            id="default_phone_us",
            name="US Phone Number",
            pattern=r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            pii_type=PIIType.PHONE,
            sensitivity=DataSensitivityLevel.MEDIUM,
            action=DLPAction.REDACT,
            description="Detects US phone numbers in common formats.",
        ),
        DLPRule(
            id="default_phone_intl",
            name="International Phone Number",
            pattern=r"(?<!\w)\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{3,}[-.\s]?\d{3,}\b",
            pii_type=PIIType.PHONE,
            sensitivity=DataSensitivityLevel.MEDIUM,
            action=DLPAction.REDACT,
            description="Detects international phone numbers with country code.",
        ),
        DLPRule(
            id="default_ssn",
            name="US Social Security Number",
            pattern=r"\b\d{3}-\d{2}-\d{4}\b",
            pii_type=PIIType.SSN,
            sensitivity=DataSensitivityLevel.HIGH,
            action=DLPAction.BLOCK,
            description="Detects US SSN in XXX-XX-XXXX format.",
        ),
        DLPRule(
            id="default_credit_card",
            name="Credit Card Number",
            pattern=r"\b(?:\d[ -]*?){13,19}\b",
            pii_type=PIIType.CREDIT_CARD,
            sensitivity=DataSensitivityLevel.CRITICAL,
            action=DLPAction.BLOCK,
            description="Detects 13-19 digit credit card number patterns.",
        ),
        DLPRule(
            id="default_ipv4",
            name="IPv4 Address",
            pattern=r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
            pii_type=PIIType.IP_ADDRESS,
            sensitivity=DataSensitivityLevel.LOW,
            action=DLPAction.AUDIT,
            description="Detects IPv4 addresses.",
        ),
        DLPRule(
            id="default_ipv6",
            name="IPv6 Address",
            pattern=r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b",
            pii_type=PIIType.IP_ADDRESS,
            sensitivity=DataSensitivityLevel.LOW,
            action=DLPAction.AUDIT,
            description="Detects full IPv6 addresses.",
        ),
        DLPRule(
            id="default_passport_us",
            name="US Passport Number",
            pattern=r"\b[A-Z]\d{8}\b",
            pii_type=PIIType.PASSPORT,
            sensitivity=DataSensitivityLevel.HIGH,
            action=DLPAction.REDACT,
            description="Detects US passport numbers (letter + 8 digits).",
        ),
        DLPRule(
            id="default_bank_account",
            name="Bank Account Number",
            pattern=r"\b\d{8,17}\b",
            pii_type=PIIType.BANK_ACCOUNT,
            sensitivity=DataSensitivityLevel.CRITICAL,
            action=DLPAction.BLOCK,
            description="Detects 8-17 digit bank account numbers.",
        ),
        DLPRule(
            id="default_id_card",
            name="ID Card Number",
            pattern=r"\b[A-Z]{1,3}\d{6,12}\b",
            pii_type=PIIType.ID_CARD,
            sensitivity=DataSensitivityLevel.HIGH,
            action=DLPAction.REDACT,
            description="Detects national ID / driver's license patterns.",
        ),
        DLPRule(
            id="default_address",
            name="Street Address",
            pattern=r"\b\d{1,6}\s+[A-Z][a-zA-Z]+\s+(?:Street|St|Avenue|Ave|"
            r"Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Place|Pl)\b",
            pii_type=PIIType.ADDRESS,
            sensitivity=DataSensitivityLevel.MEDIUM,
            action=DLPAction.AUDIT,
            description="Detects US street addresses.",
        ),
        # ------------------------------------------------------------------
        # PRC (China) specific PII patterns
        # ------------------------------------------------------------------
        DLPRule(
            id="default_chinese_id_card",
            name="PRC Resident Identity Card (18-digit)",
            pattern=r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
            r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]",
            pii_type=PIIType.CHINESE_ID_CARD,
            sensitivity=DataSensitivityLevel.CRITICAL,
            action=DLPAction.BLOCK,
            description="Detects 18-digit PRC resident identity card numbers.",
        ),
        DLPRule(
            id="default_unified_social_credit_code",
            name="PRC Unified Social Credit Code",
            pattern=r"[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}",
            pii_type=PIIType.UNIFIED_SOCIAL_CREDIT_CODE,
            sensitivity=DataSensitivityLevel.HIGH,
            action=DLPAction.REDACT,
            description="Detects 18-character PRC unified social credit codes.",
        ),
        DLPRule(
            id="default_phone_cn",
            name="PRC Mobile Phone Number",
            pattern=r"(?<!\w)1[3-9]\d{9}\b",
            pii_type=PIIType.PHONE,
            sensitivity=DataSensitivityLevel.MEDIUM,
            action=DLPAction.REDACT,
            description="Detects PRC mobile phone numbers (11 digits starting with 1).",
        ),
        DLPRule(
            id="default_chinese_bank_card",
            name="PRC Bank Card Number",
            pattern=r"\b[1-9]\d{14,18}\b",
            pii_type=PIIType.BANK_ACCOUNT,
            sensitivity=DataSensitivityLevel.CRITICAL,
            action=DLPAction.BLOCK,
            description="Detects PRC bank card numbers (15-19 digits, non-zero leading).",
        ),
        DLPRule(
            id="default_case_number",
            name="PRC Judicial Case Number",
            pattern=r"\(\d{4}\)[^\s]+\d+号",
            pii_type=PIIType.CASE_NUMBER,
            sensitivity=DataSensitivityLevel.MEDIUM,
            action=DLPAction.AUDIT,
            description="Detects PRC judicial case numbers, e.g. (2024)京01民初1号.",
        ),
        DLPRule(
            id="default_business_license",
            name="PRC Business Licence Number",
            pattern=r"\b\d{15}|\d{18}\b",
            pii_type=PIIType.BUSINESS_LICENSE,
            sensitivity=DataSensitivityLevel.MEDIUM,
            action=DLPAction.AUDIT,
            description="Detects PRC business licence numbers (15 or 18 digits).",
        ),
    ]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class DLPRule(BaseModel):
    """A named regex-based PII detection rule.

    Attributes:
        id: Unique rule identifier (auto-generated UUID4 hex).
        name: Human-readable rule name.
        pattern: Python regex pattern string.
        pii_type: The :class:`PIIType` this rule detects.
        sensitivity: Default :class:`DataSensitivityLevel` for findings.
        action: The :class:`DLPAction` to take on match.
        description: What the rule detects.
        enabled: Whether the rule is active.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    pattern: str
    pii_type: PIIType
    sensitivity: DataSensitivityLevel = DataSensitivityLevel.MEDIUM
    action: DLPAction = DLPAction.REDACT
    description: str = ""
    enabled: bool = True


class PIIFinding(BaseModel):
    """A single PII match found during scanning.

    Attributes:
        pii_type: The :class:`PIIType` of the match.
        value: The matched text.
        start: Start character offset in the scanned text.
        end: End character offset (exclusive).
        sensitivity: The :class:`DataSensitivityLevel`.
        rule_id: The ID of the rule that produced this finding.
    """

    pii_type: PIIType
    value: str
    start: int
    end: int
    sensitivity: DataSensitivityLevel = DataSensitivityLevel.MEDIUM
    rule_id: str = ""


# ---------------------------------------------------------------------------
# DLP scanner
# ---------------------------------------------------------------------------


class DLPScanner:
    """Thread-safe PII scanner with default and custom rules.

    Ships with sensible default regex patterns for common PII types.
    Custom rules can be added via :meth:`add_rule` and removed via
    :meth:`remove_rule`.

    Example::

        scanner = DLPScanner()
        findings = scanner.scan_text("Email: alice@example.com, SSN: 123-45-6789")
        for f in findings:
            print(f.pii_type, f.value, f.sensitivity)
    """

    def __init__(self, rules: list[DLPRule] | None = None) -> None:
        self._rules: dict[str, DLPRule] = {}
        self._compiled: dict[str, re.Pattern[str]] = {}
        self._lock = threading.RLock()
        source_rules = rules if rules is not None else _build_default_rules()
        for rule in source_rules:
            self._register_rule(rule)

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: DLPRule) -> DLPRule:
        """Add or replace a DLP rule.

        Raises:
            DLPError: If the rule pattern is an invalid regex.
        """

        with self._lock:
            self._register_rule(rule)
        logger.info("Added DLP rule %s (%s)", rule.id, rule.name)
        return rule

    def remove_rule(self, rule_id: str) -> DLPRule | None:
        """Remove a rule by id. Returns the removed rule or ``None``."""

        with self._lock:
            rule = self._rules.pop(rule_id, None)
            self._compiled.pop(rule_id, None)
        if rule is not None:
            logger.info("Removed DLP rule %s", rule_id)
        return rule

    def list_rules(self, *, enabled_only: bool = False) -> list[DLPRule]:
        """Return all (or only enabled) rules."""

        with self._lock:
            rules = list(self._rules.values())
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        return rules

    def get_rule(self, rule_id: str) -> DLPRule | None:
        """Return a rule by id, or ``None``."""

        with self._lock:
            return self._rules.get(rule_id)

    def _register_rule(self, rule: DLPRule) -> None:
        """Register a rule and compile its pattern (caller holds lock)."""

        try:
            compiled = re.compile(rule.pattern)
        except re.error as exc:
            raise DLPError(f"Invalid regex pattern for rule {rule.id!r}: {exc}") from exc
        self._rules[rule.id] = rule
        self._compiled[rule.id] = compiled

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_text(self, text: str) -> list[PIIFinding]:
        """Scan *text* and return all PII findings.

        Findings are sorted by start position. Overlapping findings from
        different rules are all retained.
        """

        findings: list[PIIFinding] = []
        with self._lock:
            rules = list(self._rules.values())
            compiled = dict(self._compiled)

        for rule in rules:
            if not rule.enabled:
                continue
            pattern = compiled.get(rule.id)
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                findings.append(
                    PIIFinding(
                        pii_type=rule.pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        sensitivity=rule.sensitivity,
                        rule_id=rule.id,
                    )
                )

        findings.sort(key=lambda f: (f.start, f.end))
        return findings

    async def scan_file(self, path: Path | str) -> list[PIIFinding]:
        """Read a file and scan its contents for PII.

        The file is read as UTF-8 text. Binary files that cannot be
        decoded are skipped with a warning.

        Args:
            path: Path to the file to scan.

        Returns:
            List of :class:`PIIFinding` objects.
        """

        import asyncio

        file_path = Path(path)
        try:
            content = await asyncio.to_thread(file_path.read_text, "utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Cannot read file %s: %s", file_path, exc)
            return []
        return self.scan_text(content)

    def scan_dict(self, data: dict[str, Any]) -> dict[str, list[PIIFinding]]:
        """Scan all string values in a dict recursively.

        Returns a mapping of ``"key.path"`` to the findings in that value.
        """

        results: dict[str, list[PIIFinding]] = {}

        def _scan(obj: Any, prefix: str) -> None:
            if isinstance(obj, str):
                findings = self.scan_text(obj)
                if findings:
                    results[prefix] = findings
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    _scan(value, f"{prefix}.{key}" if prefix else str(key))
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    _scan(item, f"{prefix}[{idx}]")

        _scan(data, "")
        return results

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    @property
    def rule_count(self) -> int:
        """Total number of registered rules."""

        with self._lock:
            return len(self._rules)

    def summary(self, text: str) -> dict[str, Any]:
        """Scan *text* and return a compact summary of findings."""

        findings = self.scan_text(text)
        by_type: dict[str, int] = {}
        by_sensitivity: dict[str, int] = {}
        for finding in findings:
            by_type[finding.pii_type.value] = by_type.get(finding.pii_type.value, 0) + 1
            by_sensitivity[finding.sensitivity.value] = (
                by_sensitivity.get(finding.sensitivity.value, 0) + 1
            )
        return {
            "total_findings": len(findings),
            "by_type": by_type,
            "by_sensitivity": by_sensitivity,
            "has_critical": DataSensitivityLevel.CRITICAL.value in by_sensitivity,
        }


# ---------------------------------------------------------------------------
# Data sanitizer
# ---------------------------------------------------------------------------


class DataSanitizer:
    """Redaction and masking utilities for PII in text.

    Provides three modes:

    * :meth:`redact_pii` — replace each PII match with a typed
      placeholder (e.g. ``[REDACTED_EMAIL]``).
    * :meth:`mask_partial` — partially mask the value, keeping enough
      context to identify the type without exposing the full value
      (e.g. ``ali***@example.com``).
    * :meth:`sanitize` — apply a set of :class:`DLPAction` values to
      text scanned by a :class:`DLPScanner`.

    Example::

        sanitizer = DataSanitizer()
        assert sanitizer.redact_pii("Email: alice@example.com") == \
            "Email: [REDACTED_EMAIL]"
        assert "a***" in sanitizer.mask_partial("alice@example.com")
    """

    #: Placeholder text per PII type for full redaction.
    _REDACT_PLACEHOLDERS: dict[PIIType, str] = {
        PIIType.EMAIL: "[REDACTED_EMAIL]",
        PIIType.PHONE: "[REDACTED_PHONE]",
        PIIType.SSN: "[REDACTED_SSN]",
        PIIType.CREDIT_CARD: "[REDACTED_CREDIT_CARD]",
        PIIType.IP_ADDRESS: "[REDACTED_IP]",
        PIIType.PASSPORT: "[REDACTED_PASSPORT]",
        PIIType.BANK_ACCOUNT: "[REDACTED_BANK_ACCOUNT]",
        PIIType.ID_CARD: "[REDACTED_ID]",
        PIIType.ADDRESS: "[REDACTED_ADDRESS]",
        PIIType.NAME: "[REDACTED_NAME]",
        # PRC (China) specific PII types
        PIIType.CHINESE_ID_CARD: "[REDACTED_CN_ID_CARD]",
        PIIType.UNIFIED_SOCIAL_CREDIT_CODE: "[REDACTED_USCC]",
        PIIType.CASE_NUMBER: "[REDACTED_CASE_NUMBER]",
        PIIType.BUSINESS_LICENSE: "[REDACTED_BUSINESS_LICENSE]",
    }

    def __init__(self, scanner: DLPScanner | None = None) -> None:
        self._scanner = scanner or DLPScanner()

    @property
    def scanner(self) -> DLPScanner:
        """The underlying :class:`DLPScanner`."""

        return self._scanner

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------

    def redact_pii(self, text: str) -> str:
        """Replace every PII match with a typed placeholder.

        Example::

            >>> DataSanitizer().redact_pii("SSN: 123-45-6789")
            'SSN: [REDACTED_SSN]'
        """

        findings = self._scanner.scan_text(text)
        # Process from the end to keep offsets valid.
        result = text
        for finding in reversed(findings):
            placeholder = self._REDACT_PLACEHOLDERS.get(finding.pii_type, "[REDACTED]")
            result = result[: finding.start] + placeholder + result[finding.end :]
        return result

    # ------------------------------------------------------------------
    # Partial masking
    # ------------------------------------------------------------------

    def mask_partial(self, text: str) -> str:
        """Partially mask each PII match, preserving type context.

        Keeps the first and last characters visible and replaces the
        middle with asterisks. For very short values, only the first
        character is kept.

        Example::

            >>> DataSanitizer().mask_partial("alice@example.com")
            'a***e@example.com'
        """

        findings = self._scanner.scan_text(text)
        result = text
        for finding in reversed(findings):
            masked = self._mask_value(finding.value, finding.pii_type)
            result = result[: finding.start] + masked + result[finding.end :]
        return result

    @staticmethod
    def _mask_value(value: str, pii_type: PIIType) -> str:
        """Mask a single PII value, preserving type context."""

        if pii_type is PIIType.EMAIL and "@" in value:
            local, _, domain = value.partition("@")
            if len(local) <= 1:
                return f"{local}***@{domain}"
            return f"{local[0]}{'*' * (len(local) - 1)}@{domain}"

        if pii_type in (
            PIIType.SSN,
            PIIType.CREDIT_CARD,
            PIIType.BANK_ACCOUNT,
            PIIType.CHINESE_ID_CARD,
            PIIType.BUSINESS_LICENSE,
        ):
            # Keep last 4 digits for financial / identity identifiers.
            if len(value) <= 4:
                return "*" * len(value)
            return "*" * (len(value) - 4) + value[-4:]

        if pii_type is PIIType.UNIFIED_SOCIAL_CREDIT_CODE:
            # Keep first 4 (registration authority) and last 4 (check digit).
            if len(value) <= 8:
                return "*" * len(value)
            return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"

        if pii_type is PIIType.PHONE:
            # Keep last 4 digits.
            if len(value) <= 4:
                return "*" * len(value)
            return "*" * (len(value) - 4) + value[-4:]

        if pii_type is PIIType.IP_ADDRESS:
            # Mask the last octet / group.
            parts = value.rsplit(".", 1) if "." in value else value.rsplit(":", 1)
            if len(parts) == 2:
                return f"{parts[0]}.***"
            return "***"

        # Generic masking for other types (e.g. CASE_NUMBER, ADDRESS).
        if len(value) <= 2:
            return "*" * len(value)
        return f"{value[0]}{'*' * (len(value) - 2)}{value[-1]}"

    # ------------------------------------------------------------------
    # Action-based sanitisation
    # ------------------------------------------------------------------

    def sanitize(
        self,
        text: str,
        actions: set[DLPAction] | None = None,
    ) -> str:
        """Apply a set of DLP actions to *text*.

        By default, all :attr:`DLPAction.REDACT` findings are redacted.
        When :attr:`DLPAction.BLOCK` is in the action set and any finding
        with a BLOCK action exists, the entire text is replaced with a
        block notice.

        Args:
            text: The text to sanitise.
            actions: The set of actions to apply. If ``None``, REDACT
                findings are redacted and BLOCK findings cause a block.

        Returns:
            The sanitised text.
        """

        if actions is None:
            actions = {DLPAction.REDACT, DLPAction.BLOCK}

        findings = self._scanner.scan_text(text)
        rules = {r.id: r for r in self._scanner.list_rules()}

        # Check for blocking findings.
        if DLPAction.BLOCK in actions:
            blocking = [
                f
                for f in findings
                if rules.get(
                    f.rule_id,
                    DLPRule(
                        id="",
                        name="",
                        pattern="",
                        pii_type=f.pii_type,
                        action=DLPAction.BLOCK,
                    ),
                ).action
                is DLPAction.BLOCK
            ]
            if blocking:
                logger.warning("Content blocked: %d BLOCK finding(s) detected", len(blocking))
                return "[CONTENT BLOCKED BY DLP POLICY]"

        # Apply redaction for REDACT findings.
        if DLPAction.REDACT in actions:
            redact_findings = [
                f
                for f in findings
                if rules.get(
                    f.rule_id,
                    DLPRule(
                        id="",
                        name="",
                        pattern="",
                        pii_type=f.pii_type,
                        action=DLPAction.REDACT,
                    ),
                ).action
                is DLPAction.REDACT
            ]
            result = text
            for finding in reversed(redact_findings):
                placeholder = self._REDACT_PLACEHOLDERS.get(finding.pii_type, "[REDACTED]")
                result = result[: finding.start] + placeholder + result[finding.end :]
            return result

        return text


__all__ = [
    "DLPAction",
    "DLPError",
    "DLPRule",
    "DLPScanner",
    "DataSanitizer",
    "DataSensitivityLevel",
    "PIIFinding",
    "PIIType",
]
