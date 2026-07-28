"""Compliance and audit — framework rules, policy enforcement, audit trails.

Provides the compliance layer for the Omniagent platform: policy
decisions evaluated against regulatory frameworks (GDPR, HIPAA, SOC 2,
ISO 27001, PCI-DSS, CCPA) and an append-only audit trail with
tamper-evident hash chaining.

Design:

* :class:`ComplianceFramework` — supported regulatory frameworks.
* :class:`Severity` — rule severity tiers.
* :class:`AuditResult` — outcome of an audited action.
* :class:`ComplianceRule` — a single framework requirement.
* :class:`PolicyDecision` — structured allow/deny with violated rules
  and recommendations.
* :class:`AuditTrail` — a single immutable audit event.
* :class:`ComplianceChecker` — evaluates data access and retention
  against configured rules.
* :class:`AuditTrailManager` — thread-safe append-only audit log with
  hash chaining and optional file persistence.

Example::

    checker = ComplianceChecker()
    checker.add_rule(ComplianceRule(
        id="gdpr_001",
        framework=ComplianceFramework.GDPR,
        requirement="data_minimization",
        description="Only collect data necessary for the purpose.",
        severity=Severity.HIGH,
    ))
    decision = checker.check_data_access("alice", "pii", "export")
    # -> PolicyDecision(allowed=False, violated_rules=[...], recommendations=[...])

    audit = AuditTrailManager()
    await audit.record(AuditTrail(
        actor="alice", action="export", resource="customer_data",
        result=AuditResult.SUCCESS,
    ))
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("myagent.security.compliance")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ComplianceError(Exception):
    """Raised for compliance checking or audit trail errors."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ComplianceFramework(str, Enum):  # noqa: UP042 - match existing codebase style
    """Supported regulatory compliance frameworks.

    Attributes:
        GDPR: EU General Data Protection Regulation.
        HIPAA: US Health Insurance Portability and Accountability Act.
        SOC2: AICPA SOC 2 Trust Services Criteria.
        ISO27001: ISO/IEC 27001 Information Security Management.
        PCI_DSS: Payment Card Industry Data Security Standard.
        CCPA: California Consumer Privacy Act.
        NONE: No framework (used for unconstrained operations).
    """

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    CCPA = "ccpa"
    NONE = "none"


class Severity(str, Enum):  # noqa: UP042
    """Severity tier for a compliance rule.

    Attributes:
        INFO: Informational, no enforcement action.
        LOW: Minor risk; logged but rarely blocks.
        MEDIUM: Moderate risk; may block sensitive operations.
        HIGH: Significant risk; blocks by default.
        CRITICAL: Severe risk; always blocks and alerts.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditResult(str, Enum):  # noqa: UP042
    """Outcome of an audited action.

    Attributes:
        SUCCESS: The action completed successfully.
        FAILURE: The action failed (e.g. an error occurred).
        DENIED: The action was denied by policy.
        ERROR: A system error prevented completion.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"


#: Severities at or above this level block operations by default.
_BLOCKING_SEVERITIES: frozenset[Severity] = frozenset({Severity.HIGH, Severity.CRITICAL})

#: SHA-256 of an empty string — sentinel for the genesis audit entry.
_GENESIS_PREV_HASH = "0" * 64

#: Default retention limits in days per framework.
_DEFAULT_RETENTION_DAYS: dict[ComplianceFramework, int] = {
    ComplianceFramework.GDPR: 365,  # 1 year default
    ComplianceFramework.HIPAA: 2190,  # 6 years
    ComplianceFramework.SOC2: 2555,  # 7 years
    ComplianceFramework.ISO27001: 1095,  # 3 years
    ComplianceFramework.PCI_DSS: 365,  # 1 year
    ComplianceFramework.CCPA: 730,  # 2 years
    ComplianceFramework.NONE: 0,  # no limit
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ComplianceRule(BaseModel):
    """A single compliance requirement within a framework.

    Attributes:
        id: Unique rule identifier (auto-generated UUID4 hex).
        framework: The :class:`ComplianceFramework` this rule belongs to.
        requirement: Short machine-readable requirement code
            (e.g. ``"data_minimization"``).
        description: Human-readable explanation of the requirement.
        severity: The :class:`Severity` of violating this rule.
        enabled: Whether the rule is actively enforced.
        data_types: Data types this rule applies to (e.g. ``["pii",
            "phi"]``). Empty list = applies to all data types.
        actions: Actions this rule applies to (e.g. ``["export",
            "delete"]``). Empty list = applies to all actions.
        retention_days: Maximum retention period in days (0 = no limit).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    framework: ComplianceFramework
    requirement: str
    description: str = ""
    severity: Severity = Severity.MEDIUM
    enabled: bool = True
    data_types: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    retention_days: int = 0


class AuditTrail(BaseModel):
    """A single immutable audit event.

    Attributes:
        event_id: Unique event identifier (auto-generated UUID4 hex).
        timestamp: Unix timestamp of the event.
        actor: User or service that performed the action.
        action: Short verb describing the operation (e.g. ``"export"``).
        resource: Identifier of the resource acted upon.
        result: The :class:`AuditResult` outcome.
        ip_address: Source IP address of the request.
        user_agent: User-Agent header of the request.
        metadata: Arbitrary structured metadata.
        prev_hash: Hash of the previous entry (for chain integrity).
        entry_hash: SHA-256 of this entry's content (computed on seal).
    """

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    actor: str
    action: str
    resource: str = ""
    result: AuditResult = AuditResult.SUCCESS
    ip_address: str = ""
    user_agent: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = _GENESIS_PREV_HASH
    entry_hash: str = ""

    def _signing_payload(self) -> str:
        """Canonical JSON of every field except ``entry_hash``."""

        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "result": self.result.value,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "metadata": self.metadata,
            "prev_hash": self.prev_hash,
        }
        return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)

    def compute_hash(self) -> str:
        """Return the SHA-256 hex digest of this entry's signing payload."""

        return hashlib.sha256(self._signing_payload().encode("utf-8")).hexdigest()

    def seal(self) -> AuditTrail:
        """Return a copy with ``entry_hash`` populated (idempotent)."""

        digest = self.compute_hash()
        if self.entry_hash == digest:
            return self
        return self.model_copy(update={"entry_hash": digest})

    def verify(self) -> bool:
        """True when the stored ``entry_hash`` matches a recomputation."""

        if not self.entry_hash:
            return False
        return self.compute_hash() == self.entry_hash


# ---------------------------------------------------------------------------
# Policy decision
# ---------------------------------------------------------------------------


@dataclass
class PolicyDecision:
    """Structured result of a compliance policy check.

    Attributes:
        allowed: Whether the action is permitted.
        violated_rules: Rules that were violated by the request.
        recommendations: Suggested remediation actions.
    """

    allowed: bool
    violated_rules: list[ComplianceRule] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.allowed


# ---------------------------------------------------------------------------
# Compliance checker
# ---------------------------------------------------------------------------


class ComplianceChecker:
    """Evaluates data access and retention against compliance rules.

    Maintains a set of :class:`ComplianceRule` instances organised by
    framework. On construction, a baseline set of rules for common
    frameworks (GDPR, HIPAA, PCI-DSS) is loaded.

    Example::

        checker = ComplianceChecker()
        decision = checker.check_data_access("alice", "phi", "export")
        if not decision:
            for rule in decision.violated_rules:
                print(rule.requirement, rule.description)
    """

    def __init__(self) -> None:
        self._rules: dict[str, ComplianceRule] = {}
        self._lock = threading.RLock()
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        """Register baseline rules for common frameworks."""

        defaults = [
            ComplianceRule(
                id="gdpr_consent",
                framework=ComplianceFramework.GDPR,
                requirement="lawful_basis",
                description="Processing of personal data requires a lawful basis "
                "and, for non-essential processing, user consent.",
                severity=Severity.HIGH,
                data_types=["pii"],
                actions=["export", "share"],
            ),
            ComplianceRule(
                id="gdpr_minimization",
                framework=ComplianceFramework.GDPR,
                requirement="data_minimization",
                description="Only collect and process personal data adequate, "
                "relevant and limited to what is necessary.",
                severity=Severity.MEDIUM,
                data_types=["pii"],
            ),
            ComplianceRule(
                id="gdpr_retention",
                framework=ComplianceFramework.GDPR,
                requirement="storage_limitation",
                description="Personal data must not be kept longer than necessary.",
                severity=Severity.HIGH,
                retention_days=_DEFAULT_RETENTION_DAYS[ComplianceFramework.GDPR],
            ),
            ComplianceRule(
                id="hipaa_phi_access",
                framework=ComplianceFramework.HIPAA,
                requirement="phi_access_control",
                description="Access to Protected Health Information (PHI) requires "
                "role-based authorisation and audit logging.",
                severity=Severity.CRITICAL,
                data_types=["phi"],
                actions=["read", "export", "share"],
            ),
            ComplianceRule(
                id="hipaa_retention",
                framework=ComplianceFramework.HIPAA,
                requirement="retention_period",
                description="HIPAA requires retention of PHI for at least 6 years.",
                severity=Severity.HIGH,
                retention_days=_DEFAULT_RETENTION_DAYS[ComplianceFramework.HIPAA],
            ),
            ComplianceRule(
                id="pci_card_data",
                framework=ComplianceFramework.PCI_DSS,
                requirement="cardholder_data_protection",
                description="Cardholder data must be protected with strong "
                "encryption and access controls.",
                severity=Severity.CRITICAL,
                data_types=["credit_card", "pci"],
                actions=["export", "share"],
            ),
            ComplianceRule(
                id="pci_retention",
                framework=ComplianceFramework.PCI_DSS,
                requirement="retention_limit",
                description="Cardholder data retention must be limited and documented.",
                severity=Severity.HIGH,
                retention_days=_DEFAULT_RETENTION_DAYS[ComplianceFramework.PCI_DSS],
            ),
            ComplianceRule(
                id="soc2_access_logging",
                framework=ComplianceFramework.SOC2,
                requirement="access_logging",
                description="All access to sensitive data must be logged for audit purposes.",
                severity=Severity.MEDIUM,
                data_types=["pii", "phi", "pci", "confidential"],
                actions=["read", "write", "export", "delete"],
            ),
        ]
        for rule in defaults:
            self._rules[rule.id] = rule

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: ComplianceRule) -> ComplianceRule:
        """Add or replace a compliance rule."""

        with self._lock:
            self._rules[rule.id] = rule
        logger.info("Added compliance rule %s (%s)", rule.id, rule.framework.value)
        return rule

    def remove_rule(self, rule_id: str) -> ComplianceRule | None:
        """Remove a rule by id. Returns the removed rule or ``None``."""

        with self._lock:
            return self._rules.pop(rule_id, None)

    def get_rules(
        self,
        framework: ComplianceFramework | None = None,
    ) -> list[ComplianceRule]:
        """Return rules, optionally filtered by framework."""

        with self._lock:
            rules = list(self._rules.values())
        if framework is not None:
            rules = [r for r in rules if r.framework is framework]
        return rules

    def get_rule(self, rule_id: str) -> ComplianceRule | None:
        """Return a rule by id, or ``None``."""

        with self._lock:
            return self._rules.get(rule_id)

    # ------------------------------------------------------------------
    # Policy evaluation
    # ------------------------------------------------------------------

    def check_data_access(
        self,
        user: str,
        data_type: str,
        action: str,
    ) -> PolicyDecision:
        """Evaluate whether *user* may perform *action* on *data_type*.

        Returns a :class:`PolicyDecision` with violated rules and
        recommendations. The decision is ``allowed=False`` when any
        enabled rule with a blocking severity (HIGH or CRITICAL) is
        violated.

        Only rules with at least one action specified are considered —
        rules with an empty ``actions`` list are general framework
        principles (e.g. retention) enforced via :meth:`check_data_retention`,
        not per-access.
        """

        violated: list[ComplianceRule] = []
        recommendations: list[str] = []

        with self._lock:
            rules = list(self._rules.values())

        for rule in rules:
            if not rule.enabled:
                continue
            # Skip rules that don't govern specific access actions
            # (e.g. retention-only rules).
            if not rule.actions:
                continue
            if not self._rule_applies(rule, data_type, action):
                continue
            violated.append(rule)
            if rule.severity in _BLOCKING_SEVERITIES:
                recommendations.append(
                    f"Review {rule.framework.value} requirement "
                    f"'{rule.requirement}': {rule.description}"
                )

        has_blocking = any(r.severity in _BLOCKING_SEVERITIES for r in violated)
        allowed = not has_blocking

        if not recommendations and violated:
            recommendations.append(
                "Consider reviewing the violated compliance rules before proceeding."
            )

        return PolicyDecision(
            allowed=allowed,
            violated_rules=violated,
            recommendations=recommendations,
        )

    def check_data_retention(
        self,
        data_age_days: int,
        framework: ComplianceFramework,
    ) -> bool:
        """Return ``True`` if *data_age_days* is within the retention limit.

        When no explicit retention rule exists for *framework*, the
        default from :data:`_DEFAULT_RETENTION_DAYS` is used.
        """

        with self._lock:
            rules = [
                r for r in self._rules.values() if r.framework is framework and r.retention_days > 0
            ]

        if rules:
            max_retention = max(r.retention_days for r in rules)
        else:
            max_retention = _DEFAULT_RETENTION_DAYS.get(framework, 0)

        if max_retention == 0:
            return True  # no limit

        compliant = data_age_days <= max_retention
        if not compliant:
            logger.warning(
                "Data age %d days exceeds retention limit %d for %s",
                data_age_days,
                max_retention,
                framework.value,
            )
        return compliant

    @staticmethod
    def _rule_applies(
        rule: ComplianceRule,
        data_type: str,
        action: str,
    ) -> bool:
        """Check whether a rule applies to the given data type and action."""

        if rule.data_types and data_type.lower() not in {dt.lower() for dt in rule.data_types}:
            return False
        return not rule.actions or action.lower() in {a.lower() for a in rule.actions}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def export_report(self, format: str = "json") -> str:
        """Export a compliance report as a string.

        Args:
            format: Output format — ``"json"`` (default) or ``"csv"``.

        Returns:
            The formatted report string.
        """

        with self._lock:
            rules = list(self._rules.values())

        if format.lower() == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                [
                    "rule_id",
                    "framework",
                    "requirement",
                    "description",
                    "severity",
                    "enabled",
                    "data_types",
                    "actions",
                    "retention_days",
                ]
            )
            for rule in rules:
                writer.writerow(
                    [
                        rule.id,
                        rule.framework.value,
                        rule.requirement,
                        rule.description,
                        rule.severity.value,
                        rule.enabled,
                        ";".join(rule.data_types),
                        ";".join(rule.actions),
                        rule.retention_days,
                    ]
                )
            return buf.getvalue()

        # Default: JSON
        data = [rule.model_dump(mode="json") for rule in rules]
        return json.dumps(data, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    @property
    def rule_count(self) -> int:
        """Total number of registered rules."""

        with self._lock:
            return len(self._rules)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the compliance configuration."""

        with self._lock:
            rules = list(self._rules.values())
        by_framework: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        enabled = 0
        for rule in rules:
            by_framework[rule.framework.value] = by_framework.get(rule.framework.value, 0) + 1
            by_severity[rule.severity.value] = by_severity.get(rule.severity.value, 0) + 1
            if rule.enabled:
                enabled += 1
        return {
            "total_rules": len(rules),
            "enabled_rules": enabled,
            "by_framework": by_framework,
            "by_severity": by_severity,
        }


# ---------------------------------------------------------------------------
# Audit trail manager
# ---------------------------------------------------------------------------


class AuditTrailManager:
    """Thread-safe, append-only audit log with hash chaining.

    Each :class:`AuditTrail` entry is sealed with a SHA-256 hash computed
    over its content plus the previous entry's hash, forming a
    tamper-evident chain. Entries can optionally be persisted to a JSONL
    file so the chain survives process restarts.

    Storage is strictly append-only — there is no ``update`` or
    ``delete`` method.

    Example::

        manager = AuditTrailManager()
        await manager.record(AuditTrail(
            actor="alice",
            action="data.export",
            resource="customer_records",
            result=AuditResult.SUCCESS,
            ip_address="10.0.0.1",
        ))
        entries = await manager.query({"actor": "alice"})
    """

    def __init__(self, persistence_path: Path | str | None = None) -> None:
        self._entries: list[AuditTrail] = []
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._lock = threading.Lock()
        if self._persistence_path is not None and self._persistence_path.exists():
            self._load_sync()

    # ------------------------------------------------------------------
    # Recording (the only mutation)
    # ------------------------------------------------------------------

    def record(self, event: AuditTrail) -> AuditTrail:
        """Seal and append an audit event. Returns the sealed entry.

        The ``prev_hash`` is set to the hash of the previous entry (or
        the genesis sentinel for the first entry). The ``entry_hash`` is
        computed and stored.

        Raises:
            ComplianceError: If the event already has an ``entry_hash``
                (events must be unsealed before recording).
        """

        if event.entry_hash:
            raise ComplianceError("Cannot record an already-sealed audit event")

        with self._lock:
            prev_hash = self._entries[-1].entry_hash if self._entries else _GENESIS_PREV_HASH
            event.prev_hash = prev_hash
            sealed = event.seal()
            self._entries.append(sealed)
            if self._persistence_path is not None:
                self._append_sync(sealed)
        logger.debug(
            "Audit event recorded: %s/%s by %s",
            sealed.action,
            sealed.result.value,
            sealed.actor,
        )
        return sealed

    async def record_async(self, event: AuditTrail) -> AuditTrail:
        """Async wrapper for :meth:`record` (for I/O-bound callers)."""

        import asyncio

        return await asyncio.to_thread(self.record, event)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[AuditTrail]:
        """Return entries matching *filters* (or all entries).

        Supported filter keys: ``actor``, ``action``, ``resource``,
        ``result``, ``since`` (Unix timestamp), ``until`` (Unix
        timestamp), ``ip_address``.
        """

        with self._lock:
            snapshot = list(self._entries)

        if not filters:
            return snapshot

        results: list[AuditTrail] = []
        for entry in snapshot:
            if not self._matches(entry, filters):
                continue
            results.append(entry)
        return results

    def get(self, event_id: str) -> AuditTrail | None:
        """Return the entry with *event_id*, or ``None``."""

        with self._lock:
            for entry in self._entries:
                if entry.event_id == event_id:
                    return entry
        return None

    @staticmethod
    def _matches(entry: AuditTrail, filters: dict[str, Any]) -> bool:
        """Check whether *entry* matches all *filters*."""

        if "actor" in filters and entry.actor != filters["actor"]:
            return False
        if "action" in filters and entry.action != filters["action"]:
            return False
        if "resource" in filters and entry.resource != filters["resource"]:
            return False
        if "result" in filters:
            expected = filters["result"]
            if isinstance(expected, str):
                expected = AuditResult(expected)
            if entry.result is not expected:
                return False
        if "ip_address" in filters and entry.ip_address != filters["ip_address"]:
            return False
        if "since" in filters and entry.timestamp < filters["since"]:
            return False
        return "until" not in filters or entry.timestamp <= filters["until"]

    # ------------------------------------------------------------------
    # Chain verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire hash chain.

        Checks that every entry's ``entry_hash`` matches a fresh
        recomputation and that ``prev_hash`` links to the previous
        entry's hash.
        """

        with self._lock:
            snapshot = list(self._entries)

        for idx, entry in enumerate(snapshot):
            if not entry.verify():
                logger.error("Audit chain broken at entry #%d (hash mismatch)", idx)
                return False
            if idx == 0:
                if entry.prev_hash != _GENESIS_PREV_HASH:
                    logger.error("Genesis entry prev_hash is not the zero sentinel")
                    return False
            else:
                prev = snapshot[idx - 1]
                if entry.prev_hash != prev.entry_hash:
                    logger.error("Audit chain broken at entry #%d (prev_hash mismatch)", idx)
                    return False
        return True

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, format: str = "json") -> str:
        """Export all entries as a string.

        Args:
            format: ``"json"`` (default), ``"jsonl"`` or ``"csv"``.

        Returns:
            The formatted export string.
        """

        with self._lock:
            snapshot = list(self._entries)

        if format.lower() == "jsonl":
            return "\n".join(e.model_dump_json() for e in snapshot)

        if format.lower() == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                [
                    "event_id",
                    "timestamp",
                    "actor",
                    "action",
                    "resource",
                    "result",
                    "ip_address",
                    "user_agent",
                    "entry_hash",
                ]
            )
            for e in snapshot:
                writer.writerow(
                    [
                        e.event_id,
                        e.timestamp,
                        e.actor,
                        e.action,
                        e.resource,
                        e.result.value,
                        e.ip_address,
                        e.user_agent,
                        e.entry_hash,
                    ]
                )
            return buf.getvalue()

        # Default: JSON array
        data = [json.loads(e.model_dump_json()) for e in snapshot]
        return json.dumps(data, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path | None:
        """Persist all entries to a JSONL file. Returns the path written."""

        target = Path(path) if path is not None else self._persistence_path
        if target is None:
            return None
        with self._lock:
            lines = [e.model_dump_json() for e in self._entries]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        logger.debug("Persisted %d audit entries to %s", len(lines), target)
        return target

    def _load_sync(self) -> None:
        """Load entries from the persistence file (caller holds lock)."""

        assert self._persistence_path is not None
        try:
            content = self._persistence_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to load audit trail from %s: %s", self._persistence_path, exc)
            return
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = AuditTrail.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001 - skip unparseable
                logger.warning("Skipping malformed audit entry: %s", exc)
                continue
            self._entries.append(entry)
        logger.info("Loaded %d audit entries from %s", len(self._entries), self._persistence_path)

    def _append_sync(self, entry: AuditTrail) -> None:
        """Append a single entry to the persistence file (caller holds lock)."""

        assert self._persistence_path is not None
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with self._persistence_path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of entries currently held."""

        with self._lock:
            return len(self._entries)

    def __len__(self) -> int:
        return self.count

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the audit trail."""

        with self._lock:
            snapshot = list(self._entries)
        by_result: dict[str, int] = {}
        by_action: dict[str, int] = {}
        for entry in snapshot:
            by_result[entry.result.value] = by_result.get(entry.result.value, 0) + 1
            by_action[entry.action] = by_action.get(entry.action, 0) + 1
        return {
            "total_entries": len(snapshot),
            "by_result": by_result,
            "by_action": by_action,
            "chain_valid": self.verify_chain() if snapshot else True,
        }


__all__ = [
    "AuditResult",
    "AuditTrail",
    "AuditTrailManager",
    "ComplianceChecker",
    "ComplianceError",
    "ComplianceFramework",
    "ComplianceRule",
    "PolicyDecision",
    "Severity",
]
