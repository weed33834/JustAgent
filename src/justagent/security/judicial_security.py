"""Judicial security extensions — case classification, audit and evidence chain.

Provides judicial-specific security primitives layered on top of the core
compliance and RBAC subsystems.  Designed for PRC (China) judicial
information-security practice where content is classified into tiers:
公开 (PUBLIC) < 内部 (INTERNAL) < 秘密 (SECRET) < 机密 (CONFIDENTIAL).

Design:

* :class:`JudicialSecurityLevel` — classification tiers for judicial
  content.
* :class:`CaseClassification` — a single case's security classification
  record.
* :class:`CaseSecurity` — registry and access-control manager for
  per-case security classifications (案件密级管理).
* :class:`JudicialAuditLogger` — :class:`AuditTrailManager` subclass
  with convenience methods for judicial operations (review, seal,
  archive, serve).
* :class:`EvidenceItem` — a registered piece of evidence with a content
  hash.
* :class:`CustodyRecord` — a single link in an evidence custody chain,
  sealed with a SHA-256 hash.
* :class:`EvidenceChainSecurity` — tamper-evident evidence chain with
  hash-based integrity verification (证据链完整性保护).

All classes are thread-safe (``threading.Lock`` / ``threading.RLock``)
and use Pydantic v2 for data models, consistent with the rest of
:mod:`justagent.security`.

Example::

    # Case classification
    cs = CaseSecurity()
    cs.classify_case("case-001", JudicialSecurityLevel.SECRET,
                     classified_by="judge_li")
    cs.set_user_clearance("clerk_wang", JudicialSecurityLevel.SECRET)
    assert cs.can_access("clerk_wang", "case-001")

    # Evidence chain
    ecs = EvidenceChainSecurity()
    item = ecs.register_evidence("ev-1", "case-001", b"evidence bytes",
                                 registered_by="officer_zhang")
    ecs.add_custody("ev-1", "lab_tech", "transfer", notes="to lab")
    assert ecs.verify_evidence("ev-1", b"evidence bytes")
    assert ecs.verify_chain("ev-1")
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from justagent.security.compliance import (
    AuditResult,
    AuditTrail,
    AuditTrailManager,
)

logger = logging.getLogger("justagent.security.judicial_security")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class JudicialSecurityError(Exception):
    """Raised for judicial security errors (unknown case, broken chain...)."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JudicialSecurityLevel(str, Enum):  # noqa: UP042 - match existing codebase style
    """Classification tiers for judicial content.

    Ordered from least to most restrictive.  A user's clearance must be
    at or above a case's classification level to gain access.

    Attributes:
        PUBLIC: Publicly disclosable information (公开).
        INTERNAL: Internal information, restricted to authorised
            personnel (内部).
        SECRET: Secret information, requires special clearance (秘密).
        CONFIDENTIAL: Confidential information, highest restriction
            level (机密).
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    SECRET = "secret"
    CONFIDENTIAL = "confidential"

    @classmethod
    def from_value(cls, value: str) -> JudicialSecurityLevel:
        """Resolve a level from its string value (case-insensitive).

        Raises:
            JudicialSecurityError: If *value* is not a recognised level.
        """

        normalised = value.strip().lower()
        for level in cls:
            if level.value == normalised:
                return level
        raise JudicialSecurityError(f"Unknown judicial security level: {value!r}")

    def __ge__(self, other: object) -> bool:
        """True if *self* is at least as restrictive as *other*."""

        if not isinstance(other, JudicialSecurityLevel):
            return NotImplemented
        return _LEVEL_RANK[self] >= _LEVEL_RANK[other]

    def __gt__(self, other: object) -> bool:
        """True if *self* is strictly more restrictive than *other*."""

        if not isinstance(other, JudicialSecurityLevel):
            return NotImplemented
        return _LEVEL_RANK[self] > _LEVEL_RANK[other]

    def __le__(self, other: object) -> bool:
        """True if *self* is at most as restrictive as *other*."""

        if not isinstance(other, JudicialSecurityLevel):
            return NotImplemented
        return _LEVEL_RANK[self] <= _LEVEL_RANK[other]

    def __lt__(self, other: object) -> bool:
        """True if *self* is strictly less restrictive than *other*."""

        if not isinstance(other, JudicialSecurityLevel):
            return NotImplemented
        return _LEVEL_RANK[self] < _LEVEL_RANK[other]


#: Numeric rank per level — lower = less restrictive.
_LEVEL_RANK: dict[JudicialSecurityLevel, int] = {
    JudicialSecurityLevel.PUBLIC: 0,
    JudicialSecurityLevel.INTERNAL: 1,
    JudicialSecurityLevel.SECRET: 2,
    JudicialSecurityLevel.CONFIDENTIAL: 3,
}

#: Default clearance granted to a user when none has been set.
_DEFAULT_CLEARANCE = JudicialSecurityLevel.PUBLIC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> float:
    """Return the current Unix timestamp."""

    return time.time()


def _sha256(data: bytes | str) -> str:
    """Return the SHA-256 hex digest of *data*.

    Strings are encoded as UTF-8 before hashing.
    """

    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


#: SHA-256 of an empty string — sentinel for the genesis custody record.
_GENESIS_PREV_HASH = "0" * 64


# ---------------------------------------------------------------------------
# Case classification
# ---------------------------------------------------------------------------


class CaseClassification(BaseModel):
    """Security classification record for a single judicial case.

    Attributes:
        case_id: Unique case identifier.
        security_level: The :class:`JudicialSecurityLevel` assigned.
        classified_by: User who set the classification.
        classified_at: Unix timestamp of classification.
        reason: Rationale for the classification decision.
        metadata: Arbitrary structured metadata.
    """

    case_id: str
    security_level: JudicialSecurityLevel = JudicialSecurityLevel.INTERNAL
    classified_by: str = ""
    classified_at: float = Field(default_factory=_now)
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseSecurity:
    """Manages security classifications for judicial cases (案件密级管理).

    Maintains a thread-safe registry of :class:`CaseClassification`
    records and a per-user clearance map.  Access to a case is granted
    only when the user's clearance is at or above the case's
    classification level.

    Example::

        cs = CaseSecurity()
        cs.classify_case("case-001", JudicialSecurityLevel.CONFIDENTIAL,
                         classified_by="judge_li", reason="national security")
        cs.set_user_clearance("clerk_wang", JudicialSecurityLevel.SECRET)
        assert not cs.can_access("clerk_wang", "case-001")  # SECRET < CONFIDENTIAL
    """

    def __init__(self) -> None:
        self._classifications: dict[str, CaseClassification] = {}
        self._clearance: dict[str, JudicialSecurityLevel] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Classification management
    # ------------------------------------------------------------------

    def classify_case(
        self,
        case_id: str,
        level: JudicialSecurityLevel,
        classified_by: str = "",
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CaseClassification:
        """Set or replace the classification for *case_id*.

        Returns the created :class:`CaseClassification`.
        """

        classification = CaseClassification(
            case_id=case_id,
            security_level=level,
            classified_by=classified_by,
            reason=reason,
            metadata=metadata or {},
        )
        with self._lock:
            self._classifications[case_id] = classification
        logger.info(
            "Case %s classified as %s by %s",
            case_id,
            level.value,
            classified_by or "(unknown)",
        )
        return classification

    def get_classification(self, case_id: str) -> CaseClassification | None:
        """Return the classification for *case_id*, or ``None``."""

        with self._lock:
            return self._classifications.get(case_id)

    def update_classification(
        self,
        case_id: str,
        level: JudicialSecurityLevel,
        classified_by: str = "",
        reason: str = "",
    ) -> CaseClassification:
        """Update an existing case's classification.

        Raises:
            JudicialSecurityError: If *case_id* is not registered.
        """

        with self._lock:
            if case_id not in self._classifications:
                raise JudicialSecurityError(f"Case not classified: {case_id}")
            return self.classify_case(
                case_id, level, classified_by=classified_by, reason=reason
            )

    def remove_case(self, case_id: str) -> CaseClassification | None:
        """Remove a case classification. Returns the removed record or ``None``."""

        with self._lock:
            removed = self._classifications.pop(case_id, None)
        if removed is not None:
            logger.info("Removed classification for case %s", case_id)
        return removed

    def list_cases(
        self,
        level: JudicialSecurityLevel | None = None,
    ) -> list[CaseClassification]:
        """Return all classifications, optionally filtered by *level*."""

        with self._lock:
            records = list(self._classifications.values())
        if level is not None:
            records = [r for r in records if r.security_level is level]
        return records

    # ------------------------------------------------------------------
    # User clearance
    # ------------------------------------------------------------------

    def set_user_clearance(
        self,
        user: str,
        level: JudicialSecurityLevel,
    ) -> None:
        """Set the maximum clearance level for *user*."""

        with self._lock:
            self._clearance[user] = level
        logger.info("User %s clearance set to %s", user, level.value)

    def get_user_clearance(self, user: str) -> JudicialSecurityLevel:
        """Return *user*'s clearance (defaults to PUBLIC)."""

        with self._lock:
            return self._clearance.get(user, _DEFAULT_CLEARANCE)

    def revoke_clearance(self, user: str) -> None:
        """Remove a user's explicit clearance (reverts to default)."""

        with self._lock:
            self._clearance.pop(user, None)

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def can_access(self, user: str, case_id: str) -> bool:
        """True if *user*'s clearance is at or above the case's level.

        Unregistered cases default to :attr:`JudicialSecurityLevel.INTERNAL`.
        """

        with self._lock:
            classification = self._classifications.get(case_id)
            clearance = self._clearance.get(user, _DEFAULT_CLEARANCE)

        case_level = (
            classification.security_level
            if classification is not None
            else JudicialSecurityLevel.INTERNAL
        )
        return clearance >= case_level

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    @property
    def case_count(self) -> int:
        """Number of classified cases."""

        with self._lock:
            return len(self._classifications)

    @property
    def user_count(self) -> int:
        """Number of users with explicit clearance."""

        with self._lock:
            return len(self._clearance)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the classification registry."""

        with self._lock:
            records = list(self._classifications.values())
        by_level: dict[str, int] = {}
        for record in records:
            key = record.security_level.value
            by_level[key] = by_level.get(key, 0) + 1
        return {
            "total_cases": len(records),
            "total_users_with_clearance": self.user_count,
            "by_level": by_level,
        }


# ---------------------------------------------------------------------------
# Judicial audit logger
# ---------------------------------------------------------------------------


class JudicialAuditLogger(AuditTrailManager):
    """Audit trail manager specialised for judicial operations.

    Extends :class:`AuditTrailManager` with convenience methods for the
    judicial permissions defined in :mod:`justagent.security.rbac`
    (review, seal, archive, serve) and evidence-related actions.  Each
    method records a sealed :class:`AuditTrail` entry with the case id
    stored in ``metadata`` for later querying.

    The full hash-chain integrity, persistence and querying
    capabilities of the parent class are inherited unchanged.

    Example::

        logger = JudicialAuditLogger()
        logger.log_seal("judge_li", case_id="case-001", resource="verdict.pdf")
        entries = logger.query_by_case("case-001")
    """

    def _record_judicial(
        self,
        actor: str,
        action: str,
        case_id: str,
        resource: str = "",
        result: AuditResult = AuditResult.SUCCESS,
        ip_address: str = "",
        user_agent: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuditTrail:
        """Record a judicial audit event with case context."""

        meta: dict[str, Any] = {"case_id": case_id}
        if metadata:
            meta.update(metadata)
        event = AuditTrail(
            actor=actor,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=meta,
        )
        return self.record(event)

    # ------------------------------------------------------------------
    # Judicial operation convenience methods
    # ------------------------------------------------------------------

    def log_review(
        self,
        actor: str,
        case_id: str,
        resource: str = "",
        result: AuditResult = AuditResult.SUCCESS,
        ip_address: str = "",
        user_agent: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuditTrail:
        """Log a case / document review action (审查)."""

        return self._record_judicial(
            actor, "judicial.review", case_id, resource, result,
            ip_address, user_agent, metadata,
        )

    def log_seal(
        self,
        actor: str,
        case_id: str,
        resource: str = "",
        result: AuditResult = AuditResult.SUCCESS,
        ip_address: str = "",
        user_agent: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuditTrail:
        """Log an official seal / stamp action (盖章)."""

        return self._record_judicial(
            actor, "judicial.seal", case_id, resource, result,
            ip_address, user_agent, metadata,
        )

    def log_archive(
        self,
        actor: str,
        case_id: str,
        resource: str = "",
        result: AuditResult = AuditResult.SUCCESS,
        ip_address: str = "",
        user_agent: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuditTrail:
        """Log an archival filing action (归档)."""

        return self._record_judicial(
            actor, "judicial.archive", case_id, resource, result,
            ip_address, user_agent, metadata,
        )

    def log_serve(
        self,
        actor: str,
        case_id: str,
        target: str = "",
        resource: str = "",
        result: AuditResult = AuditResult.SUCCESS,
        ip_address: str = "",
        user_agent: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuditTrail:
        """Log a legal document service / delivery action (送达)."""

        meta = dict(metadata) if metadata else {}
        if target:
            meta["target"] = target
        return self._record_judicial(
            actor, "judicial.serve", case_id, resource, result,
            ip_address, user_agent, meta,
        )

    def log_evidence_access(
        self,
        actor: str,
        case_id: str,
        evidence_id: str,
        action: str,
        result: AuditResult = AuditResult.SUCCESS,
        ip_address: str = "",
        user_agent: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuditTrail:
        """Log an evidence access / handling action.

        Args:
            action: Short verb describing the evidence operation
                (e.g. ``"collect"``, ``"inspect"``, ``"transfer"``).
        """

        meta = dict(metadata) if metadata else {}
        meta["evidence_id"] = evidence_id
        meta["evidence_action"] = action
        return self._record_judicial(
            actor, "judicial.evidence", case_id, evidence_id, result,
            ip_address, user_agent, meta,
        )

    # ------------------------------------------------------------------
    # Case-scoped queries
    # ------------------------------------------------------------------

    def query_by_case(self, case_id: str) -> list[AuditTrail]:
        """Return all audit entries associated with *case_id*."""

        with self._lock:
            snapshot = list(self._entries)
        return [
            e for e in snapshot if e.metadata.get("case_id") == case_id
        ]

    def query_by_evidence(self, evidence_id: str) -> list[AuditTrail]:
        """Return all audit entries associated with *evidence_id*."""

        with self._lock:
            snapshot = list(self._entries)
        return [
            e for e in snapshot if e.metadata.get("evidence_id") == evidence_id
        ]


# ---------------------------------------------------------------------------
# Evidence chain security
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """A registered piece of judicial evidence.

    Attributes:
        evidence_id: Unique evidence identifier (auto-generated UUID4 hex).
        case_id: The case this evidence belongs to.
        content_hash: SHA-256 hex digest of the evidence content.
        registered_by: User who registered the evidence.
        registered_at: Unix timestamp of registration.
        description: Human-readable description of the evidence.
        metadata: Arbitrary structured metadata.
    """

    evidence_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    case_id: str
    content_hash: str
    registered_by: str = ""
    registered_at: float = Field(default_factory=_now)
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CustodyRecord(BaseModel):
    """A single link in an evidence custody chain.

    Each record is sealed with a SHA-256 hash computed over its content
    plus the previous record's hash, forming a tamper-evident chain
    analogous to :class:`AuditTrail`.

    Attributes:
        record_id: Unique record identifier (auto-generated UUID4 hex).
        evidence_id: The evidence this record pertains to.
        handler: User who performed the custody action.
        action: Short verb (e.g. ``"collect"``, ``"transfer"``,
            ``"inspect"``, ``"return"``).
        timestamp: Unix timestamp of the action.
        notes: Free-text notes about the action.
        prev_hash: Hash of the previous custody record (genesis
            sentinel for the first record).
        record_hash: SHA-256 of this record's content (computed on seal).
    """

    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    evidence_id: str
    handler: str
    action: str
    timestamp: float = Field(default_factory=_now)
    notes: str = ""
    prev_hash: str = _GENESIS_PREV_HASH
    record_hash: str = ""

    def _signing_payload(self) -> str:
        """Canonical JSON of every field except ``record_hash``."""

        payload: dict[str, Any] = {
            "record_id": self.record_id,
            "evidence_id": self.evidence_id,
            "handler": self.handler,
            "action": self.action,
            "timestamp": self.timestamp,
            "notes": self.notes,
            "prev_hash": self.prev_hash,
        }
        return json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)

    def compute_hash(self) -> str:
        """Return the SHA-256 hex digest of this record's signing payload."""

        return _sha256(self._signing_payload())

    def seal(self) -> CustodyRecord:
        """Return a copy with ``record_hash`` populated (idempotent)."""

        digest = self.compute_hash()
        if self.record_hash == digest:
            return self
        return self.model_copy(update={"record_hash": digest})

    def verify(self) -> bool:
        """True when the stored ``record_hash`` matches a recomputation."""

        if not self.record_hash:
            return False
        return self.compute_hash() == self.record_hash


class EvidenceChainSecurity:
    """Tamper-evident evidence chain with hash verification (证据链完整性保护).

    Maintains a registry of :class:`EvidenceItem` objects, each storing a
    SHA-256 content hash, and a hash-chained custody trail per evidence
    item.  Content integrity is verified by recomputing the hash of
    provided content; chain integrity is verified by checking that every
    :class:`CustodyRecord`'s ``record_hash`` and ``prev_hash`` link
    correctly.

    Example::

        ecs = EvidenceChainSecurity()
        item = ecs.register_evidence("ev-1", "case-001", b"file bytes",
                                     registered_by="officer_zhang")
        ecs.add_custody("ev-1", "lab", "transfer")
        assert ecs.verify_evidence("ev-1", b"file bytes")
        assert ecs.verify_chain("ev-1")
    """

    def __init__(self) -> None:
        self._evidence: dict[str, EvidenceItem] = {}
        self._chains: dict[str, list[CustodyRecord]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Evidence registration
    # ------------------------------------------------------------------

    def register_evidence(
        self,
        evidence_id: str | None,
        case_id: str,
        content: bytes | str,
        registered_by: str = "",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceItem:
        """Register a new piece of evidence with a content hash.

        Args:
            evidence_id: Unique evidence id.  If ``None`` a UUID4 hex
                is generated.
            content: The evidence content (bytes or UTF-8 string).  The
                content itself is **not** stored — only its SHA-256 hash.
            registered_by: User registering the evidence.

        Returns:
            The created :class:`EvidenceItem`.

        Raises:
            JudicialSecurityError: If *evidence_id* already exists.
        """

        eid = evidence_id or uuid.uuid4().hex
        content_hash = _sha256(content)
        item = EvidenceItem(
            evidence_id=eid,
            case_id=case_id,
            content_hash=content_hash,
            registered_by=registered_by,
            description=description,
            metadata=metadata or {},
        )
        with self._lock:
            if eid in self._evidence:
                raise JudicialSecurityError(f"Evidence already registered: {eid}")
            self._evidence[eid] = item
            self._chains[eid] = []
        logger.info(
            "Evidence %s registered for case %s (hash=%s)",
            eid, case_id, content_hash[:12] + "...",
        )
        return item

    def get_evidence(self, evidence_id: str) -> EvidenceItem | None:
        """Return the evidence item, or ``None``."""

        with self._lock:
            return self._evidence.get(evidence_id)

    def list_evidence(
        self,
        case_id: str | None = None,
    ) -> list[EvidenceItem]:
        """Return all evidence items, optionally filtered by *case_id*."""

        with self._lock:
            items = list(self._evidence.values())
        if case_id is not None:
            items = [i for i in items if i.case_id == case_id]
        return items

    # ------------------------------------------------------------------
    # Custody chain
    # ------------------------------------------------------------------

    def add_custody(
        self,
        evidence_id: str,
        handler: str,
        action: str,
        notes: str = "",
    ) -> CustodyRecord:
        """Append a custody record to the evidence's chain.

        Raises:
            JudicialSecurityError: If *evidence_id* is not registered.
        """

        with self._lock:
            if evidence_id not in self._evidence:
                raise JudicialSecurityError(f"Evidence not registered: {evidence_id}")
            chain = self._chains[evidence_id]
            prev_hash = chain[-1].record_hash if chain else _GENESIS_PREV_HASH
            record = CustodyRecord(
                evidence_id=evidence_id,
                handler=handler,
                action=action,
                notes=notes,
                prev_hash=prev_hash,
            )
            sealed = record.seal()
            chain.append(sealed)
        logger.info(
            "Custody record added for evidence %s: %s by %s",
            evidence_id, action, handler,
        )
        return sealed

    def get_custody_chain(self, evidence_id: str) -> list[CustodyRecord]:
        """Return the full custody chain for *evidence_id* (empty if none)."""

        with self._lock:
            return list(self._chains.get(evidence_id, []))

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_evidence(
        self,
        evidence_id: str,
        content: bytes | str,
    ) -> bool:
        """Verify that *content* matches the stored content hash.

        Returns ``False`` if the evidence is not registered or the hash
        does not match (indicating tampering or corruption).
        """

        with self._lock:
            item = self._evidence.get(evidence_id)
        if item is None:
            logger.warning("Evidence not found for verification: %s", evidence_id)
            return False
        recomputed = _sha256(content)
        if recomputed != item.content_hash:
            logger.error(
                "Evidence content hash mismatch for %s: expected %s, got %s",
                evidence_id,
                item.content_hash[:12] + "...",
                recomputed[:12] + "...",
            )
            return False
        return True

    def verify_chain(self, evidence_id: str | None = None) -> bool:
        """Verify the integrity of one or all custody chains.

        When *evidence_id* is ``None``, every registered evidence chain
        is verified.

        Checks that each :class:`CustodyRecord`'s ``record_hash`` matches
        a fresh recomputation and that ``prev_hash`` links to the
        previous record's hash.
        """

        with self._lock:
            if evidence_id is not None:
                targets = [evidence_id]
            else:
                targets = list(self._chains.keys())

            snapshots: dict[str, list[CustodyRecord]] = {
                eid: list(self._chains.get(eid, [])) for eid in targets
            }

        for eid in targets:
            chain = snapshots[eid]
            for idx, record in enumerate(chain):
                if not record.verify():
                    logger.error(
                        "Custody chain broken for evidence %s at record #%d "
                        "(hash mismatch)", eid, idx,
                    )
                    return False
                if idx == 0:
                    if record.prev_hash != _GENESIS_PREV_HASH:
                        logger.error(
                            "Genesis custody record for %s has invalid prev_hash",
                            eid,
                        )
                        return False
                else:
                    prev = chain[idx - 1]
                    if record.prev_hash != prev.record_hash:
                        logger.error(
                            "Custody chain broken for evidence %s at record #%d "
                            "(prev_hash mismatch)", eid, idx,
                        )
                        return False
        return True

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    @property
    def evidence_count(self) -> int:
        """Number of registered evidence items."""

        with self._lock:
            return len(self._evidence)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the evidence registry."""

        with self._lock:
            total_evidence = len(self._evidence)
            total_records = sum(len(c) for c in self._chains.values())
            cases = {i.case_id for i in self._evidence.values()}
        return {
            "total_evidence": total_evidence,
            "total_custody_records": total_records,
            "total_cases": len(cases),
            "all_chains_valid": self.verify_chain() if total_evidence else True,
        }


__all__ = [
    "CaseClassification",
    "CaseSecurity",
    "CustodyRecord",
    "EvidenceChainSecurity",
    "EvidenceItem",
    "JudicialAuditLogger",
    "JudicialSecurityError",
    "JudicialSecurityLevel",
]
