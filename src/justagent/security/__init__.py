"""Enterprise security module for the JustAgent platform.

This package provides four integrated security subsystems that together
deliver enterprise-grade protection for a local-first AI agent platform:

* **Encryption** (:mod:`justagent.security.encryption`) — authenticated
  encryption (AES-256-GCM, ChaCha20-Poly1305, Fernet) with a full key
  lifecycle manager (create, rotate, revoke, persist) and AEAD support.
  Key derivation via PBKDF2, scrypt or Argon2id (lazy import).

* **RBAC** (:mod:`justagent.security.rbac`) — role-based access control
  with a permission matrix keyed by resource type, transitive role
  inheritance, built-in system roles (admin, editor, viewer) and
  structured access decisions for audit logging.

* **Data Protection** (:mod:`justagent.security.data_protection`) — Data
  Loss Prevention with regex-based PII scanning (email, phone, SSN,
  credit card, IP, passport, bank account, ID card, address), sensitivity
  classification, and content sanitisation (full redaction or partial
  masking).

* **Compliance** (:mod:`justagent.security.compliance`) — compliance
  policy enforcement for GDPR, HIPAA, SOC 2, ISO 27001, PCI-DSS and
  CCPA. Includes data access and retention checks, and a thread-safe
  append-only audit trail with SHA-256 hash chaining for tamper
  evidence.

All subsystems use Pydantic v2 for data models, are thread-safe
(``threading.Lock`` / ``threading.RLock``), and follow the
``justagent.security.<submodule>`` logging namespace. Async I/O is used
where appropriate (file scanning, audit recording).

Architecture overview::

    +-------------------+     +-------------------+
    |   Encryption      |     |      RBAC         |
    |  (KeyManager,     |     |  (RBACEngine,     |
    |   EncryptionEngine)|    |   Permission)      |
    +-------------------+     +-------------------+
             |                         |
             v                         v
    +-------------------+     +-------------------+
    |  Data Protection  |     |   Compliance      |
    |  (DLPScanner,     |     |  (ComplianceChecker,|
    |   DataSanitizer)  |     |   AuditTrailManager)|
    +-------------------+     +-------------------+

Quick start::

    from justagent.security import (
        # Encryption
        KeyManager, EncryptionEngine, EncryptionAlgorithm,
        # RBAC
        RBACEngine, Permission, ResourceType, User,
        # Data Protection
        DLPScanner, DataSanitizer, PIIType,
        # Compliance
        ComplianceChecker, AuditTrailManager, ComplianceFramework,
    )

    # --- Encryption ---
    km = KeyManager()
    km.create_key(EncryptionAlgorithm.AES_256_GCM)
    engine = EncryptionEngine(km)
    payload = engine.encrypt_string("secret", associated_data=b"ctx")

    # --- RBAC ---
    rbac = RBACEngine()
    rbac.register_user(User(username="alice", roles=[rbac.ADMIN_ROLE]))
    assert rbac.check_access("alice", ResourceType.DOCUMENT, Permission.DELETE)

    # --- Data Protection ---
    scanner = DLPScanner()
    findings = scanner.scan_text("Contact alice@example.com")
    clean = DataSanitizer(scanner).redact_pii("Email: alice@example.com")

    # --- Compliance ---
    from justagent.security.compliance import AuditTrail, AuditResult
    checker = ComplianceChecker()
    decision = checker.check_data_access("alice", "pii", "export")
    audit = AuditTrailManager()
    audit.record(AuditTrail(
        actor="alice", action="export", resource="records",
        result=AuditResult.SUCCESS,
    ))
"""

from __future__ import annotations

from justagent.security.compliance import (
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
from justagent.security.data_protection import (
    DataSanitizer,
    DataSensitivityLevel,
    DLPAction,
    DLPError,
    DLPRule,
    DLPScanner,
    PIIFinding,
    PIIType,
)
from justagent.security.encryption import (
    DataClassification,
    EncryptedPayload,
    EncryptionAlgorithm,
    EncryptionEngine,
    EncryptionError,
    EncryptionKey,
    KeyDerivationMethod,
    KeyManagementError,
    KeyManager,
    SecurityError,
)
from justagent.security.rbac import (
    AccessDecision,
    Permission,
    RBACEngine,
    RBACError,
    ResourceType,
    Role,
    User,
    UserStatus,
)

__all__ = [
    # encryption
    "DataClassification",
    "EncryptedPayload",
    "EncryptionAlgorithm",
    "EncryptionEngine",
    "EncryptionError",
    "EncryptionKey",
    "KeyDerivationMethod",
    "KeyManagementError",
    "KeyManager",
    "SecurityError",
    # rbac
    "AccessDecision",
    "Permission",
    "RBACEngine",
    "RBACError",
    "ResourceType",
    "Role",
    "User",
    "UserStatus",
    # data protection
    "DLPAction",
    "DLPError",
    "DLPRule",
    "DLPScanner",
    "DataSanitizer",
    "DataSensitivityLevel",
    "PIIFinding",
    "PIIType",
    # compliance
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
