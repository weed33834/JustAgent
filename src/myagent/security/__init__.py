"""Enterprise security module for the Omniagent platform.

This package provides five integrated security subsystems that together
deliver enterprise-grade protection for a local-first AI agent platform:

* **Encryption** (:mod:`myagent.security.encryption`) — authenticated
  encryption (AES-256-GCM, ChaCha20-Poly1305, Fernet) with a full key
  lifecycle manager (create, rotate, revoke, persist) and AEAD support.
  Key derivation via PBKDF2, scrypt or Argon2id (lazy import).

* **RBAC** (:mod:`myagent.security.rbac`) — role-based access control
  with a permission matrix keyed by resource type, transitive role
  inheritance, built-in system roles (admin, editor, viewer) and
  structured access decisions for audit logging.

* **SSO** (:mod:`myagent.security.sso`) — single sign-on integration
  for SAML, OIDC, LDAP and SCIM protocols. Token validation uses lazy
  imports for protocol-specific libraries (PyJWT, signxml) so the module
  works out-of-the-box with pre-parsed tokens for testing. Supports
  auto-provisioning of users and email domain allow-listing.

* **Data Protection** (:mod:`myagent.security.data_protection`) — Data
  Loss Prevention with regex-based PII scanning (email, phone, SSN,
  credit card, IP, passport, bank account, ID card, address), sensitivity
  classification, and content sanitisation (full redaction or partial
  masking).

* **Compliance** (:mod:`myagent.security.compliance`) — compliance
  policy enforcement for GDPR, HIPAA, SOC 2, ISO 27001, PCI-DSS and
  CCPA. Includes data access and retention checks, and a thread-safe
  append-only audit trail with SHA-256 hash chaining for tamper
  evidence.

All subsystems use Pydantic v2 for data models, are thread-safe
(``threading.Lock`` / ``threading.RLock``), and follow the
``myagent.security.<submodule>`` logging namespace. Async I/O is used
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
    |  Data Protection  |     |       SSO         |
    |  (DLPScanner,     |     |  (SSOManager,     |
    |   DataSanitizer)  |     |   TokenPayload)   |
    +-------------------+     +-------------------+
             |                         |
             v                         v
    +-------------------------------------------+
    |              Compliance                    |
    |  (ComplianceChecker, AuditTrailManager)   |
    +-------------------------------------------+

Quick start::

    from myagent.security import (
        # Encryption
        KeyManager, EncryptionEngine, EncryptionAlgorithm,
        # RBAC
        RBACEngine, Permission, ResourceType, User,
        # SSO
        SSOManager, SSOConfig, SSOProvider, SSOProtocol,
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

    # --- SSO ---
    sso = SSOManager(SSOConfig(
        providers=[SSOProvider(id="okta", name="Okta",
                                protocol=SSOProtocol.OIDC, enabled=True)],
        auto_provision=True,
    ))
    user = sso.authenticate({"sub": "alice", "iss": "okta"}, provider_id="okta")

    # --- Data Protection ---
    scanner = DLPScanner()
    findings = scanner.scan_text("Contact alice@example.com")
    clean = DataSanitizer(scanner).redact_pii("Email: alice@example.com")

    # --- Compliance ---
    from myagent.security.compliance import AuditTrail, AuditResult
    checker = ComplianceChecker()
    decision = checker.check_data_access("alice", "pii", "export")
    audit = AuditTrailManager()
    audit.record(AuditTrail(
        actor="alice", action="export", resource="records",
        result=AuditResult.SUCCESS,
    ))
"""

from __future__ import annotations

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
from myagent.security.data_protection import (
    DataSanitizer,
    DataSensitivityLevel,
    DLPAction,
    DLPError,
    DLPRule,
    DLPScanner,
    PIIFinding,
    PIIType,
)
from myagent.security.encryption import (
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
from myagent.security.rbac import (
    AccessDecision,
    Permission,
    RBACEngine,
    RBACError,
    ResourceType,
    Role,
    User,
    UserStatus,
)
from myagent.security.sso import (
    SSOConfig,
    SSOError,
    SSOManager,
    SSOProtocol,
    SSOProvider,
    TokenPayload,
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
    # sso
    "SSOConfig",
    "SSOError",
    "SSOManager",
    "SSOProtocol",
    "SSOProvider",
    "TokenPayload",
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
