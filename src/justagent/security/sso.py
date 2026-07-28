"""Single sign-on — SAML, OIDC, LDAP and SCIM integration.

Provides enterprise identity federation for the JustAgent platform.
Users authenticate once against an external identity provider (IdP) and
receive a platform :class:`~justagent.security.rbac.User` session.

Protocol-specific validation is performed via **lazy imports** so the
module works out-of-the-box with pre-parsed tokens for testing, and
only pulls in the heavy protocol libraries (``PyJWT``, ``xmlsec``,
``signxml``, ``python3-saml``) when real token strings are processed.

Design:

* :class:`SSOProtocol` — supported federation protocols.
* :class:`SSOProvider` — a registered identity provider configuration.
* :class:`SSOConfig` — the full SSO configuration (providers, defaults).
* :class:`TokenPayload` — normalised claims extracted from any token.
* :class:`SSOManager` — thread-safe manager: register providers,
  validate tokens, authenticate and auto-provision users.

Example::

    config = SSOConfig(
        providers=[
            SSOProvider(
                id="okta",
                name="Okta",
                protocol=SSOProtocol.OIDC,
                entity_id="https://okta.example.com",
                enabled=True,
            )
        ],
        default_provider="okta",
        auto_provision=True,
        allowed_domains=["example.com"],
    )
    manager = SSOManager(config)

    # With a pre-parsed token (for testing):
    payload = manager.validate_token(
        {"sub": "alice", "iss": "https://okta.example.com", "email": "alice@example.com"},
        provider_id="okta",
    )
    user = manager.authenticate(payload, provider_id="okta")
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from justagent.security.rbac import User, UserStatus

logger = logging.getLogger("justagent.security.sso")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SSOError(Exception):
    """Raised for SSO authentication, validation or provisioning failures."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SSOProtocol(str, Enum):  # noqa: UP042 - match existing codebase style
    """Supported federation protocols.

    Attributes:
        SAML: Security Assertion Markup Language 2.0.
        OIDC: OpenID Connect (JWT-based).
        LDAP: Lightweight Directory Access Protocol.
        SCIM: System for Cross-domain Identity Management (provisioning).
    """

    SAML = "saml"
    OIDC = "oidc"
    LDAP = "ldap"
    SCIM = "scim"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SSOProvider(BaseModel):
    """Configuration for a single identity provider.

    Attributes:
        id: Unique provider identifier (auto-generated UUID4 hex).
        name: Human-readable provider name (e.g. ``"Okta"``).
        protocol: The :class:`SSOProtocol` this provider uses.
        entity_id: Provider entity ID / issuer URL.
        metadata_url: URL to the provider's metadata document.
        certificate: PEM-encoded public certificate for signature
            verification (SAML / OIDC).
        client_id: OAuth/OIDC client ID.
        client_secret: OAuth/OIDC client secret (stored in memory only).
        enabled: Whether the provider is active.
        created_at: Unix timestamp of registration.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    protocol: SSOProtocol
    entity_id: str = ""
    metadata_url: str = ""
    certificate: str = ""
    client_id: str = ""
    client_secret: str = ""
    enabled: bool = True
    created_at: float = Field(default_factory=time.time)


class SSOConfig(BaseModel):
    """Full SSO configuration.

    Attributes:
        providers: List of registered :class:`SSOProvider` instances.
        default_provider: Provider ID used when none is specified.
        auto_provision: Whether to automatically create users on first
            successful authentication.
        allowed_domains: Email domains permitted for auto-provisioning
            (empty list = all domains allowed).
        session_timeout: Session lifetime in seconds (default 8h).
    """

    providers: list[SSOProvider] = Field(default_factory=list)
    default_provider: str | None = None
    auto_provision: bool = True
    allowed_domains: list[str] = Field(default_factory=list)
    session_timeout: float = 28_800.0  # 8 hours


class TokenPayload(BaseModel):
    """Normalised claims extracted from any authentication token.

    Attributes:
        subject: Unique identifier for the user at the IdP (``sub``).
        issuer: Token issuer (``iss``).
        audience: Intended audience / client ID (``aud``).
        expires_at: Unix timestamp after which the token is invalid.
        issued_at: Unix timestamp the token was issued.
        claims: All raw claims from the token.
    """

    subject: str
    issuer: str = ""
    audience: str = ""
    expires_at: float = 0.0
    issued_at: float = Field(default_factory=time.time)
    claims: dict[str, Any] = Field(default_factory=dict)

    def is_expired(self, now: float | None = None) -> bool:
        """True if the token has passed its expiry timestamp."""

        if self.expires_at == 0.0:
            return False
        current = time.time() if now is None else now
        return current >= self.expires_at

    @property
    def email(self) -> str:
        """Extract the email claim (``email`` or ``email_address``)."""

        return str(
            self.claims.get("email")
            or self.claims.get("email_address")
            or self.claims.get("upn")
            or ""
        )

    @property
    def display_name(self) -> str:
        """Extract a display name from common claim keys."""

        return str(
            self.claims.get("name")
            or self.claims.get("display_name")
            or self.claims.get("given_name", "")
        )

    @property
    def department(self) -> str:
        """Extract the department / organisational unit claim."""

        return str(
            self.claims.get("department")
            or self.claims.get("ou")
            or self.claims.get("organization")
            or ""
        )


# ---------------------------------------------------------------------------
# SSO manager
# ---------------------------------------------------------------------------


class SSOManager:
    """Thread-safe SSO authentication and user provisioning manager.

    Manages identity provider configurations and processes authentication
    tokens. Protocol-specific validation is performed via lazy imports
    so the module degrades gracefully when optional dependencies are
    absent — pre-parsed tokens (dicts) always work.

    Example::

        manager = SSOManager(config)
        payload = manager.validate_token(token_string, provider_id="okta")
        user = manager.authenticate(payload, provider_id="okta")
    """

    def __init__(self, config: SSOConfig | None = None) -> None:
        self._config = config or SSOConfig()
        self._providers: dict[str, SSOProvider] = {p.id: p for p in self._config.providers}
        self._provisioned_users: dict[str, User] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def register_provider(self, provider: SSOProvider) -> SSOProvider:
        """Register a new identity provider.

        Raises:
            SSOError: If a provider with the same ID already exists.
        """

        with self._lock:
            if provider.id in self._providers:
                raise SSOError(f"Provider already exists: {provider.id}")
            self._providers[provider.id] = provider
            if self._config.default_provider is None:
                self._config.default_provider = provider.id
        logger.info(
            "Registered SSO provider %s (%s, %s)",
            provider.id,
            provider.name,
            provider.protocol.value,
        )
        return provider

    def unregister_provider(self, provider_id: str) -> SSOProvider | None:
        """Remove a provider. Returns the removed provider or ``None``."""

        with self._lock:
            provider = self._providers.pop(provider_id, None)
            if provider is not None:
                if self._config.default_provider == provider_id:
                    self._config.default_provider = next(iter(self._providers), None)
                logger.info("Unregistered SSO provider %s", provider_id)
            return provider

    def remove_provider(self, provider_id: str) -> SSOProvider | None:
        """Alias for :meth:`unregister_provider`."""

        return self.unregister_provider(provider_id)

    def get_provider(self, provider_id: str) -> SSOProvider | None:
        """Return a provider by id, or ``None``."""

        with self._lock:
            return self._providers.get(provider_id)

    def list_providers(self, *, enabled_only: bool = False) -> list[SSOProvider]:
        """Return all (or only enabled) providers."""

        with self._lock:
            providers = list(self._providers.values())
        if enabled_only:
            providers = [p for p in providers if p.enabled]
        return providers

    def _resolve_provider(self, provider_id: str | None) -> SSOProvider:
        """Resolve a provider by id, falling back to the default.

        Raises:
            SSOError: If the provider is unknown or disabled.
        """

        pid = provider_id or self._config.default_provider
        if pid is None:
            raise SSOError("No provider specified and no default configured")
        provider = self._providers.get(pid)
        if provider is None:
            raise SSOError(f"Unknown provider: {pid}")
        if not provider.enabled:
            raise SSOError(f"Provider is disabled: {pid}")
        return provider

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    def validate_token(
        self,
        token: str | dict[str, Any] | TokenPayload,
        provider_id: str | None = None,
    ) -> TokenPayload:
        """Validate an authentication token and return normalised claims.

        Accepts:

        * A **dict** of pre-parsed claims (for testing or when the
          caller has already decoded the token).
        * A :class:`TokenPayload` instance (passed through).
        * A **string** token — for OIDC this is decoded as a JWT (lazy
          ``PyJWT`` import); for SAML the XML is parsed for signature
          verification (lazy ``signxml``/``xmlsec`` import). If the
          protocol library is unavailable, the raw token is rejected.

        Args:
            token: The token to validate.
            provider_id: The provider to validate against. Defaults to
                the configured default provider.

        Raises:
            SSOError: If the token is invalid, expired, or the issuer
                does not match the provider.
        """

        provider = self._resolve_provider(provider_id)

        # Pre-parsed dict — wrap directly.
        if isinstance(token, dict):
            payload = self._payload_from_dict(token)
        elif isinstance(token, TokenPayload):
            payload = token
        elif isinstance(token, str):
            payload = self._validate_token_string(token, provider)
        else:
            raise SSOError(f"Unsupported token type: {type(token).__name__}")

        # Verify issuer matches the provider (when both are set).
        if payload.issuer and provider.entity_id and payload.issuer != provider.entity_id:
            raise SSOError(
                f"Token issuer {payload.issuer!r} does not match "
                f"provider entity_id {provider.entity_id!r}"
            )

        # Check expiry.
        if payload.is_expired():
            raise SSOError("Token has expired")

        logger.debug(
            "Validated token for subject %s via provider %s",
            payload.subject,
            provider.id,
        )
        return payload

    def _payload_from_dict(self, data: dict[str, Any]) -> TokenPayload:
        """Build a :class:`TokenPayload` from a pre-parsed claims dict."""

        now = time.time()
        return TokenPayload(
            subject=str(
                data.get("sub")
                or data.get("subject")
                or data.get("user_id")
                or data.get("name_id")
                or ""
            ),
            issuer=str(data.get("iss") or data.get("issuer") or ""),
            audience=str(data.get("aud") or data.get("audience") or ""),
            expires_at=float(data.get("exp") or data.get("expires_at") or 0.0),
            issued_at=float(data.get("iat") or data.get("issued_at") or now),
            claims=data,
        )

    def _validate_token_string(
        self,
        token: str,
        provider: SSOProvider,
    ) -> TokenPayload:
        """Validate a raw token string using the provider's protocol.

        Uses lazy imports for protocol-specific libraries. Falls back to
        a graceful error when the library is not installed.
        """

        if provider.protocol is SSOProtocol.OIDC:
            return self._validate_oidc_jwt(token, provider)

        if provider.protocol is SSOProtocol.SAML:
            return self._validate_saml_assertion(token, provider)

        if provider.protocol is SSOProtocol.LDAP:
            # LDAP tokens are typically bind results, not standalone tokens.
            raise SSOError("LDAP authentication requires a bind operation, not a token string")

        if provider.protocol is SSOProtocol.SCIM:
            # SCIM is a provisioning protocol; treat the string as JSON.
            import json

            try:
                data = json.loads(token)
            except json.JSONDecodeError as exc:
                raise SSOError(f"Invalid SCIM JSON token: {exc}") from exc
            return self._payload_from_dict(data)

        raise SSOError(f"Unsupported protocol: {provider.protocol!r}")

    def _validate_oidc_jwt(
        self,
        token: str,
        provider: SSOProvider,
    ) -> TokenPayload:
        """Validate an OIDC JWT using PyJWT (lazy import)."""

        try:
            import jwt  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "PyJWT not installed; cannot validate raw JWT strings. "
                "Pass a pre-parsed dict for testing."
            )
            raise SSOError(
                "PyJWT is required to validate raw OIDC JWT tokens. "
                "Install 'PyJWT' or pass a pre-parsed claims dict."
            ) from None

        options: dict[str, Any] = {"verify_aud": bool(provider.client_id)}
        audience = provider.client_id or None

        try:
            if provider.client_secret:
                decoded = jwt.decode(
                    token,
                    provider.client_secret,
                    algorithms=["HS256", "RS256"],
                    audience=audience,
                    options=options,
                )
            elif provider.certificate:
                decoded = jwt.decode(
                    token,
                    provider.certificate,
                    algorithms=["RS256"],
                    audience=audience,
                    options=options,
                )
            else:
                # No verification key — decode without verification
                # (for development/testing only).
                logger.warning(
                    "Decoding JWT without signature verification "
                    "(no secret or certificate configured for provider %s)",
                    provider.id,
                )
                decoded = jwt.decode(token, options={"verify_signature": False})
        except Exception as exc:  # noqa: BLE001 - surface all JWT errors
            raise SSOError(f"JWT validation failed: {exc}") from exc

        return self._payload_from_dict(decoded)

    def _validate_saml_assertion(
        self,
        token: str,
        provider: SSOProvider,
    ) -> TokenPayload:
        """Validate a SAML assertion using signxml (lazy import)."""

        try:
            from signxml import XMLVerifier  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "signxml not installed; cannot validate raw SAML assertions. "
                "Pass a pre-parsed dict for testing."
            )
            raise SSOError(
                "signxml is required to validate raw SAML tokens. "
                "Install 'signxml' or pass a pre-parsed claims dict."
            ) from None

        try:
            cert = provider.certificate or None
            verified = XMLVerifier().verify(token, x509_cert=cert).signed_xml
            # Extract claims from the verified SAML assertion.
            claims = self._extract_saml_claims(verified)
        except Exception as exc:  # noqa: BLE001 - surface all SAML errors
            raise SSOError(f"SAML validation failed: {exc}") from exc

        return self._payload_from_dict(claims)

    @staticmethod
    def _extract_saml_claims(signed_xml: Any) -> dict[str, Any]:
        """Extract attribute claims from a verified SAML assertion XML.

        This is a best-effort extraction that looks for common SAML
        attribute names. The *signed_xml* object may be an
        ``lxml.etree._Element`` or a parsed DOM.
        """

        claims: dict[str, Any] = {}
        # Try lxml-style iteration.
        try:
            ns = {"saml": "urn:oasis:names:tc:SAML:2.0:assertion"}
            for attr in signed_xml.findall(".//saml:Attribute", ns):
                name = attr.get("Name", "")
                values = [v.text for v in attr.findall("saml:AttributeValue", ns) if v.text]
                if name:
                    claims[name] = values[0] if len(values) == 1 else values
        except Exception:  # noqa: BLE001 - best-effort extraction
            pass

        # Map common SAML attribute names to standard claim keys.
        mapping = {
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress": "email",
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name": "name",
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier": "sub",
        }
        for saml_name, standard_name in mapping.items():
            if saml_name in claims and standard_name not in claims:
                claims[standard_name] = claims[saml_name]

        return claims

    # ------------------------------------------------------------------
    # Authentication & provisioning
    # ------------------------------------------------------------------

    def authenticate(
        self,
        token: str | dict[str, Any] | TokenPayload,
        provider_id: str | None = None,
    ) -> User:
        """Validate a token and return (or provision) a platform user.

        If ``auto_provision`` is enabled and the user does not yet exist,
        a new :class:`User` is created from the token claims. Domain
        restrictions (``allowed_domains``) are enforced.

        Raises:
            SSOError: If validation fails or the domain is not allowed.
        """

        payload = self.validate_token(token, provider_id)
        return self.provision_user(payload, provider_id)

    def provision_user(
        self,
        payload: TokenPayload | None = None,
        provider_id: str | None = None,
        *,
        username: str | None = None,
        email: str | None = None,
        department: str | None = None,
    ) -> User:
        """Create or retrieve a platform user from a validated token.

        Accepts either a :class:`TokenPayload` (from token validation)
        or explicit keyword arguments for direct provisioning.

        When the user already exists (matched by subject or email) the
        existing record is returned and its ``last_active`` timestamp is
        refreshed.

        Raises:
            SSOError: If auto-provisioning is disabled and the user is
                unknown, or if the email domain is not allowed.
        """

        # Build from keyword arguments when no payload is given.
        if payload is None:
            if username is None:
                raise SSOError("Either payload or username must be provided")
            payload = TokenPayload(
                subject=username,
                claims={
                    "username": username,
                    "email": email or "",
                    "department": department or "",
                },
            )
            # Skip provider resolution for direct provisioning.
            provider = None
        else:
            provider = self._resolve_provider(provider_id)

        user_email = email or payload.email

        # Enforce allowed domains.
        if user_email and self._config.allowed_domains:
            domain = user_email.rsplit("@", 1)[-1].lower() if "@" in user_email else ""
            allowed = {d.lower() for d in self._config.allowed_domains}
            if domain not in allowed:
                raise SSOError(f"Email domain {domain!r} is not in the allowed list")

        with self._lock:
            # Look up existing user by subject or email.
            user = self._find_user(payload.subject, user_email)
            if user is not None:
                user.last_active = time.time()
                logger.debug("Authenticated existing user %s", user.username)
                return user

            # Auto-provision new user.
            if not self._config.auto_provision and provider is not None:
                raise SSOError(
                    f"User {payload.subject!r} is not registered and auto-provisioning is disabled"
                )

            # Prefer username claim, then email, then subject.
            claim_username = str(
                payload.claims.get("username")
                or payload.claims.get("preferred_username")
                or ""
            )
            resolved_username = claim_username or user_email or payload.subject or f"user-{uuid.uuid4().hex[:8]}"
            user_department = department or payload.department
            user = User(
                username=resolved_username,
                display_name=payload.display_name or resolved_username,
                email=user_email,
                department=user_department,
                status=UserStatus.ACTIVE,
                metadata={
                    "sso_provider": provider.id if provider else "direct",
                    "sso_subject": payload.subject,
                    "sso_issuer": payload.issuer,
                },
            )
            self._provisioned_users[user.id] = user
        logger.info(
            "Provisioned new user %s via %s",
            user.username,
            provider.id if provider else "direct provisioning",
        )
        return user

    def _find_user(self, subject: str, email: str) -> User | None:
        """Find a provisioned user by SSO subject or email (caller holds lock)."""

        for user in self._provisioned_users.values():
            user_subject = user.metadata.get("sso_subject", "")
            if subject and user_subject == subject:
                return user
            if email and user.email.lower() == email.lower():
                return user
        return None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_user(self, user_id: str) -> User | None:
        """Return a provisioned user by id, or ``None``."""

        with self._lock:
            return self._provisioned_users.get(user_id)

    def list_users(self) -> list[User]:
        """Return all provisioned users."""

        with self._lock:
            return list(self._provisioned_users.values())

    def verify_domain(self, email: str) -> bool:
        """Return ``True`` if *email*'s domain is allowed (or no restriction)."""

        if not self._config.allowed_domains:
            return True
        if "@" not in email:
            return False
        domain = email.rsplit("@", 1)[-1].lower()
        return domain in {d.lower() for d in self._config.allowed_domains}

    @property
    def config(self) -> SSOConfig:
        """The current SSO configuration."""

        return self._config

    @property
    def provider_count(self) -> int:
        """Total number of registered providers."""

        with self._lock:
            return len(self._providers)

    @property
    def user_count(self) -> int:
        """Total number of provisioned users."""

        with self._lock:
            return len(self._provisioned_users)


__all__ = [
    "SSOConfig",
    "SSOError",
    "SSOManager",
    "SSOProtocol",
    "SSOProvider",
    "TokenPayload",
]
