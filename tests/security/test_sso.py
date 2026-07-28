"""Tests for the SSO module."""

from __future__ import annotations

import time

import pytest

from justagent.security.sso import (
    SSOConfig,
    SSOError,
    SSOManager,
    SSOProtocol,
    SSOProvider,
    TokenPayload,
)


class TestSSOManager:
    def test_register_provider(self) -> None:
        manager = SSOManager()
        provider = SSOProvider(
            name="Corporate OIDC",
            protocol=SSOProtocol.OIDC,
            entity_id="https://corp.example.com",
            metadata_url="https://corp.example.com/.well-known/openid-configuration",
        )
        manager.register_provider(provider)
        retrieved = manager.get_provider(provider.id)
        assert retrieved is not None
        assert retrieved.name == "Corporate OIDC"

    def test_get_provider_unknown_returns_none(self) -> None:
        manager = SSOManager()
        assert manager.get_provider("nonexistent") is None

    def test_list_providers(self) -> None:
        manager = SSOManager()
        manager.register_provider(SSOProvider(
            name="OIDC", protocol=SSOProtocol.OIDC, entity_id="oidc-1",
        ))
        manager.register_provider(SSOProvider(
            name="SAML", protocol=SSOProtocol.SAML, entity_id="saml-1",
        ))
        providers = manager.list_providers()
        assert len(providers) == 2

    def test_remove_provider(self) -> None:
        manager = SSOManager()
        provider = SSOProvider(
            name="Test", protocol=SSOProtocol.OIDC, entity_id="test-1",
        )
        manager.register_provider(provider)
        removed = manager.remove_provider(provider.id)
        assert removed is not None
        assert manager.get_provider(provider.id) is None

    def test_authenticate_with_valid_token(self) -> None:
        manager = SSOManager()
        provider = SSOProvider(
            name="Test OIDC",
            protocol=SSOProtocol.OIDC,
            entity_id="https://test.example.com",
            enabled=True,
        )
        manager.register_provider(provider)

        token = TokenPayload(
            subject="user-123",
            issuer="https://test.example.com",
            audience="justagent",
            expires_at=time.time() + 3600,
            issued_at=time.time(),
            claims={"username": "alice", "email": "alice@example.com", "department": "eng"},
        )
        user = manager.authenticate(token, provider.id)
        assert user is not None
        assert user.username == "alice"
        assert user.email == "alice@example.com"

    def test_authenticate_unknown_provider_raises(self) -> None:
        manager = SSOManager()
        token = TokenPayload(
            subject="user-123",
            issuer="https://test.example.com",
            audience="justagent",
            expires_at=time.time() + 3600,
            issued_at=time.time(),
        )
        with pytest.raises(SSOError, match="Unknown provider"):
            manager.authenticate(token, "nonexistent")

    def test_authenticate_disabled_provider_raises(self) -> None:
        manager = SSOManager()
        provider = SSOProvider(
            name="Disabled",
            protocol=SSOProtocol.OIDC,
            entity_id="https://disabled.example.com",
            enabled=False,
        )
        manager.register_provider(provider)
        token = TokenPayload(
            subject="user-123",
            issuer="https://disabled.example.com",
            audience="justagent",
            expires_at=time.time() + 3600,
            issued_at=time.time(),
        )
        with pytest.raises(SSOError, match="disabled"):
            manager.authenticate(token, provider.id)

    def test_authenticate_expired_token_raises(self) -> None:
        manager = SSOManager()
        provider = SSOProvider(
            name="Test",
            protocol=SSOProtocol.OIDC,
            entity_id="https://test.example.com",
        )
        manager.register_provider(provider)
        token = TokenPayload(
            subject="user-123",
            issuer="https://test.example.com",
            audience="justagent",
            expires_at=time.time() - 100,  # expired
            issued_at=time.time() - 200,
        )
        with pytest.raises(SSOError, match="expired|Expired"):
            manager.authenticate(token, provider.id)

    def test_authenticate_wrong_issuer_raises(self) -> None:
        manager = SSOManager()
        provider = SSOProvider(
            name="Test",
            protocol=SSOProtocol.OIDC,
            entity_id="https://test.example.com",
        )
        manager.register_provider(provider)
        token = TokenPayload(
            subject="user-123",
            issuer="https://wrong.example.com",
            audience="justagent",
            expires_at=time.time() + 3600,
            issued_at=time.time(),
        )
        with pytest.raises(SSOError, match="issuer|Issuer"):
            manager.authenticate(token, provider.id)

    def test_provision_user(self) -> None:
        manager = SSOManager()
        user = manager.provision_user(
            username="newuser",
            email="newuser@example.com",
            department="sales",
        )
        assert user.username == "newuser"
        assert user.email == "newuser@example.com"
        assert user.department == "sales"

    def test_sso_config_with_allowed_domains(self) -> None:
        config = SSOConfig(
            providers=[],
            default_provider=None,
            auto_provision=True,
            allowed_domains=["example.com", "corp.example.com"],
        )
        assert "example.com" in config.allowed_domains
        assert config.auto_provision is True

    def test_sso_protocol_values(self) -> None:
        assert SSOProtocol.SAML.value == "saml"
        assert SSOProtocol.OIDC.value == "oidc"
        assert SSOProtocol.LDAP.value == "ldap"
        assert SSOProtocol.SCIM.value == "scim"

    def test_token_payload_model(self) -> None:
        payload = TokenPayload(
            subject="user-1",
            issuer="https://idp.example.com",
            audience="justagent",
            expires_at=time.time() + 3600,
            issued_at=time.time(),
            claims={"role": "admin"},
        )
        assert payload.subject == "user-1"
        assert payload.claims["role"] == "admin"

    def test_sso_error_is_exception(self) -> None:
        assert issubclass(SSOError, Exception)
