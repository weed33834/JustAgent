"""Tests for the SSO module."""

from __future__ import annotations

import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from autoship.core.sso import (
    Identity,
    SsoError,
    StubSsoProvider,
    get_current_identity,
    get_provider,
    register_provider,
)
from autoship.models.config import AppConfig, SsoConfig
from autoship.utils.permissions import ensure_dir_permissions


def _sso_config(token_cache: Path) -> SsoConfig:
    return SsoConfig(enabled=True, provider="stub", token_cache=token_cache)


def _write_identity(cache_path: Path, identity: Identity) -> None:
    ensure_dir_permissions(cache_path.parent, 0o700)
    cache_path.write_text(identity.model_dump_json(), encoding="utf-8")
    cache_path.chmod(0o600)


def _clear_sso_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove AUTOSHIP_SSO_* env vars so they don't leak between tests."""
    for key in list(os.environ):
        if key.startswith("AUTOSHIP_SSO_"):
            monkeypatch.delenv(key, raising=False)


def test_get_current_identity_disabled_returns_none(tmp_path: Path) -> None:
    config = AppConfig(project_root=tmp_path, sso=SsoConfig(enabled=False))
    assert get_current_identity(config) is None


def test_stub_provider_reads_token_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    cache = tmp_path / "sso" / "token.json"
    identity = Identity(
        user="alice@example.com",
        subject="alice",
        groups=["eng"],
        provider="stub",
        expires_at=None,
    )
    _write_identity(cache, identity)

    provider = StubSsoProvider()
    result = provider.get_identity(_sso_config(cache))
    assert result.user == "alice@example.com"
    assert result.groups == ["eng"]


def test_stub_provider_missing_cache_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_sso_env(monkeypatch)
    cache = tmp_path / "missing.json"
    provider = StubSsoProvider()
    assert provider.get_identity(_sso_config(cache)) is None


def test_stub_provider_expired_token_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_sso_env(monkeypatch)
    cache = tmp_path / "sso" / "token.json"
    expired = Identity(
        user="alice",
        subject="alice",
        groups=[],
        provider="stub",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    _write_identity(cache, expired)

    provider = StubSsoProvider()
    assert provider.get_identity(_sso_config(cache)) is None


def test_stub_provider_reads_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    monkeypatch.setenv("AUTOSHIP_SSO_USER", "bob@example.com")
    monkeypatch.setenv("AUTOSHIP_SSO_SUBJECT", "bob-sub")
    monkeypatch.setenv("AUTOSHIP_SSO_GROUPS", "eng,staff")

    cache = tmp_path / "unused.json"
    provider = StubSsoProvider()
    identity = provider.get_identity(_sso_config(cache))
    assert identity is not None
    assert identity.user == "bob@example.com"
    assert identity.subject == "bob-sub"
    assert identity.groups == ["eng", "staff"]


def test_stub_login_writes_cache_0600(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    monkeypatch.setenv("AUTOSHIP_SSO_USER", "carol@example.com")
    monkeypatch.setenv("AUTOSHIP_SSO_GROUPS", "eng")

    cache = tmp_path / "sso" / "token.json"
    provider = StubSsoProvider()
    identity = provider.login(_sso_config(cache))

    assert identity.user == "carol@example.com"
    assert cache.exists()
    mode = stat.S_IMODE(cache.stat().st_mode)
    assert mode == 0o600
    # Cache contents should round-trip back into the same identity.
    reloaded = StubSsoProvider().get_identity(_sso_config(cache))
    assert reloaded is not None
    assert reloaded.user == "carol@example.com"


def test_stub_logout_deletes_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    cache = tmp_path / "sso" / "token.json"
    _write_identity(cache, Identity(user="alice", subject="alice"))

    provider = StubSsoProvider()
    provider.logout(_sso_config(cache))
    assert not cache.exists()


def test_stub_logout_missing_cache_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)
    cache = tmp_path / "missing.json"
    provider = StubSsoProvider()
    # Should not raise.
    provider.logout(_sso_config(cache))


def test_get_current_identity_corrupt_cache_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_sso_env(monkeypatch)
    cache = tmp_path / "sso" / "token.json"
    ensure_dir_permissions(cache.parent, 0o700)
    cache.write_text("{not valid json", encoding="utf-8")
    cache.chmod(0o600)

    config = AppConfig(project_root=tmp_path, sso=_sso_config(cache))
    # get_current_identity no longer swallows provider
    # exceptions — the caller (cli.main_callback) is responsible for
    # soft-failing and recording an ``sso.identity_failed`` audit record.
    # A corrupt cache surfaces as SsoError so the failure is observable.
    with pytest.raises(SsoError, match="Failed to read SSO token cache"):
        get_current_identity(config)


def test_unknown_provider_raises_sso_error(tmp_path: Path) -> None:
    config = SsoConfig(enabled=True, provider="nonexistent")
    with pytest.raises(SsoError, match="Unknown SSO provider"):
        get_provider(config)


def test_register_provider_custom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sso_env(monkeypatch)

    class FakeProvider(StubSsoProvider):
        pass

    register_provider("fake", FakeProvider)
    try:
        config = SsoConfig(enabled=True, provider="fake")
        assert isinstance(get_provider(config), FakeProvider)
    finally:
        # Restore registry — keep test isolation.
        from autoship.core import sso as sso_module

        sso_module._PROVIDERS.pop("fake", None)
