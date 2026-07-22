"""Tests for the caching registry client."""

from __future__ import annotations

import base64
import hashlib
import json
import stat
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from myagent.core.registry_client import RegistryClient
from myagent.exceptions import RegistryError
from myagent.models.config import RegistryConfig


def _canonical_payload(data: dict[str, Any]) -> bytes:
    """Return the canonical bytes used for signing and hashing."""
    stripped = {k: v for k, v in data.items() if k not in ("sha256", "signature")}
    return json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign_index(data: dict[str, Any]) -> tuple[bytes, bytes]:
    """Return (public_key_bytes, signature_bytes) for the given index data."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    signature = private_key.sign(_canonical_payload(data))
    return public_key.public_bytes_raw(), signature


def test_returns_fresh_cache(tmp_path: Path) -> None:
    cache = tmp_path / "registry.json"
    cached_data = {"version": 2, "plugins": [{"name": "cached"}]}
    cache.write_text(json.dumps(cached_data), encoding="utf-8")

    client = RegistryClient(cache_file=cache)
    result = client.get()
    assert result == cached_data


def test_fetches_remote_and_caches(tmp_path: Path) -> None:
    cache = tmp_path / "registry.json"
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}

    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(cache_file=cache)
        result = client.get()

    assert result == remote_data
    assert cache.exists()
    assert json.loads(cache.read_text(encoding="utf-8")) == remote_data


def test_falls_back_to_stale_cache_on_remote_failure(tmp_path: Path) -> None:
    cache = tmp_path / "registry.json"
    stale_data = {"version": 2, "plugins": [{"name": "stale"}]}
    cache.write_text(json.dumps(stale_data), encoding="utf-8")

    config = RegistryConfig(cache_enabled=True, cache_ttl_seconds=0)
    with patch(
        "myagent.core.registry_client.httpx.get", side_effect=httpx.ConnectError("offline")
    ):
        client = RegistryClient(config=config, cache_file=cache)
        result = client.get()

    assert result == stale_data


def test_no_cache_option_bypasses_cache(tmp_path: Path) -> None:
    cache = tmp_path / "registry.json"
    cached_data = {"version": 2, "plugins": [{"name": "cached"}]}
    remote_data = {"version": 2, "plugins": [{"name": "fresh"}]}
    cache.write_text(json.dumps(cached_data), encoding="utf-8")

    config = RegistryConfig(cache_enabled=True, cache_ttl_seconds=3600)
    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(config=config, cache_file=cache)
        result = client.get(no_cache=True)

    assert result == remote_data


def test_clear_cache_removes_file(tmp_path: Path) -> None:
    cache = tmp_path / "registry.json"
    cache.write_text("{}", encoding="utf-8")
    client = RegistryClient(cache_file=cache)
    client.clear_cache()
    assert not cache.exists()


def test_valid_signed_index_is_cached(tmp_path: Path) -> None:
    cache = tmp_path / "registry.json"
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}
    public_key, signature = _sign_index(remote_data)
    remote_data["sha256"] = hashlib.sha256(_canonical_payload(remote_data)).hexdigest()
    remote_data["signature"] = base64.b64encode(signature).decode("ascii")

    config = RegistryConfig(public_key=base64.b64encode(public_key).decode("ascii"))
    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(config=config, cache_file=cache)
        result = client.get()

    assert result["plugins"] == [{"name": "remote"}]
    assert cache.exists()
    assert json.loads(cache.read_text(encoding="utf-8"))["signature"] == remote_data["signature"]


def test_tampered_signature_raises_and_does_not_cache(tmp_path: Path) -> None:
    cache = tmp_path / "registry.json"
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}
    public_key, signature = _sign_index(remote_data)
    remote_data["sha256"] = hashlib.sha256(_canonical_payload(remote_data)).hexdigest()
    remote_data["signature"] = base64.b64encode(signature).decode("ascii")

    # Tamper with the payload after signing.
    remote_data["plugins"][0]["name"] = "evil"

    config = RegistryConfig(public_key=base64.b64encode(public_key).decode("ascii"))
    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(config=config, cache_file=cache)

        with pytest.raises(RegistryError):
            client.get()

    assert not cache.exists()


def test_tampered_cache_is_rejected(tmp_path: Path) -> None:
    cache = tmp_path / "registry.json"
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}
    public_key, signature = _sign_index(remote_data)
    remote_data["sha256"] = hashlib.sha256(_canonical_payload(remote_data)).hexdigest()
    remote_data["signature"] = base64.b64encode(signature).decode("ascii")

    # Write a cache file that has a valid structure but does not match the signature.
    tampered = dict(remote_data)
    tampered["plugins"] = [{"name": "evil"}]
    cache.write_text(json.dumps(tampered), encoding="utf-8")

    config = RegistryConfig(
        public_key=base64.b64encode(public_key).decode("ascii"), cache_ttl_seconds=3600
    )
    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(config=config, cache_file=cache)
        result = client.get()

    assert result["plugins"] == [{"name": "remote"}]


def test_missing_signature_with_public_key_raises(tmp_path: Path) -> None:
    cache = tmp_path / "registry.json"
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}
    public_key, _signature = _sign_index(remote_data)

    config = RegistryConfig(public_key=base64.b64encode(public_key).decode("ascii"))
    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(config=config, cache_file=cache)
        with pytest.raises(RegistryError):
            client.get()


def test_invalid_sha256_raises(tmp_path: Path) -> None:
    cache = tmp_path / "registry.json"
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}
    public_key, signature = _sign_index(remote_data)
    remote_data["sha256"] = "0" * 64
    remote_data["signature"] = base64.b64encode(signature).decode("ascii")

    config = RegistryConfig(public_key=base64.b64encode(public_key).decode("ascii"))
    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(config=config, cache_file=cache)
        with pytest.raises(RegistryError):
            client.get()


def test_registry_cache_has_restrictive_permissions(tmp_path: Path) -> None:
    """Registry cache directory and file are only owner-readable/writable."""
    cache = tmp_path / "registry.json"
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}

    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(cache_file=cache)
        client.get()

    assert cache.parent.exists()
    assert stat.S_IMODE(cache.parent.stat().st_mode) == 0o700
    assert cache.exists()
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600


def test_cache_disabled_treats_existing_cache_as_stale(tmp_path: Path) -> None:
    """When caching is disabled, an existing cache file is not considered fresh."""
    cache = tmp_path / "registry.json"
    cached_data = {"version": 2, "plugins": [{"name": "cached"}]}
    cache.write_text(json.dumps(cached_data), encoding="utf-8")
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}

    config = RegistryConfig(cache_enabled=False)
    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(config=config, cache_file=cache)
        result = client.get()

    assert result == remote_data
    # Cache file is never written when caching is disabled.
    assert json.loads(cache.read_text(encoding="utf-8")) == cached_data


def test_cache_stat_oserror_treated_as_not_fresh(tmp_path: Path) -> None:
    """An OSError reading cache mtime is swallowed and treated as stale."""
    cache = tmp_path / "registry.json"
    cache.write_text(json.dumps({"version": 2, "plugins": []}), encoding="utf-8")
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}

    client = RegistryClient(cache_file=cache)
    with (
        patch.object(Path, "stat", side_effect=OSError("stat denied")),
        patch("myagent.core.registry_client.httpx.get") as mock_get,
    ):
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        result = client.get()

    assert result == remote_data


def test_corrupt_cache_is_ignored_and_refetches(tmp_path: Path) -> None:
    """A cache file with invalid JSON is dropped and the remote is fetched."""
    cache = tmp_path / "registry.json"
    cache.write_text("{not valid json", encoding="utf-8")
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}

    config = RegistryConfig(cache_enabled=True, cache_ttl_seconds=3600)
    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(config=config, cache_file=cache)
        result = client.get()

    assert result == remote_data


def test_write_cache_oserror_is_swallowed(tmp_path: Path) -> None:
    """A failure writing the cache must not propagate to callers."""
    cache = tmp_path / "registry.json"
    remote_data = {"version": 2, "plugins": [{"name": "remote"}]}

    with (
        patch("myagent.core.registry_client.httpx.get") as mock_get,
        patch(
            "myagent.utils.json_io.ensure_file_permissions",
            side_effect=OSError("disk full"),
        ),
    ):
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(cache_file=cache)
        result = client.get()  # must not raise

    assert result == remote_data


def test_invalid_public_key_value_raises_registry_error(tmp_path: Path) -> None:
    """A public_key that is not valid base64 surfaces as RegistryError (ValueError branch)."""
    cache = tmp_path / "registry.json"
    remote_data = {
        "version": 2,
        "plugins": [{"name": "remote"}],
        "sha256": hashlib.sha256(
            _canonical_payload({"version": 2, "plugins": [{"name": "remote"}]})
        ).hexdigest(),
        "signature": "deadbeef",
    }

    config = RegistryConfig(public_key="@@@ not base64 @@@")
    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(config=config, cache_file=cache)
        with pytest.raises(RegistryError, match="signature verification failed"):
            client.get()


def test_bundled_index_returned_when_no_cache_and_no_remote(tmp_path: Path) -> None:
    """With no cache and an unreachable remote, the bundled index is the final fallback."""
    cache = tmp_path / "registry.json"
    with patch(
        "myagent.core.registry_client.httpx.get", side_effect=httpx.ConnectError("offline")
    ):
        client = RegistryClient(cache_file=cache)
        result = client.get()

    bundled = json.loads(
        (
            Path(__file__).resolve().parents[2] / "src" / "myagent" / "registry" / "plugins.json"
        ).read_text(encoding="utf-8")
    )
    assert result["plugins"] == bundled["plugins"]


def test_bundled_index_returns_empty_when_bundled_missing(tmp_path: Path) -> None:
    """If the bundled index file is absent, an empty index is returned."""
    cache = tmp_path / "registry.json"
    with patch(
        "myagent.core.registry_client.httpx.get", side_effect=httpx.ConnectError("offline")
    ):
        client = RegistryClient(cache_file=cache)
        with patch.object(Path, "exists", return_value=False):
            result = client._bundled_index()

    assert result == {"version": 1, "plugins": []}


def test_clear_cache_oserror_is_swallowed(tmp_path: Path) -> None:
    """A failure unlinking the cache must not propagate."""
    cache = tmp_path / "registry.json"
    cache.write_text("{}", encoding="utf-8")
    client = RegistryClient(cache_file=cache)
    with patch.object(Path, "unlink", side_effect=OSError("permission denied")):
        client.clear_cache()  # must not raise


def test_fetch_index_force_clears_cache_first(tmp_path: Path) -> None:
    """fetch_index(force=True) clears the cache before contacting the remote."""
    cache = tmp_path / "registry.json"
    cache.write_text(json.dumps({"version": 2, "plugins": [{"name": "stale"}]}), encoding="utf-8")
    remote_data = {"version": 2, "plugins": [{"name": "fresh"}]}

    with patch("myagent.core.registry_client.httpx.get") as mock_get:
        mock_get.return_value.json.return_value = remote_data
        mock_get.return_value.raise_for_status = lambda: None
        client = RegistryClient(cache_file=cache)
        result = client.fetch_index(force=True)

    assert result == remote_data


def test_fetch_index_returns_none_on_remote_failure(tmp_path: Path) -> None:
    """When the remote is unreachable, fetch_index returns None (no fallback)."""
    cache = tmp_path / "registry.json"
    with patch(
        "myagent.core.registry_client.httpx.get", side_effect=httpx.ConnectError("offline")
    ):
        client = RegistryClient(cache_file=cache)
        assert client.fetch_index() is None


def test_get_registry_client_factory() -> None:
    """get_registry_client builds a client from an AppConfig or a default RegistryConfig."""
    from myagent.core.registry_client import get_registry_client
    from myagent.models.config import AppConfig

    default_client = get_registry_client()
    assert default_client.config.url == RegistryConfig().url

    config = AppConfig()
    configured_client = get_registry_client(config)
    assert configured_client.config.url == config.registry.url
