"""Remote plugin registry client with local caching and offline fallback."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import httpx
from cryptography.exceptions import InvalidSignature  # pyright: ignore[reportMissingImports]
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,  # pyright: ignore[reportMissingImports]
)

from autoship.core.metrics import get_registry
from autoship.exceptions import RegistryError
from autoship.models.config import AppConfig, RegistryConfig
from autoship.utils.json_io import load_json, save_json

logger = logging.getLogger("autoship")


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "autoship"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "registry.json"


class RegistryClient:
    """Fetch the plugin registry index from a remote URL with local cache.

    The client follows this resolution order:

    1. If ``use_cache`` is enabled and the cache is fresh, return it.
    2. Fetch from the configured remote ``url`` and update the cache.
    3. If the remote is unreachable, return a stale cache if present.
    4. Fall back to the bundled index shipped with the CLI.
    """

    def __init__(
        self,
        config: RegistryConfig | None = None,
        cache_file: Path | None = None,
    ) -> None:
        self.config = config or RegistryConfig()
        self.cache_file = cache_file or DEFAULT_CACHE_FILE

    def _cache_is_fresh(self) -> bool:
        if not self.config.cache_enabled:
            return False
        try:
            # NOTE: On Python 3.12, Path.exists() re-raises non-ENOENT
            # OSError from Path.stat(); on 3.13+ it swallows all OSError.
            # To behave consistently across versions and to treat any stat
            # failure as "not fresh", we go straight to stat() and let the
            # except below catch FileNotFoundError (a subclass of OSError)
            # as well as permission/IO errors.
            mtime = self.cache_file.stat().st_mtime
        except OSError:
            return False
        return (time.time() - mtime) < self.config.cache_ttl_seconds

    def _read_cache(self) -> dict[str, Any] | None:
        data = load_json(self.cache_file, label="registry cache")
        if data is None:
            return None
        data = cast(dict[str, Any], data)
        try:
            self._verify_index(data)
        except RegistryError as exc:
            logger.warning("Registry cache failed verification: %s", exc)
            return None
        return data

    def _write_cache(self, data: dict[str, Any]) -> None:
        if not self.config.cache_enabled:
            return
        save_json(self.cache_file, data, label="registry cache")

    @staticmethod
    def _canonical_payload(data: dict[str, Any]) -> bytes:
        """Return canonical bytes used for hashing and signing.

        The ``sha256`` and ``signature`` fields are excluded because they depend on
        the payload itself.
        """
        stripped = {k: v for k, v in data.items() if k not in ("sha256", "signature")}
        return json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _verify_index(self, data: dict[str, Any]) -> None:
        """Validate the registry index integrity and optional signature.

        Raises:
            RegistryError: If the sha256 hash does not match or if signature
                verification fails / is required but missing.
        """
        payload = self._canonical_payload(data)

        expected_sha256 = data.get("sha256")
        if expected_sha256 is not None:
            actual_sha256 = hashlib.sha256(payload).hexdigest()
            if actual_sha256 != expected_sha256:
                raise RegistryError(
                    "Registry index sha256 mismatch",
                    details={"expected": expected_sha256, "actual": actual_sha256},
                )

        signature_b64 = data.get("signature")
        public_key_b64 = self.config.public_key

        if signature_b64 is not None and public_key_b64 is not None:
            try:
                public_key_bytes = base64.b64decode(public_key_b64, validate=True)
                signature_bytes = base64.b64decode(signature_b64, validate=True)
                public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
                public_key.verify(signature_bytes, payload)
            except (ValueError, InvalidSignature) as exc:
                raise RegistryError(
                    "Registry index signature verification failed",
                    details={"reason": str(exc)},
                ) from exc
        elif public_key_b64 is not None and signature_b64 is None:
            raise RegistryError(
                "Registry index is not signed but a public key is configured",
            )
        elif signature_b64 is not None:
            # Mirror package_verifier.verify_package: a signature present
            # without a configured public_key is a downgrade, not a warning.
            # Refuse so a tampered index carrying a fake signature cannot be
            # silently accepted just because the sha256 (attacker-computable)
            # happens to match.
            raise RegistryError(
                "Registry index is signed but no public_key is configured — "
                "set `registry.public_key` in your config to enable "
                "verification, or remove the signature from the index.",
            )

    def _fetch_remote(self) -> dict[str, Any] | None:
        registry = get_registry()
        start = time.perf_counter()
        try:
            response = httpx.get(str(self.config.url), timeout=10.0)
            response.raise_for_status()
            data = cast(dict[str, Any], response.json())
            self._verify_index(data)
            self._write_cache(data)
            registry.inc("registry_sync_success", description="Successful registry syncs")
            return data
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("Failed to fetch remote registry index: %s", exc)
            registry.inc("registry_sync_errors", description="Registry sync errors")
            return None
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            registry.record(
                "registry_sync_latency_ms", elapsed_ms, description="Registry sync latency"
            )

    def _bundled_index(self) -> dict[str, Any]:
        package_root = Path(__file__).resolve().parents[1]
        bundled = package_root / "registry" / "plugins.json"
        data = load_json(bundled, label="bundled registry index")
        if isinstance(data, dict):
            return cast(dict[str, Any], data)
        return {"version": 1, "plugins": []}

    def get(self, *, no_cache: bool = False) -> dict[str, Any]:
        """Return the registry index, using cache and fallbacks as needed."""
        if not no_cache and self._cache_is_fresh():
            cached = self._read_cache()
            if cached is not None:
                return cached

        remote = self._fetch_remote()
        if remote is not None:
            return remote

        cached = self._read_cache()
        if cached is not None:
            logger.info("Using stale registry cache because remote is unavailable")
            return cached

        return self._bundled_index()

    def clear_cache(self) -> None:
        """Remove the local registry cache file."""
        try:
            self.cache_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to clear registry cache: %s", exc)

    def fetch_index(self, *, force: bool = False) -> dict[str, Any] | None:
        """Fetch the registry index from the remote URL.

        This bypasses the freshness check and always contacts the remote.
        If ``force`` is True, the local cache is cleared first so a stale
        cached index cannot be returned.
        """
        if force:
            self.clear_cache()
        return self._fetch_remote()


def get_registry_client(config: AppConfig | None = None) -> RegistryClient:
    """Factory for the default registry client."""
    registry_config = config.registry if config else RegistryConfig()
    return RegistryClient(config=registry_config)
