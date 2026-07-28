"""Enterprise encryption — authenticated encryption, key lifecycle and AEAD.

Provides the cryptographic foundation for the Omniagent platform's
local-first security model. All sensitive data at rest is encrypted with
authenticated encryption (AEAD) so that tampering is detected on
decryption.

Design:

* :class:`EncryptionAlgorithm` — supported ciphers (AES-256-GCM,
  ChaCha20-Poly1305, Fernet).
* :class:`KeyDerivationMethod` — password-to-key derivation functions
  (PBKDF2-HMAC-SHA256, Argon2id via lazy import, scrypt).
* :class:`DataClassification` — sensitivity tiers driving policy.
* :class:`EncryptionKey` — a managed key with material, algorithm,
  derivation metadata and lifecycle timestamps.
* :class:`EncryptedPayload` — self-describing ciphertext bundle carrying
  the key id, algorithm, nonce, tag and optional associated data.
* :class:`KeyManager` — thread-safe key lifecycle: create, rotate,
  revoke, list, with optional JSON file persistence.
* :class:`EncryptionEngine` — thread-safe encrypt/decrypt for strings
  and bytes with AEAD support and automatic algorithm selection.

The module uses the ``cryptography`` library for all primitives.
External KDFs (Argon2) are imported lazily so the module degrades
gracefully when the optional dependency is absent.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("myagent.security.encryption")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SecurityError(Exception):
    """Base exception for all security-related errors in this module."""


class EncryptionError(SecurityError):
    """Raised when an encryption or decryption operation fails."""


class KeyManagementError(SecurityError):
    """Raised for invalid key lifecycle operations (unknown, expired, revoked)."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EncryptionAlgorithm(str, Enum):  # noqa: UP042 - match existing codebase style
    """Supported authenticated encryption algorithms.

    Attributes:
        AES_256_GCM: AES-256 in Galois/Counter Mode — NIST standard AEAD.
        CHACHA20_POLY1305: ChaCha20 stream cipher with Poly1305 MAC —
            fast on architectures without AES-NI.
        FERNET: Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256) —
            simple, self-contained token format without AEAD support.
    """

    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"


class KeyDerivationMethod(str, Enum):  # noqa: UP042
    """Password-to-key derivation functions.

    Attributes:
        PBKDF2: PBKDF2-HMAC-SHA256 (always available via ``cryptography``).
        ARGON2: Argon2id (requires the optional ``argon2-cffi`` package).
        SCRYPT: scrypt PBKDF (always available via ``cryptography``).
    """

    PBKDF2 = "pbkdf2"
    ARGON2 = "argon2"
    SCRYPT = "scrypt"


class DataClassification(str, Enum):  # noqa: UP042
    """Sensitivity tier for classifying protected data.

    Attributes:
        PUBLIC: Approved for public release; no encryption required.
        INTERNAL: Internal use only; encryption recommended.
        CONFIDENTIAL: Business-sensitive; encryption required.
        RESTRICTED: Highest sensitivity; encryption + access logging required.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


# ---------------------------------------------------------------------------
# Key length constants
# ---------------------------------------------------------------------------

#: All supported algorithms use a 256-bit (32-byte) symmetric key.
_KEY_LENGTH_BYTES = 32

#: Nonce length for AES-GCM and ChaCha20-Poly1305 (96-bit).
_NONCE_LENGTH_BYTES = 12

#: Authentication tag length for AEAD ciphers (128-bit).
_TAG_LENGTH_BYTES = 16

#: PBKDF2 iteration count (OWASP 2023 recommendation for SHA-256).
_PBKDF2_ITERATIONS = 600_000


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class EncryptionKey(BaseModel):
    """A managed encryption key with lifecycle metadata.

    Attributes:
        key_id: Unique key identifier (auto-generated UUID4 hex).
        algorithm: The :class:`EncryptionAlgorithm` this key is for.
        created_at: Unix timestamp of creation.
        expires_at: Unix timestamp after which the key must not be used
            for *new* encryption (``None`` means no expiry).
        material: Base64-encoded raw key bytes (always 32 bytes).
        derivation_method: How the key material was derived from a
            passphrase (``PBKDF2`` when generated randomly without a
            passphrase).
        salt: Base64-encoded salt used during derivation (may be empty
            for randomly generated keys).
        revoked: Whether the key has been administratively revoked.
        rotated_from: The ``key_id`` this key replaced, if created via
            :meth:`KeyManager.rotate_key`.
        classification: Optional data classification this key protects.
    """

    key_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    algorithm: EncryptionAlgorithm
    created_at: float = Field(default_factory=time.time)
    expires_at: float | None = None
    material: str
    derivation_method: KeyDerivationMethod = KeyDerivationMethod.PBKDF2
    salt: str = ""
    revoked: bool = False
    rotated_from: str | None = None
    classification: DataClassification | None = None

    @property
    def raw_material(self) -> bytes:
        """Decode and return the raw key bytes."""

        return base64.b64decode(self.material)

    @property
    def raw_salt(self) -> bytes:
        """Decode and return the salt bytes (empty if unset)."""

        return base64.b64decode(self.salt) if self.salt else b""

    def is_expired(self, now: float | None = None) -> bool:
        """True if the key has passed its expiry timestamp."""

        if self.expires_at is None:
            return False
        current = time.time() if now is None else now
        return current >= self.expires_at

    def is_usable(self, now: float | None = None) -> bool:
        """True if the key is neither revoked nor expired."""

        return not self.revoked and not self.is_expired(now)


class EncryptedPayload(BaseModel):
    """A self-describing encrypted data bundle.

    Carries everything needed to decrypt the ciphertext back to
    plaintext, *except* the key material itself (which is resolved via
    ``key_id`` from the :class:`KeyManager`).

    Attributes:
        key_id: Identifier of the :class:`EncryptionKey` used.
        algorithm: The :class:`EncryptionAlgorithm` that produced the
            ciphertext.
        nonce: The per-message nonce / IV (empty for Fernet).
        ciphertext: The encrypted plaintext (without the tag).
        tag: The authentication tag (empty for Fernet, which embeds it).
        associated_data: Optional AEAD associated data (authenticated
            but not encrypted).
        created_at: Unix timestamp of encryption.
        classification: Optional data classification of the plaintext.
    """

    key_id: str
    algorithm: EncryptionAlgorithm
    nonce: bytes = b""
    ciphertext: bytes
    tag: bytes = b""
    associated_data: bytes = b""
    created_at: float = Field(default_factory=time.time)
    classification: DataClassification | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict with binary fields base64-encoded."""

        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "nonce": base64.b64encode(self.nonce).decode("ascii"),
            "ciphertext": base64.b64encode(self.ciphertext).decode("ascii"),
            "tag": base64.b64encode(self.tag).decode("ascii"),
            "associated_data": base64.b64encode(self.associated_data).decode("ascii"),
            "created_at": self.created_at,
            "classification": self.classification.value if self.classification else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncryptedPayload:
        """Reconstruct an :class:`EncryptedPayload` from a JSON-safe dict."""

        return cls(
            key_id=data["key_id"],
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            nonce=base64.b64decode(data.get("nonce", "")),
            ciphertext=base64.b64decode(data["ciphertext"]),
            tag=base64.b64decode(data.get("tag", "")),
            associated_data=base64.b64decode(data.get("associated_data", "")),
            created_at=data.get("created_at", time.time()),
            classification=(
                DataClassification(data["classification"]) if data.get("classification") else None
            ),
        )


# ---------------------------------------------------------------------------
# Key derivation helpers
# ---------------------------------------------------------------------------


def _derive_key(
    password: str | bytes,
    salt: bytes,
    method: KeyDerivationMethod,
    length: int = _KEY_LENGTH_BYTES,
) -> bytes:
    """Derive a *length*-byte key from *password* and *salt*.

    Args:
        password: The passphrase (str or bytes).
        salt: Cryptographic salt.
        method: The derivation method to use.
        length: Desired key length in bytes.

    Returns:
        The derived key bytes.

    Raises:
        EncryptionError: If the requested method is unavailable.
    """

    password_bytes = password.encode("utf-8") if isinstance(password, str) else password

    if method is KeyDerivationMethod.PBKDF2:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            iterations=_PBKDF2_ITERATIONS,
        )
        return kdf.derive(password_bytes)

    if method is KeyDerivationMethod.SCRYPT:
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

        kdf = Scrypt(
            salt=salt,
            length=length,
            n=2**14,
            r=8,
            p=1,
        )
        return kdf.derive(password_bytes)

    if method is KeyDerivationMethod.ARGON2:
        try:
            from argon2.low_level import Type as Argon2Type
            from argon2.low_level import hash_secret_raw
        except ImportError:
            logger.warning("argon2-cffi not installed; falling back to PBKDF2 for key derivation")
            return _derive_key(password_bytes, salt, KeyDerivationMethod.PBKDF2, length)

        return hash_secret_raw(
            secret=password_bytes,
            salt=salt,
            time_cost=3,
            memory_cost=65_536,
            parallelism=4,
            hash_len=length,
            type=Argon2Type.ID,
        )

    raise EncryptionError(f"Unsupported key derivation method: {method!r}")


def _generate_random_key() -> bytes:
    """Generate a cryptographically random 256-bit key."""

    return os.urandom(_KEY_LENGTH_BYTES)


def _get_aead_cipher(algorithm: EncryptionAlgorithm, raw_key: bytes) -> Any:
    """Return the AEAD cipher object for *algorithm* initialised with *raw_key*.

    For :attr:`EncryptionAlgorithm.FERNET` a :class:`Fernet` instance is
    returned (Fernet does not support AEAD associated data).

    Raises:
        EncryptionError: If the key length is invalid for the algorithm.
    """

    if len(raw_key) != _KEY_LENGTH_BYTES:
        raise EncryptionError(f"Key must be {_KEY_LENGTH_BYTES} bytes, got {len(raw_key)}")

    if algorithm is EncryptionAlgorithm.AES_256_GCM:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        return AESGCM(raw_key)

    if algorithm is EncryptionAlgorithm.CHACHA20_POLY1305:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        return ChaCha20Poly1305(raw_key)

    if algorithm is EncryptionAlgorithm.FERNET:
        from cryptography.fernet import Fernet

        return Fernet(base64.urlsafe_b64encode(raw_key))

    raise EncryptionError(f"Unsupported algorithm: {algorithm!r}")


# ---------------------------------------------------------------------------
# Key manager
# ---------------------------------------------------------------------------


class KeyManager:
    """Thread-safe lifecycle manager for encryption keys.

    Holds keys in memory keyed by ``key_id`` and optionally persists them
    to a JSON file. Supports creation (random or passphrase-derived),
    rotation, revocation and lookup by id or active status.

    Example::

        manager = KeyManager()
        key = manager.create_key(EncryptionAlgorithm.AES_256_GCM)
        engine = EncryptionEngine(manager)
        payload = engine.encrypt_string("secret")
        assert engine.decrypt_string(payload) == "secret"
        new_key = manager.rotate_key(key.key_id)
    """

    def __init__(self, persistence_path: Path | str | None = None) -> None:
        self._keys: dict[str, EncryptionKey] = {}
        self._lock = threading.RLock()
        self._persistence_path = Path(persistence_path) if persistence_path else None
        if self._persistence_path is not None and self._persistence_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Key creation
    # ------------------------------------------------------------------

    def create_key(
        self,
        algorithm: EncryptionAlgorithm,
        *,
        password: str | None = None,
        derivation_method: KeyDerivationMethod = KeyDerivationMethod.PBKDF2,
        expires_in: float | None = None,
        classification: DataClassification | None = None,
    ) -> EncryptionKey:
        """Create and register a new encryption key.

        When *password* is provided the key material is derived from it
        using *derivation_method*; otherwise a cryptographically random
        key is generated.

        Args:
            algorithm: The cipher algorithm for this key.
            password: Optional passphrase to derive the key from.
            derivation_method: KDF to use when *password* is given.
            expires_in: Key lifetime in seconds from now (``None`` = no
                expiry).
            classification: Optional data classification this key
                protects.

        Returns:
            The newly created :class:`EncryptionKey`.
        """

        salt = os.urandom(16)
        if password is not None:
            raw_key = _derive_key(password, salt, derivation_method)
        else:
            raw_key = _generate_random_key()
            # Random keys are not "derived" but we keep derivation_method
            # for metadata consistency.

        created_at = time.time()
        key = EncryptionKey(
            algorithm=algorithm,
            created_at=created_at,
            expires_at=(created_at + expires_in) if expires_in is not None else None,
            material=base64.b64encode(raw_key).decode("ascii"),
            derivation_method=derivation_method,
            salt=base64.b64encode(salt).decode("ascii"),
            classification=classification,
        )

        with self._lock:
            self._keys[key.key_id] = key
            self._persist()
        logger.info(
            "Created key %s (%s, derived=%s)",
            key.key_id,
            algorithm.value,
            password is not None,
        )
        return key

    # ------------------------------------------------------------------
    # Key lifecycle
    # ------------------------------------------------------------------

    def rotate_key(
        self,
        key_id: str,
        *,
        password: str | None = None,
    ) -> EncryptionKey:
        """Create a successor key and revoke the original.

        The new key inherits the algorithm, derivation method and
        classification of the old key. The old key is marked revoked so
        it can still decrypt existing payloads but will not be selected
        for new encryption.

        Args:
            key_id: The key to rotate.
            password: Optional passphrase for the new derived key. If
                ``None`` a random key is generated.

        Returns:
            The new :class:`EncryptionKey`.

        Raises:
            KeyManagementError: If *key_id* is unknown.
        """

        with self._lock:
            old = self._keys.get(key_id)
            if old is None:
                raise KeyManagementError(f"Key not found: {key_id}")

            new_key = self.create_key(
                old.algorithm,
                password=password,
                derivation_method=old.derivation_method,
                classification=old.classification,
            )
            new_key.rotated_from = old.key_id
            old.revoked = True
            self._persist()
        logger.info(
            "Rotated key %s -> %s (algorithm=%s)",
            old.key_id,
            new_key.key_id,
            old.algorithm.value,
        )
        return new_key

    def revoke_key(self, key_id: str) -> EncryptionKey:
        """Mark a key as revoked. It can no longer be used for encryption.

        Raises:
            KeyManagementError: If *key_id* is unknown.
        """

        with self._lock:
            key = self._keys.get(key_id)
            if key is None:
                raise KeyManagementError(f"Key not found: {key_id}")
            key.revoked = True
            self._persist()
        logger.warning("Revoked key %s", key_id)
        return key

    def delete_key(self, key_id: str) -> EncryptionKey | None:
        """Permanently remove a key from the manager.

        .. warning::

            Removing a key makes all payloads encrypted with it
            unrecoverable. Prefer :meth:`revoke_key` in production.

        Returns:
            The removed key, or ``None`` if not found.
        """

        with self._lock:
            key = self._keys.pop(key_id, None)
            if key is not None:
                self._persist()
                logger.warning("Deleted key %s", key_id)
            return key

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_key(self, key_id: str) -> EncryptionKey | None:
        """Return a key by id, or ``None``."""

        with self._lock:
            return self._keys.get(key_id)

    def list_keys(
        self,
        *,
        include_revoked: bool = True,
        include_expired: bool = True,
    ) -> list[EncryptionKey]:
        """Return all keys, optionally filtering revoked and expired ones."""

        now = time.time()
        with self._lock:
            keys = list(self._keys.values())
        result: list[EncryptionKey] = []
        for key in keys:
            if not include_revoked and key.revoked:
                continue
            if not include_expired and key.is_expired(now):
                continue
            result.append(key)
        return result

    def get_active_key(
        self,
        algorithm: EncryptionAlgorithm,
    ) -> EncryptionKey | None:
        """Return the most recently created usable key for *algorithm*.

        A usable key is neither revoked nor expired. Returns ``None``
        when no such key exists.
        """

        now = time.time()
        with self._lock:
            candidates = [
                k for k in self._keys.values() if k.algorithm is algorithm and k.is_usable(now)
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda k: k.created_at, reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path | None:
        """Persist all keys to a JSON file.

        Args:
            path: Destination path. Defaults to the manager's configured
                persistence path.

        Returns:
            The path written, or ``None`` if no path is configured.
        """

        target = Path(path) if path is not None else self._persistence_path
        if target is None:
            return None
        with self._lock:
            data = [k.model_dump(mode="json") for k in self._keys.values()]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Persisted %d key(s) to %s", len(data), target)
        return target

    def load(self, path: Path | str | None = None) -> int:
        """Load keys from a JSON file, replacing the in-memory store.

        Returns the number of keys loaded.
        """

        source = Path(path) if path is not None else self._persistence_path
        if source is None or not source.exists():
            return 0
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load keys from %s: %s", source, exc)
            return 0
        loaded = 0
        with self._lock:
            self._keys.clear()
            for item in data:
                try:
                    key = EncryptionKey.model_validate(item)
                except Exception as exc:  # noqa: BLE001 - best-effort import
                    logger.warning("Skipping invalid key entry: %s", exc)
                    continue
                self._keys[key.key_id] = key
                loaded += 1
        logger.info("Loaded %d key(s) from %s", loaded, source)
        return loaded

    def _persist(self) -> None:
        """Persist to the configured path (caller must hold the lock)."""

        if self._persistence_path is None:
            return
        data = [k.model_dump(mode="json") for k in self._keys.values()]
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self._persistence_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    @property
    def key_count(self) -> int:
        """Total number of keys (including revoked and expired)."""

        with self._lock:
            return len(self._keys)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the key store."""

        now = time.time()
        with self._lock:
            keys = list(self._keys.values())
        by_algo: dict[str, int] = {}
        active = 0
        revoked = 0
        expired = 0
        for key in keys:
            by_algo[key.algorithm.value] = by_algo.get(key.algorithm.value, 0) + 1
            if key.revoked:
                revoked += 1
            elif key.is_expired(now):
                expired += 1
            else:
                active += 1
        return {
            "total": len(keys),
            "active": active,
            "revoked": revoked,
            "expired": expired,
            "by_algorithm": by_algo,
        }


# ---------------------------------------------------------------------------
# Encryption engine
# ---------------------------------------------------------------------------


class EncryptionEngine:
    """Thread-safe encrypt/decrypt engine backed by a :class:`KeyManager`.

    Supports authenticated encryption with associated data (AEAD) for
    AES-256-GCM and ChaCha20-Poly1305. Fernet is supported as a simpler
    alternative without AEAD (associated data is stored but not
    cryptographically bound to the ciphertext).

    Example::

        manager = KeyManager()
        manager.create_key(EncryptionAlgorithm.AES_256_GCM)
        engine = EncryptionEngine(manager)

        payload = engine.encrypt_string(
            "top secret",
            associated_data=b"tenant=acme",
        )
        plaintext = engine.decrypt_string(payload)
    """

    def __init__(self, key_manager: KeyManager) -> None:
        self._key_manager = key_manager
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Encrypt
    # ------------------------------------------------------------------

    def encrypt(
        self,
        data: bytes,
        *,
        algorithm: EncryptionAlgorithm | None = None,
        associated_data: bytes = b"",
        key_id: str | None = None,
        classification: DataClassification | None = None,
    ) -> EncryptedPayload:
        """Encrypt *data* (bytes) and return an :class:`EncryptedPayload`.

        Args:
            data: Plaintext bytes to encrypt.
            algorithm: Preferred algorithm. If ``None`` the engine
                auto-selects AES-256-GCM. Ignored when *key_id* is given.
            associated_data: Optional AEAD associated data (authenticated
                but not encrypted).
            key_id: Use a specific key by id. If ``None`` the most
                recent active key for the (auto-)selected algorithm is
                used.
            classification: Optional data classification recorded on the
                payload.

        Raises:
            EncryptionError: If no usable key is available or encryption
                fails.
        """

        key = self._resolve_key(algorithm, key_id)
        cipher = _get_aead_cipher(key.algorithm, key.raw_material)

        try:
            if key.algorithm is EncryptionAlgorithm.FERNET:
                token = cipher.encrypt(data)
                nonce = b""
                ciphertext = token
                tag = b""
                aad = b""
            else:
                nonce = os.urandom(_NONCE_LENGTH_BYTES)
                combined = cipher.encrypt(nonce, data, associated_data or None)
                ciphertext = combined[:-_TAG_LENGTH_BYTES]
                tag = combined[-_TAG_LENGTH_BYTES:]
                aad = associated_data
        except Exception as exc:  # noqa: BLE001 - surface crypto errors uniformly
            raise EncryptionError(f"Encryption failed: {exc}") from exc

        payload = EncryptedPayload(
            key_id=key.key_id,
            algorithm=key.algorithm,
            nonce=nonce,
            ciphertext=ciphertext,
            tag=tag,
            associated_data=aad,
            classification=classification,
        )
        logger.debug(
            "Encrypted %d byte(s) with %s (key=%s)",
            len(data),
            key.algorithm.value,
            key.key_id,
        )
        return payload

    def encrypt_string(
        self,
        text: str,
        *,
        algorithm: EncryptionAlgorithm | None = None,
        associated_data: bytes = b"",
        key_id: str | None = None,
        classification: DataClassification | None = None,
    ) -> EncryptedPayload:
        """Encrypt a string (UTF-8 encoded). See :meth:`encrypt`."""

        return self.encrypt(
            text.encode("utf-8"),
            algorithm=algorithm,
            associated_data=associated_data,
            key_id=key_id,
            classification=classification,
        )

    # ------------------------------------------------------------------
    # Decrypt
    # ------------------------------------------------------------------

    def decrypt(self, payload: EncryptedPayload) -> bytes:
        """Decrypt an :class:`EncryptedPayload` back to plaintext bytes.

        Raises:
            EncryptionError: If the key is missing, the payload is
                tampered with, or the associated data does not match.
        """

        key = self._key_manager.get_key(payload.key_id)
        if key is None:
            raise EncryptionError(f"Key not found: {payload.key_id}")

        cipher = _get_aead_cipher(payload.algorithm, key.raw_material)

        try:
            if payload.algorithm is EncryptionAlgorithm.FERNET:
                plaintext = cipher.decrypt(payload.ciphertext)
            else:
                combined = payload.ciphertext + payload.tag
                aad = payload.associated_data or None
                plaintext = cipher.decrypt(payload.nonce, combined, aad)
        except Exception as exc:  # noqa: BLE001 - surface crypto errors uniformly
            raise EncryptionError(f"Decryption failed (key={payload.key_id}): {exc}") from exc

        logger.debug(
            "Decrypted %d byte(s) with %s (key=%s)",
            len(plaintext),
            payload.algorithm.value,
            payload.key_id,
        )
        return plaintext

    def decrypt_string(self, payload: EncryptedPayload) -> str:
        """Decrypt an :class:`EncryptedPayload` and decode as UTF-8."""

        return self.decrypt(payload).decode("utf-8")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_key(
        self,
        algorithm: EncryptionAlgorithm | None,
        key_id: str | None,
    ) -> EncryptionKey:
        """Resolve the key to use for encryption (caller holds lock)."""

        with self._lock:
            if key_id is not None:
                key = self._key_manager.get_key(key_id)
                if key is None:
                    raise EncryptionError(f"Key not found: {key_id}")
                if not key.is_usable():
                    raise EncryptionError(f"Key is not usable: {key_id}")
                return key

            # If a specific algorithm is requested, look for an active key.
            if algorithm is not None:
                key = self._key_manager.get_active_key(algorithm)
                if key is not None:
                    return key
                raise EncryptionError(
                    f"No usable key available for {algorithm.value}; "
                    "create a key with KeyManager.create_key() first"
                )

            # No algorithm specified — try the preferred order, then any key.
            for algo in (
                EncryptionAlgorithm.AES_256_GCM,
                EncryptionAlgorithm.CHACHA20_POLY1305,
                EncryptionAlgorithm.FERNET,
            ):
                key = self._key_manager.get_active_key(algo)
                if key is not None:
                    return key

            # Last resort: any active key of any algorithm.
            all_keys = self._key_manager.list_keys(include_revoked=False, include_expired=False)
            if all_keys:
                all_keys.sort(key=lambda k: k.created_at, reverse=True)
                return all_keys[0]

            raise EncryptionError(
                "No usable key available; "
                "create a key with KeyManager.create_key() first"
            )

    @property
    def key_manager(self) -> KeyManager:
        """The underlying :class:`KeyManager`."""

        return self._key_manager


__all__ = [
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
]
