"""Tests for the enterprise encryption module."""

from __future__ import annotations

import time

import pytest

from justagent.security.encryption import (
    DataClassification,
    EncryptionAlgorithm,
    EncryptionEngine,
    EncryptionError,
    EncryptionKey,
    EncryptedPayload,
    KeyDerivationMethod,
    KeyManagementError,
    KeyManager,
    SecurityError,
)


# ---------------------------------------------------------------------------
# KeyManager
# ---------------------------------------------------------------------------


class TestKeyManager:
    def test_create_key_aes256gcm(self) -> None:
        km = KeyManager()
        key = km.create_key(EncryptionAlgorithm.AES_256_GCM)
        assert key.algorithm is EncryptionAlgorithm.AES_256_GCM
        assert key.key_id
        assert key.material
        assert not key.revoked
        assert key.is_usable()

    def test_create_key_chacha20(self) -> None:
        km = KeyManager()
        key = km.create_key(EncryptionAlgorithm.CHACHA20_POLY1305)
        assert key.algorithm is EncryptionAlgorithm.CHACHA20_POLY1305

    def test_create_key_fernet(self) -> None:
        km = KeyManager()
        key = km.create_key(EncryptionAlgorithm.FERNET)
        assert key.algorithm is EncryptionAlgorithm.FERNET

    def test_create_key_with_password(self) -> None:
        km = KeyManager()
        key = km.create_key(
            EncryptionAlgorithm.AES_256_GCM,
            password="my-passphrase",
            derivation_method=KeyDerivationMethod.PBKDF2,
        )
        assert key.derivation_method is KeyDerivationMethod.PBKDF2
        assert key.salt

    def test_create_key_with_expiry(self) -> None:
        km = KeyManager()
        key = km.create_key(EncryptionAlgorithm.AES_256_GCM, expires_in=3600)
        assert key.expires_at is not None
        assert key.expires_at > time.time()
        assert not key.is_expired()
        assert key.is_usable()

    def test_create_key_with_classification(self) -> None:
        km = KeyManager()
        key = km.create_key(
            EncryptionAlgorithm.AES_256_GCM,
            classification=DataClassification.RESTRICTED,
        )
        assert key.classification is DataClassification.RESTRICTED

    def test_get_key(self) -> None:
        km = KeyManager()
        key = km.create_key(EncryptionAlgorithm.AES_256_GCM)
        retrieved = km.get_key(key.key_id)
        assert retrieved is not None
        assert retrieved.key_id == key.key_id

    def test_get_key_unknown_returns_none(self) -> None:
        km = KeyManager()
        assert km.get_key("nonexistent") is None

    def test_list_keys(self) -> None:
        km = KeyManager()
        km.create_key(EncryptionAlgorithm.AES_256_GCM)
        km.create_key(EncryptionAlgorithm.FERNET)
        keys = km.list_keys()
        assert len(keys) == 2

    def test_list_keys_exclude_revoked(self) -> None:
        km = KeyManager()
        k1 = km.create_key(EncryptionAlgorithm.AES_256_GCM)
        km.create_key(EncryptionAlgorithm.FERNET)
        km.revoke_key(k1.key_id)
        active = km.list_keys(include_revoked=False)
        assert len(active) == 1
        assert active[0].algorithm is EncryptionAlgorithm.FERNET

    def test_revoke_key(self) -> None:
        km = KeyManager()
        key = km.create_key(EncryptionAlgorithm.AES_256_GCM)
        revoked = km.revoke_key(key.key_id)
        assert revoked.revoked is True
        assert not revoked.is_usable()

    def test_revoke_key_unknown_raises(self) -> None:
        km = KeyManager()
        with pytest.raises(KeyManagementError, match="Key not found"):
            km.revoke_key("nonexistent")

    def test_rotate_key(self) -> None:
        km = KeyManager()
        old_key = km.create_key(EncryptionAlgorithm.AES_256_GCM)
        new_key = km.rotate_key(old_key.key_id)
        assert new_key.key_id != old_key.key_id
        assert new_key.rotated_from == old_key.key_id
        assert new_key.algorithm == old_key.algorithm
        old_refreshed = km.get_key(old_key.key_id)
        assert old_refreshed is not None
        assert old_refreshed.revoked is True

    def test_rotate_key_unknown_raises(self) -> None:
        km = KeyManager()
        with pytest.raises(KeyManagementError, match="Key not found"):
            km.rotate_key("nonexistent")

    def test_delete_key(self) -> None:
        km = KeyManager()
        key = km.create_key(EncryptionAlgorithm.AES_256_GCM)
        deleted = km.delete_key(key.key_id)
        assert deleted is not None
        assert km.get_key(key.key_id) is None

    def test_delete_key_unknown_returns_none(self) -> None:
        km = KeyManager()
        assert km.delete_key("nonexistent") is None

    def test_persistence_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "keys.json"
        km1 = KeyManager(persistence_path=path)
        key = km1.create_key(EncryptionAlgorithm.AES_256_GCM)
        assert path.exists()

        km2 = KeyManager(persistence_path=path)
        loaded = km2.get_key(key.key_id)
        assert loaded is not None
        assert loaded.material == key.material


# ---------------------------------------------------------------------------
# EncryptionEngine
# ---------------------------------------------------------------------------


class TestEncryptionEngine:
    def test_encrypt_decrypt_roundtrip_aes256gcm(self) -> None:
        km = KeyManager()
        km.create_key(EncryptionAlgorithm.AES_256_GCM)
        engine = EncryptionEngine(km)
        payload = engine.encrypt_string("hello world")
        assert engine.decrypt_string(payload) == "hello world"

    def test_encrypt_decrypt_roundtrip_chacha20(self) -> None:
        km = KeyManager()
        km.create_key(EncryptionAlgorithm.CHACHA20_POLY1305)
        engine = EncryptionEngine(km)
        payload = engine.encrypt_string("hello chacha")
        assert engine.decrypt_string(payload) == "hello chacha"

    def test_encrypt_decrypt_roundtrip_fernet(self) -> None:
        km = KeyManager()
        km.create_key(EncryptionAlgorithm.FERNET)
        engine = EncryptionEngine(km)
        payload = engine.encrypt_string("hello fernet")
        assert engine.decrypt_string(payload) == "hello fernet"

    def test_encrypt_decrypt_bytes(self) -> None:
        km = KeyManager()
        km.create_key(EncryptionAlgorithm.AES_256_GCM)
        engine = EncryptionEngine(km)
        payload = engine.encrypt(b"binary data")
        assert engine.decrypt(payload) == b"binary data"

    def test_encrypt_with_associated_data(self) -> None:
        km = KeyManager()
        km.create_key(EncryptionAlgorithm.AES_256_GCM)
        engine = EncryptionEngine(km)
        aad = b"tenant=acme"
        payload = engine.encrypt_string("secret", associated_data=aad)
        assert engine.decrypt_string(payload) == "secret"

    def test_encrypt_with_wrong_aad_fails(self) -> None:
        km = KeyManager()
        km.create_key(EncryptionAlgorithm.AES_256_GCM)
        engine = EncryptionEngine(km)
        payload = engine.encrypt_string("secret", associated_data=b"correct_aad")
        payload.associated_data = b"wrong_aad"
        with pytest.raises(EncryptionError, match="Decryption failed"):
            engine.decrypt_string(payload)

    def test_tampered_ciphertext_fails(self) -> None:
        km = KeyManager()
        km.create_key(EncryptionAlgorithm.AES_256_GCM)
        engine = EncryptionEngine(km)
        payload = engine.encrypt_string("secret")
        # Tamper with ciphertext.
        payload.ciphertext = b"\x00" * len(payload.ciphertext)
        with pytest.raises(EncryptionError, match="Decryption failed"):
            engine.decrypt_string(payload)

    def test_decrypt_with_missing_key_fails(self) -> None:
        km1 = KeyManager()
        km1.create_key(EncryptionAlgorithm.AES_256_GCM)
        engine1 = EncryptionEngine(km1)
        payload = engine1.encrypt_string("secret")

        km2 = KeyManager()
        engine2 = EncryptionEngine(km2)
        with pytest.raises(EncryptionError, match="Key not found"):
            engine2.decrypt_string(payload)

    def test_encrypt_no_keys_raises(self) -> None:
        km = KeyManager()
        engine = EncryptionEngine(km)
        with pytest.raises(EncryptionError):
            engine.encrypt_string("secret")

    def test_key_rotation_old_data_still_decryptable(self) -> None:
        km = KeyManager()
        old_key = km.create_key(EncryptionAlgorithm.AES_256_GCM)
        engine = EncryptionEngine(km)
        payload = engine.encrypt_string("old data")

        # Rotate the key.
        km.rotate_key(old_key.key_id)

        # Old data should still be decryptable.
        assert engine.decrypt_string(payload) == "old data"

    def test_encrypted_payload_to_dict_from_dict_roundtrip(self) -> None:
        km = KeyManager()
        km.create_key(EncryptionAlgorithm.AES_256_GCM)
        engine = EncryptionEngine(km)
        payload = engine.encrypt_string("test data")
        d = payload.to_dict()
        restored = EncryptedPayload.from_dict(d)
        assert engine.decrypt_string(restored) == "test data"

    def test_encryption_key_is_usable(self) -> None:
        key = EncryptionKey(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            material="dGVzdGtleW1hdGVyaWFsdGVzdGtleW1hdGVyaWFs",
        )
        assert key.is_usable()
        assert not key.is_expired()

    def test_encryption_key_expired(self) -> None:
        key = EncryptionKey(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            material="dGVzdGtleW1hdGVyaWFsdGVzdGtleW1hdGVyaWFs",
            expires_at=time.time() - 100,
        )
        assert key.is_expired()
        assert not key.is_usable()

    def test_security_error_hierarchy(self) -> None:
        assert issubclass(EncryptionError, SecurityError)
        assert issubclass(KeyManagementError, SecurityError)
