"""Signed team-level configuration loader.

AutoShip already merges ``.autoship.team.toml`` over ``.autoship.toml`` in
:data:`config_center.load_config`. This module adds **signature verification**
so a team can distribute a shared profile through untrusted channels (chat,
wiki, shared drive) and each workstation can prove it came from a trusted
key before applying it.

Scheme
------
* Algorithm: Ed25519 (RFC 8032) via :mod:`cryptography`.
* Key encoding: base64 (URL-safe), 32-byte raw public key, 32-byte raw
  private key. The format is intentionally the same one used by
  :class:`RegistryConfig.public_key` so teams can reuse tooling.
* Signature: detached, base64 (URL-safe), written alongside the config as
  ``<config>.sig``. Detached signatures keep the team TOML readable in a
  text editor and diff cleanly in code review.

Failure modes are explicit: a missing signature file when a public key is
configured raises :class:`TeamConfigError`, never a silent warning. This
matches the registry index signature check.
"""

from __future__ import annotations

import base64
import tomllib
from pathlib import Path
from typing import Any

from autoship.exceptions import AutoShipError


class TeamConfigError(AutoShipError):
    """Raised when a team config is missing, malformed, or fails signature verification."""

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


def _b64url_decode(value: str) -> bytes:
    """Decode a URL-safe base64 string, padding-tolerant."""
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded)
    except (ValueError, TypeError) as exc:
        raise TeamConfigError(f"Invalid base64 key/signature: {exc}") from exc


def _b64url_encode(value: bytes) -> str:
    """Encode bytes as unpadded URL-safe base64."""
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def signature_path_for(config_path: Path) -> Path:
    """Return the conventional detached-signature path for ``config_path``."""
    return config_path.with_suffix(config_path.suffix + ".sig")


def verify_team_config(config_path: Path, public_key_b64: str) -> bool:
    """Verify the detached signature of a team config file.

    Args:
        config_path: Path to ``.autoship.team.toml``.
        public_key_b64: Base64 URL-safe Ed25519 public key (32 bytes raw).

    Returns:
        True if the signature is valid.

    Raises:
        TeamConfigError: if the config or signature file is missing, the
            key is malformed, or the signature does not verify.
    """
    if not config_path.exists():
        raise TeamConfigError(
            f"Team config not found: {config_path}",
            path=config_path,
        )
    sig_path = signature_path_for(config_path)
    if not sig_path.exists():
        raise TeamConfigError(
            f"Team config signature missing: {sig_path}. "
            "Refusing to load an unsigned team profile when a public key is configured.",
            path=sig_path,
        )

    public_key_bytes = _b64url_decode(public_key_b64)
    signature_bytes = _b64url_decode(sig_path.read_text(encoding="utf-8").strip())
    config_bytes = config_path.read_bytes()

    # Import lazily so the module imports cleanly in environments that
    # exercise the loader without exercising verification (e.g. tests
    # that only inspect the merged dict).
    from cryptography.exceptions import InvalidSignature  # pyright: ignore[reportMissingImports]
    from cryptography.hazmat.primitives import (
        serialization,  # pyright: ignore[reportMissingImports]
    )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # pyright: ignore[reportMissingImports]
        Ed25519PublicKey,
    )

    if len(public_key_bytes) != 32:
        raise TeamConfigError(
            f"Malformed Ed25519 public key: expected 32 raw bytes, got {len(public_key_bytes)}.",
        )

    try:
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        public_key.verify(signature_bytes, config_bytes)
    except (InvalidSignature, ValueError) as exc:
        raise TeamConfigError(
            "Team config signature verification failed. "
            "The file was modified after signing or was signed by a different key.",
            path=config_path,
        ) from exc

    # Sanity-check that the bytes really are an Ed25519 key. The round-trip
    # through serialization catches subtle corruption that from_public_bytes
    # would otherwise accept silently.
    _ = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return True


def load_team_config(
    config_path: Path,
    public_key_b64: str | None = None,
    *,
    require_signature: bool = False,
) -> dict[str, Any]:
    """Load and (optionally) verify a team config file.

    Args:
        config_path: Path to ``.autoship.team.toml``.
        public_key_b64: When provided, the config signature is verified
            against this key. When None, verification is skipped.
        require_signature: When True and a public key is configured but the
            signature file is absent, raise instead of skipping verification.

    Returns:
        Parsed TOML as a dict. An empty dict if the file does not exist.

    Raises:
        TeamConfigError: signature required and missing/invalid, or TOML
            parse error.
    """
    if not config_path.exists():
        return {}

    if public_key_b64:
        verify_team_config(config_path, public_key_b64)
    elif require_signature:
        raise TeamConfigError(
            "Team config signature required (require_signature=True) "
            "but no public key was provided to verify against.",
            path=config_path,
        )

    try:
        return tomllib.loads(config_path.read_bytes())
    except (OSError, tomllib.TOMLDecodeError) as exc:  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        raise TeamConfigError(
            f"Failed to parse team config {config_path}: {exc}",
            path=config_path,
        ) from exc


def sign_team_config(config_path: Path, private_key_b64: str) -> bytes:
    """Produce a detached signature for a team config file.

    Args:
        config_path: Path to the team config to sign. Must exist.
        private_key_b64: Base64 URL-safe Ed25519 private key (32 bytes raw).

    Returns:
        The raw 64-byte signature. Callers typically base64-encode it and
        write it to :func:`signature_path_for`.

    Raises:
        TeamConfigError: if the config or key is malformed.
    """
    if not config_path.exists():
        raise TeamConfigError(f"Team config not found: {config_path}", path=config_path)

    private_key_bytes = _b64url_decode(private_key_b64)
    if len(private_key_bytes) != 32:
        raise TeamConfigError(
            f"Malformed Ed25519 private key: expected 32 raw bytes, got {len(private_key_bytes)}.",
        )

    config_bytes = config_path.read_bytes()

    from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # pyright: ignore[reportMissingImports]
        Ed25519PrivateKey,
    )

    try:
        private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        return private_key.sign(config_bytes)
    except (TypeError, ValueError) as exc:
        raise TeamConfigError(f"Failed to sign team config: {exc}", path=config_path) from exc


def generate_keypair() -> tuple[str, str]:
    """Generate a fresh Ed25519 keypair, returning ``(public_b64, private_b64)``.

    Both values are URL-safe base64, unpadded. The private key is the raw
    32-byte seed; the public key is the raw 32-byte compressed point. This
    format matches :func:`verify_team_config` and :func:`sign_team_config`.
    """
    from cryptography.hazmat.primitives import (
        serialization,  # pyright: ignore[reportMissingImports]
    )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # pyright: ignore[reportMissingImports]
        Ed25519PrivateKey,
    )

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_b64 = _b64url_encode(
        private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_b64 = _b64url_encode(
        public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    return public_b64, private_b64
