"""Unified sensitive-data redaction for JustAgent.

This module provides the single source of truth for:
- Sensitive key names that must always be masked.
- Secret-like value patterns (API tokens, private keys, JWTs, etc.).
- Helpers to redact free-form text, dictionary values, and nested structures.

All callers (audit logger, telemetry, config display) import from here so that
redaction behaviour never diverges between subsystems.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

# ── Sensitive key names ──────────────────────────────────────────────
# Union of all previously separate key sets (audit_logger, telemetry, config).
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "token",
        "api_key",
        "apikey",
        "api-key",
        "password",
        "passwd",
        "pwd",
        "secret",
        "siem_token",
        "key",
        "private",
        "private_key",
        "privatekey",
        "credentials",
        "auth",
        "authorization",
        "access_token",
        "refresh_token",
        "cookie",
        "session",
        "email",
        "phone",
        "cx",
        "public_key",
        "base_url",
    }
)

# ── Secret-like value patterns ───────────────────────────────────────
# Union of all previously separate pattern sets.
#
# Unambiguous secret formats (token prefixes, PEM blocks, JWTs, emails) are
# kept separate from the bare-hex pattern so that hash/identifier fields can
# opt out of hex redaction while still masking known secret formats.
_SECRET_FORMAT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # GitHub personal access token (classic)
    re.compile(r"ghp_[A-Za-z0-9_]{36}"),
    # GitHub fine-grained personal access token
    re.compile(r"github_pat_[A-Za-z0-9_]{22}_[A-Za-z0-9_]{59}"),
    # OpenAI API key
    re.compile(r"sk-[a-zA-Z0-9]{48}"),
    # AWS access key id
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # PEM/SSH private key block
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA |PGP )?PRIVATE KEY-----"),
    # JWT (header.payload.signature)
    re.compile(r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*"),
    # Generic JWT-like dotted tokens
    re.compile(r"[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}"),
    # Email addresses
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
)

# Bare hexadecimal runs of 32+ chars. These are ambiguous: they may be a
# secret token OR a hash/UUID. Standard hash/UUID lengths — 32 (MD5 / UUID
# without dashes), 40 (git SHA-1), 64 (SHA-256) — are excluded so that
# digests and commit SHAs remain visible for observability. The lookbehind
# and lookahead anchors ensure the exclusion is evaluated against the full
# maximal hex run rather than a suffix of it (a naive ``(?![a-f0-9]{64}$)``
# would still match 63 chars starting at offset 1 of a 64-char hash).
_BARE_HEX_PATTERN = re.compile(
    r"(?i)"
    r"(?<![a-f0-9])"
    r"(?![a-f0-9]{32}(?![a-f0-9]))"
    r"(?![a-f0-9]{40}(?![a-f0-9]))"
    r"(?![a-f0-9]{64}(?![a-f0-9]))"
    r"[a-f0-9]{32,}"
    r"(?![a-f0-9])"
)

# Public tuple (backward-compatible name) consumed by telemetry and other
# modules: all secret-format patterns plus the bare-hex pattern.
SENSITIVE_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    *_SECRET_FORMAT_PATTERNS,
    _BARE_HEX_PATTERN,
)

# Field-name tokens that denote a hash/identifier rather than a secret. When
# the enclosing field matches one of these tokens, bare hex values are left
# intact (known secret formats are still masked) so digests, commit SHAs and
# UUIDs stay observable.
_HASH_FIELD_TOKENS: frozenset[str] = frozenset(
    {
        "hash",
        "sha",
        "sha1",
        "sha256",
        "sha512",
        "commit",
        "commits",
        "digest",
        "checksum",
        "uuid",
        "oid",
        "fingerprint",
        "ref",
    }
)


def _is_hash_field(name: str) -> bool:
    """Return True if *name* denotes a field that carries a hash/identifier.

    Matching is token-based (split on ``_-. `` and a space) to avoid substring
    false positives such as ``preference`` matching ``ref``.
    """
    parts = re.split(r"[_\-. ]", name.lower())
    return any(part in _HASH_FIELD_TOKENS for part in parts if part)


def is_sensitive_key(key: str) -> bool:
    """Return True if *key* (case-insensitive) indicates a sensitive field.

    Uses exact match against the known set of sensitive key names, plus
    substring matching for compound keys (e.g. ``user_token``, ``api_key_id``)
    to avoid missing variants.
    """
    lower = key.lower()
    if lower in SENSITIVE_KEYS:
        return True
    # Check compound keys: split on common separators and check each part.
    parts = re.split(r"[_\-. ]", lower)
    return any(part in SENSITIVE_KEYS for part in parts if part)


# Identity fields that are intentionally retained verbatim even when their
# value happens to match a secret-like pattern (e.g. an SSO user identifier in
# email form). These keys are known to carry actor identity, not secrets, so
# masking them would destroy audit traceability.
IDENTITY_KEYS: frozenset[str] = frozenset(
    {
        "user",
        "sso_subject",
        "sso_provider",
        "role",
    }
)


def redact_text(text: str, *, include_hex: bool = True) -> str:
    """Redact a free-form string when it contains a secret-like pattern.

    Returns ``"***"`` if any known secret pattern matches, otherwise the
    original text unchanged. When *include_hex* is False, bare hexadecimal
    strings are not treated as secret-like — used for fields that are known
    to carry hashes/identifiers (e.g. ``sha256``, ``commit``) rather than
    tokens, so known secret formats are still masked but digests stay visible.
    """
    patterns: tuple[re.Pattern[str], ...] = (
        SENSITIVE_VALUE_PATTERNS if include_hex else _SECRET_FORMAT_PATTERNS
    )
    if any(pattern.search(text) for pattern in patterns):
        return "***"
    return text


def redact_paths(text: str, project_root: Path | None = None, *, redact_home: bool = True) -> str:
    """Mask absolute local paths that may leak from stdout/stderr.

    The project root absolute path prefix (when provided and present in
    ``text``) is replaced with ``.`` so that file references become relative.
    Afterwards the user's home directory prefix is replaced with ``~`` so that
    paths under the home directory (but outside the project) are not disclosed
    in full either.

    The function is intentionally conservative: it only rewrites leading path
    prefixes, never substrings in the middle of a token, so command output is
    still useful for debugging. It is a pure function and safe to call on any
    string.
    """
    result = text
    if project_root is not None:
        root_str = str(project_root.resolve())
        # Avoid replacing the empty string when ``project_root`` resolves to "".
        if root_str and root_str in result:
            result = result.replace(root_str, ".")
    if redact_home:
        home_str = str(Path.home())
        if home_str and home_str in result:
            result = result.replace(home_str, "~")
    return result


def redact_scalar(value: Any, *, include_hex: bool = True) -> Any:
    """Redact a scalar value if it contains a secret-like pattern."""
    if isinstance(value, str):
        return redact_text(value, include_hex=include_hex)
    return value


def redact_dict(
    data: dict[str, Any],
    *,
    mask: str = "***",
    redact_unknown: bool = False,
    safe_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Recursively redact sensitive keys and secret-like values in *data*.

    Parameters
    ----------
    data:
        The dictionary to redact.
    mask:
        The string to replace sensitive values with.
    redact_unknown:
        When True, any key not in *safe_keys* is treated as sensitive.
    safe_keys:
        Keys considered safe when *redact_unknown* is True.
    """
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()
        # Hash/identifier fields keep bare hex values visible; known secret
        # formats are still masked via include_hex=False.
        include_hex = not _is_hash_field(key_lower)
        if is_sensitive_key(key_lower):
            redacted[key] = mask
        elif redact_unknown and safe_keys is not None and key_lower not in safe_keys:
            redacted[key] = _redact_unknown_value(value, mask=mask)
        elif key_lower in IDENTITY_KEYS:
            # Identity fields (user, sso_subject, sso_provider, role) are
            # retained verbatim even when the value matches a secret-like
            # pattern (e.g. an email-shaped SSO user). Masking them would
            # destroy audit traceability without any security benefit, since
            # these keys are explicitly bound by the audit logger.
            redacted[key] = value
        elif isinstance(value, dict):
            redacted[key] = redact_dict(
                cast(dict[str, Any], value),
                mask=mask,
                redact_unknown=redact_unknown,
                safe_keys=safe_keys,
            )
        elif isinstance(value, list):
            redacted[key] = [
                redact_dict(
                    cast(dict[str, Any], item),
                    mask=mask,
                    redact_unknown=redact_unknown,
                    safe_keys=safe_keys,
                )
                if isinstance(item, dict)
                else redact_scalar(item, include_hex=include_hex)
                for item in cast(list[Any], value)
            ]
        else:
            redacted[key] = redact_scalar(value, include_hex=include_hex)
    return redacted


def _redact_unknown_value(value: Any, *, mask: str = "***") -> Any:
    """Redact an unknown value recursively."""
    if isinstance(value, dict):
        return {
            k: _redact_unknown_value(v, mask=mask) for k, v in cast(dict[str, Any], value).items()
        }
    if isinstance(value, list):
        return [_redact_unknown_value(item, mask=mask) for item in cast(list[Any], value)]
    return mask
