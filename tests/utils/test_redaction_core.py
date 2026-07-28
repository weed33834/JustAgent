"""Unit tests for the core redaction primitives in ``justagent.utils.redaction``.

These tests focus on the lower-level helpers (``redact_text``, ``redact_scalar``,
``redact_dict``, ``is_sensitive_key``, ``_is_hash_field``, ``_BARE_HEX_PATTERN``)
that the higher-level ``redact_paths`` tests in :mod:`tests.utils.test_redaction`
do not cover directly.
"""

from __future__ import annotations

from justagent.utils.redaction import (
    _BARE_HEX_PATTERN,
    _is_hash_field,
    is_sensitive_key,
    redact_dict,
    redact_scalar,
    redact_text,
)

# ============================================================
# is_sensitive_key
# ============================================================


def test_is_sensitive_key_exact_match() -> None:
    assert is_sensitive_key("api_key") is True
    assert is_sensitive_key("API_KEY") is True
    assert is_sensitive_key("password") is True


def test_is_sensitive_key_compound_match() -> None:
    """Compound keys split on separators are still detected."""
    assert is_sensitive_key("user_token") is True
    assert is_sensitive_key("api-key-id") is True
    assert is_sensitive_key("auth.header") is True
    assert is_sensitive_key("db password") is True


def test_is_sensitive_key_non_sensitive() -> None:
    assert is_sensitive_key("username") is False
    assert is_sensitive_key("command") is False
    assert is_sensitive_key("duration_ms") is False


def test_is_sensitive_key_avoids_substring_false_positive() -> None:
    """``preference`` must not match ``ref`` because matching is token-based."""
    assert is_sensitive_key("preference") is False
    assert is_sensitive_key("reflog_entry") is False


# ============================================================
# _is_hash_field
# ============================================================


def test_is_hash_field_recognises_known_tokens() -> None:
    assert _is_hash_field("sha256") is True
    assert _is_hash_field("commit") is True
    assert _is_hash_field("package_digest") is True
    assert _is_hash_field("release.fingerprint") is True
    assert _is_hash_field("oid") is True


def test_is_hash_field_rejects_non_hash_names() -> None:
    assert _is_hash_field("username") is False
    assert _is_hash_field("command") is False
    assert _is_hash_field("token") is False


def test_is_hash_field_is_token_based_not_substring() -> None:
    """Substring matches must not produce false positives (``preference``)."""
    assert _is_hash_field("preference") is False
    assert _is_hash_field("refresh_token") is False


# ============================================================
# redact_text
# ============================================================


def test_redact_text_masks_github_token() -> None:
    text = "using token ghp_" + "a" * 36 + " for auth"
    assert redact_text(text) == "***"


def test_redact_text_masks_openai_key() -> None:
    text = "key=sk-" + "a" * 48
    assert redact_text(text) == "***"


def test_redact_text_masks_jwt() -> None:
    text = "Bearer eyJ" + "a" * 10 + ".eyJ" + "b" * 10 + "." + "c" * 10
    assert redact_text(text) == "***"


def test_redact_text_masks_email() -> None:
    assert redact_text("contact me at user@example.com") == "***"


def test_redact_text_passes_through_benign_text() -> None:
    assert redact_text("just a regular log message") == "just a regular log message"


def test_redact_text_include_hex_false_keeps_hashes_visible() -> None:
    """``include_hex=False`` is used for hash/identifier fields."""
    sha1 = "a" * 40
    assert redact_text(sha1, include_hex=False) == sha1


def test_redact_text_include_hex_true_masks_long_hex() -> None:
    """A bare hex run longer than 64 chars is masked by default."""
    long_hex = "a" * 80
    assert redact_text(long_hex) == "***"


# ============================================================
# _BARE_HEX_PATTERN
# ============================================================


def test_bare_hex_pattern_excludes_32_char_md5_uuid() -> None:
    """Standard 32-char (MD5/UUID-without-dashes) hashes are NOT matched."""
    md5 = "a" * 32
    assert _BARE_HEX_PATTERN.search(md5) is None


def test_bare_hex_pattern_excludes_40_char_sha1() -> None:
    sha1 = "a" * 40
    assert _BARE_HEX_PATTERN.search(sha1) is None


def test_bare_hex_pattern_excludes_64_char_sha256() -> None:
    sha256 = "a" * 64
    assert _BARE_HEX_PATTERN.search(sha256) is None


def test_bare_hex_pattern_matches_longer_runs() -> None:
    """Hex runs longer than 64 chars (not a standard hash) are matched."""
    long_hex = "a" * 80
    assert _BARE_HEX_PATTERN.search(long_hex) is not None


def test_bare_hex_pattern_is_case_insensitive() -> None:
    assert _BARE_HEX_PATTERN.search("F" * 80) is not None


# ============================================================
# redact_scalar
# ============================================================


def test_redact_scalar_redacts_string_with_token() -> None:
    assert redact_scalar("ghp_" + "a" * 36) == "***"


def test_redact_scalar_passes_through_benign_string() -> None:
    assert redact_scalar("hello world") == "hello world"


def test_redact_scalar_passes_through_non_string_types() -> None:
    """Non-string scalars are returned unchanged."""
    assert redact_scalar(42) == 42
    assert redact_scalar(3.14) == 3.14
    assert redact_scalar(True) is True
    assert redact_scalar(None) is None


# ============================================================
# redact_dict
# ============================================================


def test_redact_dict_masks_sensitive_keys() -> None:
    data = {"api_key": "ghp_" + "a" * 36, "command": "clean"}
    result = redact_dict(data)
    assert result["api_key"] == "***"
    assert result["command"] == "clean"


def test_redact_dict_recurses_into_nested_dicts() -> None:
    data = {"outer": {"inner_token": "secret-value", "ok": "fine"}}
    result = redact_dict(data)
    assert result["outer"]["inner_token"] == "***"
    assert result["outer"]["ok"] == "fine"


def test_redact_dict_recurses_into_lists_of_dicts() -> None:
    data = {"items": [{"token": "x", "name": "n"}, {"token": "y", "name": "m"}]}
    result = redact_dict(data)
    assert result["items"][0]["token"] == "***"
    assert result["items"][0]["name"] == "n"
    assert result["items"][1]["token"] == "***"
    assert result["items"][1]["name"] == "m"


def test_redact_dict_keeps_hash_field_hex_visible() -> None:
    """``sha256`` field keeps its bare-hex value (include_hex=False path)."""
    sha256 = "a" * 64
    data = {"sha256": sha256, "name": "package"}
    result = redact_dict(data)
    assert result["sha256"] == sha256
    assert result["name"] == "package"


def test_redact_dict_still_masks_known_secret_format_in_hash_field() -> None:
    """Even hash fields mask unambiguous secret formats (e.g. ghp_ tokens)."""
    token = "ghp_" + "a" * 36
    data = {"commit": token}
    result = redact_dict(data)
    assert result["commit"] == "***"


def test_redact_dict_redact_unknown_masks_unsafe_keys() -> None:
    data = {"command": "clean", "mystery": "value"}
    result = redact_dict(data, redact_unknown=True, safe_keys=frozenset({"command"}))
    assert result["command"] == "clean"
    assert result["mystery"] == "***"


def test_redact_dict_identity_keys_retained_verbatim() -> None:
    """``user``/``sso_subject``/etc. are kept even if email-shaped."""
    data = {"user": "admin@example.com", "command": "clean"}
    result = redact_dict(data)
    assert result["user"] == "admin@example.com"
    assert result["command"] == "clean"
