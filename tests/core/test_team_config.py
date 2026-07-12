"""Tests for the team_config module — signature verify / load / sign / keygen."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoship.core.team_config import (
    TeamConfigError,
    generate_keypair,
    load_team_config,
    sign_team_config,
    signature_path_for,
    verify_team_config,
)


def _write_team_config(path: Path, content: str = "") -> None:
    if not content:
        content = '[clean]\ntools = ["ruff", "black"]\n\n[commit]\nconventional_commits = true\n'
    path.write_text(content, encoding="utf-8")


def test_load_team_config_returns_empty_when_missing(tmp_path: Path) -> None:
    result = load_team_config(tmp_path / ".autoship.team.toml")
    assert result == {}


def test_load_team_config_returns_dict_when_unsigned_and_no_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    _write_team_config(cfg_path)
    result = load_team_config(cfg_path)
    assert isinstance(result, dict)
    assert "clean" in result
    assert result["clean"]["tools"] == ["ruff", "black"]


def test_load_team_config_require_signature_without_key_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    _write_team_config(cfg_path)
    with pytest.raises(TeamConfigError, match="require_signature"):
        load_team_config(cfg_path, public_key_b64=None, require_signature=True)


def test_verify_missing_signature_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    _write_team_config(cfg_path)
    public_b64, _ = generate_keypair()
    with pytest.raises(TeamConfigError, match="signature missing"):
        verify_team_config(cfg_path, public_b64)


def test_verify_missing_config_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    public_b64, _ = generate_keypair()
    with pytest.raises(TeamConfigError, match="not found"):
        verify_team_config(cfg_path, public_b64)


def test_sign_then_verify_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    _write_team_config(cfg_path)

    public_b64, private_b64 = generate_keypair()
    signature_bytes = sign_team_config(cfg_path, private_b64)

    sig_path = signature_path_for(cfg_path)
    import base64

    sig_path.write_text(
        base64.urlsafe_b64encode(signature_bytes).rstrip(b"=").decode("ascii") + "\n",
        encoding="utf-8",
    )

    assert verify_team_config(cfg_path, public_b64) is True


def test_verify_rejects_tampered_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    _write_team_config(cfg_path, content='[clean]\ntools = ["ruff"]\n')

    public_b64, private_b64 = generate_keypair()
    signature_bytes = sign_team_config(cfg_path, private_b64)

    import base64

    sig_path = signature_path_for(cfg_path)
    sig_path.write_text(
        base64.urlsafe_b64encode(signature_bytes).rstrip(b"=").decode("ascii") + "\n",
        encoding="utf-8",
    )

    # Tamper with the config after signing.
    _write_team_config(cfg_path, content='[clean]\ntools = ["black"]\n')

    with pytest.raises(TeamConfigError, match="verification failed"):
        verify_team_config(cfg_path, public_b64)


def test_verify_rejects_wrong_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    _write_team_config(cfg_path)

    signer_public, signer_private = generate_keypair()
    other_public, _ = generate_keypair()

    signature_bytes = sign_team_config(cfg_path, signer_private)
    import base64

    sig_path = signature_path_for(cfg_path)
    sig_path.write_text(
        base64.urlsafe_b64encode(signature_bytes).rstrip(b"=").decode("ascii") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TeamConfigError, match="verification failed"):
        verify_team_config(cfg_path, other_public)


def test_malformed_public_key_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    _write_team_config(cfg_path)
    sig_path = signature_path_for(cfg_path)
    sig_path.write_text("YQ==\n", encoding="utf-8")  # 1-byte signature
    with pytest.raises(TeamConfigError, match="expected 32 raw bytes"):
        verify_team_config(cfg_path, "YQ==")  # 1-byte key


def test_malformed_private_key_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    _write_team_config(cfg_path)
    with pytest.raises(TeamConfigError, match="expected 32 raw bytes"):
        sign_team_config(cfg_path, "YQ==")


def test_signature_path_for_appends_sig_suffix(tmp_path: Path) -> None:
    cfg = tmp_path / ".autoship.team.toml"
    assert signature_path_for(cfg) == tmp_path / ".autoship.team.toml.sig"


def test_generate_keypair_returns_32_byte_keys() -> None:
    import base64

    public_b64, private_b64 = generate_keypair()
    public_bytes = base64.urlsafe_b64decode(public_b64 + "=" * (-len(public_b64) % 4))
    private_bytes = base64.urlsafe_b64decode(private_b64 + "=" * (-len(private_b64) % 4))
    assert len(public_bytes) == 32
    assert len(private_bytes) == 32
    # Each call must produce a fresh keypair.
    public2, _ = generate_keypair()
    assert public2 != public_b64


def test_load_team_config_malformed_toml_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    cfg_path.write_text("not = valid = toml =\n", encoding="utf-8")
    with pytest.raises(TeamConfigError, match="Failed to parse"):
        load_team_config(cfg_path)
