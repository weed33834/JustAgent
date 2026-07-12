"""Tests for the team command."""

from __future__ import annotations

import base64
from pathlib import Path

from typer.testing import CliRunner

from autoship.cli.main import app
from autoship.core.team_config import generate_keypair, sign_team_config

runner = CliRunner()


def _write_config(tmp_path: Path, *, public_key: str | None = None) -> Path:
    config_path = tmp_path / ".autoship.toml"
    lines = [
        f'schema_version = 1\nproject_root = "{tmp_path}"\n',
    ]
    if public_key is not None:
        lines.append(f'\n[team]\npublic_key = "{public_key}"\n')
    config_path.write_text("".join(lines), encoding="utf-8")
    return config_path


def _write_signed_team_config(tmp_path: Path, content: str) -> tuple[Path, str, str]:
    """Create a team config + its detached signature, return (path, public_b64, private_b64)."""
    cfg_path = tmp_path / ".autoship.team.toml"
    cfg_path.write_text(content, encoding="utf-8")
    public_b64, private_b64 = generate_keypair()
    signature_bytes = sign_team_config(cfg_path, private_b64)
    sig_path = cfg_path.with_suffix(cfg_path.suffix + ".sig")
    sig_path.write_text(
        base64.urlsafe_b64encode(signature_bytes).rstrip(b"=").decode("ascii") + "\n",
        encoding="utf-8",
    )
    return cfg_path, public_b64, private_b64


def test_team_verify_no_key_returns_zero_with_warning(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / ".autoship.team.toml").write_text("[clean]\ntools = []\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["--config", str(config_path), "team", "verify"],
    )
    assert result.exit_code == 0
    assert "No [team] public_key configured" in result.output


def test_team_verify_missing_team_config_exits_two(tmp_path: Path) -> None:
    public, _ = generate_keypair()
    config_path = _write_config(tmp_path, public_key=public)

    result = runner.invoke(
        app,
        ["--config", str(config_path), "team", "verify"],
    )
    assert result.exit_code == 2
    assert "not found" in result.output.lower()


def test_team_verify_valid_signature(tmp_path: Path) -> None:
    cfg_path, public_b64, _ = _write_signed_team_config(tmp_path, '[clean]\ntools = ["ruff"]\n')
    config_path = _write_config(tmp_path, public_key=public_b64)

    result = runner.invoke(
        app,
        ["--config", str(config_path), "team", "verify"],
    )
    assert result.exit_code == 0
    assert "signature verified" in result.output.lower()


def test_team_verify_tampered_signature_fails(tmp_path: Path) -> None:
    cfg_path, public_b64, _ = _write_signed_team_config(tmp_path, '[clean]\ntools = ["ruff"]\n')
    # Tamper after signing.
    cfg_path.write_text('[clean]\ntools = ["black"]\n', encoding="utf-8")
    config_path = _write_config(tmp_path, public_key=public_b64)

    result = runner.invoke(
        app,
        ["--config", str(config_path), "team", "verify"],
    )
    assert result.exit_code == 1
    assert "verification failed" in result.output.lower()


def test_team_verify_missing_signature_file_fails(tmp_path: Path) -> None:
    public, _ = generate_keypair()
    config_path = _write_config(tmp_path, public_key=public)
    (tmp_path / ".autoship.team.toml").write_text("[clean]\ntools = []\n", encoding="utf-8")
    # No .sig file written.

    result = runner.invoke(
        app,
        ["--config", str(config_path), "team", "verify"],
    )
    assert result.exit_code == 1
    assert "signature missing" in result.output.lower()


def test_team_show_prints_config_and_validity(tmp_path: Path) -> None:
    cfg_path, public_b64, _ = _write_signed_team_config(tmp_path, '[clean]\ntools = ["ruff"]\n')
    config_path = _write_config(tmp_path, public_key=public_b64)

    result = runner.invoke(
        app,
        ["--config", str(config_path), "team", "show"],
    )
    assert result.exit_code == 0
    assert "signature: VALID" in result.output
    assert "tools" in result.output


def test_team_show_invalid_signature_exits_one(tmp_path: Path) -> None:
    cfg_path, public_b64, _ = _write_signed_team_config(tmp_path, '[clean]\ntools = ["ruff"]\n')
    cfg_path.write_text('[clean]\ntools = ["black"]\n', encoding="utf-8")
    config_path = _write_config(tmp_path, public_key=public_b64)

    result = runner.invoke(
        app,
        ["--config", str(config_path), "team", "show"],
    )
    assert result.exit_code == 1
    assert "INVALID" in result.output


def test_team_show_missing_team_config_exits_zero_with_message(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["--config", str(config_path), "team", "show"],
    )
    assert result.exit_code == 0
    assert "not found" in result.output.lower()


def test_team_sign_writes_signature_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    cfg_path.write_text('[clean]\ntools = ["ruff"]\n', encoding="utf-8")
    config_path = _write_config(tmp_path)
    _, private_b64 = generate_keypair()

    result = runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "team",
            "sign",
            str(cfg_path),
            "--key",
            private_b64,
        ],
    )
    assert result.exit_code == 0
    assert "Wrote detached signature" in result.output
    sig_path = cfg_path.with_suffix(cfg_path.suffix + ".sig")
    assert sig_path.exists()
    # The signature file should be valid base64.
    content = sig_path.read_text(encoding="utf-8").strip()
    decoded = base64.urlsafe_b64decode(content + "=" * (-len(content) % 4))
    assert len(decoded) == 64  # Ed25519 signatures are 64 bytes


def test_team_sign_stdout_prints_signature(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    cfg_path.write_text('[clean]\ntools = ["ruff"]\n', encoding="utf-8")
    config_path = _write_config(tmp_path)
    _, private_b64 = generate_keypair()

    result = runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "team",
            "sign",
            str(cfg_path),
            "--key",
            private_b64,
            "--stdout",
        ],
    )
    assert result.exit_code == 0
    # The output is just the base64 signature.
    output = result.output.strip()
    decoded = base64.urlsafe_b64decode(output + "=" * (-len(output) % 4))
    assert len(decoded) == 64
    # No .sig file should be written.
    assert not cfg_path.with_suffix(cfg_path.suffix + ".sig").exists()


def test_team_keygen_prints_keypair(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["--config", str(config_path), "team", "keygen"],
    )
    assert result.exit_code == 0
    assert "Generated fresh Ed25519 keypair" in result.output
    assert "[team]" in result.output
    assert "public_key" in result.output
    assert "Sign with:" in result.output


def test_team_sign_malformed_key_fails(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".autoship.team.toml"
    cfg_path.write_text('[clean]\ntools = ["ruff"]\n', encoding="utf-8")
    config_path = _write_config(tmp_path)

    result = runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "team",
            "sign",
            str(cfg_path),
            "--key",
            "YQ==",  # 1-byte key
        ],
    )
    assert result.exit_code == 1
    assert "32 raw bytes" in result.output
