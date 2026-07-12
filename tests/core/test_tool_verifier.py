"""Tests for ToolVerifier."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from autoship.core.tool_verifier import ToolVerifier
from autoship.exceptions import ConfigError
from autoship.models.config import ToolConfig, ToolsConfig


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_executable(path: Path) -> Path:
    """Create an executable file at ``path`` and return it."""
    path.write_bytes(b"#!/bin/sh\necho ok\n")
    path.chmod(0o755)
    return path


def test_resolve_returns_logical_name_when_not_pinned(tmp_path: Path) -> None:
    """When no path/hash is configured, keep the original tool name."""
    verifier = ToolVerifier(ToolsConfig())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_executable(bin_dir / "mytool")

    with patch.dict(os.environ, {"PATH": str(bin_dir)}):
        assert verifier.resolve("mytool") == "mytool"


def test_resolve_raises_when_tool_not_in_path(tmp_path: Path) -> None:
    """A missing tool produces a ConfigError."""
    verifier = ToolVerifier(ToolsConfig())
    with (
        patch.dict(os.environ, {"PATH": str(tmp_path)}),
        pytest.raises(ConfigError, match="not found in PATH"),
    ):
        verifier.resolve("missingtool")


def test_resolve_uses_configured_absolute_path(tmp_path: Path) -> None:
    """A configured absolute path is resolved and returned."""
    tool = _make_executable(tmp_path / "custom-git")
    config = ToolsConfig(git=ToolConfig(path=str(tool)))
    verifier = ToolVerifier(config)
    assert verifier.resolve("git") == str(tool)


def test_resolve_rejects_relative_configured_path(tmp_path: Path) -> None:
    """Configured paths must be absolute."""
    config = ToolsConfig(git=ToolConfig(path="relative/path"))
    verifier = ToolVerifier(config)
    with pytest.raises(ConfigError, match="must be absolute"):
        verifier.resolve("git")


def test_resolve_rejects_missing_configured_path(tmp_path: Path) -> None:
    """A configured path that does not exist is rejected."""
    config = ToolsConfig(git=ToolConfig(path=str(tmp_path / "does-not-exist")))
    verifier = ToolVerifier(config)
    with pytest.raises(ConfigError, match="does not exist"):
        verifier.resolve("git")


def test_resolve_validates_sha256_hash(tmp_path: Path) -> None:
    """When a hash is configured, the resolved file must match."""
    tool = _make_executable(tmp_path / "pinned")
    digest = _sha256(tool)
    config = ToolsConfig(git=ToolConfig(path=str(tool), sha256=digest))
    verifier = ToolVerifier(config)
    assert verifier.resolve("git") == str(tool)


def test_resolve_rejects_sha256_mismatch(tmp_path: Path) -> None:
    """A hash mismatch raises ConfigError."""
    tool = _make_executable(tmp_path / "pinned")
    config = ToolsConfig(git=ToolConfig(path=str(tool), sha256="0" * 64))
    verifier = ToolVerifier(config)
    with pytest.raises(ConfigError, match="SHA-256 mismatch"):
        verifier.resolve("git")


def test_resolve_hashes_binary_found_in_path(tmp_path: Path) -> None:
    """SHA-256 validation also works for PATH-resolved tools."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = _make_executable(bin_dir / "git")
    digest = _sha256(tool)

    config = ToolsConfig(git=ToolConfig(sha256=digest))
    verifier = ToolVerifier(config)
    with patch.dict(os.environ, {"PATH": str(bin_dir)}):
        assert verifier.resolve("git") == str(tool)


def test_resolve_disables_path_search(tmp_path: Path) -> None:
    """When ``search_path=False`` only configured paths are allowed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_executable(bin_dir / "mytool")
    verifier = ToolVerifier(ToolsConfig())
    with (
        patch.dict(os.environ, {"PATH": str(bin_dir)}),
        pytest.raises(ConfigError, match="search_path is disabled"),
    ):
        verifier.resolve("mytool", search_path=False)


def test_check_returns_false_on_failure(tmp_path: Path) -> None:
    """``check`` returns False when resolution fails."""
    verifier = ToolVerifier(ToolsConfig())
    with patch.dict(os.environ, {"PATH": str(tmp_path)}):
        assert verifier.check("missingtool") is False


def test_check_returns_true_on_success(tmp_path: Path) -> None:
    """``check`` returns True when resolution succeeds."""
    tool = _make_executable(tmp_path / "tool")
    config = ToolsConfig(git=ToolConfig(path=str(tool)))
    verifier = ToolVerifier(config)
    assert verifier.check("git") is True
