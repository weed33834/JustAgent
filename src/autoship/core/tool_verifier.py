"""Helpers for verifying external tool binaries before execution.

This module protects against PATH pollution attacks by allowing operators to
pin either the absolute path or the SHA-256 hash of the external binaries that
AutoShip invokes (git, docker, twine, gh, patch, etc.).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from autoship.exceptions import ConfigError
from autoship.models.config import ToolsConfig
from autoship.utils.hashing import compute_sha256


class ToolVerifier:
    """Resolve and optionally pin external tool binaries.

    The verifier consults the ``tools`` section of the application config.  For
    each supported tool the operator may configure:

    - ``path``: an absolute path to the binary that must be used.
    - ``sha256``: the expected SHA-256 digest of the resolved binary.

    When ``path`` is set the verifier uses that exact executable and validates
    that the file exists.  Otherwise it falls back to ``shutil.which`` and the
    current ``PATH``.  When ``sha256`` is set the verifier hashes the resolved
    executable and rejects it if the digest does not match.
    """

    def __init__(self, config: ToolsConfig | None = None) -> None:
        """Create a verifier from a ``ToolsConfig`` model.

        If no config is provided the verifier operates in PATH-only mode.
        """
        self.config = config or ToolsConfig()

    def resolve(self, name: str, *, search_path: bool = True) -> str:
        """Return the executable to use for ``name`` after validation.

        Args:
            name: the logical tool name (e.g. ``"git"``, ``"docker"``).
            search_path: when ``True`` and no explicit path is configured, fall
                back to ``shutil.which(name)``.

        Returns:
            The configured absolute path when ``path`` or ``sha256`` is set,
            otherwise the original tool name so that the existing tests and
            PATH resolution continue to work unchanged.

        Raises:
            ConfigError: when the tool cannot be resolved or fails validation.
        """
        tool_config = self.config.get(name)
        resolved: Path | None = None

        if tool_config.path:
            raw = Path(tool_config.path).expanduser()
            if not raw.is_absolute():
                raise ConfigError(
                    f"Configured path for tool '{name}' must be absolute: {tool_config.path}"
                )
            resolved = raw.resolve()
            if not resolved.is_file():
                raise ConfigError(f"Configured tool '{name}' does not exist: {tool_config.path}")
        elif search_path:
            found = shutil.which(name)
            if found is None:
                raise ConfigError(f"Tool '{name}' not found in PATH")
            resolved = Path(found).resolve()
        else:
            raise ConfigError(f"Tool '{name}' has no configured path and search_path is disabled")

        if tool_config.sha256:
            actual = compute_sha256(resolved)
            expected = tool_config.sha256.lower()
            if actual != expected:
                raise ConfigError(
                    f"SHA-256 mismatch for tool '{name}': expected {expected}, got {actual}"
                )

        # Keep the original logical name unless an operator explicitly pinned the
        # tool. This preserves backward compatibility while still validating the
        # configured path/hash when provided.
        if tool_config.path or tool_config.sha256:
            return str(resolved)
        return name

    def check(self, name: str) -> bool:
        """Return ``True`` if ``name`` can be resolved without raising."""
        try:
            self.resolve(name)
        except ConfigError:
            return False
        return True
