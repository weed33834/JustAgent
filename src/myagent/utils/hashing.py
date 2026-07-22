"""Shared hashing and package-installer helpers."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path


def compute_sha256(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file.

    Accepts either a :class:`Path` or a string path; callers that pass a
    plain string previously hit ``AttributeError: 'str' object has no
    attribute 'open'``.
    """
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _in_virtualenv() -> bool:
    """Return True when the active interpreter is inside a virtual environment."""
    return hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix


def pip_cmd() -> list[str]:
    """Return the preferred package installer command (uv or pip).

    ``uv pip`` is used only when ``uv`` is available *and* the active interpreter
    is running inside a virtual environment, because ``uv pip install`` refuses
    to install into a non-virtual environment without ``--system``.
    """
    if shutil.which("uv") and _in_virtualenv():
        return ["uv", "pip"]
    return ["pip"]


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

    def __init__(self, tools_config=None) -> None:
        """Create a verifier from a ``ToolsConfig`` model.

        If no config is provided the verifier operates in PATH-only mode.
        """
        self._config = tools_config

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
        from myagent.exceptions import ConfigError

        tool_config = self._config.get(name) if self._config else None
        resolved = None

        if tool_config and tool_config.path:
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

        if tool_config and tool_config.sha256:
            actual = compute_sha256(resolved)
            expected = tool_config.sha256.lower()
            if actual != expected:
                raise ConfigError(
                    f"SHA-256 mismatch for tool '{name}': expected {expected}, got {actual}"
                )

        if tool_config and (tool_config.path or tool_config.sha256):
            return str(resolved)
        return name

    def check(self, name: str) -> bool:
        """Return ``True`` if ``name`` can be resolved without raising."""
        try:
            self.resolve(name)
        except Exception:
            return False
        return True
