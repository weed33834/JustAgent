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


# Re-exported for backward compatibility. The canonical ``ToolVerifier``
# implementation lives in :mod:`justagent.core.tool_verifier` and stores its
# configuration as ``self.config``; the duplicate previously defined here used
# ``self._config``, which broke tests asserting on ``uploader._verifier.config``.
# This import is deferred to the end of the module so that ``compute_sha256``
# (imported by ``core.tool_verifier``) is already defined when the import cycle
# closes, avoiding a partially-initialised-module error.
from justagent.core.tool_verifier import ToolVerifier as ToolVerifier  # noqa: E402
