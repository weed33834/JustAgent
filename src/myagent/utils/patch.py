"""Shared patch application helpers used by ``verify`` and ``fix``.

Both the ``verify`` (apply a suggested fix) and ``fix`` (apply an LLM-proposed
patch) commands need to apply a unified diff to the working tree. The
mechanics — try ``git apply`` first, fall back to ``patch -p1`` — are identical
and live here so the two command modules cannot drift apart.

This module also hosts :func:`patch_paths_are_safe`, the path-traversal /
test-file guard used by both commands before a patch is applied. Centralising
it here means ``verify --fix`` (which applies patches produced by plugins)
gets the same safety check as the ``fix`` command (which applies patches
produced by an LLM) — a malicious or buggy plugin cannot bypass it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from myagent.exceptions import ConfigError
from myagent.utils.hashing import ToolVerifier

#: Default subprocess timeout for ``git apply`` / ``patch -p1``. 30s is well
#: above the time a sane patch takes to apply, but bounded enough that a
#: hanging ``patch`` command (which prompts on /dev/tty when a hunk is
#: ambiguous) cannot block the CLI indefinitely in non-interactive contexts.
DEFAULT_PATCH_TIMEOUT = 30


def collect_patch_paths(patch: str) -> set[str]:
    """Return the file paths referenced in a unified diff.

    Strips the optional ``a/`` / ``b/`` prefixes and the trailing timestamp
    that ``git diff`` appends, so the returned paths are project-relative.
    ``/dev/null`` (used by git for file creation/deletion) is skipped.
    """
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            # Strip the optional timestamp suffix that ``git diff`` appends.
            raw = line[4:].split("\t", 1)[0].strip()
            if raw in ("/dev/null", "dev/null"):
                continue
            # Unified diffs prefix old paths with ``a/`` and new paths with ``b/``.
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            paths.add(raw)
    return paths


def patch_paths_are_safe(project_root: Path, patch: str) -> bool:
    """Return True when every path in ``patch`` stays inside ``project_root``.

    Also rejects patches that would modify test files, keeping the ``fix``
    command focused on implementation/source code only. This guard is the
    shared pre-check used by both ``fix`` and ``verify --fix`` before a
    patch is handed to :func:`apply_patch`.
    """
    root = project_root.resolve()
    for raw in collect_patch_paths(patch):
        # Reject absolute paths and path traversal attempts outright.
        if Path(raw).is_absolute() or ".." in Path(raw).parts:
            return False
        if not (root / raw).resolve().is_relative_to(root):
            return False
        raw_lower = raw.lower()
        if "tests/" in raw_lower or "test_" in raw_lower or raw_lower.startswith("test"):
            return False
    return True


def apply_patch(
    project_root: Path,
    patch_text: str,
    tools: ToolVerifier,
    *,
    timeout: float = DEFAULT_PATCH_TIMEOUT,
) -> tuple[bool, str | None]:
    """Apply a unified diff patch. Returns ``(success, error_message)``.

    Prefers ``git apply`` (via ``--check`` first) and falls back to the
    ``patch`` command so patches can still be applied when the working tree
    differs from HEAD. The supplied :class:`ToolVerifier` resolves the pinned
    ``git`` / ``patch`` binaries (or PATH defaults) and is honoured for both
    sha256 pinning and explicit path overrides.

    All subprocess invocations are bounded by ``timeout`` seconds (default
    :data:`DEFAULT_PATCH_TIMEOUT`). A timeout returns ``(False, ...)`` rather
    than raising so the caller (``fix`` / ``verify --fix``) can surface the
    failure through its normal error path. The ``patch`` fallback additionally
    passes ``--batch`` so an ambiguous hunk fails the subprocess instead of
    prompting on ``/dev/tty`` and hanging non-interactive invocations (CI,
    LSP, dogfood scripts).
    """
    last_reason: str | None = None

    try:
        git = tools.resolve("git")
    except ConfigError:
        git = None

    if git:
        try:
            check = subprocess.run(
                [git, "apply", "--check"],
                input=patch_text,
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"patch apply timed out after {int(timeout)}s"
        if check.returncode == 0:
            try:
                apply = subprocess.run(
                    [git, "apply"],
                    input=patch_text,
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return False, f"patch apply timed out after {int(timeout)}s"
            if apply.returncode == 0:
                return True, None
            last_reason = apply.stderr.strip() or "git apply failed"
        else:
            last_reason = check.stderr.strip() or "git apply --check failed"

    try:
        patch_cmd = tools.resolve("patch")
    except ConfigError:
        patch_cmd = None

    if patch_cmd:
        try:
            proc = subprocess.run(
                # ``--batch`` makes patch(1) fail on ambiguous hunks instead
                # of prompting on /dev/tty, which would hang non-interactive
                # invocations (CI, LSP, the fix command's LLM-driven flow).
                [patch_cmd, "-p1", "--batch"],
                input=patch_text,
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"patch apply timed out after {int(timeout)}s"
        if proc.returncode == 0:
            return True, None
        last_reason = proc.stderr.strip() or "patch command failed"
        # Clean up the `.orig` / `.rej` scratch files that `patch(1)` writes
        # next to a target file when a hunk is rejected or the original is
        # backed up. They provide no value to the caller (the failure reason
        # is already in ``last_reason``) and leak as untracked files into the
        # working tree, which then show up in `git status` and confuse VCS
        # workflows. Only files referenced by this patch are touched.
        for raw in collect_patch_paths(patch_text):
            if raw.startswith("/") or ".." in Path(raw).parts:
                continue
            for suffix in (".orig", ".rej"):
                scratch = project_root / f"{raw}{suffix}"
                try:
                    scratch.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    return False, last_reason or "Neither git nor patch is available on PATH"
