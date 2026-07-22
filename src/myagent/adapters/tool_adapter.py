"""Adapter for running external formatting/cleanup tools."""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

from myagent.models.config import ToolsConfig
from myagent.utils.hashing import ToolVerifier


def _glob_to_regex(pattern: str) -> str:
    """Translate a glob ``pattern`` (with ``**`` support) into a regex string."""
    # fnmatch.translate gives a Python regex; we use it so the same glob
    # semantics as Path.match / glob are applied. ``**`` is handled by
    # fnmatch as ``*`` which is good enough for directory-prefix excludes.
    return fnmatch.translate(pattern)


def _build_force_exclude_regex(exclude: list[str]) -> str:
    """Combine a list of glob exclude patterns into a single black regex.

    black's ``--force-exclude`` takes a single regex; multiple patterns are
    joined with ``|``. Each glob is translated via :func:`_glob_to_regex`
    so users can write ``migrations/`` or ``vendor/**`` in config.
    """
    if not exclude:
        return ""
    parts = [_glob_to_regex(p.strip()) for p in exclude if p and p.strip()]
    return "|".join(parts) if parts else ""


class ToolChain:
    """Run a configurable sequence of cleanup/formatting tools."""

    def __init__(
        self,
        tools: list[str],
        project_root: Path,
        *,
        dry_run: bool = False,
        verbose: bool = False,
        tool_verifier: ToolVerifier | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        self.tools = tools
        self.project_root = project_root
        self.dry_run = dry_run
        self.verbose = verbose
        self.exclude = list(exclude) if exclude else []
        self._verifier = tool_verifier or ToolVerifier(ToolsConfig())

    def _resolve(self, name: str) -> str | None:
        """Return the verified absolute path to ``name`` or ``None``."""
        try:
            return self._verifier.resolve(name)
        except Exception:  # noqa: BLE001
            return None

    def _is_excluded(self, path: Path) -> bool:
        """Return True if ``path`` matches any configured exclude glob."""
        if not self.exclude:
            return False
        # Match against both the absolute and project-root-relative forms so
        # that ``migrations/`` excludes ``<root>/migrations/foo.py`` whether
        # the caller passes ``migrations/foo.py`` or an absolute path.
        candidates = [str(path)]
        try:
            rel = path.resolve().relative_to(self.project_root.resolve())
            candidates.append(str(rel))
        except (OSError, ValueError):
            pass
        for pattern in self.exclude:
            for candidate in candidates:
                if fnmatch.fnmatch(candidate, pattern):
                    return True
            # Also match path components so ``migrations/`` excludes any
            # file living under a ``migrations`` directory.
            if any(part == pattern.rstrip("/") for part in path.parts):
                return True
        return False

    def _filter_paths(self, paths: list[Path]) -> list[str]:
        """Drop paths matching the configured exclude globs and return str form."""
        if not self.exclude:
            return [str(p) for p in paths]
        return [str(p) for p in paths if not self._is_excluded(p)]

    def _run(
        self,
        cmd: list[str],
        *,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if self.dry_run:
            print(f"[dry-run] {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        if self.verbose:
            print(f"[exec] {' '.join(cmd)}")
        return subprocess.run(
            cmd,
            cwd=self.project_root,
            check=check,
            capture_output=capture_output,
            text=True,
        )

    def preview(self, paths: list[Path]) -> str:
        """Return a diff/preview of the changes that would be applied."""
        targets = self._filter_paths(paths)
        if not targets:
            return ""
        black = self._resolve("black")
        if "black" in self.tools and black:
            cmd: list[str] = [black, "--diff"]
            force_exclude = _build_force_exclude_regex(self.exclude)
            if force_exclude:
                cmd += ["--force-exclude", force_exclude]
            cmd += targets
            result = self._run(cmd, capture_output=True)
            return result.stdout
        return ""

    def apply(self, paths: list[Path]) -> None:
        """Apply formatting/cleanup tools in place."""
        targets = self._filter_paths(paths)
        if not targets:
            return
        autoflake = self._resolve("autoflake")
        black = self._resolve("black")
        if "autoflake" in self.tools and autoflake:
            self._run(
                [autoflake, "--remove-all-unused-imports", "--in-place", "-r"] + targets,
            )
        if "black" in self.tools and black:
            cmd: list[str] = [black]
            force_exclude = _build_force_exclude_regex(self.exclude)
            if force_exclude:
                cmd += ["--force-exclude", force_exclude]
            cmd += targets
            self._run(cmd)
