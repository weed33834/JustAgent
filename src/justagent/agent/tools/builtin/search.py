"""``search`` tool — search files by content (regex) or filename pattern."""

from __future__ import annotations

import re
import typing
from pathlib import Path

from pydantic import BaseModel, Field

from justagent.agent.tools.base import Tool, ToolContext, ToolResult
from justagent.agent.tools.builtin._paths import resolve_under_cwd
from justagent.agent.tools.truncation import TruncationService


class SearchInput(BaseModel):
    """Input for the ``search`` tool."""

    pattern: str = Field(
        ..., description="Regular expression pattern to search for."
    )
    path: str | None = Field(
        None,
        description="Directory to search in (defaults to the agent's cwd).",
    )
    glob: str | None = Field(
        None,
        description=(
            "Optional filename glob (e.g. ``*.py``) to restrict the search "
            "to specific file types."
        ),
    )
    output_mode: str = Field(
        "content",
        description=(
            "Output mode: ``content`` (default, shows matching lines with "
            "context), ``files_with_matches`` (just filenames), or "
            "``count`` (per-file match counts)."
        ),
    )


_SEARCH_DESCRIPTION = """\
Search file contents using a regular expression.

Walks the directory tree (skipping ``.git/``, ``.venv/``, ``__pycache__/``,
``node_modules/``, and other common ignore patterns) and returns matches.

Three output modes:

* ``content`` (default): matching lines with file:line prefixes, like
  ``ripgrep`` output.
* ``files_with_matches``: just the file paths that contain matches.
* ``count``: per-file match counts.

Use ``glob`` to restrict by filename pattern (e.g. ``*.py``).
"""


# Directories we never descend into during search.
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
        ".eggs",
        "target",
        ".next",
        ".nuxt",
        "coverage",
        ".coverage",
    }
)

# Files we skip (binary blobs).
_IGNORED_FILE_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".bin",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".pdf",
        ".docx",
        ".xlsx",
        ".pptx",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".a",
        ".o",
        ".class",
        ".jar",
        ".war",
    }
)

MAX_FILE_SIZE_BYTES = 1_000_000  # 1 MB
MAX_MATCHES_PER_FILE = 100
MAX_TOTAL_MATCHES = 500


async def _search_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    assert isinstance(args, SearchInput)

    # Validate output mode.
    if args.output_mode not in {"content", "files_with_matches", "count"}:
        return ToolResult.failure(
            f"Invalid output_mode: {args.output_mode!r}. "
            "Must be one of: content, files_with_matches, count."
        )

    # Compile the regex.
    try:
        regex = re.compile(args.pattern)
    except re.error as exc:
        return ToolResult.failure(
            f"Invalid regex pattern {args.pattern!r}: {exc}"
        )

    # Resolve the search root.
    if args.path:
        try:
            root = resolve_under_cwd(ctx.cwd, args.path)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(str(exc))
    else:
        root = Path(ctx.cwd).resolve()

    if not root.exists():
        return ToolResult.failure(f"Search path does not exist: {args.path}")
    if not root.is_dir():
        return ToolResult.failure(
            f"Search path is not a directory: {args.path}"
        )

    # Compile the glob pattern if provided.
    glob_pattern = args.glob
    if glob_pattern:
        try:
            import fnmatch

            glob_re = re.compile(fnmatch.translate(glob_pattern))
        except re.error as exc:
            return ToolResult.failure(
                f"Invalid glob pattern {glob_pattern!r}: {exc}"
            )
    else:
        glob_re = None

    # Walk the tree.
    matches: list[tuple[Path, int, str]] = []
    file_counts: dict[Path, int] = {}
    files_with_matches: list[Path] = []
    total_matches = 0

    for file_path in _iter_source_files(root):
        ctx.check_aborted()
        if total_matches >= MAX_TOTAL_MATCHES:
            break

        # Filter by glob.
        if glob_re is not None and not glob_re.match(file_path.name):
            continue

        # Skip large files.
        try:
            if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
        except OSError:
            continue

        # Read and search.
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        file_match_count = 0
        for line_no, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                if args.output_mode == "content":
                    matches.append((file_path, line_no, line.rstrip()))
                file_match_count += 1
                total_matches += 1
                if file_match_count >= MAX_MATCHES_PER_FILE:
                    break
                if total_matches >= MAX_TOTAL_MATCHES:
                    break

        if file_match_count > 0:
            files_with_matches.append(file_path)
            file_counts[file_path] = file_match_count

    # Format output.
    if args.output_mode == "files_with_matches":
        file_lines = [str(p.relative_to(root)) for p in files_with_matches]
        content = "\n".join(sorted(file_lines))
        return ToolResult.success(
            content,
            match_count=len(files_with_matches),
            truncated=total_matches >= MAX_TOTAL_MATCHES,
        )

    if args.output_mode == "count":
        count_lines = [
            f"{file_counts[p]:>6} {p.relative_to(root)}"
            for p in sorted(file_counts.keys())
        ]
        content = "\n".join(count_lines)
        return ToolResult.success(
            content,
            file_count=len(file_counts),
            total_matches=total_matches,
            truncated=total_matches >= MAX_TOTAL_MATCHES,
        )

    # content mode.
    content_lines: list[str] = []
    for file_path, line_no, line in matches:
        rel = file_path.relative_to(root)
        content_lines.append(f"{rel}:{line_no}:{line}")
    content = "\n".join(content_lines)

    # Truncate.
    truncator = TruncationService()
    trunc = truncator.truncate(content, tool_id="search")

    metadata: dict[str, object] = {
        "match_count": total_matches,
        "file_count": len(files_with_matches),
    }
    if total_matches >= MAX_TOTAL_MATCHES:
        metadata["truncated"] = True
    if trunc.truncated:
        metadata["output_truncated"] = True
        if trunc.output_path is not None:
            metadata["output_path"] = trunc.output_path

    return ToolResult(output=trunc.content, metadata=metadata)


def _iter_source_files(root: Path) -> typing.Iterator[Path]:
    """Yield source files under ``root``, skipping ignored dirs/files."""

    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        # Skip if any path component is an ignored dir.
        try:
            rel_parts = entry.relative_to(root).parts
        except ValueError:
            continue
        if any(part in _IGNORED_DIRS for part in rel_parts[:-1]):
            continue
        if entry.suffix.lower() in _IGNORED_FILE_SUFFIXES:
            continue
        yield entry


def make_search_tool() -> Tool:
    """Construct the ``search`` tool."""

    return Tool(
        id="search",
        description=_SEARCH_DESCRIPTION,
        parameters=SearchInput,
        execute=_search_execute,
        timeout_ms=60_000,
    )


__all__ = ["SearchInput", "make_search_tool"]
