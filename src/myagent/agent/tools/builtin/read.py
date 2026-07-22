"""``read_file`` tool — read a file or list a directory."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from myagent.agent.tools.base import Tool, ToolContext, ToolResult
from myagent.agent.tools.builtin._paths import resolve_under_cwd
from myagent.agent.tools.truncation import TruncationService


class ReadFileInput(BaseModel):
    """Input for the ``read_file`` tool."""

    path: str = Field(
        ..., description="The absolute or cwd-relative path to read."
    )
    offset: int | None = Field(
        None,
        description="Line number to start reading from (1-indexed).",
        ge=1,
    )
    limit: int | None = Field(
        None,
        description="Maximum number of lines to read.",
        ge=1,
    )


_READ_DESCRIPTION = """\
Read the contents of a file or list a directory.

When ``path`` is a file, returns its contents with line numbers (1-indexed,
matching the ``cat -n`` format). When ``path`` is a directory, returns a
listing of its immediate children (directories are suffixed with ``/``).

Use ``offset`` and ``limit`` to read a specific range of lines from a large
file (e.g. ``offset=100, limit=50`` reads lines 100-150).

Output is truncated to 2000 lines / 50KB by default. When truncated, the
full output is saved to a temp file and the path is noted in the result.
"""


async def _read_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    assert isinstance(args, ReadFileInput)

    try:
        resolved = resolve_under_cwd(ctx.cwd, args.path)
    except Exception as exc:  # noqa: BLE001 — surface path errors to LLM
        return ToolResult.failure(str(exc))

    if not resolved.exists():
        return ToolResult.failure(f"Path does not exist: {args.path}")

    if resolved.is_dir():
        return _list_directory(resolved, args.path)

    return _read_file(resolved, args.path, args.offset, args.limit)


def _list_directory(path: Path, display: str) -> ToolResult:
    """Return a directory listing."""

    try:
        entries = sorted(
            path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())
        )
    except OSError as exc:
        return ToolResult.failure(f"Cannot list directory {display}: {exc}")

    lines: list[str] = []
    for entry in entries:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{entry.name}{suffix}")

    return ToolResult.success(
        "\n".join(lines), entry_count=len(lines), is_directory=True
    )


def _read_file(
    path: Path,
    display: str,
    offset: int | None,
    limit: int | None,
) -> ToolResult:
    """Read a file with optional offset/limit, applying truncation."""

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult.failure(f"Cannot read file {display}: {exc}")

    lines = text.splitlines(keepends=True)
    total = len(lines)

    start = (offset - 1) if offset and offset > 0 else 0
    if start > total:
        start = total
    end = (start + limit) if limit else total
    if end > total:
        end = total
    selected = lines[start:end]

    # Format with line numbers (1-indexed, ``cat -n`` style).
    numbered: list[str] = []
    for i, line in enumerate(selected, start=start + 1):
        if line.endswith("\n"):
            numbered.append(f"{i:>6}\t{line[:-1]}\n")
        else:
            numbered.append(f"{i:>6}\t{line}")
    content = "".join(numbered)

    # Truncate if needed.
    truncator = TruncationService()
    trunc = truncator.truncate(content, tool_id="read_file")

    metadata: dict[str, object] = {
        "line_count": total,
        "bytes": len(text.encode("utf-8")),
    }
    if offset:
        metadata["offset"] = offset
    if limit:
        metadata["limit"] = limit
    if trunc.truncated:
        metadata["truncated"] = True
        if trunc.output_path is not None:
            metadata["output_path"] = trunc.output_path

    # Range header if we sliced.
    if offset or limit:
        range_end = min(start + len(selected), total)
        header = f"[Showing lines {start + 1}-{range_end} of {total}]\n\n"
        final_content = header + trunc.content
    else:
        final_content = trunc.content

    return ToolResult(output=final_content, metadata=metadata)


def make_read_file_tool() -> Tool:
    """Construct the ``read_file`` tool."""

    return Tool(
        id="read_file",
        description=_READ_DESCRIPTION,
        parameters=ReadFileInput,
        execute=_read_execute,
        timeout_ms=30_000,
    )


__all__ = ["ReadFileInput", "make_read_file_tool"]
