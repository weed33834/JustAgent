"""``write_to_file`` tool — create or overwrite a file."""

from __future__ import annotations

from pydantic import BaseModel, Field

from justagent.agent.tools.base import Tool, ToolContext, ToolResult
from justagent.agent.tools.builtin._paths import resolve_under_cwd
from justagent.utils.atomic_write import atomic_write_text


class WriteToFileInput(BaseModel):
    """Input for the ``write_to_file`` tool."""

    path: str = Field(..., description="The path to write to.")
    content: str = Field(..., description="The full content to write.")
    create_dirs: bool = Field(
        True, description="Create parent directories if they don't exist."
    )


_WRITE_DESCRIPTION = """\
Create a new file or overwrite an existing file with the given content.

The write is atomic — the file is either fully written or left unchanged
(no partial writes on error).

Set ``create_dirs=false`` to fail if parent directories don't exist.

WARNING: This tool replaces the entire file content. To make a partial
edit, use ``replace_in_file`` or ``apply_patch`` instead.
"""


async def _write_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    assert isinstance(args, WriteToFileInput)

    try:
        resolved = resolve_under_cwd(ctx.cwd, args.path)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(str(exc))

    # Request permission before any filesystem side effect.
    approved = await ctx.request_permission(
        {
            "tool": "write_to_file",
            "path": str(resolved),
            "description": f"Write {len(args.content)} chars to {args.path}",
            "is_new_file": not resolved.exists(),
        }
    )
    if not approved:
        return ToolResult.failure("Permission denied by user")

    # Read old content *before* writing so the runtime can compute a
    # diff for the change tracker. ``None`` means the file didn't exist.
    old_content: str | None = None
    if resolved.exists():
        try:
            old_content = resolved.read_text(encoding="utf-8")
        except OSError:
            old_content = None

    if args.create_dirs:
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult.failure(
                f"Cannot create parent directories for "
                f"{args.path}: {exc}"
            )
    elif not resolved.parent.exists():
        return ToolResult.failure(
            f"Parent directory does not exist: {resolved.parent} "
            f"(set create_dirs=true to auto-create)"
        )

    try:
        atomic_write_text(resolved, args.content)
    except OSError as exc:
        return ToolResult.failure(f"Cannot write file {args.path}: {exc}")

    line_count = args.content.count("\n") + (
        0 if args.content.endswith("\n") else 1
    )
    return ToolResult.success(
        f"Successfully wrote {line_count} lines to {args.path}",
        line_count=line_count,
        bytes=len(args.content.encode("utf-8")),
        path=str(resolved),
        old_content=old_content,
        new_content=args.content,
    )


def make_write_to_file_tool() -> Tool:
    """Construct the ``write_to_file`` tool."""

    return Tool(
        id="write_to_file",
        description=_WRITE_DESCRIPTION,
        parameters=WriteToFileInput,
        execute=_write_execute,
        timeout_ms=30_000,
    )


__all__ = ["WriteToFileInput", "make_write_to_file_tool"]
