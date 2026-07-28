"""``replace_in_file`` tool — apply SEARCH/REPLACE blocks to a file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from justagent.agent.search_replace import (
    SearchReplaceError,
    apply_search_replace,
)
from justagent.agent.tools.base import Tool, ToolContext, ToolResult
from justagent.agent.tools.builtin._paths import resolve_under_cwd


class ReplaceInFileInput(BaseModel):
    """Input for the ``replace_in_file`` tool."""

    path: str = Field(..., description="The path to the file to edit.")
    diff: str = Field(
        ...,
        description=(
            "The SEARCH/REPLACE blocks (Aider format) to apply. Each block "
            "is delimited by ``<<<<<<< SEARCH`` / ``=======`` / "
            "``>>>>>>> REPLACE`` markers."
        ),
    )


_REPLACE_DESCRIPTION = """\
Apply SEARCH/REPLACE blocks to edit an existing file.

The ``diff`` parameter must contain one or more SEARCH/REPLACE blocks in
the Aider format:

```
<<<<<<< SEARCH
original code to find
=======
new code to replace it with
>>>>>>> REPLACE
```

Matching is tolerant: perfect match is tried first, then leading-whitespace
tolerance, then ``...`` elision handling, then fuzzy fallback (similarity
>= 0.8). An empty SEARCH block appends to the file.

If a block fails to match, the tool returns an error listing the failed
blocks (other blocks in the same diff that succeeded are still applied).
"""


async def _replace_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    assert isinstance(args, ReplaceInFileInput)

    # Resolve the target path so we can describe the permission request.
    try:
        resolved = resolve_under_cwd(ctx.cwd, args.path)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(str(exc))

    # Request permission before applying any edits.
    approved = await ctx.request_permission(
        {
            "tool": "replace_in_file",
            "path": str(resolved),
            "description": f"Edit {args.path} ({len(args.diff)} chars of changes)",
            "diff_preview": args.diff[:500],
        }
    )
    if not approved:
        return ToolResult.failure("Permission denied by user")

    # Snapshot old contents of every file referenced in the diff so the
    # runtime can compute line deltas for the change tracker.
    old_contents: dict[str, str | None] = {}
    try:
        from justagent.agent.search_replace import parse_search_replace

        edits = parse_search_replace(args.diff)
        for edit in edits:
            if edit.filename not in old_contents:
                try:
                    fp = resolve_under_cwd(ctx.cwd, edit.filename)
                    old_contents[edit.filename] = (
                        fp.read_text(encoding="utf-8") if fp.exists() else None
                    )
                except Exception:  # noqa: BLE001
                    old_contents[edit.filename] = None
    except Exception:  # noqa: BLE001
        # If pre-parsing fails, we still let apply_search_replace try.
        pass

    try:
        result = apply_search_replace(
            args.diff,
            cwd=Path(ctx.cwd),
            restrict_to_cwd=True,
        )
    except SearchReplaceError as exc:
        return ToolResult.failure(f"Failed to parse SEARCH/REPLACE blocks: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(str(exc))

    if result.failed:
        failures = "\n\n".join(
            f"Block {i + 1} ({edit.filename}): {reason}"
            for i, (edit, reason) in enumerate(result.failed)
        )
        return ToolResult.failure(
            f"{len(result.failed)} block(s) failed to apply:\n{failures}",
            touched=result.touched,
        )

    # Build per-file change metadata for the change tracker.
    changes: list[dict[str, Any]] = []
    for fname in result.touched:
        try:
            fp = resolve_under_cwd(ctx.cwd, fname)
            new_text = fp.read_text(encoding="utf-8") if fp.exists() else ""
        except Exception:  # noqa: BLE001
            new_text = ""
        changes.append(
            {
                "path": fname,
                "old_content": old_contents.get(fname),
                "new_content": new_text,
            }
        )

    return ToolResult.success(
        f"Successfully applied {len(result.touched)} edit(s) to: "
        f"{', '.join(result.touched)}",
        touched=result.touched,
        changes=changes,
    )


def make_replace_in_file_tool() -> Tool:
    """Construct the ``replace_in_file`` tool."""

    return Tool(
        id="replace_in_file",
        description=_REPLACE_DESCRIPTION,
        parameters=ReplaceInFileInput,
        execute=_replace_execute,
        timeout_ms=30_000,
    )


__all__ = ["ReplaceInFileInput", "make_replace_in_file_tool"]
