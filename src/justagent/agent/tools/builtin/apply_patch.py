"""``apply_patch`` tool — apply Cline-style ``*** Begin Patch`` format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from justagent.agent.patch import DiffError, apply_patch_text, compute_patch_changes
from justagent.agent.tools.base import Tool, ToolContext, ToolResult


class ApplyPatchInput(BaseModel):
    """Input for the ``apply_patch`` tool."""

    patch: str = Field(
        ...,
        description=(
            "The patch text in Cline apply_patch format. Begins with "
            "``*** Begin Patch`` and ends with ``*** End Patch``. "
            "Supports Add File / Delete File / Update File actions."
        ),
    )


_PATCH_DESCRIPTION = """\
Apply a multi-file patch in the Cline ``apply_patch`` format.

The ``patch`` parameter must be a single patch block delimited by
``*** Begin Patch`` and ``*** End Patch`` sentinels. Inside, each action
is introduced by:

* ``*** Add File: <path>`` — create a new file (must not already exist).
* ``*** Delete File: <path>`` — remove an existing file.
* ``*** Update File: <path>`` — modify an existing file with context hunks.

Update hunks use ``@@`` section markers and ``*** End of File`` anchors.
Fuzzy matching (similarity >= 0.66) is used when context lines don't
exactly match — useful for tolerant LLM outputs.

All actions are applied atomically: if any action fails to parse or
apply, no files are written.
"""


async def _patch_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    assert isinstance(args, ApplyPatchInput)

    # Request permission before applying any file changes.
    approved = await ctx.request_permission(
        {
            "tool": "apply_patch",
            "description": f"Apply patch ({len(args.patch)} chars)",
            "patch_preview": args.patch[:500],
        }
    )
    if not approved:
        return ToolResult.failure("Permission denied by user")

    # Pre-compute the per-file changes (without writing) so we can
    # capture old/new content for the change tracker.
    changes_meta: list[dict[str, Any]] = []
    try:
        computed, _fuzz = compute_patch_changes(args.patch, cwd=Path(ctx.cwd), restrict_to_cwd=True)
        for fpath, change in computed.items():
            changes_meta.append(
                {
                    "path": fpath,
                    "action": (
                        "created"
                        if change.old_content is None and change.new_content is not None
                        else "deleted"
                        if change.new_content is None
                        else "modified"
                    ),
                    "old_content": change.old_content,
                    "new_content": change.new_content,
                }
            )
    except Exception:  # noqa: BLE001
        # If pre-computation fails, proceed without change metadata —
        # apply_patch_text will raise the real error below.
        pass

    try:
        touched, _count = apply_patch_text(args.patch, cwd=Path(ctx.cwd), restrict_to_cwd=True)
    except DiffError as exc:
        return ToolResult.failure(f"Failed to apply patch: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(str(exc))

    if not touched:
        return ToolResult.success(
            "Patch applied (no files were changed)",
            touched=[],
        )

    return ToolResult.success(
        f"Successfully applied patch to {len(touched)} file(s): {', '.join(touched)}",
        touched=touched,
        changes=changes_meta,
    )


def make_apply_patch_tool() -> Tool:
    """Construct the ``apply_patch`` tool."""

    return Tool(
        id="apply_patch",
        description=_PATCH_DESCRIPTION,
        parameters=ApplyPatchInput,
        execute=_patch_execute,
        timeout_ms=60_000,
    )


__all__ = ["ApplyPatchInput", "make_apply_patch_tool"]
