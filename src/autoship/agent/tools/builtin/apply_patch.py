"""``apply_patch`` tool — apply Cline-style ``*** Begin Patch`` format."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from autoship.agent.patch import DiffError, apply_patch_text
from autoship.agent.tools.base import Tool, ToolContext, ToolResult


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

    try:
        touched, _count = apply_patch_text(
            args.patch, cwd=Path(ctx.cwd), restrict_to_cwd=True
        )
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
        f"Successfully applied patch to {len(touched)} file(s): "
        f"{', '.join(touched)}",
        touched=touched,
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
