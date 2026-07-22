"""``replace_in_file`` tool — apply SEARCH/REPLACE blocks to a file."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from autoship.agent.search_replace import (
    SearchReplaceError,
    apply_search_replace,
)
from autoship.agent.tools.base import Tool, ToolContext, ToolResult


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

    return ToolResult.success(
        f"Successfully applied {len(result.touched)} edit(s) to: "
        f"{', '.join(result.touched)}",
        touched=result.touched,
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
