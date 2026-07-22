"""``ask_question`` tool — ask the user a clarifying question."""

from __future__ import annotations

from pydantic import BaseModel, Field

from autoship.agent.tools.base import Tool, ToolContext, ToolResult


class AskQuestionInput(BaseModel):
    """Input for the ``ask_question`` tool."""

    question: str = Field(
        ..., description="The question to ask the user."
    )
    options: list[str] | None = Field(
        None,
        description=(
            "Optional multiple-choice options. When provided, the user "
            "selects one; the tool returns the selected option text. "
            "When omitted, the user enters a free-text answer."
        ),
    )
    default: str | None = Field(
        None,
        description="Optional default answer if the user skips the prompt.",
    )


_ASK_DESCRIPTION = """\
Ask the user a clarifying question.

Use this tool when you need information that isn't in the file system or
when you're unsure about the user's intent. The runtime displays the
question and waits for the user's response.

When ``options`` is provided, the user selects one of the listed options
(or types a custom answer); the tool returns the selected text. When
``options`` is omitted, the user enters free text.

The ``default`` field is returned if the user dismisses the prompt
without answering.

This tool is also used for permission prompts when a side-effecting tool
(write, run_command, etc.) needs user approval — in that case the
runtime constructs the question automatically and the LLM does not
need to call this tool explicitly for permissions.
"""


async def _ask_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    assert isinstance(args, AskQuestionInput)

    # If the runtime hasn't configured an ask callback, we can't actually
    # ask the user. Fall back to the default (or return an error).
    if ctx.ask is None:
        if args.default is not None:
            return ToolResult.success(
                args.default,
                source="default",
                reason="no ask callback configured",
            )
        return ToolResult.failure(
            "Cannot ask the user a question: no ask callback is "
            "configured on the tool context. Either configure a callback "
            "or supply a 'default' answer."
        )

    # Construct the permission/question request.
    request: dict[str, object] = {
        "type": "question",
        "question": args.question,
        "default": args.default,
    }
    if args.options:
        request["options"] = args.options

    try:
        answer = await ctx.ask(request)
    except Exception as exc:  # noqa: BLE001
        if args.default is not None:
            return ToolResult.success(
                args.default,
                source="default",
                reason=f"ask callback raised: {exc}",
            )
        return ToolResult.failure(f"Failed to ask question: {exc}")

    if not answer:
        # User dismissed or denied.
        if args.default is not None:
            return ToolResult.success(
                args.default, source="default", reason="user dismissed"
            )
        return ToolResult.failure("User dismissed the question without answering.")

    return ToolResult.success(str(answer), source="user")


def make_ask_question_tool() -> Tool:
    """Construct the ``ask_question`` tool."""

    return Tool(
        id="ask_question",
        description=_ASK_DESCRIPTION,
        parameters=AskQuestionInput,
        execute=_ask_execute,
        timeout_ms=0,  # no timeout — wait indefinitely for user response
        completes_run=False,
    )


__all__ = ["AskQuestionInput", "make_ask_question_tool"]
