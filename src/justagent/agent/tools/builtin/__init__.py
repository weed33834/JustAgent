"""Built-in agent tools.

Eight tools shipped with justagent, inspired by Cline's
``createDefaultTools`` and OpenCode's built-in ``Tool.define`` calls:

* :func:`make_read_file_tool` — read a file or list a directory.
* :func:`make_write_to_file_tool` — create or overwrite a file.
* :func:`make_replace_in_file_tool` — apply SEARCH/REPLACE blocks.
* :func:`make_apply_patch_tool` — apply Cline apply_patch format.
* :func:`make_run_command_tool` — execute a shell command.
* :func:`make_search_tool` — regex search across files.
* :func:`make_web_fetch_tool` — fetch a URL and return markdown.
* :func:`make_ask_question_tool` — ask the user a clarifying question.

Use :func:`make_default_tools` to construct all eight at once.
"""

from __future__ import annotations

from pathlib import Path

from justagent.agent.tools.base import Tool
from justagent.agent.tools.builtin.apply_patch import (
    ApplyPatchInput,
    make_apply_patch_tool,
)
from justagent.agent.tools.builtin.ask_question import (
    AskQuestionInput,
    make_ask_question_tool,
)
from justagent.agent.tools.builtin.edit import (
    ReplaceInFileInput,
    make_replace_in_file_tool,
)
from justagent.agent.tools.builtin.judicial import (
    JudicialInput,
    make_judicial_tool,
)
from justagent.agent.tools.builtin.read import (
    ReadFileInput,
    make_read_file_tool,
)
from justagent.agent.tools.builtin.run_command import (
    RunCommandInput,
    make_run_command_tool,
)
from justagent.agent.tools.builtin.search import (
    SearchInput,
    make_search_tool,
)
from justagent.agent.tools.builtin.web_fetch import (
    WebFetchInput,
    make_web_fetch_tool,
)
from justagent.agent.tools.builtin.write import (
    WriteToFileInput,
    make_write_to_file_tool,
)


def make_default_tools(state_path: str | None = None) -> list[Tool]:
    """Return all built-in tools in canonical order.

    The order matters for prompt-building: tools earlier in the list
    appear earlier in the LLM's system prompt.

    When ``state_path`` (the persisted judicial state file) is given, the
    ``judicial`` tool is appended so the conversational agent can manage
    cases / evidence / legal knowledge / documents directly from the chat.
    """

    tools = [
        make_read_file_tool(),
        make_write_to_file_tool(),
        make_replace_in_file_tool(),
        make_apply_patch_tool(),
        make_search_tool(),
        make_run_command_tool(),
        make_web_fetch_tool(),
        make_ask_question_tool(),
    ]
    if state_path:
        tools.append(make_judicial_tool(Path(state_path)))
    return tools


__all__ = [
    "ApplyPatchInput",
    "AskQuestionInput",
    "JudicialInput",
    "ReadFileInput",
    "ReplaceInFileInput",
    "RunCommandInput",
    "SearchInput",
    "WebFetchInput",
    "WriteToFileInput",
    "make_apply_patch_tool",
    "make_ask_question_tool",
    "make_default_tools",
    "make_judicial_tool",
    "make_read_file_tool",
    "make_replace_in_file_tool",
    "make_run_command_tool",
    "make_search_tool",
    "make_web_fetch_tool",
    "make_write_to_file_tool",
]
