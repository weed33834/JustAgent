"""Built-in agent tools.

Eight tools shipped with myagent, inspired by Cline's
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

from myagent.agent.tools.base import Tool
from myagent.agent.tools.builtin.apply_patch import (
    ApplyPatchInput,
    make_apply_patch_tool,
)
from myagent.agent.tools.builtin.ask_question import (
    AskQuestionInput,
    make_ask_question_tool,
)
from myagent.agent.tools.builtin.edit import (
    ReplaceInFileInput,
    make_replace_in_file_tool,
)
from myagent.agent.tools.builtin.read import (
    ReadFileInput,
    make_read_file_tool,
)
from myagent.agent.tools.builtin.run_command import (
    RunCommandInput,
    make_run_command_tool,
)
from myagent.agent.tools.builtin.search import (
    SearchInput,
    make_search_tool,
)
from myagent.agent.tools.builtin.web_fetch import (
    WebFetchInput,
    make_web_fetch_tool,
)
from myagent.agent.tools.builtin.write import (
    WriteToFileInput,
    make_write_to_file_tool,
)


def make_default_tools() -> list[Tool]:
    """Return all eight built-in tools in canonical order.

    The order matters for prompt-building: tools earlier in the list
    appear earlier in the LLM's system prompt.
    """

    return [
        make_read_file_tool(),
        make_write_to_file_tool(),
        make_replace_in_file_tool(),
        make_apply_patch_tool(),
        make_search_tool(),
        make_run_command_tool(),
        make_web_fetch_tool(),
        make_ask_question_tool(),
    ]


__all__ = [
    "ApplyPatchInput",
    "AskQuestionInput",
    "ReadFileInput",
    "ReplaceInFileInput",
    "RunCommandInput",
    "SearchInput",
    "WebFetchInput",
    "WriteToFileInput",
    "make_apply_patch_tool",
    "make_ask_question_tool",
    "make_default_tools",
    "make_read_file_tool",
    "make_replace_in_file_tool",
    "make_run_command_tool",
    "make_search_tool",
    "make_web_fetch_tool",
    "make_write_to_file_tool",
]
