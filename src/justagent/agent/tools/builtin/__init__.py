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

Use :func:`make_default_tools` to construct all eight at once. Tools
contributed by installed vertical packages (entry-point group
``justagent.tools``) are appended automatically when a state root is given;
the engine never imports a vertical directly.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points
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


def vertical_tool_factories() -> list:
    """Discover tool factories published by vertical packages.

    Verticals register via the ``justagent.tools`` entry-point group; each
    entry must resolve to a callable ``factory(state_root) -> Tool | None``.
    A broken vertical logs a warning and is skipped — it can never prevent
    the agent from starting.
    """
    factories = []
    eps = entry_points(group="justagent.tools")
    for ep in eps:
        try:
            obj = ep.load()
            if callable(obj):
                factories.append(obj)
        except Exception as exc:  # noqa: BLE001 - broken optional integration
            warnings.warn(f"Skipping tool entry point {ep.name}: {exc}", RuntimeWarning, stacklevel=2)
    return factories


def make_default_tools(project_root: str | None = None) -> list[Tool]:
    """Return all built-in tools in canonical order.

    The order matters for prompt-building: tools earlier in the list
    appear earlier in the LLM's system prompt.

    When ``project_root`` is given, tools contributed by installed verticals
    (entry-point group ``justagent.tools``) are appended after the built-ins.
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
    if project_root:
        root = Path(project_root)
        for factory in vertical_tool_factories():
            try:
                tool = factory(root)
            except Exception as exc:  # noqa: BLE001 - broken optional integration
                warnings.warn(f"Vertical tool factory {factory} failed: {exc}", RuntimeWarning, stacklevel=2)
                continue
            if tool is not None:
                tools.append(tool)
    return tools


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
