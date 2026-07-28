"""Integration tests for the permission system.

Verifies that the destructive built-in tools (``write_to_file``,
``replace_in_file``, ``run_command``, ``apply_patch``) consult
``ctx.request_permission()`` before executing, and that backward
compatibility is preserved when no ``ask`` callback is wired (the
default-allow behaviour).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from justagent.agent.tools.base import ToolContext
from justagent.agent.tools.builtin.apply_patch import make_apply_patch_tool
from justagent.agent.tools.builtin.edit import make_replace_in_file_tool
from justagent.agent.tools.builtin.run_command import make_run_command_tool
from justagent.agent.tools.builtin.write import make_write_to_file_tool

# ---------------------------------------------------------------------------
# Mock ask callbacks
# ---------------------------------------------------------------------------


async def _mock_ask_approved(_request: dict[str, Any]) -> bool:
    """Always approve the permission request."""

    return True


async def _mock_ask_denied(_request: dict[str, Any]) -> bool:
    """Always deny the permission request."""

    return False


def _mock_ask_capture(captured: list[dict[str, Any]]):
    """Return an ask callback that records each request then approves."""

    async def _ask(request: dict[str, Any]) -> bool:
        captured.append(request)
        return True

    return _ask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    cwd: str | Path,
    *,
    ask: Any = None,
) -> ToolContext:
    """Build a ToolContext, optionally wiring an ``ask`` callback."""

    return ToolContext(
        tool_call_id="call-1",
        iteration=1,
        cwd=str(cwd),
        ask=ask,
    )


def _block(fname: str, search: str, replace: str) -> str:
    """Build a single SEARCH/REPLACE block for the edit tool."""

    return "\n".join(
        [
            fname,
            "```",
            "<<<<<<< SEARCH",
            search,
            "=======",
            replace,
            ">>>>>>> REPLACE",
            "```",
        ]
    )


def _patch(body: str) -> str:
    """Wrap a patch body in the Cline Begin/End Patch sentinels."""

    body = body.rstrip("\n")
    if body:
        return f"*** Begin Patch\n{body}\n*** End Patch\n"
    return "*** Begin Patch\n*** End Patch\n"


# ---------------------------------------------------------------------------
# write_to_file
# ---------------------------------------------------------------------------


class TestWriteToFilePermission:
    @pytest.mark.asyncio
    async def test_calls_request_permission_when_ask_provided(
        self, tmp_path: Path
    ) -> None:
        """write_to_file must consult the ask callback before writing."""

        captured: list[dict[str, Any]] = []
        tool = make_write_to_file_tool()
        result = await tool.invoke(
            {"path": "new.txt", "content": "hello\n"},
            _make_ctx(tmp_path, ask=_mock_ask_capture(captured)),
        )
        assert not result.is_error
        assert (tmp_path / "new.txt").read_text() == "hello\n"
        # The ask callback was invoked exactly once.
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_auto_approves_when_ask_is_none(
        self, tmp_path: Path
    ) -> None:
        """When ask is None, the tool should still work (backward compat)."""

        tool = make_write_to_file_tool()
        result = await tool.invoke(
            {"path": "new.txt", "content": "hello\n"},
            _make_ctx(tmp_path),  # ask=None
        )
        assert not result.is_error
        assert (tmp_path / "new.txt").read_text() == "hello\n"

    @pytest.mark.asyncio
    async def test_denied_permission_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """A denied permission must prevent the write and return failure."""

        tool = make_write_to_file_tool()
        result = await tool.invoke(
            {"path": "new.txt", "content": "hello\n"},
            _make_ctx(tmp_path, ask=_mock_ask_denied),
        )
        assert result.is_error
        assert "Permission denied" in (result.error or "")
        # The file must NOT have been created.
        assert not (tmp_path / "new.txt").exists()

    @pytest.mark.asyncio
    async def test_ask_receives_correct_request_format(
        self, tmp_path: Path
    ) -> None:
        """The request dict must include tool, path, description, is_new_file."""

        captured: list[dict[str, Any]] = []
        tool = make_write_to_file_tool()
        await tool.invoke(
            {"path": "new.txt", "content": "hello\n"},
            _make_ctx(tmp_path, ask=_mock_ask_capture(captured)),
        )
        req = captured[0]
        assert req["tool"] == "write_to_file"
        assert req["path"] == str(tmp_path / "new.txt")
        assert "description" in req
        assert "hello" not in req["description"]  # description is the summary
        assert "Write" in req["description"]
        assert req["is_new_file"] is True

    @pytest.mark.asyncio
    async def test_is_new_file_false_for_existing_file(
        self, tmp_path: Path
    ) -> None:
        """is_new_file should be False when overwriting an existing file."""

        (tmp_path / "existing.txt").write_text("old\n")
        captured: list[dict[str, Any]] = []
        tool = make_write_to_file_tool()
        await tool.invoke(
            {"path": "existing.txt", "content": "new\n"},
            _make_ctx(tmp_path, ask=_mock_ask_capture(captured)),
        )
        assert captured[0]["is_new_file"] is False


# ---------------------------------------------------------------------------
# replace_in_file (edit)
# ---------------------------------------------------------------------------


class TestReplaceInFilePermission:
    @pytest.mark.asyncio
    async def test_calls_request_permission_when_ask_provided(
        self, tmp_path: Path
    ) -> None:
        """replace_in_file must consult the ask callback before editing."""

        (tmp_path / "f.py").write_text("def hello():\n    return 1\n")
        captured: list[dict[str, Any]] = []
        tool = make_replace_in_file_tool()
        diff = _block("f.py", "    return 1", "    return 2")
        result = await tool.invoke(
            {"path": "f.py", "diff": diff},
            _make_ctx(tmp_path, ask=_mock_ask_capture(captured)),
        )
        assert not result.is_error
        assert (tmp_path / "f.py").read_text() == "def hello():\n    return 2\n"
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_auto_approves_when_ask_is_none(
        self, tmp_path: Path
    ) -> None:
        """When ask is None, the edit should still apply (backward compat)."""

        (tmp_path / "f.py").write_text("def hello():\n    return 1\n")
        tool = make_replace_in_file_tool()
        diff = _block("f.py", "    return 1", "    return 2")
        result = await tool.invoke(
            {"path": "f.py", "diff": diff},
            _make_ctx(tmp_path),  # ask=None
        )
        assert not result.is_error
        assert (tmp_path / "f.py").read_text() == "def hello():\n    return 2\n"

    @pytest.mark.asyncio
    async def test_denied_permission_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """A denied permission must prevent the edit."""

        (tmp_path / "f.py").write_text("def hello():\n    return 1\n")
        tool = make_replace_in_file_tool()
        diff = _block("f.py", "    return 1", "    return 2")
        result = await tool.invoke(
            {"path": "f.py", "diff": diff},
            _make_ctx(tmp_path, ask=_mock_ask_denied),
        )
        assert result.is_error
        assert "Permission denied" in (result.error or "")
        # The file must be unchanged.
        assert (tmp_path / "f.py").read_text() == "def hello():\n    return 1\n"

    @pytest.mark.asyncio
    async def test_ask_receives_correct_request_format(
        self, tmp_path: Path
    ) -> None:
        """The request dict must include tool, path, description, diff_preview."""

        (tmp_path / "f.py").write_text("def hello():\n    return 1\n")
        captured: list[dict[str, Any]] = []
        tool = make_replace_in_file_tool()
        diff = _block("f.py", "    return 1", "    return 2")
        await tool.invoke(
            {"path": "f.py", "diff": diff},
            _make_ctx(tmp_path, ask=_mock_ask_capture(captured)),
        )
        req = captured[0]
        assert req["tool"] == "replace_in_file"
        assert req["path"] == str(tmp_path / "f.py")
        assert "description" in req
        assert "Edit" in req["description"]
        assert "diff_preview" in req
        assert "SEARCH" in req["diff_preview"]


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------


class TestRunCommandPermission:
    @pytest.mark.asyncio
    async def test_calls_request_permission_when_ask_provided(
        self, tmp_path: Path
    ) -> None:
        """run_command must consult the ask callback before spawning."""

        captured: list[dict[str, Any]] = []
        tool = make_run_command_tool()
        result = await tool.invoke(
            {"command": "echo hello"},
            _make_ctx(tmp_path, ask=_mock_ask_capture(captured)),
        )
        assert not result.is_error
        assert "hello" in result.output
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_auto_approves_when_ask_is_none(
        self, tmp_path: Path
    ) -> None:
        """When ask is None, the command should still run (backward compat)."""

        tool = make_run_command_tool()
        result = await tool.invoke(
            {"command": "echo hello"},
            _make_ctx(tmp_path),  # ask=None
        )
        assert not result.is_error
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_denied_permission_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """A denied permission must prevent the command from running."""

        tool = make_run_command_tool()
        result = await tool.invoke(
            {"command": "echo hello"},
            _make_ctx(tmp_path, ask=_mock_ask_denied),
        )
        assert result.is_error
        assert "Permission denied" in (result.error or "")

    @pytest.mark.asyncio
    async def test_ask_receives_correct_request_format(
        self, tmp_path: Path
    ) -> None:
        """The request dict must include tool, command, description."""

        captured: list[dict[str, Any]] = []
        tool = make_run_command_tool()
        await tool.invoke(
            {"command": "echo hello"},
            _make_ctx(tmp_path, ask=_mock_ask_capture(captured)),
        )
        req = captured[0]
        assert req["tool"] == "run_command"
        assert req["command"] == "echo hello"
        assert "description" in req
        assert "Run" in req["description"]


# ---------------------------------------------------------------------------
# apply_patch
# ---------------------------------------------------------------------------


class TestApplyPatchPermission:
    @pytest.mark.asyncio
    async def test_calls_request_permission_when_ask_provided(
        self, tmp_path: Path
    ) -> None:
        """apply_patch must consult the ask callback before applying."""

        captured: list[dict[str, Any]] = []
        tool = make_apply_patch_tool()
        patch = _patch("*** Add File: new.txt\n+hello\n")
        result = await tool.invoke(
            {"patch": patch},
            _make_ctx(tmp_path, ask=_mock_ask_capture(captured)),
        )
        assert not result.is_error
        assert (tmp_path / "new.txt").read_text() == "hello"
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_auto_approves_when_ask_is_none(
        self, tmp_path: Path
    ) -> None:
        """When ask is None, the patch should still apply (backward compat)."""

        tool = make_apply_patch_tool()
        patch = _patch("*** Add File: new.txt\n+hello\n")
        result = await tool.invoke(
            {"patch": patch},
            _make_ctx(tmp_path),  # ask=None
        )
        assert not result.is_error
        assert (tmp_path / "new.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_denied_permission_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """A denied permission must prevent the patch from applying."""

        tool = make_apply_patch_tool()
        patch = _patch("*** Add File: new.txt\n+hello\n")
        result = await tool.invoke(
            {"patch": patch},
            _make_ctx(tmp_path, ask=_mock_ask_denied),
        )
        assert result.is_error
        assert "Permission denied" in (result.error or "")
        assert not (tmp_path / "new.txt").exists()

    @pytest.mark.asyncio
    async def test_ask_receives_correct_request_format(
        self, tmp_path: Path
    ) -> None:
        """The request dict must include tool, description, patch_preview."""

        captured: list[dict[str, Any]] = []
        tool = make_apply_patch_tool()
        patch = _patch("*** Add File: new.txt\n+hello\n")
        await tool.invoke(
            {"patch": patch},
            _make_ctx(tmp_path, ask=_mock_ask_capture(captured)),
        )
        req = captured[0]
        assert req["tool"] == "apply_patch"
        assert "description" in req
        assert "Apply patch" in req["description"]
        assert "patch_preview" in req
        assert "Begin Patch" in req["patch_preview"]


# ---------------------------------------------------------------------------
# Cross-tool: denied permission never mutates state
# ---------------------------------------------------------------------------


class TestDeniedPermissionNoSideEffects:
    @pytest.mark.asyncio
    async def test_write_denied_leaves_no_file(self, tmp_path: Path) -> None:
        tool = make_write_to_file_tool()
        result = await tool.invoke(
            {"path": "deep/nested/file.txt", "content": "x"},
            _make_ctx(tmp_path, ask=_mock_ask_denied),
        )
        assert result.is_error
        assert not (tmp_path / "deep").exists()

    @pytest.mark.asyncio
    async def test_request_permission_default_allow_when_ask_none(
        self, tmp_path: Path
    ) -> None:
        """ctx.request_permission returns True when ask is None."""

        ctx = _make_ctx(tmp_path)
        approved = await ctx.request_permission({"tool": "any"})
        assert approved is True

    @pytest.mark.asyncio
    async def test_request_permission_calls_ask(self, tmp_path: Path) -> None:
        """ctx.request_permission delegates to the ask callback."""

        ctx = _make_ctx(tmp_path, ask=_mock_ask_denied)
        approved = await ctx.request_permission({"tool": "any"})
        assert approved is False

        ctx2 = _make_ctx(tmp_path, ask=_mock_ask_approved)
        approved2 = await ctx2.request_permission({"tool": "any"})
        assert approved2 is True
