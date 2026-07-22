"""Tests for the ``run_command`` built-in tool."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from myagent.agent.tools.base import (
    InvalidArgumentsError,
    ToolAbortedError,
    ToolContext,
)
from myagent.agent.tools.builtin.run_command import (
    DEFAULT_TIMEOUT_MS,
    RunCommandInput,
    make_run_command_tool,
)


def _make_ctx(
    cwd: str | Path,
    *,
    abort: asyncio.Event | None = None,
) -> ToolContext:
    ctx = ToolContext(
        tool_call_id="call-1",
        iteration=1,
        cwd=str(cwd),
    )
    if abort is not None:
        ctx.abort = abort
    return ctx


@pytest.mark.asyncio
async def test_run_command_echo(tmp_path: Path) -> None:
    tool = make_run_command_tool()
    cmd = 'echo "hello world"' if sys.platform != "win32" else 'echo "hello world"'
    result = await tool.invoke({"command": cmd}, _make_ctx(tmp_path))
    assert not result.is_error
    assert "hello world" in result.output
    assert "[exit code: 0]" in result.output
    assert result.metadata["exit_code"] == 0


@pytest.mark.asyncio
async def test_run_command_writes_to_cwd(tmp_path: Path) -> None:
    tool = make_run_command_tool()
    cmd = "echo content > out.txt" if sys.platform != "win32" else "echo content > out.txt"
    result = await tool.invoke({"command": cmd}, _make_ctx(tmp_path))
    assert not result.is_error
    assert (tmp_path / "out.txt").exists()


@pytest.mark.asyncio
async def test_run_command_nonzero_exit(tmp_path: Path) -> None:
    tool = make_run_command_tool()
    cmd = "exit 42" if sys.platform != "win32" else "exit /b 42"
    result = await tool.invoke({"command": cmd}, _make_ctx(tmp_path))
    # Non-zero exit produces a result with error set (not an exception).
    assert result.is_error
    assert "42" in result.output or "42" in (result.error or "")


@pytest.mark.asyncio
async def test_run_command_with_custom_cwd(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    tool = make_run_command_tool()
    cmd = "pwd" if sys.platform != "win32" else "cd"
    result = await tool.invoke(
        {"command": cmd, "cwd": "sub"}, _make_ctx(tmp_path)
    )
    assert not result.is_error
    assert "sub" in result.output


@pytest.mark.asyncio
async def test_run_command_rejects_cwd_escape(tmp_path: Path) -> None:
    tool = make_run_command_tool()
    cmd = "echo hi" if sys.platform != "win32" else "echo hi"
    result = await tool.invoke(
        {"command": cmd, "cwd": "../escape"}, _make_ctx(tmp_path)
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_run_command_timeout(tmp_path: Path) -> None:
    tool = make_run_command_tool()
    # sleep for 2 seconds, timeout after 100ms.
    cmd = "sleep 2" if sys.platform != "win32" else "timeout /t 2 /nobreak"
    result = await tool.invoke(
        {"command": cmd, "timeout_ms": 100}, _make_ctx(tmp_path)
    )
    assert result.is_error
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_run_command_abort_signal(tmp_path: Path) -> None:
    tool = make_run_command_tool()
    abort = asyncio.Event()
    ctx = _make_ctx(tmp_path, abort=abort)

    # Trigger abort almost immediately.
    async def _trigger() -> None:
        await asyncio.sleep(0.1)
        abort.set()

    trigger_task = asyncio.create_task(_trigger())
    try:
        cmd = "sleep 5" if sys.platform != "win32" else "timeout /t 5 /nobreak"
        with pytest.raises(ToolAbortedError):
            await tool.invoke({"command": cmd, "timeout_ms": 10_000}, ctx)
    finally:
        await trigger_task


@pytest.mark.asyncio
async def test_run_command_input_validation(tmp_path: Path) -> None:
    tool = make_run_command_tool()
    with pytest.raises(InvalidArgumentsError):
        await tool.invoke({}, _make_ctx(tmp_path))


@pytest.mark.asyncio
async def test_run_command_input_validation_timeout_zero(tmp_path: Path) -> None:
    """timeout_ms must be >= 1."""
    tool = make_run_command_tool()
    with pytest.raises(InvalidArgumentsError):
        await tool.invoke(
            {"command": "echo hi", "timeout_ms": 0}, _make_ctx(tmp_path)
        )


@pytest.mark.asyncio
async def test_run_command_stderr_captured(tmp_path: Path) -> None:
    tool = make_run_command_tool()
    cmd = 'echo "err msg" >&2' if sys.platform != "win32" else 'echo "err msg" 1>&2'
    result = await tool.invoke({"command": cmd}, _make_ctx(tmp_path))
    assert "err msg" in result.output


@pytest.mark.asyncio
async def test_run_command_json_schema() -> None:
    tool = make_run_command_tool()
    schema = tool.json_schema()
    assert schema["type"] == "object"
    assert "command" in schema["properties"]
    assert "cwd" in schema["properties"]
    assert "timeout_ms" in schema["properties"]


def test_run_command_input_model() -> None:
    inp = RunCommandInput(command="ls")
    assert inp.command == "ls"
    assert inp.cwd is None
    assert inp.timeout_ms is None


def test_make_run_command_tool_metadata() -> None:
    tool = make_run_command_tool()
    assert tool.id == "run_command"
    # timeout_ms=0 means "use per-call timeout via args".
    assert tool.timeout_ms == 0
    assert tool.completes_run is False


def test_default_timeout_constant() -> None:
    assert DEFAULT_TIMEOUT_MS == 60_000
