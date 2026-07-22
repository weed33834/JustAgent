"""``run_command`` tool — execute a shell command."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from autoship.agent.tools.base import Tool, ToolAbortedError, ToolContext, ToolResult
from autoship.agent.tools.builtin._paths import resolve_under_cwd


class RunCommandInput(BaseModel):
    """Input for the ``run_command`` tool."""

    command: str = Field(..., description="The shell command to execute.")
    cwd: str | None = Field(
        None,
        description="Working directory (defaults to the agent's cwd).",
    )
    timeout_ms: int | None = Field(
        None,
        description="Per-call timeout override in milliseconds.",
        ge=1,
    )


_RUN_DESCRIPTION = """\
Execute a shell command and return its stdout/stderr.

The command runs via ``/bin/sh -c`` (POSIX) or ``cmd /c`` (Windows).
Both stdout and stderr are captured and returned, with a clear
``[exit code: N]`` footer.

Set ``timeout_ms`` to override the default 60-second timeout. The
agent's abort signal (Ctrl-C) cancels the running process.

For long-running commands, prefer appending ``&`` (background) or
using ``nohup`` so the command survives tool timeout — the tool will
return immediately with a "still running" notice.

DANGEROUS: This tool executes arbitrary commands. The runtime should
prompt for user permission before invoking it (via the ``ask``
callback on the tool context).
"""

DEFAULT_TIMEOUT_MS = 60_000
MAX_OUTPUT_BYTES = 50_000


async def _run_execute(args: BaseModel, ctx: ToolContext) -> ToolResult:
    assert isinstance(args, RunCommandInput)

    # Resolve working directory.
    if args.cwd:
        try:
            cwd_path = resolve_under_cwd(ctx.cwd, args.cwd)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(str(exc))
        cwd_str = str(cwd_path)
    else:
        cwd_str = ctx.cwd

    timeout = (args.timeout_ms or DEFAULT_TIMEOUT_MS) / 1000

    # Run via /bin/sh -c on POSIX, cmd /c on Windows.
    import sys

    if sys.platform == "win32":
        shell_args = ["cmd", "/c", args.command]
    else:
        shell_args = ["/bin/sh", "-c", args.command]

    try:
        proc = await asyncio.create_subprocess_exec(
            *shell_args,
            cwd=cwd_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return ToolResult.failure(
            f"Failed to spawn command {args.command!r}: {exc}"
        )

    # Poll for completion, abort signal, or timeout.
    try:
        stdout_bytes, stderr_bytes = await _wait_for_output(
            proc, timeout=timeout, ctx=ctx
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return ToolResult.failure(
            f"Command timed out after {int(timeout * 1000)}ms: "
            f"{args.command!r}"
        )
    except ToolAbortedError:
        proc.kill()
        await proc.wait()
        raise

    exit_code = proc.returncode or 0

    # Decode and truncate output.
    stdout = stdout_bytes[:MAX_OUTPUT_BYTES].decode(
        "utf-8", errors="replace"
    )
    stderr = stderr_bytes[:MAX_OUTPUT_BYTES].decode(
        "utf-8", errors="replace"
    )

    output_parts: list[str] = []
    if stdout:
        output_parts.append(stdout)
    if stderr:
        output_parts.append(f"[stderr]\n{stderr}")
    output_parts.append(f"[exit code: {exit_code}]")

    output = "\n".join(output_parts)
    metadata: dict[str, object] = {
        "exit_code": exit_code,
        "command": args.command,
        "cwd": cwd_str,
    }
    if len(stdout_bytes) > MAX_OUTPUT_BYTES or len(stderr_bytes) > MAX_OUTPUT_BYTES:
        metadata["truncated"] = True

    if exit_code != 0:
        return ToolResult(
            output=output,
            error=f"Command exited with non-zero status: {exit_code}",
            metadata=metadata,
        )

    return ToolResult(output=output, metadata=metadata)


async def _wait_for_output(
    proc: asyncio.subprocess.Process,
    *,
    timeout: float,
    ctx: ToolContext,
) -> tuple[bytes, bytes]:
    """Wait for the process to finish, with timeout and abort handling."""

    # Create three tasks: process completion, timeout, abort signal.
    proc_task = asyncio.create_task(proc.communicate())
    timeout_task = asyncio.create_task(asyncio.sleep(timeout))

    async def _wait_abort() -> None:
        await ctx.abort.wait()

    abort_task = asyncio.create_task(_wait_abort())

    try:
        done, pending = await asyncio.wait(
            {proc_task, timeout_task, abort_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if proc_task in done:
            return proc_task.result()
        if abort_task in done:
            raise ToolAbortedError(
                f"Command aborted by user: {ctx.tool_call_id}"
            )
        # Timeout fired.
        raise TimeoutError()
    finally:
        import contextlib

        for task in (proc_task, timeout_task, abort_task):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, BaseException):
                    await task


def make_run_command_tool() -> Tool:
    """Construct the ``run_command`` tool."""

    return Tool(
        id="run_command",
        description=_RUN_DESCRIPTION,
        parameters=RunCommandInput,
        execute=_run_execute,
        timeout_ms=0,  # use per-call timeout via args, not Tool-level
    )


__all__ = ["DEFAULT_TIMEOUT_MS", "RunCommandInput", "make_run_command_tool"]
