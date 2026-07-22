"""``myagent ship`` 命令：按序执行 clean → verify → commit → upload 全流程。

通过 subprocess 调用各个子命令，保持隔离与复用，任一步失败即中止。
"""

from __future__ import annotations

import subprocess
import sys

import typer

from myagent.core.audit_logger import AuditLogger
from myagent.models.config import AppConfig

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.command(name="ship")(ship)


# 交付流水线的阶段顺序；每个阶段对应一个 myagent 子命令。
_STAGES: tuple[str, ...] = ("clean", "verify", "commit", "upload")


def _resolve_verify_command(config: AppConfig) -> str:
    """从 [hooks] on_save 中取 verify 命令，否则默认 pytest。"""
    for hook in config.hooks.on_save:
        if hook.command == "verify" and hook.verify_command:
            return hook.verify_command
    return "pytest"


def _run_stage(
    stage: str,
    config: AppConfig,
    audit: AuditLogger,
    verify_command: str,
    upload_target: str | None,
    dry_run: bool,
    yes: bool,
    verbose: bool,
) -> int:
    """执行单个阶段子命令，返回子进程退出码。"""
    cmd: list[str] = [sys.executable, "-m", "myagent"]
    # 全局选项透传
    if verbose:
        cmd.append("--verbose")
    if dry_run:
        cmd.append("--dry-run")
    if yes:
        cmd.append("--yes")

    cmd.append(stage)
    # 各阶段-specific 参数
    if stage == "verify":
        cmd.append(verify_command)
    elif stage == "upload":
        if upload_target:
            cmd.extend(["--target", upload_target])
        else:
            # upload 需要 --target，未提供时给出明确提示并跳过
            typer.secho(
                "ship: 未提供 --upload-target，跳过 upload 阶段。",
                fg=typer.colors.YELLOW,
                err=True,
            )
            audit.record("ship.skip", {"stage": stage, "reason": "no_upload_target"})
            return 0

    typer.secho(f"\n=== ship ▶ {stage} ===", fg=typer.colors.CYAN)
    audit.record("ship.stage_start", {"stage": stage, "cmd": cmd})

    # 直接继承当前 stdin/stdout/stderr，子命令的交互与输出如实呈现
    result = subprocess.run(cmd, cwd=str(config.project_root), check=False)
    audit.record(
        "ship.stage_done",
        {"stage": stage, "returncode": result.returncode},
    )
    return result.returncode


def ship(
    ctx: typer.Context,
    skip: list[str] = typer.Option(
        [],
        "--skip",
        "-s",
        help="跳过指定阶段（可重复，可选：clean/verify/commit/upload）",
    ),
    verify_command: str | None = typer.Option(
        None, "--verify-command", help="verify 阶段使用的命令（默认从 [hooks] 读取或 pytest）"
    ),
    upload_target: str | None = typer.Option(
        None, "--upload-target", help="upload 阶段的目标，例如 pypi/docker/github"
    ),
    stop_on_failure: bool = typer.Option(
        True, "--stop-on-failure/--no-stop-on-failure", help="某阶段失败即中止（默认开启）"
    ),
) -> None:
    """按序执行 clean → verify → commit → upload 交付流水线。"""
    config: AppConfig = ctx.obj["config"]
    audit: AuditLogger = ctx.obj["audit_logger"]
    dry_run: bool = ctx.obj.get("dry_run", False)
    yes: bool = ctx.obj.get("yes", False)
    verbose: bool = ctx.obj.get("verbose", False)

    # 校验 skip 值，避免拼写错误静默生效
    invalid = [s for s in skip if s not in _STAGES]
    if invalid:
        typer.secho(
            f"ship: 无效的 --skip 值：{invalid}。可选：{list(_STAGES)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    effective_verify = verify_command or _resolve_verify_command(config)
    stages = [s for s in _STAGES if s not in skip]

    audit.record(
        "ship.start",
        {"stages": stages, "verify_command": effective_verify, "upload_target": upload_target},
    )
    typer.secho(f"ship: 将执行阶段 {stages}", fg=typer.colors.CYAN)

    failures: list[str] = []
    for stage in stages:
        code = _run_stage(
            stage,
            config,
            audit,
            effective_verify,
            upload_target,
            dry_run,
            yes,
            verbose,
        )
        if code != 0:
            failures.append(stage)
            typer.secho(
                f"ship: 阶段 {stage} 失败（退出码 {code}）。",
                fg=typer.colors.RED,
                err=True,
            )
            if stop_on_failure:
                audit.record("ship.aborted", {"stage": stage, "returncode": code})
                raise typer.Exit(code=code)

    audit.record("ship.done", {"failures": failures})
    if failures:
        typer.secho(
            f"ship: 完成（有失败阶段：{failures}）。",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    typer.secho("ship: 全部阶段完成。", fg=typer.colors.GREEN)
