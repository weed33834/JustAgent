"""The ``autoship upload`` command."""

from __future__ import annotations

import subprocess
from typing import Any, cast

import typer

from autoship.adapters.upload import get_uploader
from autoship.core.audit_logger import AuditLogger
from autoship.core.context import CommandContext
from autoship.exceptions import ConfigError, UploadError
from autoship.plugin_manager import manager as plugin_manager

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.command(name="upload", help="将产物上传到已配置的目标。")(upload)


@app.command(name="upload")
def upload(
    ctx: typer.Context,
    target: str = typer.Option(..., "--target", help="上传目标，例如 pypi/docker/github"),
    image: str | None = typer.Option(None, "--image", help="Docker 镜像名称"),
    tag: str | None = typer.Option(None, "--tag", "-t", help="Docker 镜像标签或 GitHub 发布标签"),
    artifacts: list[str] | None = typer.Option(None, "--artifact", help="要上传的产物"),
    repository: str | None = typer.Option(None, "--repository", help="PyPI 仓库名称（默认：testpypi）"),
    repository_url: str | None = typer.Option(
        None, "--repository-url", help="PyPI 仓库上传地址"
    ),
    registry: str | None = typer.Option(None, "--registry", help="Docker 仓库前缀（例如 localhost:5000）"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="仅显示将要上传的内容而不实际执行"),
) -> None:
    """将产物上传到已配置的目标。"""
    from autoship.adapters.upload.pypi import PyPIUploader

    config = ctx.obj["config"]

    audit: AuditLogger = ctx.obj["audit_logger"]

    # 合并全局 --dry-run 与命令局部 --dry-run。
    # 直接单元测试调用可能传入 typer.OptionInfo 作为默认值，此时按 True 判断。
    local_dry_run: bool = dry_run is True
    dry_run = ctx.obj.get("dry_run", False) or local_dry_run
    yes: bool = ctx.obj.get("yes", False)
    verbose: bool = ctx.obj.get("verbose", False)

    # 直接单元测试调用可能收到 typer.Option 对象作为默认值。
    image = image if isinstance(image, str) else None
    tag = tag if isinstance(tag, str) else None
    artifacts = artifacts if isinstance(artifacts, list) else None
    repository = repository if isinstance(repository, str) else None
    repository_url = repository_url if isinstance(repository_url, str) else None
    registry = registry if isinstance(registry, str) else None

    uploader_cfg: dict[str, Any] = {"target": target}
    if image:
        uploader_cfg["image"] = image
    if tag:
        uploader_cfg["tag"] = tag
    if artifacts:
        uploader_cfg["artifacts"] = artifacts
    if repository:
        uploader_cfg["repository"] = repository
    if repository_url:
        if not PyPIUploader.is_safe_repository_url(repository_url):
            raise UploadError("--repository-url 必须使用 HTTPS，或指向 localhost/127.0.0.1")
        uploader_cfg["repository_url"] = repository_url
    if registry:
        uploader_cfg["registry"] = registry

    context = CommandContext(
        command="upload",
        project_root=config.project_root,
        config=config,
        dry_run=dry_run,
        yes=yes,
        trace_id=audit.trace_id,
        extras=uploader_cfg,
    )

    audit.record("upload.start", {"target": target, "config": uploader_cfg})
    plugin_manager.call("pre_upload", context=context, fail_fast=False)

    try:
        uploader = get_uploader(target, config.project_root, uploader_cfg, tools=config.tools)
    except ConfigError as exc:
        if dry_run:
            typer.echo(f"[dry-run] 目标 '{target}' 尚未完整配置：{exc}")
            audit.record("upload.dry_run_not_configured", {"target": target, "reason": str(exc)})
            return
        raise

    if not dry_run and not yes and not typer.confirm(f"上传到 {target}？"):
        typer.echo("已中止。")
        audit.record("upload.aborted", {"reason": "user_declined"})
        raise typer.Exit(code=0)

    try:
        result = uploader.upload(dry_run=dry_run, verbose=verbose)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        error = UploadError(f"上传到 {target} 失败：{exc}")
        audit.record(
            "upload.error",
            {"target": target, "error": str(exc)},
        )
        plugin_manager.call("on_error", context=context, error=error, fail_fast=False)
        raise error from exc

    if dry_run:
        audit.record(
            "upload.dry_run",
            {"target": target, "details": result.details},
        )
        plugin_manager.call("post_upload", context=context, fail_fast=False)
        details_str = _format_dry_run_details(result.details)
        typer.echo(f"[dry-run] 将上传到 {target}：\n{details_str}")
        return

    audit.record("upload.done", {"target": target, "result": result.details})
    plugin_manager.call("post_upload", context=context, fail_fast=False)
    if result.url:
        typer.echo(f"已上传到 {target}：{result.url}")
    else:
        typer.echo(f"已上传到 {target}")


def _format_dry_run_details(details: dict[str, Any] | None) -> str:
    """Format dry-run details for terminal output."""
    if not details:
        return ""
    lines: list[str] = []
    for key, value in details.items():
        if key == "dry_run":
            continue
        if isinstance(value, list):
            seq = cast(list[object], value)
            parts: list[str] = [str(v) for v in seq]
            display_value = ", ".join(parts)
        else:
            display_value = str(value)
        lines.append(f"  {key}: {display_value}")
    return "\n".join(lines)
