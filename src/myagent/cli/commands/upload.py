"""The ``myagent upload`` command."""

from __future__ import annotations

import subprocess
from typing import Any, cast

import typer

from myagent.adapters.upload import get_uploader
from myagent.core.audit_logger import AuditLogger
from myagent.core.context import CommandContext
from myagent.core.i18n import I18n, get_i18n_from_ctx
from myagent.exceptions import ConfigError, UploadError
from myagent.plugin_manager import manager as plugin_manager

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.command(name="upload", help="Upload artifacts to a configured target.")(upload)


@app.command(name="upload")
def upload(
    ctx: typer.Context,
    target: str = typer.Option(..., "--target", help="Upload target, e.g. pypi/docker/github"),
    image: str | None = typer.Option(None, "--image", help="Docker image name"),
    tag: str | None = typer.Option(None, "--tag", "-t", help="Docker image tag or GitHub release tag"),
    artifacts: list[str] | None = typer.Option(None, "--artifact", help="Artifacts to upload"),
    repository: str | None = typer.Option(None, "--repository", help="PyPI repository name (default: testpypi)"),
    repository_url: str | None = typer.Option(
        None, "--repository-url", help="PyPI repository upload URL"
    ),
    registry: str | None = typer.Option(None, "--registry", help="Docker registry prefix (e.g. localhost:5000)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be uploaded without uploading"),
) -> None:
    """Upload artifacts to a configured target."""
    from myagent.adapters.upload.pypi import PyPIUploader

    config = ctx.obj["config"]
    i18n: I18n = get_i18n_from_ctx(ctx)

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
            raise UploadError(i18n._("upload.repository_url_invalid"))
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
            typer.echo(i18n._("upload.dry_run_not_configured", target=target, reason=exc))
            audit.record("upload.dry_run_not_configured", {"target": target, "reason": str(exc)})
            return
        raise

    if not dry_run and not yes and not typer.confirm(i18n._("upload.confirm", target=target)):
        typer.echo(i18n._("upload.aborted"))
        audit.record("upload.aborted", {"reason": "user_declined"})
        raise typer.Exit(code=0)

    try:
        result = uploader.upload(dry_run=dry_run, verbose=verbose)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        error = UploadError(i18n._("upload.failed", target=target, exc=exc))
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
        typer.echo(i18n._("upload.dry_run", target=target, details=details_str))
        return

    audit.record("upload.done", {"target": target, "result": result.details})
    plugin_manager.call("post_upload", context=context, fail_fast=False)
    if result.url:
        typer.echo(i18n._("upload.result_url", target=target, url=result.url))
    else:
        typer.echo(i18n._("upload.result", target=target))


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
