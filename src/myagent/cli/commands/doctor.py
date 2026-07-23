"""``myagent doctor`` 命令：诊断环境和依赖。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import typer

from myagent.core.cache import DiskCache
from myagent.core.config_center import load_config
from myagent.core.model_router import ModelRouter
from myagent.exceptions import ConfigError
from myagent.models.config import AppConfig


class Status(StrEnum):
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    suggestion: str = ""


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=lambda: list[CheckResult]())

    def add(self, name: str, status: Status, message: str, suggestion: str = "") -> None:
        self.checks.append(CheckResult(name, status, message, suggestion))

    def summary(self) -> tuple[int, int, int]:
        ok = sum(1 for c in self.checks if c.status == Status.OK)
        warnings = sum(1 for c in self.checks if c.status == Status.WARNING)
        errors = sum(1 for c in self.checks if c.status == Status.ERROR)
        return ok, warnings, errors


def _run_cmd(cmd: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10.0)
        return True, result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
        subprocess.TimeoutExpired,
    ) as exc:
        return False, str(exc)


def check_python() -> CheckResult:
    version = sys.version_info
    version_str = f"Python {version.major}.{version.minor}.{version.micro}"
    if version < (3, 11):
        return CheckResult("python", Status.ERROR, version_str, "请升级到 Python 3.11+。")
    return CheckResult("python", Status.OK, version_str)


def check_git() -> CheckResult:
    ok, output = _run_cmd(["git", "--version"])
    if not ok:
        return CheckResult("git", Status.ERROR, "未找到 git", "请安装 git 并加入 PATH。")
    return CheckResult("git", Status.OK, output)


def check_clean_toolchain(config: AppConfig) -> CheckResult:
    project_type = config.project_type
    if project_type not in ("python", "unknown", ""):
        return CheckResult("clean-toolchain", Status.OK, "非 Python 项目，已跳过。")
    tools = config.clean.tools
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        return CheckResult(
            "clean-toolchain", Status.WARNING,
            f"缺少工具：{', '.join(missing)}",
            "安装所需工具或将其加入 PATH。",
        )
    return CheckResult("clean-toolchain", Status.OK, "所有清理工具可用。")


def check_model_backend(config: AppConfig) -> CheckResult:
    if not config.model.backends:
        return CheckResult(
            "model-backend", Status.WARNING,
            "未配置模型后端。",
            "编辑 .myagent.toml 配置模型后端。",
        )
    with ModelRouter(config) as router:
        healthy = router.select_backend(tier=config.model.default_tier)
        if healthy is None:
            return CheckResult(
                "model-backend", Status.WARNING,
                "所有模型后端不可达。",
                "检查 API key 或本地模型服务是否运行。",
            )
        return CheckResult("model-backend", Status.OK, "模型后端可用。")


def check_plugin_dependencies() -> CheckResult:
    optional = {
        "semgrep": "security-scan 插件",
        "docker": "docker-ship 插件",
    }
    missing = {tool: reason for tool, reason in optional.items() if shutil.which(tool) is None}
    if missing:
        details = "; ".join(f"{tool} ({reason})" for tool, reason in missing.items())
        return CheckResult(
            "plugin-dependencies", Status.WARNING,
            f"缺少可选依赖：{details}",
            "安装所需工具以启用对应插件。",
        )
    return CheckResult("plugin-dependencies", Status.OK, "可选依赖齐全。")


def _resolve_directory_paths(config: AppConfig) -> list[Path]:
    audit_dir = config.audit_log_dir or config.audit.log_dir or (Path.home() / ".myagent" / "logs")
    cache_dir = Path.home() / ".myagent" / "cache"
    return [config.project_root, audit_dir, cache_dir]


def _is_writable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    return path.parent.exists() and os.access(path.parent, os.W_OK)


def check_directories(config: AppConfig) -> CheckResult:
    paths = _resolve_directory_paths(config)
    bad = [str(p) for p in paths if not (p.exists() or p.parent.exists())]
    if bad:
        return CheckResult("directories", Status.WARNING, f"路径不存在：{bad}", "创建所需目录。")

    not_writable = [str(p) for p in paths if not _is_writable(p)]
    if not_writable:
        return CheckResult("directories", Status.WARNING, f"路径不可写：{not_writable}", "检查目录权限。")
    return CheckResult("directories", Status.OK, "所有目录可写。")


def check_cache(config: AppConfig) -> CheckResult:
    try:
        cache = DiskCache(cache_dir=config.cache.dir)
        cache.set("__doctor_probe__", "ok", ttl=10)
        value = cache.get("__doctor_probe__")
        cache.invalidate("__doctor_probe__")
        if value == "ok":
            return CheckResult("cache", Status.OK, "缓存正常。")
    except OSError:
        return CheckResult("cache", Status.ERROR, "缓存读写失败。", "检查缓存目录权限。")
    return CheckResult("cache", Status.WARNING, "缓存返回异常值。", "检查缓存目录权限。")


def build_report() -> DoctorReport:
    report = DoctorReport()
    report.add(**check_python().__dict__)
    report.add(**check_git().__dict__)
    report.add(**check_plugin_dependencies().__dict__)

    try:
        config = load_config()
    except ConfigError:
        report.add("config", Status.ERROR, "配置加载失败。", "运行 `myagent init` 创建配置。")
        return report

    report.add(**check_clean_toolchain(config).__dict__)
    report.add(**check_model_backend(config).__dict__)
    report.add(**check_directories(config).__dict__)
    report.add(**check_cache(config).__dict__)
    return report


def register(parent: typer.Typer) -> None:
    parent.command(name="doctor")(doctor)


def doctor(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="以 JSON 格式输出"),
    fail_on_error: bool = typer.Option(False, "--fail-on-error", help="有错误时返回非零退出码"),
) -> None:
    """诊断 MyAgent 环境和依赖。"""
    report = build_report()
    ok, warnings, errors = report.summary()

    if json_output:
        import json as _json

        data: dict[str, object] = {
            "summary": {"ok": ok, "warning": warnings, "error": errors},
            "checks": [
                {"name": c.name, "status": c.status.value.lower(), "message": c.message, "suggestion": c.suggestion}
                for c in report.checks
            ],
        }
        typer.echo(_json.dumps(data, indent=2, ensure_ascii=False))
        if fail_on_error and errors:
            raise typer.Exit(code=1)
        return

    typer.echo("MyAgent 环境诊断")
    typer.echo("-" * 60)
    for check in report.checks:
        status_color = {
            Status.OK: typer.colors.GREEN,
            Status.WARNING: typer.colors.YELLOW,
            Status.ERROR: typer.colors.RED,
        }[check.status]
        typer.secho(f"[{check.status.value}] {check.name:<20} {check.message}", fg=status_color)
        if check.suggestion:
            typer.echo(f"  → {check.suggestion}")
    typer.echo("-" * 60)
    typer.echo(f"总计：{ok} OK，{warnings} 警告，{errors} 错误")
    if fail_on_error and errors:
        raise typer.Exit(code=1)
