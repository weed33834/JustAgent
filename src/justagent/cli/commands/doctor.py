"""``justagent doctor`` 命令：诊断环境和依赖。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import typer

from justagent.core.cache import Cache
from justagent.core.config_center import load_config
from justagent.core.i18n import I18n, get_i18n, get_i18n_from_ctx
from justagent.core.model_router import ModelRouter
from justagent.exceptions import ConfigError
from justagent.models.config import AppConfig


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
    i18n = get_i18n()
    version = sys.version_info
    version_str = f"Python {version.major}.{version.minor}.{version.micro}"
    if version < (3, 11):
        return CheckResult("python", Status.ERROR, version_str, i18n._("doctor.python_old"))
    return CheckResult("python", Status.OK, version_str)


def check_git() -> CheckResult:
    i18n = get_i18n()
    ok, output = _run_cmd(["git", "--version"])
    if not ok:
        return CheckResult(
            "git", Status.ERROR, i18n._("doctor.git_missing"), i18n._("doctor.git_suggestion")
        )
    return CheckResult("git", Status.OK, output)


def check_clean_toolchain(config: AppConfig) -> CheckResult:
    i18n = get_i18n()
    project_type = config.project_type
    if project_type not in ("python", "unknown", ""):
        return CheckResult(
            "clean-toolchain", Status.OK, i18n._("doctor.clean_skipped_non_python", type=project_type)
        )
    tools = config.clean.tools
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        return CheckResult(
            "clean-toolchain",
            Status.WARNING,
            i18n._("doctor.clean_missing", tools=", ".join(missing)),
            i18n._("doctor.clean_suggestion"),
        )
    return CheckResult("clean-toolchain", Status.OK, i18n._("doctor.clean_ok"))


def check_model_backend(config: AppConfig) -> CheckResult:
    i18n = get_i18n()
    if not config.model.backends:
        return CheckResult(
            "model-backend",
            Status.WARNING,
            i18n._("doctor.model_none"),
            i18n._("doctor.model_suggestion"),
        )
    with ModelRouter(config) as router:
        healthy = router.select_backend(tier=config.model.default_tier)
        if healthy is None:
            return CheckResult(
                "model-backend",
                Status.WARNING,
                i18n._("doctor.model_unreachable"),
                i18n._("doctor.model_start"),
            )
        return CheckResult(
            "model-backend",
            Status.OK,
            i18n._("doctor.model_ok", model=healthy.cfg.model or str(healthy.cfg.base_url)),
        )


def check_plugin_dependencies() -> CheckResult:
    i18n = get_i18n()
    optional = {
        "semgrep": i18n._("doctor.plugin_desc", name="security-scan"),
        "docker": i18n._("doctor.plugin_desc", name="docker-ship"),
    }
    missing = {tool: reason for tool, reason in optional.items() if shutil.which(tool) is None}
    if missing:
        details = "; ".join(f"{tool} ({reason})" for tool, reason in missing.items())
        return CheckResult(
            "plugin-dependencies",
            Status.WARNING,
            i18n._("doctor.plugin_missing", details=details),
            i18n._("doctor.plugin_suggestion"),
        )
    return CheckResult("plugin-dependencies", Status.OK, i18n._("doctor.plugin_ok"))


def _resolve_directory_paths(config: AppConfig) -> list[Path]:
    audit_dir = config.audit_log_dir or config.audit.log_dir or (Path.home() / ".justagent" / "logs")
    cache_dir = Path.home() / ".justagent" / "cache"
    return [config.project_root, audit_dir, cache_dir]


def _is_writable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    return path.parent.exists() and os.access(path.parent, os.W_OK)


def check_directories(config: AppConfig, i18n: I18n) -> CheckResult:
    paths = _resolve_directory_paths(config)
    bad = [str(p) for p in paths if not (p.exists() or p.parent.exists())]
    if bad:
        return CheckResult(
            "directories",
            Status.WARNING,
            i18n._("doctor.dirs_bad", paths=", ".join(bad)),
            i18n._("doctor.dirs_suggestion"),
        )

    not_writable = [str(p) for p in paths if not _is_writable(p)]
    if not_writable:
        return CheckResult(
            "directories",
            Status.WARNING,
            i18n._("doctor.dirs_not_writable", paths=", ".join(not_writable)),
            i18n._("doctor.dirs_writable_suggestion"),
        )
    return CheckResult("directories", Status.OK, i18n._("doctor.dirs_ok"))


def check_cache(config: AppConfig) -> CheckResult:
    i18n = get_i18n()
    try:
        cache = Cache(cache_dir=config.cache.dir)
        cache.set("__doctor_probe__", "ok", ttl=10)
        value = cache.get("__doctor_probe__")
        cache.invalidate("__doctor_probe__")
        if value == "ok":
            return CheckResult("cache", Status.OK, i18n._("doctor.cache_ok"))
    except OSError as exc:
        return CheckResult(
            "cache",
            Status.ERROR,
            i18n._("doctor.cache_error", error=exc),
            i18n._("doctor.cache_suggestion"),
        )
    return CheckResult(
        "cache",
        Status.WARNING,
        i18n._("doctor.cache_warning"),
        i18n._("doctor.cache_suggestion"),
    )


def build_report() -> DoctorReport:
    report = DoctorReport()
    i18n = get_i18n()
    report.add(**check_python().__dict__)
    report.add(**check_git().__dict__)
    report.add(**check_plugin_dependencies().__dict__)

    try:
        config = load_config()
    except ConfigError as exc:
        report.add(
            "config",
            Status.ERROR,
            i18n._("doctor.config_error", exc=exc),
            i18n._("error.suggestion.init"),
        )
        return report

    report.add(**check_clean_toolchain(config).__dict__)
    report.add(**check_model_backend(config).__dict__)
    report.add(**check_directories(config, i18n).__dict__)
    report.add(**check_cache(config).__dict__)
    return report


def register(parent: typer.Typer) -> None:
    parent.command(name="doctor")(doctor)


def doctor(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output report as JSON"),
    fail_on_error: bool = typer.Option(
        False, "--fail-on-error", help="Exit with non-zero code when errors are present"
    ),
) -> None:
    """Diagnose the environment and dependencies."""
    i18n = get_i18n_from_ctx(ctx)
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

    typer.echo(i18n._("doctor.title"))
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
    typer.echo(i18n._("doctor.summary", ok=ok, warnings=warnings, errors=errors))
    if fail_on_error and errors:
        raise typer.Exit(code=1)
