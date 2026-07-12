"""The ``autoship doctor`` command."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import typer

from autoship.core.cache import DiskCache
from autoship.core.config_center import load_config
from autoship.core.metrics import get_registry
from autoship.core.model_router import ModelRouter
from autoship.exceptions import ConfigError
from autoship.models.config import AppConfig


class Status(str, Enum):
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


def check_python(i18n) -> CheckResult:
    version = sys.version_info
    version_str = f"Python {version.major}.{version.minor}.{version.micro}"
    if version < (3, 10):
        return CheckResult(
            "python",
            Status.ERROR,
            version_str,
            "doctor.python_old",
        )
    return CheckResult("python", Status.OK, version_str)


def check_git(i18n) -> CheckResult:
    ok, output = _run_cmd(["git", "--version"])
    if not ok:
        return CheckResult("git", Status.ERROR, "doctor.git_missing", "doctor.git_suggestion")
    return CheckResult("git", Status.OK, output)


def check_clean_toolchain(config: AppConfig, i18n) -> CheckResult:
    # The default clean toolchain is Python-centric; skip for non-Python
    # projects to avoid spurious warnings. "unknown" falls back to Python.
    project_type = config.project_type
    if project_type not in ("python", "unknown", ""):
        return CheckResult(
            "clean-toolchain",
            Status.OK,
            "doctor.clean_skipped_non_python",
        )
    tools = config.clean.tools
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        return CheckResult(
            "clean-toolchain",
            Status.WARNING,
            "doctor.clean_missing",
            "doctor.clean_suggestion",
        )
    return CheckResult("clean-toolchain", Status.OK, "doctor.clean_ok")


def check_model_backend(config: AppConfig, i18n) -> CheckResult:
    if not config.model.backends:
        return CheckResult(
            "model-backend",
            Status.WARNING,
            "doctor.model_none",
            "doctor.model_suggestion",
        )
    with ModelRouter(config) as router:
        healthy = router.select_backend(tier=config.model.default_tier)
        if healthy is None:
            return CheckResult(
                "model-backend",
                Status.WARNING,
                "doctor.model_unreachable",
                "doctor.model_start",
            )
        return CheckResult(
            "model-backend",
            Status.OK,
            "doctor.model_ok",
        )


def check_plugin_dependencies(i18n) -> CheckResult:
    optional = {
        "semgrep": "security-scan plugin",
        "docker": "docker-ship plugin",
        "gitleaks": "security-scan plugin",
    }
    missing = {tool: reason for tool, reason in optional.items() if shutil.which(tool) is None}
    if missing:
        _details = "; ".join(f"{tool} ({reason})" for tool, reason in missing.items())
        return CheckResult(
            "plugin-dependencies",
            Status.WARNING,
            "doctor.plugin_missing",
            "doctor.plugin_suggestion",
        )
    return CheckResult("plugin-dependencies", Status.OK, "doctor.plugin_ok")


def _resolve_directory_paths(config: AppConfig) -> list[Path]:
    from autoship.core.registry_client import DEFAULT_CACHE_DIR

    audit_dir = config.audit_log_dir or config.audit.log_dir or (Path.home() / ".autoship" / "logs")
    return [config.project_root, audit_dir, DEFAULT_CACHE_DIR]


def _is_writable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    return path.parent.exists() and os.access(path.parent, os.W_OK)


def check_directories(config: AppConfig, i18n) -> CheckResult:
    paths = _resolve_directory_paths(config)
    bad = [str(p) for p in paths if not (p.exists() or p.parent.exists())]
    if bad:
        return CheckResult(
            "directories",
            Status.WARNING,
            "doctor.dirs_bad",
            "doctor.dirs_suggestion",
        )

    not_writable = [str(p) for p in paths if not _is_writable(p)]
    if not_writable:
        return CheckResult(
            "directories",
            Status.WARNING,
            "doctor.dirs_not_writable",
            "doctor.dirs_writable_suggestion",
        )
    return CheckResult("directories", Status.OK, "doctor.dirs_ok")


def check_cache(config: AppConfig, i18n) -> CheckResult:
    try:
        cache = DiskCache(cache_dir=config.cache.dir)
        cache.set("__doctor_probe__", "ok", ttl=10)
        value = cache.get("__doctor_probe__")
        cache.invalidate("__doctor_probe__")
        if value == "ok":
            return CheckResult("cache", Status.OK, "doctor.cache_ok")
    except OSError:
        return CheckResult(
            "cache",
            Status.ERROR,
            "doctor.cache_error",
            "doctor.cache_suggestion",
        )
    return CheckResult(
        "cache",
        Status.WARNING,
        "doctor.cache_warning",
        "doctor.cache_suggestion",
    )


def check_observability(i18n) -> CheckResult:
    registry = get_registry()
    snapshot = registry.snapshot()
    if not snapshot:
        return CheckResult(
            "observability",
            Status.OK,
            "doctor.observability_ok_empty",
        )
    _counter_count = sum(1 for d in snapshot.values() if d.get("type") == "counter")
    _hist_count = sum(1 for d in snapshot.values() if d.get("type") == "histogram")
    return CheckResult(
        "observability",
        Status.OK,
        "doctor.observability_ok",
    )


def check_registry_signature(config: AppConfig, i18n) -> CheckResult:
    """Warn when the registry public_key is not configured."""
    public_key = config.registry.public_key if config.registry else None
    if not public_key:
        return CheckResult(
            "registry-signature",
            Status.WARNING,
            "doctor.registry_sig_unconfigured",
            "doctor.registry_sig_suggestion",
        )
    return CheckResult("registry-signature", Status.OK, "doctor.registry_sig_ok")


def build_report(i18n=None) -> DoctorReport:
    report = DoctorReport()
    report.add(**check_python(i18n).__dict__)
    report.add(**check_git(i18n).__dict__)
    report.add(**check_plugin_dependencies(i18n).__dict__)

    try:
        config = load_config()
    except ConfigError:
        report.add(
            "config",
            Status.ERROR,
            "doctor.config_error",
            "error.suggestion.init",
        )
        return report

    report.add(**check_clean_toolchain(config, i18n).__dict__)
    report.add(**check_model_backend(config, i18n).__dict__)
    report.add(**check_directories(config, i18n).__dict__)
    report.add(**check_cache(config, i18n).__dict__)
    report.add(**check_registry_signature(config, i18n).__dict__)
    report.add(**check_observability(i18n).__dict__)
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
    """Diagnose the AutoShip environment and dependencies."""

    report = build_report(None)
    ok, warnings, errors = report.summary()

    if json_output:
        import json as _json

        checks_index: dict[str, CheckResult] = {c.name: c for c in report.checks}

        def _group(keys: list[str]) -> dict[str, str]:
            entry: dict[str, str] = {}
            for k in keys:
                chk = checks_index.get(k)
                if chk is not None:
                    entry["status"] = chk.status.value.lower()
                    entry["message"] = chk.message
                    if chk.suggestion:
                        entry["suggestion"] = chk.suggestion
            if not entry:
                entry["status"] = "unknown"
                entry["message"] = ""
            return entry

        def _worst(states: list[str]) -> str:
            for s in ("error", "warning", "ok"):
                if s in states:
                    return s
            return "ok"

        # Gather sub-group statuses
        py_status = checks_index.get("python", CheckResult("python", Status.OK, ""))
        git_status = checks_index.get("git", CheckResult("git", Status.OK, ""))
        tools_status = _worst(
            [
                checks_index.get("clean-toolchain", CheckResult("", Status.OK, "")).status.value,
                checks_index.get(
                    "plugin-dependencies", CheckResult("", Status.OK, "")
                ).status.value,
            ]
        )

        data: dict[str, object] = {
            "summary": {"ok": ok, "warning": warnings, "error": errors},
            "python": {
                "status": py_status.status.value.lower(),
                "version": py_status.message,
            },
            "git": {
                "status": git_status.status.value.lower(),
                "version": git_status.message,
            },
            "tools": {
                "status": tools_status,
                "clean": checks_index.get(
                    "clean-toolchain",
                    CheckResult("clean-toolchain", Status.OK, ""),
                ).message,
                "plugin_deps": checks_index.get(
                    "plugin-dependencies",
                    CheckResult("plugin-dependencies", Status.OK, ""),
                ).message,
            },
            "model": _group(["model-backend"]),
            "dirs": _group(["directories"]),
            "cache": _group(["cache"]),
            "observability": _group(["observability"]),
        }
        typer.echo(_json.dumps(data, indent=2, ensure_ascii=False))
        if fail_on_error and errors:
            raise typer.Exit(code=1)
        return

    typer.echo("doctor.title")
    typer.echo("-" * 60)
    for check in report.checks:
        status_color = {
            Status.OK: typer.colors.GREEN,
            Status.WARNING: typer.colors.YELLOW,
            Status.ERROR: typer.colors.RED,
        }[check.status]
        typer.secho(f"[{check.status.value}] {check.name:<20} {check.message}", fg=status_color)
        if check.suggestion:
            typer.echo("doctor.suggestion")
    typer.echo("-" * 60)
    typer.echo("doctor.summary")
    if fail_on_error and errors:
        raise typer.Exit(code=1)
