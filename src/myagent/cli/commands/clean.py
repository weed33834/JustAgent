"""The ``myagent clean`` command."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer

from myagent.adapters.tool_adapter import ToolChain
from myagent.core.audit_logger import AuditLogger
from myagent.core.context import CommandContext
from myagent.core.i18n import I18n, get_i18n, get_i18n_from_ctx
from myagent.exceptions import ToolChainError
from myagent.models.config import AppConfig
from myagent.plugin_manager import manager as plugin_manager
from myagent.utils import is_within_project
from myagent.utils.json_io import atomic_write_text

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.command(name="clean")(clean)


_PYTHON_EXTENSIONS = frozenset({".py", ".pyi", ".pyx", ".pxd"})

# Source file extensions handled by the built-in formatter. The built-in
# whitespace rules (trailing whitespace, blank line collapsing, inline space
# compression, trailing newline) apply uniformly to all of these languages.
_SOURCE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".pyx",
        ".pxd",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rb",
    }
)

# Directories that should never be scanned by the built-in formatter.
_EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "target",
        "build",
        "dist",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
)


def _builtin_format_file(file_path: Path) -> bool:
    """Apply built-in formatting to a single file.

    Returns True if the file was modified.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    original = content
    lines = content.splitlines(keepends=True)

    # 1. Strip trailing whitespace from each line
    lines = [(line.rstrip() + "\n") if line.endswith("\n") else line.rstrip() for line in lines]

    # 2. Collapse multiple consecutive blank lines into a single blank line
    deduped: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        deduped.append(line)
        prev_blank = is_blank

    # 3. Ensure file ends with exactly one trailing newline
    new_content = "".join(deduped).rstrip("\n") + "\n"

    if new_content != original:
        atomic_write_text(file_path, new_content)
        return True
    return False


def _collect_source_files(paths: list[Path], project_root: Path) -> list[Path]:
    """Collect source files from the requested paths.

    Covers all extensions in :data:`_SOURCE_EXTENSIONS`. Directories listed in
    :data:`_EXCLUDED_DIRS` (e.g. ``node_modules``, ``.git``, ``target``) are
    pruned from the recursive scan so that dependency trees are not formatted.

    Paths that resolve outside ``project_root`` are silently skipped to
    prevent path-traversal attacks (e.g. ``../../etc/passwd``).
    """

    def _is_source(p: Path) -> bool:
        return p.suffix in _SOURCE_EXTENSIONS

    def _is_excluded(p: Path) -> bool:
        return any(part in _EXCLUDED_DIRS for part in p.parts)

    result: list[Path] = []
    for p in paths:
        target = (project_root / p).resolve() if not p.is_absolute() else p.resolve()
        if not is_within_project(target, project_root):
            continue
        if target.is_file() and _is_source(target) and not _is_excluded(target):
            result.append(target)
        elif target.is_dir():
            for ext in _SOURCE_EXTENSIONS:
                for f in target.rglob(f"*{ext}"):
                    if f.is_file() and not _is_excluded(f) and is_within_project(f, project_root):
                        result.append(f)
    return result


def _run_builtin_format_fallback(
    config: AppConfig,
    paths: list[Path],
    project_root: Path,
    dry_run: bool,
    verbose: bool,
    audit: AuditLogger,
    context: CommandContext,
) -> None:
    """Fall back to built-in formatting when external tools produce no diff.

    This covers two cases:
    1. Configured Python tools are missing -> format every source file.
    2. External tools are present but only handle Python -> format the
       non-Python source files (e.g. .js, .rs) that they skip.
    """
    missing = [t for t in config.clean.tools if shutil.which(t) is None]
    source_files = _collect_source_files(paths, project_root)
    fallback_due_to_missing = bool(
        missing and any(t in config.clean.tools for t in ("autoflake", "black"))
    )
    non_python_files = [f for f in source_files if f.suffix not in _PYTHON_EXTENSIONS]

    i18n: I18n = get_i18n()

    if fallback_due_to_missing:
        typer.echo(
            i18n._("clean.builtin_fallback", tools=", ".join(missing)),
            err=True,
        )
        files_to_format: list[Path] = source_files
    elif non_python_files:
        files_to_format = non_python_files
    else:
        files_to_format = []

    if files_to_format:
        changed = 0
        for f in files_to_format:
            if dry_run:
                typer.echo(f"[dry-run] would format {f}")
                changed += 1
            elif _builtin_format_file(f):
                changed += 1
                if verbose:
                    typer.echo(f"Formatted: {f}")
        if changed:
            audit.record("clean.builtin", {"changed": changed})
            typer.echo(i18n._("clean.builtin_done", count=changed))
            plugin_manager.call("post_clean", context=context, fail_fast=False)
            return

    typer.echo(i18n._("clean.noop"))
    audit.record("clean.noop")


@app.command()
def clean(
    ctx: typer.Context,
    paths: list[Path] = typer.Argument(default_factory=lambda: [Path(".")]),
    check: bool = typer.Option(False, "--check", help="Exit with error if changes are needed"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmations"),
) -> None:
    """Clean and format the project code."""
    # RBAC gate: clean mutates source files.

    config = ctx.obj["config"]
    i18n: I18n = get_i18n_from_ctx(ctx)

    audit: AuditLogger = ctx.obj["audit_logger"]
    dry_run: bool = ctx.obj.get("dry_run", False)
    yes = yes or ctx.obj.get("yes", False)
    verbose: bool = ctx.obj.get("verbose", False)

    context = CommandContext(
        command="clean",
        project_root=config.project_root,
        config=config,
        dry_run=dry_run,
        yes=yes,
        trace_id=audit.trace_id,
    )

    audit.record("clean.start", {"paths": [str(p) for p in paths]})
    plugin_manager.call("pre_clean", context=context, fail_fast=False)

    toolchain = ToolChain(
        tools=config.clean.tools,
        project_root=config.project_root,
        dry_run=dry_run,
        verbose=verbose,
        exclude=config.clean.exclude,
    )

    try:
        diff = toolchain.preview(paths)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise ToolChainError(i18n._("clean.preview_failed", exc=exc)) from exc

    if not diff.strip():
        _run_builtin_format_fallback(
            config,
            paths,
            config.project_root,
            dry_run,
            verbose,
            audit,
            context,
        )
        return

    if verbose or dry_run:
        typer.echo(diff)

    if check:
        raise ToolChainError(i18n._("clean.not_clean"))

    if not dry_run and not yes and not typer.confirm(i18n._("clean.confirm")):
        typer.echo(i18n._("clean.aborted"))
        audit.record("clean.aborted", {"reason": "user_declined"})
        raise typer.Exit(code=0)

    try:
        toolchain.apply(paths)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise ToolChainError(i18n._("clean.apply_failed", exc=exc)) from exc

    audit.record("clean.done", {"paths": [str(p) for p in paths]})
    plugin_manager.call("post_clean", context=context, fail_fast=False)
    typer.echo(i18n._("clean.complete"))
