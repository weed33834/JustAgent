"""The ``myagent artifacts`` command: language-aware build output removal.

Distinct from ``clean`` (which formats source files), ``artifacts``
removes the language-native build outputs that this project's toolchain
produces: ``bin/`` for Go, ``dist/`` and ``*.tsbuildinfo`` for Node,
``target/classes`` and loose ``*.class`` for Java, and so on. Rust's
``target/`` cache is intentionally NOT removed wholesale; see the
``rust-ship`` example plugin for the narrower ``target/{debug,release}``
opt-in.

Rules live in :mod:`myagent.core.language_rules`; this command is a thin
CLI surface over :func:`plan_artifact_removal` /
:func:`apply_artifact_removal`.

Hook reuse
----------
``artifacts`` intentionally reuses the ``pre_clean`` / ``post_clean`` hooks
rather than defining a separate ``pre_artifacts`` / ``post_artifacts``
hookspec, so plugins gating cleanup fire uniformly for both ``clean``
(source formatting) and ``artifacts`` (build-output removal). This keeps
the command a thin CLI surface for cleanup operations.

Examples::

    myagent artifacts --dry-run       # show what would be removed
    myagent artifacts                 # remove build outputs
    myagent artifacts --list          # show which rules apply, no scanning
    myagent artifacts --language go   # restrict to one language
"""

from __future__ import annotations

from pathlib import Path

import typer

from myagent.core.audit_logger import AuditLogger
from myagent.core.context import CommandContext
from myagent.core.i18n import I18n, get_i18n_from_ctx
from myagent.core.language_rules import (
    RULES,
    apply_artifact_removal,
    plan_artifact_removal,
)
from myagent.models.config import AppConfig
from myagent.plugin_manager import manager as plugin_manager

app = typer.Typer()


def register(parent: typer.Typer) -> None:
    parent.command(name="artifacts")(artifacts)


def _format_plan(plan_paths_by_lang: dict[str, list[Path]]) -> str:
    """Render the planned removal list as a human-readable summary."""
    if not plan_paths_by_lang:
        return "(nothing to remove)"
    lines: list[str] = []
    for language, paths in plan_paths_by_lang.items():
        if not paths:
            continue
        lines.append(f"  {language}:")
        for p in paths:
            lines.append(f"    - {p}")
    return "\n".join(lines)


def artifacts(
    ctx: typer.Context,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="List what would be removed without deleting anything.",
    ),
    list_rules: bool = typer.Option(
        False,
        "--list",
        help="Show which language rules apply to this project, then exit.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Restrict to a single language (go, rust, node, java).",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmation."),
) -> None:
    """Remove language-native build artifacts (bin/, dist/, *.class, ...)."""
    config: AppConfig = ctx.obj["config"]
    i18n: I18n = get_i18n_from_ctx(ctx)
    audit: AuditLogger = ctx.obj["audit_logger"]
    project_root = Path(config.project_root)

    context = CommandContext(
        command="artifacts",
        project_root=project_root,
        config=config,
        dry_run=dry_run,
        yes=yes,
        trace_id=audit.trace_id,
    )

    # artifacts intentionally reuses the clean hooks (pre_clean/post_clean)
    # as a thin CLI surface for cleanup operations; see module docstring.
    plugin_manager.call("pre_clean", context=context, fail_fast=False)

    if list_rules:
        applicable = [
            rule
            for rule in RULES.values()
            if any((project_root / m).exists() for m in rule.manifests)
            and (language is None or rule.language == language)
        ]
        if not applicable:
            typer.echo(i18n._("artifacts.no_rules"))
            raise typer.Exit(code=0)
        for rule in applicable:
            typer.echo(f"[{rule.language}]")
            typer.echo(f"  manifests:    {', '.join(rule.manifests)}")
            typer.echo(f"  artifact_dirs: {', '.join(rule.artifact_dirs) or '(none)'}")
            typer.echo(f"  artifact_globs: {', '.join(rule.artifact_globs) or '(none)'}")
            typer.echo(f"  test_command: {rule.test_command or '(none)'}")
            typer.echo(f"  lint_command: {rule.lint_command or '(none)'}")
            if rule.notes:
                typer.echo(f"  notes: {rule.notes}")
            typer.echo("")
        raise typer.Exit(code=0)

    # Distinguish "no language detected" from "language detected but clean".
    # Both produce an empty plan, but the operator cares about the difference:
    # the former means artifacts won't ever apply to this project, the latter
    # means the rule set matched and there's simply nothing to remove right now.
    applicable_rule_count = sum(
        1
        for rule in RULES.values()
        if any((project_root / m).exists() for m in rule.manifests)
        and (language is None or rule.language == language)
    )
    if applicable_rule_count == 0:
        typer.echo(i18n._("artifacts.no_rules"))
        # post_clean hook reused for artifacts; see module docstring.
        plugin_manager.call("post_clean", context=context, fail_fast=False)
        raise typer.Exit(code=0)

    plan = plan_artifact_removal(project_root)
    if language is not None:
        plan.by_language = {
            lang: paths for lang, paths in plan.by_language.items() if lang == language
        }

    audit.record(
        "artifacts.plan",
        {
            "by_language": {
                lang: [str(p) for p in paths] for lang, paths in plan.by_language.items()
            },
            "skipped": [str(p) for p in plan.skipped],
            "dry_run": dry_run,
        },
    )

    if plan.total == 0:
        typer.echo(i18n._("artifacts.nothing_to_remove"))
        plugin_manager.call("post_clean", context=context, fail_fast=False)
        raise typer.Exit(code=0)

    typer.echo(i18n._("artifacts.plan_header"))
    typer.echo(_format_plan(plan.by_language))

    if plan.skipped:
        typer.secho(
            i18n._("artifacts.skipped", count=len(plan.skipped)),
            fg=typer.colors.YELLOW,
        )
        for p in plan.skipped:
            typer.echo(f"    {p}", err=True)

    if dry_run:
        typer.echo(i18n._("artifacts.dry_run_summary", count=plan.total))
        plugin_manager.call("post_clean", context=context, fail_fast=False)
        raise typer.Exit(code=0)

    skip_confirm = yes or ctx.obj.get("yes", False)
    if not skip_confirm and not typer.confirm(i18n._("artifacts.confirm", count=plan.total)):
        typer.echo(i18n._("artifacts.aborted"))
        audit.record("artifacts.aborted", {"reason": "user_declined"})
        raise typer.Exit(code=0)

    removed = apply_artifact_removal(plan, dry_run=False)
    audit.record("artifacts.removed", {"count": removed})
    typer.echo(i18n._("artifacts.done", count=removed))
    plugin_manager.call("post_clean", context=context, fail_fast=False)
