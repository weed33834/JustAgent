"""``justagent skill`` command — SKILL.md lifecycle management.

Exposes the :class:`justagent.context.skill.SkillLoader` through the CLI:

* ``justagent skill list``      — 列出所有已发现的技能。
* ``justagent skill show``      — 查看技能完整正文。
* ``justagent skill create``    — 交互式或通过参数创建新技能。
* ``justagent skill delete``    — 删除一个技能。
* ``justagent skill update``    — 更新技能字段。
* ``justagent skill import``    — 从外部 ``SKILL.md`` 导入技能。
* ``justagent skill generate``  — 调用 LLM 自动生成技能。

本模块遵循与其他 ``justagent`` 命令一致的约定：通过模块级 ``register``
函数注册子命令组，通过 ``ctx.obj`` 访问共享的配置 / 审计日志 / 详细 /
试运行标志，使用 Rich 表格渲染列表，并在 ``generate`` 子命令中延迟导入
较重的 LLM 依赖，保证未配置模型时 ``justagent skill ...`` 仍能正常加载。
"""

from __future__ import annotations

import json
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from justagent.cli.commands import _common as common
from justagent.cli.display import get_console
from justagent.context.skill import (
    Skill,
    SkillError,
    SkillLoader,
    SkillTrigger,
    parse_skill_file,
)
from justagent.models.config import AppConfig

# ---------------------------------------------------------------------------
# Typer sub-app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="skill",
    help="技能（SKILL.md）管理：列出 / 查看 / 创建 / 删除 / 更新 / 导入 / 生成。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def register(parent: typer.Typer) -> None:
    """Register the ``skill`` command group."""

    parent.add_typer(app, name="skill")


# ---------------------------------------------------------------------------
# Context accessors — defensive like the other command modules (build fallbacks when ctx.obj
# is missing keys, e.g. when a command is invoked directly in tests).
# ---------------------------------------------------------------------------


def _get_yes(ctx: typer.Context, explicit: bool = False) -> bool:
    """Return whether to skip confirmations (global ``yes`` or an explicit flag)."""

    obj = getattr(ctx, "obj", None)
    return bool(explicit or (obj.get("yes") if obj else False))


def _get_loader(ctx: typer.Context) -> SkillLoader:
    """Build a :class:`SkillLoader` rooted at the current working directory."""

    # Per the CLI contract, skills are discovered relative to the directory
    # the user invokes the command from.
    return SkillLoader(project_root=Path.cwd())


# ---------------------------------------------------------------------------
# Formatting / parsing helpers
# ---------------------------------------------------------------------------


def _format_triggers(triggers: list[SkillTrigger]) -> str:
    """Render triggers as a compact ``type:value`` comma-separated string."""

    if not triggers:
        return "-"
    return ", ".join(f"{t.type}:{t.value}" for t in triggers)


def _parse_triggers(raw: list[str] | None) -> list[SkillTrigger]:
    """Parse repeated ``--trigger`` strings into :class:`SkillTrigger` objects.

    Each item may be ``"type:value"`` (e.g. ``"keyword:migration"``) or a bare
    token (e.g. ``"migration"``) which defaults to a ``keyword`` trigger.
    """

    if not raw:
        return []
    result: list[SkillTrigger] = []
    for item in raw:
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            t, _, v = item.partition(":")
            result.append(SkillTrigger(type=t.strip(), value=v.strip()))
        else:
            result.append(SkillTrigger(type="keyword", value=item))
    return result


def _skill_to_dict(skill: Skill, *, include_body: bool = False) -> dict[str, Any]:
    """Serialise a skill to a JSON-friendly dict."""

    data: dict[str, Any] = {
        "name": skill.name,
        "description": skill.description,
        "path": str(skill.path),
        "triggers": [{"type": t.type, "value": t.value} for t in skill.triggers],
    }
    if include_body:
        data["body"] = skill.body
    return data


def _emit_json(data: Any) -> None:
    """Print *data* as indented JSON to stdout."""

    typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


def _resolve_body(body: str | None, body_file: Path | None) -> str | None:
    """Resolve the skill body from ``--body`` or ``--body-file``.

    Raises :class:`typer.BadParameter` if both are given or the file cannot be
    read. Returns ``None`` when neither is provided (caller may prompt).
    """

    if body is not None and body_file is not None:
        raise typer.BadParameter("--body 与 --body-file 不能同时使用")
    if body_file is not None:
        try:
            return body_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise typer.BadParameter(f"读取正文文件失败：{exc}") from exc
    return body


def _require_skill(loader: SkillLoader, name: str) -> Skill:
    """Return an existing skill or raise a Typer error (exit 1)."""

    skill = loader.get(name)
    if skill is None:
        raise typer.BadParameter(f"未找到技能：{name}")
    return skill


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command("list", help="列出所有已发现的技能。")
def skill_list(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """列出所有已发现的技能（名称、描述、触发器）。"""

    loader = _get_loader(ctx)
    skills = loader.discover()

    if json_output:
        _emit_json([_skill_to_dict(s) for s in skills])
        return

    console = get_console()
    if not skills:
        console.print("[dim]暂无技能。使用 `skill create` 或 `skill import` 创建一个。[/dim]")
        return

    table = Table(title=f"技能列表（共 {len(skills)} 个）", border_style="cyan")
    table.add_column("名称", style="bold")
    table.add_column("描述")
    table.add_column("触发器")
    table.add_column("路径", style="dim")

    for s in skills:
        table.add_row(
            s.name,
            common.short(s.description, 50),
            _format_triggers(s.triggers),
            common.short(str(s.path), 45),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@app.command("show", help="查看技能完整正文。")
def skill_show(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="技能名称"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """查看指定技能的元信息与完整正文。"""

    loader = _get_loader(ctx)
    skill = _require_skill(loader, name)

    if json_output:
        _emit_json(_skill_to_dict(skill, include_body=True))
        return

    console = get_console()
    header = (
        f"[bold]名称:[/bold]        {skill.name}\n"
        f"[bold]描述:[/bold]        {skill.description or '-'}\n"
        f"[bold]触发器:[/bold]      {_format_triggers(skill.triggers)}\n"
        f"[bold]路径:[/bold]        {skill.path}"
    )
    console.print(Panel(header, title=f"技能 — {skill.name}", border_style="cyan"))
    console.print(Panel(skill.body or "(空正文)", title="正文", border_style="blue"))


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


@app.command("create", help="创建一个新技能。")
def skill_create(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="技能名称（唯一标识）"),
    description: str | None = typer.Option(
        None, "--description", "-d", help="技能描述；未提供则交互式询问"
    ),
    body: str | None = typer.Option(None, "--body", "-b", help="技能正文；未提供则交互式询问"),
    body_file: Path | None = typer.Option(
        None,
        "--body-file",
        exists=True,
        dir_okay=False,
        readable=True,
        help="从文件读取技能正文（与 --body 二选一，适合多行 Markdown）",
    ),
    trigger: list[str] | None = typer.Option(
        None,
        "--trigger",
        "-t",
        help='触发器，可重复指定，格式 "type:value"（如 keyword:migration）；'
        "省略 type 时默认 keyword",
    ),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """创建一个新技能。

    可通过 ``--description`` / ``--body`` / ``--trigger`` 参数一次性提供全部
    字段，也可省略部分参数进入交互式提示。正文为多行 Markdown 时建议使用
    ``--body-file``。若同名技能已存在则报错（请改用 ``skill update``）。
    """

    loader = _get_loader(ctx)
    dry_run = common.get_dry_run(ctx)

    if loader.get(name) is not None:
        raise typer.BadParameter(f"技能已存在：{name}（如需修改请使用 `skill update`）")

    resolved_body = _resolve_body(body, body_file)

    # Interactive prompts for anything not supplied via flags.
    if description is None:
        description = typer.prompt("描述")
    if resolved_body is None:
        resolved_body = typer.prompt("正文（单行；多行内容请使用 --body-file）", default="")
    if not description.strip():
        raise typer.BadParameter("描述不能为空")

    triggers = _parse_triggers(trigger)

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将创建技能\n"
                f"名称:    {name}\n"
                f"描述:    {description}\n"
                f"触发器:  {_format_triggers(triggers)}\n"
                f"正文:    {common.short(resolved_body, 80)}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    try:
        skill = loader.create_skill(
            name=name,
            description=description,
            body=resolved_body,
            triggers=triggers,
        )
    except SkillError as exc:
        get_console().print(f"[red]✗ 创建失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    common.audit(
        ctx,
        "skill.create",
        {"name": name, "description": description, "triggers": len(triggers)},
    )

    if json_output:
        _emit_json(_skill_to_dict(skill, include_body=True))
        return

    get_console().print(f"[green]✓[/green] 已创建技能 [bold]{skill.name}[/bold]（{skill.path}）")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@app.command("delete", help="删除一个技能。")
def skill_delete(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="技能名称"),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认提示"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """删除一个技能及其目录（不可恢复，会要求确认）。"""

    loader = _get_loader(ctx)
    dry_run = common.get_dry_run(ctx)
    skip_confirm = _get_yes(ctx, yes)

    skill = _require_skill(loader, name)

    if not skip_confirm and not typer.confirm(
        f"确认删除技能 '{name}'（位于 {skill.path.parent}）？此操作不可恢复。",
        default=False,
    ):
        get_console().print("[dim]已取消。[/dim]")
        raise typer.Exit()

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将删除技能 {name}\n路径: {skill.path.parent}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    deleted = loader.delete_skill(name)
    if not deleted:
        get_console().print(f"[red]✗ 删除失败：未找到技能 {name}[/red]")
        raise typer.Exit(code=1)

    common.audit(ctx, "skill.delete", {"name": name, "path": str(skill.path)})

    if json_output:
        _emit_json({"name": name, "deleted": True, "path": str(skill.path)})
        return

    get_console().print(f"[green]✓[/green] 已删除技能 [bold]{name}[/bold]")


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


@app.command("update", help="更新一个技能的字段。")
def skill_update(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="技能名称"),
    description: str | None = typer.Option(None, "--description", "-d", help="新的描述"),
    body: str | None = typer.Option(None, "--body", "-b", help="新的正文"),
    body_file: Path | None = typer.Option(
        None,
        "--body-file",
        exists=True,
        dir_okay=False,
        readable=True,
        help="从文件读取新的正文（与 --body 二选一）",
    ),
    trigger: list[str] | None = typer.Option(
        None,
        "--trigger",
        "-t",
        help='触发器，可重复指定，格式 "type:value"；提供时将整体替换现有触发器',
    ),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """更新一个已有技能的字段。

    仅更新显式提供的字段；未提供的字段保持不变。``--trigger`` 一旦提供
    即整体替换现有触发器列表（传入空值会清空触发器）。
    """

    loader = _get_loader(ctx)
    dry_run = common.get_dry_run(ctx)
    existing = _require_skill(loader, name)

    resolved_body = _resolve_body(body, body_file)
    # ``trigger is None`` → keep existing; a (possibly empty) list → replace.
    new_triggers: list[SkillTrigger] | None = (
        _parse_triggers(trigger) if trigger is not None else None
    )

    if description is None and resolved_body is None and new_triggers is None:
        raise typer.BadParameter("请至少指定 --description / --body / --body-file / --trigger 之一")

    if dry_run:
        changes: list[str] = []
        if description is not None:
            changes.append(f"描述: {existing.description} → {description}")
        if resolved_body is not None:
            changes.append(
                f"正文: {common.short(existing.body, 30)} → {common.short(resolved_body, 30)}"
            )
        if new_triggers is not None:
            changes.append(
                f"触发器: {_format_triggers(existing.triggers)} → {_format_triggers(new_triggers)}"
            )
        get_console().print(
            Panel(
                f"[dry-run] 将更新技能 {name}\n" + "\n".join(changes),
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    try:
        skill = loader.update_skill(
            name,
            description=description,
            body=resolved_body,
            triggers=new_triggers,
        )
    except SkillError as exc:
        get_console().print(f"[red]✗ 更新失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    common.audit(ctx, "skill.update", {"name": name})

    if json_output:
        _emit_json(_skill_to_dict(skill, include_body=True))
        return

    get_console().print(f"[green]✓[/green] 已更新技能 [bold]{name}[/bold]")


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


@app.command("import", help="从外部 SKILL.md 文件导入技能。")
def skill_import(
    ctx: typer.Context,
    file_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="外部 SKILL.md 文件路径",
    ),
    name: str | None = typer.Option(
        None, "--name", "-n", help="导入后重命名技能（默认沿用源文件中的 name）"
    ),
    force: bool = typer.Option(False, "--force", help="目标技能已存在时强制覆盖"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """从外部 ``SKILL.md`` 文件导入一个技能。

    源文件需包含合法的 frontmatter（``---`` 分隔）。可通过 ``--name`` 在
    导入时重命名；若目标名称已存在且未指定 ``--force`` 则报错。
    """

    loader = _get_loader(ctx)
    dry_run = common.get_dry_run(ctx)
    target_name = name

    # Pre-validate the source file so we can fail fast with a clear message.
    try:
        source_skill = parse_skill_file(file_path)
    except SkillError as exc:
        get_console().print(f"[red]✗ 源文件解析失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    final_name = target_name or source_skill.name
    if loader.get(final_name) is not None and not force:
        raise typer.BadParameter(f"技能已存在：{final_name}（使用 --force 覆盖）")

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将导入技能\n"
                f"源文件:  {file_path}\n"
                f"名称:    {final_name}\n"
                f"描述:    {source_skill.description}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    try:
        skill = loader.import_skill(file_path, name=target_name)
    except SkillError as exc:
        get_console().print(f"[red]✗ 导入失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    common.audit(
        ctx,
        "skill.import",
        {"name": skill.name, "source": str(file_path), "renamed": bool(target_name)},
    )

    if json_output:
        _emit_json(_skill_to_dict(skill, include_body=True))
        return

    get_console().print(f"[green]✓[/green] 已导入技能 [bold]{skill.name}[/bold]（{skill.path}）")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

# The system prompt instructs the LLM to produce a complete SKILL.md file with
# ``---`` frontmatter delimiters. Kept at module scope (a plain string, no
# heavy deps) so it is available without importing the model stack.
_GENERATE_SYSTEM_PROMPT = (
    "You are a skill authoring assistant for the JustAgent platform.\n"
    "A 'skill' is a SKILL.md markdown file that extends the agent with "
    "domain-specific, reusable instructions.\n\n"
    "Generate a COMPLETE SKILL.md file. It MUST start with YAML-like "
    "frontmatter delimited by lines containing exactly '---':\n\n"
    "---\n"
    "name: <unique-skill-name>\n"
    "description: <one-line human-readable description>\n"
    "triggers:\n"
    "  - type: keyword\n"
    "    value: <trigger-keyword>\n"
    "---\n\n"
    "Frontmatter fields:\n"
    "- name: required, a short unique identifier (lowercase, hyphen-separated).\n"
    "- description: required, a concise one-line summary.\n"
    "- triggers: optional list. Each item has 'type' (one of 'keyword', "
    "'tool', 'manual') and 'value'. 'keyword' matches user text, 'tool' "
    "matches a tool name, 'manual' is only activated by explicit invocation.\n\n"
    "After the closing '---' delimiter, write the skill BODY as markdown: "
    "clear, actionable, step-by-step instructions the agent should follow when "
    "the skill is activated. Use headings, lists, and examples where helpful.\n\n"
    "Output ONLY the SKILL.md content — begin with '---' and end with the "
    "body. Do NOT wrap the output in code fences and do NOT add any commentary."
)


def _build_model_router(config: AppConfig) -> Any:
    """Build a :class:`ModelRouter`, bridging the legacy ``[llm]`` config.

    Mirrors the approach in ``justagent/cli/commands/fix.py`` so the command
    works whether the user configured ``[[model.backends]]`` or the simpler
    top-level ``[llm]`` section.
    """

    from typing import cast

    from pydantic import HttpUrl

    from justagent.adapters.providers.unified_gateway import (
        _PROVIDER_BASE_URLS as PROVIDER_BASE_URLS,
    )
    from justagent.core.model_router import ModelRouter
    from justagent.models.config import LlmProvider, ModelBackendConfig, Provider

    _provider_map: dict[LlmProvider, Provider] = {
        LlmProvider.OPENAI: Provider.OPENAI,
        LlmProvider.OPENROUTER: Provider.OPENROUTER,
        LlmProvider.OLLAMA: Provider.OLLAMA,
    }
    # ``PROVIDER_BASE_URLS`` is imported from ``unified_gateway._PROVIDER_BASE_URLS``
    # (single source of truth for provider -> base URL defaults).

    if not config.model.backends and config.llm.provider in _provider_map:
        backend_provider = _provider_map[config.llm.provider]
        base_url = config.llm.base_url or cast(HttpUrl, PROVIDER_BASE_URLS[backend_provider])
        legacy_backend = ModelBackendConfig(
            provider=backend_provider,
            base_url=base_url,
            api_key=config.llm.api_key,
            api_version=config.llm.api_version,
            model=config.llm.model,
            timeout=config.llm.timeout,
        )
        compat_model = config.model.model_copy(update={"backends": [legacy_backend]})
        compat_config = config.model_copy(update={"model": compat_model})
        return ModelRouter(compat_config)
    return ModelRouter(config)


def _extract_skill_markdown(raw: str) -> str:
    """Extract the SKILL.md portion from raw LLM output.

    The LLM is instructed to start with ``---``, but defensively strip any
    leading prose / code-fence line so the frontmatter parser succeeds. A
    trailing code fence left over from wrapping is also removed.
    """

    lines = raw.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "---":
            start = i
            break
    if start is None:
        return raw  # no frontmatter found — let the parser raise a clear error
    content = "\n".join(lines[start:])
    # Remove a trailing ``` fence if the LLM wrapped the whole output.
    content = re.sub(r"\n```\s*$", "", content)
    return content


@app.command("generate", help="调用 LLM 自动生成一个技能。")
def skill_generate(
    ctx: typer.Context,
    description: str = typer.Argument(..., help="用自然语言描述想要生成的技能"),
    name: str | None = typer.Option(
        None, "--name", "-n", help="覆盖生成技能的名称（默认使用 LLM 生成的 name）"
    ),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """调用 LLM 自动生成一个 ``SKILL.md`` 技能。

    根据自然语言 *description* 构造提示词，要求模型按 ``---`` frontmatter
    格式输出完整的技能文件；解析输出后调用 :meth:`SkillLoader.create_skill`
    落盘。需要已配置模型后端（``[[model.backends]]`` 或 ``[llm]`` 段）。
    """

    # Lazy imports — keep heavy / optional LLM deps out of module import time
    # so `justagent skill ...` (without a model) still loads under the
    # auto-discovery in ``commands/__init__.py``.
    from justagent.adapters.model_gateway import (
        ChatCompletionRequest,
        ChatMessage,
        ModelGateway,
    )
    from justagent.exceptions import ModelGatewayError

    config = common.get_config(ctx)
    loader = _get_loader(ctx)
    dry_run = common.get_dry_run(ctx)
    verbose = common.get_verbose(ctx)
    console = get_console()

    user_prompt = (
        f"Generate a JustAgent skill for the following request:\n\n"
        f"{description}\n\n"
        f"Choose an appropriate unique `name` (unless a name is implied), "
        f"write a concise `description`, add sensible `triggers`, and produce "
        f"a thorough markdown body. Respond with only the SKILL.md content."
    )

    if dry_run:
        console.print(
            Panel(
                f"[dry-run] 将调用 LLM 生成技能\n"
                f"描述: {description}\n"
                f"覆盖名称: {name or '(由模型决定)'}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    router = _build_model_router(config)

    messages = [
        ChatMessage(role="system", content=_GENERATE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]
    raw_output: str
    model_used: str | None = None

    try:
        # ``select_backend`` returns a concrete ``ModelGateway`` (with health
        # checking + fallback tier logic already applied) so we can issue a
        # ``ChatCompletionRequest`` directly.
        gateway: ModelGateway | None = None
        with suppress(Exception):
            gateway = router.select_backend()

        if gateway is not None:
            request = ChatCompletionRequest(
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            response = gateway.chat(request)
            raw_output = response.content
            model_used = getattr(response, "model", None)
        else:
            # Fall back to the router's own chat() which iterates all backends.
            raw_output = router.chat(messages, "skill-generate")
    except ModelGatewayError as exc:
        console.print(f"[red]✗ 模型后端不可用：{exc}[/red]")
        console.print("[dim]请先配置模型后端（[[model.backends]] 或 [llm] 段）后重试。[/dim]")
        raise typer.Exit(code=1) from exc
    finally:
        with suppress(Exception):
            router.close()

    if verbose:
        console.print(
            Panel(
                raw_output,
                title="LLM 原始输出",
                border_style="dim",
            )
        )

    # Parse the LLM output into a Skill via the shared parser.
    skill_content = _extract_skill_markdown(raw_output)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(skill_content)
        tmp_path = Path(tmp.name)
    try:
        try:
            parsed = parse_skill_file(tmp_path)
        except SkillError as exc:
            console.print(f"[red]✗ 无法解析模型输出为 SKILL.md：{exc}[/red]")
            console.print("[dim]模型原始输出见上方，请手动修正后使用 `skill import` 导入。[/dim]")
            raise typer.Exit(code=1) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    final_name = name or parsed.name
    if loader.get(final_name) is not None:
        console.print(
            f"[red]✗ 技能已存在：{final_name}（如需覆盖请先删除或使用 `skill update`）[/red]"
        )
        raise typer.Exit(code=1)

    try:
        skill = loader.create_skill(
            name=final_name,
            description=parsed.description,
            body=parsed.body,
            triggers=list(parsed.triggers),
        )
    except SkillError as exc:
        console.print(f"[red]✗ 创建失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    common.audit(
        ctx,
        "skill.generate",
        {
            "name": final_name,
            "description": parsed.description,
            "model": model_used,
        },
    )

    if json_output:
        _emit_json(_skill_to_dict(skill, include_body=True))
        return

    console.print(f"[green]✓[/green] 已生成并创建技能 [bold]{skill.name}[/bold]（{skill.path}）")


__all__ = ["app", "register"]
