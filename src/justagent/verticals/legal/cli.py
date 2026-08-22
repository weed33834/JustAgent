"""``justagent judicial`` command — litigation case management, evidence review,
legal document generation, and statute knowledge base.

Exposes the :mod:`justagent.verticals.legal` package through the CLI:

* ``justagent judicial case``     — create / list / show / import case files.
* ``justagent judicial evidence`` — add / list / review / analyze evidence.
* ``justagent judicial doc``      — generate legal documents from templates.
* ``justagent judicial law``      — manage and search the legal knowledge base.

The judicial registries (:class:`CaseManager`, :class:`EvidenceChain`,
:class:`LegalKnowledgeBase`) are in-memory by design. To make the CLI usable
across invocations, this module transparently persists state to a JSON file
(``<project_root>/.justagent/judicial_state.json`` by default, overridable via
the ``MYAGENT_JUDICIAL_STATE`` environment variable). Each command loads the
state, performs its operation, and writes the state back (read-only commands
skip the write).

The module follows the same conventions as the other ``justagent`` commands:
a ``register(parent: typer.Typer)`` entry point, ``ctx.obj`` for shared
config / audit / verbosity, Rich tables for listings, and Google-style
docstrings with full type hints.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from justagent.cli.display import get_console
from justagent.models.config import AppConfig
from justagent.verticals.legal.case_manager import (
    CaseFile,
    CaseManager,
    CaseManagerError,
    CaseMaterial,
    CaseStatus,
    MaterialType,
)
from justagent.verticals.legal.document_generator import (
    DocumentGenerationError,
    GeneratedDocument,
    LegalDocumentGenerator,
    LegalDocumentTemplateManager,
    LegalDocumentType,
)
from justagent.verticals.legal.evidence import (
    Admissibility,
    ChainAnalysisResult,
    ChainAuditResult,
    Evidence,
    EvidenceAuditor,
    EvidenceChain,
    EvidenceError,
    EvidenceRelation,
    EvidenceRelationType,
    EvidenceReviewer,
    EvidenceType,
    ProbativeStrength,
    ReviewResult,
)
from justagent.verticals.legal.legal_knowledge import (
    ArticleStatus,
    LegalArticle,
    LegalCase,
    LegalDomain,
    LegalKnowledgeBase,
    LegalKnowledgeError,
)

# ---------------------------------------------------------------------------
# Typer sub-apps
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="judicial",
    help="司法案件管理：案件卷宗、证据审查、文书生成与法律知识库。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

case_app = typer.Typer(
    name="case",
    help="案件卷宗管理（创建 / 列出 / 查看 / 导入材料）。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

evidence_app = typer.Typer(
    name="evidence",
    help="证据管理（添加 / 列出 / 审查 / 证据链分析）。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

doc_app = typer.Typer(
    name="doc",
    help="法律文书生成（按模板生成 / 查看可用模板）。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

law_app = typer.Typer(
    name="law",
    help="法律知识库（添加法条 / 检索法条）。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def register(parent: typer.Typer) -> None:
    """Register the ``judicial`` command group and its sub-groups."""

    app.add_typer(case_app, name="case")
    app.add_typer(evidence_app, name="evidence")
    app.add_typer(doc_app, name="doc")
    app.add_typer(law_app, name="law")
    parent.add_typer(app, name="judicial")


# ---------------------------------------------------------------------------
# Context accessors — defensive like audit.py (build fallbacks when ctx.obj
# is missing keys, e.g. when a command is invoked directly in tests).
# ---------------------------------------------------------------------------


def _get_config(ctx: typer.Context) -> AppConfig:
    """Return the ``AppConfig`` from ``ctx.obj`` or a default instance."""

    obj = getattr(ctx, "obj", None)
    config = obj.get("config") if obj else None
    return config if isinstance(config, AppConfig) else AppConfig()


def _get_audit(ctx: typer.Context) -> Any:
    """Return the audit logger from ``ctx.obj``, or ``None``."""

    obj = getattr(ctx, "obj", None)
    return obj.get("audit_logger") if obj else None


def _get_verbose(ctx: typer.Context) -> bool:
    """Return the global ``--verbose`` flag."""

    obj = getattr(ctx, "obj", None)
    return bool(obj.get("verbose")) if obj else False


def _get_dry_run(ctx: typer.Context) -> bool:
    """Return the global ``--dry-run`` flag."""

    obj = getattr(ctx, "obj", None)
    return bool(obj.get("dry_run")) if obj else False


def _audit(ctx: typer.Context, event: str, payload: dict[str, Any] | None = None) -> None:
    """Record an audit event best-effort (never raises)."""

    audit = _get_audit(ctx)
    if audit is None:
        return
    with suppress(Exception):  # audit must never break a command
        audit.record(event, payload or {})


# ---------------------------------------------------------------------------
# Persistence layer
# ---------------------------------------------------------------------------


def _state_path(ctx: typer.Context) -> Path:
    """Resolve the judicial state file path.

    Priority: ``JUSTAGENT_JUDICIAL_STATE`` (or legacy ``MYAGENT_JUDICIAL_STATE``)
    env var > ``<project_root>/.justagent/judicial_state.json`` > ``./.justagent/
    judicial_state.json``.
    """

    env = os.environ.get("JUSTAGENT_JUDICIAL_STATE") or os.environ.get("MYAGENT_JUDICIAL_STATE")
    if env:
        return Path(env).expanduser()
    config = _get_config(ctx)
    root = Path(getattr(config, "project_root", ".") or ".")
    return root / ".justagent" / "judicial_state.json"


class _JudicialState:
    """Container holding the judicial managers with JSON save/restore.

    The underlying managers (:class:`CaseManager`, :class:`EvidenceChain`,
    :class:`LegalKnowledgeBase`) only offer incremental mutation APIs and
    generate fresh IDs on creation. To round-trip persisted state we restore
    the internal registries directly — this is the integration layer's
    concern and is clearly isolated here rather than spread across commands.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.case_manager = CaseManager()
        self.evidence_chain = EvidenceChain()
        self.knowledge_base = LegalKnowledgeBase()
        # The reviewer always wraps the same chain so reviews persist.
        self.reviewer = EvidenceReviewer(self.evidence_chain)

    @classmethod
    def load(cls, path: Path) -> _JudicialState:
        """Load state from *path*, or return an empty state on any error."""

        state = cls(path)
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            get_console().print(
                f"[yellow]⚠ 无法读取司法状态文件 {path}：{exc}（将以空状态启动）[/yellow]"
            )
            return state
        state._restore(data)
        return state

    def save(self) -> None:
        """Persist the current registries to the state file."""

        data = self._snapshot()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            get_console().print(
                f"[red]✗ 无法保存司法状态文件 {self.path}：{exc}[/red]", style="red"
            )

    # -- snapshot / restore ------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        """Serialise all registries to a JSON-friendly dict."""

        cm = self.case_manager
        ec = self.evidence_chain
        kb = self.knowledge_base
        return {
            "schema_version": 1,
            "saved_at": time.time(),
            "cases": [c.model_dump(mode="json") for c in cm._cases.values()],
            "materials": [m.model_dump(mode="json") for m in cm._materials.values()],
            "evidence": [e.model_dump(mode="json") for e in ec._evidence.values()],
            "relations": [r.model_dump(mode="json") for r in ec._relations.values()],
            "articles": [a.model_dump(mode="json") for a in kb._articles.values()],
            "legal_cases": [c.model_dump(mode="json") for c in kb._cases.values()],
        }

    def _restore(self, data: dict[str, Any]) -> None:
        """Rebuild the managers' internal registries from *data*."""

        cm = self.case_manager
        ec = self.evidence_chain
        kb = self.knowledge_base

        for raw in data.get("cases", []):
            case = CaseFile.model_validate(raw)
            cm._cases[case.id] = case
            if case.case_number:
                cm._case_number_index[case.case_number] = case.id

        for raw in data.get("materials", []):
            material = CaseMaterial.model_validate(raw)
            cm._materials[material.id] = material

        for raw in data.get("evidence", []):
            evidence = Evidence.model_validate(raw)
            ec._evidence[evidence.id] = evidence

        for raw in data.get("relations", []):
            relation = EvidenceRelation.model_validate(raw)
            ec._relations[relation.id] = relation

        for raw in data.get("articles", []):
            article = LegalArticle.model_validate(raw)
            kb._articles[article.id] = article
            key = LegalKnowledgeBase._article_key(article.law_name, article.article_number)
            kb._article_number_index[key] = article.id

        for raw in data.get("legal_cases", []):
            legal_case = LegalCase.model_validate(raw)
            kb._cases[legal_case.id] = legal_case
            kb._case_number_index[legal_case.case_number] = legal_case.id


@contextmanager
def _state_session(ctx: typer.Context, *, save: bool = True) -> Iterator[_JudicialState]:
    """Load state, yield it, and persist on exit (unless dry-run or read-only).

    Args:
        ctx: The Typer context (for path resolution and dry-run flag).
        save: When ``True`` (default) the state is written back on a clean
            exit. Read-only commands pass ``False`` to avoid needless writes.
    """

    state = _JudicialState.load(_state_path(ctx))
    try:
        yield state
    finally:
        if save and not _get_dry_run(ctx):
            state.save()


# ---------------------------------------------------------------------------
# Output / formatting helpers
# ---------------------------------------------------------------------------


def _format_ts(ts: float) -> str:
    """Format a Unix timestamp as ``YYYY-MM-DD HH:MM``."""

    if not ts:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _short(text: str, width: int) -> str:
    """Truncate *text* to *width* chars, appending ``…`` when cut."""

    text = (text or "").replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _id_short(value: str, width: int = 8) -> str:
    """Show the leading characters of an ID for compact table display."""

    return value[:width] if value else "-"


def _require_case(state: _JudicialState, case_id: str) -> CaseFile:
    """Return a case or raise a Typer error (exit 1)."""

    case = state.case_manager.get_case(case_id)
    if case is None:
        # Tolerate a partial / short ID match against known case IDs.
        matches = [c for c in state.case_manager.list_cases() if c.id.startswith(case_id)]
        if len(matches) == 1:
            return matches[0]
        raise typer.BadParameter(f"未找到案件：{case_id}")
    return case


def _parse_enum(value: str, enum_cls: type[Any], label: str) -> Any:
    """Parse *value* into an enum member, raising BadParameter on failure."""

    try:
        return enum_cls(value)
    except ValueError:
        valid = ", ".join(str(m.value) for m in enum_cls)
        raise typer.BadParameter(f"无效的 {label}：{value!r}（可选值：{valid}）") from None


# ---------------------------------------------------------------------------
# Case commands
# ---------------------------------------------------------------------------


@case_app.command("create", help="创建一个新的案件卷宗。")
def case_create(
    ctx: typer.Context,
    case_number: str = typer.Option(
        ..., "--case-number", "-n", help="案号，例如 (2024)京01民初1号"
    ),
    cause: str = typer.Option(..., "--cause", help="案由，例如 买卖合同纠纷"),
    court: str = typer.Option("", "--court", help="审理法院"),
    judge: str = typer.Option("", "--judge", help="承办法官"),
    description: str = typer.Option("", "--description", "-d", help="案件描述"),
    domain: str = typer.Option("", "--domain", help="法律领域（civil/criminal/...）"),
) -> None:
    """创建一个新的案件卷宗。

    法官与描述等 CaseFile 未直接建模的字段会存入案件 ``metadata``。
    """

    verbose = _get_verbose(ctx)
    dry_run = _get_dry_run(ctx)

    metadata: dict[str, Any] = {}
    if judge:
        metadata["judge"] = judge
    if description:
        metadata["description"] = description

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将创建案件\n案号: {case_number}\n案由: {cause}\n"
                f"法院: {court or '(未填写)'}\n法官: {judge or '(未填写)'}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    with _state_session(ctx) as state:
        try:
            case = state.case_manager.create_case(
                case_number=case_number,
                cause_of_action=cause,
                court=court,
                domain=domain,
                metadata=metadata,
            )
        except CaseManagerError as exc:
            get_console().print(f"[red]✗ {exc}[/red]")
            raise typer.Exit(code=1) from exc

    _audit(
        ctx,
        "judicial.case.create",
        {"case_id": case.id, "case_number": case_number, "cause": cause},
    )

    console = get_console()
    if verbose:
        console.print(
            Panel(
                f"案件 ID:    {case.id}\n"
                f"案号:       {case.case_number}\n"
                f"案由:       {case.cause_of_action}\n"
                f"法院:       {case.court or '-'}\n"
                f"法官:       {judge or '-'}\n"
                f"描述:       {description or '-'}\n"
                f"状态:       {case.status.value}\n"
                f"创建时间:   {_format_ts(case.created_at)}",
                title=f"已创建案件 {case.case_number or case.id[:8]}",
                border_style="green",
            )
        )
    else:
        console.print(
            f"[green]✓[/green] 已创建案件 "
            f"[bold]{case.case_number or case.id[:8]}[/bold]（ID: {case.id}）"
        )


@case_app.command("list", help="列出所有案件卷宗。")
def case_list(
    ctx: typer.Context,
    status: str | None = typer.Option(
        None, "--status", help="按状态过滤（draft/active/under_review/closed/archived）"
    ),
    cause: str | None = typer.Option(None, "--cause", help="按案由过滤"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """列出所有案件卷宗，可按状态或案由过滤。"""

    status_filter = _parse_enum(status, CaseStatus, "状态") if status is not None else None

    with _state_session(ctx, save=False) as state:
        cases = state.case_manager.list_cases(status=status_filter, cause_of_action=cause)

    if json_output:
        rows = [
            {
                "id": c.id,
                "case_number": c.case_number,
                "cause_of_action": c.cause_of_action,
                "court": c.court,
                "status": c.status.value,
                "parties": len(c.parties),
                "materials": len(c.material_ids),
                "evidence": len(c.evidence_ids),
                "updated_at": c.updated_at,
            }
            for c in cases
        ]
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    console = get_console()
    if not cases:
        console.print("[dim]暂无案件。使用 `judicial case create` 创建一个。[/dim]")
        return

    table = Table(title=f"案件列表（共 {len(cases)} 件）", border_style="cyan")
    table.add_column("ID", style="dim", width=8)
    table.add_column("案号", style="white")
    table.add_column("案由", style="white")
    table.add_column("法院", style="white")
    table.add_column("状态", style="bold")
    table.add_column("当事人", justify="right")
    table.add_column("材料", justify="right")
    table.add_column("证据", justify="right")
    table.add_column("更新时间", style="dim")

    for c in cases:
        status_style = {
            CaseStatus.ACTIVE: "green",
            CaseStatus.UNDER_REVIEW: "yellow",
            CaseStatus.CLOSED: "red",
            CaseStatus.ARCHIVED: "dim",
            CaseStatus.DRAFT: "blue",
        }.get(c.status, "white")
        table.add_row(
            _id_short(c.id),
            c.case_number or "-",
            _short(c.cause_of_action, 18),
            _short(c.court, 18),
            Text(c.status.value, style=status_style),
            str(len(c.parties)),
            str(len(c.material_ids)),
            str(len(c.evidence_ids)),
            _format_ts(c.updated_at),
        )
    console.print(table)


@case_app.command("show", help="查看案件详情。")
def case_show(
    ctx: typer.Context,
    case_id: str = typer.Argument(..., help="案件 ID（支持前缀匹配）"),
) -> None:
    """查看指定案件的详细信息，包括当事人、材料、时间线等。"""

    with _state_session(ctx, save=False) as state:
        case = _require_case(state, case_id)
        materials = state.case_manager.list_materials(case.id)
        evidence = state.evidence_chain.list_evidence(case_id=case.id)

    console = get_console()
    judge = case.metadata.get("judge", "-")
    description = case.metadata.get("description", "-")

    header = (
        f"[bold]案件 ID:[/bold] {case.id}\n"
        f"[bold]案号:[/bold]     {case.case_number or '-'}\n"
        f"[bold]案由:[/bold]     {case.cause_of_action or '-'}\n"
        f"[bold]法院:[/bold]     {case.court or '-'}\n"
        f"[bold]承办法官:[/bold] {judge}\n"
        f"[bold]领域:[/bold]     {case.domain or '-'}\n"
        f"[bold]状态:[/bold]     {case.status.value}\n"
        f"[bold]创建时间:[/bold] {_format_ts(case.created_at)}\n"
        f"[bold]更新时间:[/bold] {_format_ts(case.updated_at)}\n"
        f"[bold]描述:[/bold]     {description}"
    )
    console.print(
        Panel(header, title=f"案件 {case.case_number or case.id[:8]}", border_style="cyan")
    )

    # Parties
    if case.parties:
        ptable = Table(title="当事人", border_style="blue", show_lines=False)
        ptable.add_column("角色", style="bold")
        ptable.add_column("姓名/名称")
        ptable.add_column("联系方式")
        ptable.add_column("法定代表人")
        for p in case.parties:
            ptable.add_row(
                p.role.value,
                p.name,
                _short(p.contact, 24) or "-",
                p.legal_representative or "-",
            )
        console.print(ptable)
    else:
        console.print("[dim]当事人：（暂无）[/dim]")

    # Materials
    if materials:
        mtable = Table(title="已导入材料", border_style="blue")
        mtable.add_column("ID", style="dim", width=8)
        mtable.add_column("标题")
        mtable.add_column("类型")
        mtable.add_column("导入时间", style="dim")
        for m in materials:
            mtable.add_row(
                _id_short(m.id),
                _short(m.document.title, 30),
                m.material_type.value,
                _format_ts(m.imported_at),
            )
        console.print(mtable)
    else:
        console.print("[dim]已导入材料：（暂无，使用 `judicial case import` 导入）[/dim]")

    # Evidence
    if evidence:
        etable = Table(title="关联证据", border_style="blue")
        etable.add_column("ID", style="dim", width=8)
        etable.add_column("名称")
        etable.add_column("类型")
        etable.add_column("可采性")
        etable.add_column("证明力")
        for e in evidence:
            etable.add_row(
                _id_short(e.id),
                _short(e.name, 24),
                e.type.value,
                e.admissibility.value,
                e.probative_strength.value,
            )
        console.print(etable)
    else:
        console.print("[dim]关联证据：（暂无，使用 `judicial evidence add` 添加）[/dim]")

    # Timeline
    if case.timeline:
        console.print("\n[bold]时间线：[/bold]")
        for ev in case.timeline:
            console.print(f"  - {ev.date or '未知日期'}: {_short(ev.description, 70)}")


@case_app.command("import", help="将文件导入案件作为材料。")
def case_import(
    ctx: typer.Context,
    case_id: str = typer.Argument(..., help="目标案件 ID（支持前缀匹配）"),
    file_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="要导入的文件路径"
    ),
    material_type: str = typer.Option(
        "other",
        "--type",
        "-t",
        help="材料类型（complaint/defense/evidence/judgment/contract/correspondence/other）",
    ),
    title: str | None = typer.Option(None, "--title", help="材料标题（默认取文件名）"),
    notes: str = typer.Option("", "--notes", help="备注"),
    no_extract: bool = typer.Option(
        False, "--no-extract", help="跳过自动结构化抽取（当事人/诉求/事实）"
    ),
) -> None:
    """将一个文件导入案件卷宗。

    支持 PDF、Word、Excel、PPT、Markdown、HTML 与纯文本。导入后会自动
    抽取当事人、诉讼请求、事实要素与时间线（除非指定 ``--no-extract``）。
    """

    mtype = _parse_enum(material_type, MaterialType, "材料类型")
    dry_run = _get_dry_run(ctx)

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将把文件 {file_path} 作为 {mtype.value} 导入案件 {case_id}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    with _state_session(ctx) as state:
        _require_case(state, case_id)  # validate existence
        try:
            material = state.case_manager.import_file(
                case_id,
                file_path,
                material_type=mtype,
                title=title,
                notes=notes,
                auto_extract=not no_extract,
            )
        except (CaseManagerError, FileNotFoundError) as exc:
            get_console().print(f"[red]✗ 导入失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _audit(
        ctx,
        "judicial.case.import",
        {
            "case_id": case_id,
            "file": str(file_path),
            "material_id": material.id,
            "material_type": mtype.value,
        },
    )

    doc = material.document
    console = get_console()
    console.print(
        f"[green]✓[/green] 已导入材料 [bold]{_short(doc.title, 40)}[/bold]"
        f"（类型: {mtype.value}，ID: {material.id}）"
    )
    if doc.content:
        console.print(
            f"[dim]  解析内容 {len(doc.content)} 字符"
            f"{'（已自动抽取结构化信息）' if not no_extract else ''}[/dim]"
        )


# ---------------------------------------------------------------------------
# Evidence commands
# ---------------------------------------------------------------------------


@evidence_app.command("add", help="向案件添加一项证据。")
def evidence_add(
    ctx: typer.Context,
    case_id: str = typer.Argument(..., help="目标案件 ID（支持前缀匹配）"),
    name: str = typer.Option(..., "--name", help="证据名称，例如 购销合同原件"),
    type: str = typer.Option(
        "documentary",
        "--type",
        "-t",
        help="证据类型（documentary/physical/testimony/expert_opinion/"
        "inspection_record/audio_visual/electronic_data）",
    ),
    description: str = typer.Option("", "--description", "-d", help="证据描述"),
    source: str = typer.Option("", "--source", help="证据来源"),
    collector: str = typer.Option("", "--collector", help="收集人"),
    collection_date: str = typer.Option("", "--collection-date", help="收集日期（YYYY-MM-DD）"),
    collection_method: str = typer.Option("", "--collection-method", help="收集方式"),
    proving_object: str = typer.Option("", "--proving-object", help="证明对象"),
    proving_target: str = typer.Option("", "--proving-target", help="证明目的"),
) -> None:
    """向案件添加一项证据并自动关联到该案件。"""

    etype = _parse_enum(type, EvidenceType, "证据类型")
    dry_run = _get_dry_run(ctx)

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将向案件 {case_id} 添加证据「{name}」（类型: {etype.value}）",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    with _state_session(ctx) as state:
        _require_case(state, case_id)
        evidence = Evidence(
            name=name,
            type=etype,
            description=description,
            source=source,
            collector=collector,
            collection_date=collection_date,
            collection_method=collection_method,
            proving_object=proving_object,
            proving_target=proving_target,
            case_id=case_id,
        )
        state.evidence_chain.add_evidence(evidence)
        state.case_manager.link_evidence(case_id, evidence.id)

    _audit(
        ctx,
        "judicial.evidence.add",
        {"case_id": case_id, "evidence_id": evidence.id, "name": name, "type": etype.value},
    )

    get_console().print(
        f"[green]✓[/green] 已添加证据 [bold]{name}[/bold]"
        f"（类型: {etype.value}，ID: {evidence.id}）到案件 {case_id[:8]}"
    )


@evidence_app.command("list", help="列出案件的所有证据。")
def evidence_list(
    ctx: typer.Context,
    case_id: str = typer.Argument(..., help="案件 ID（支持前缀匹配）"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """列出指定案件的所有证据及其审查状态。"""

    with _state_session(ctx, save=False) as state:
        _require_case(state, case_id)
        evidence = state.evidence_chain.list_evidence(case_id=case_id)

    if json_output:
        rows = [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type.value,
                "proving_object": e.proving_object,
                "admissibility": e.admissibility.value,
                "probative_strength": e.probative_strength.value,
                "relevance_score": e.relevance_score,
                "probative_score": e.probative_score,
                "reviewed": e.reviewed,
            }
            for e in evidence
        ]
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    console = get_console()
    if not evidence:
        console.print("[dim]该案件暂无证据。[/dim]")
        return

    table = Table(title=f"证据列表（共 {len(evidence)} 项）", border_style="cyan")
    table.add_column("ID", style="dim", width=8)
    table.add_column("名称", style="white")
    table.add_column("类型")
    table.add_column("证明对象")
    table.add_column("可采性", style="bold")
    table.add_column("证明力", style="bold")
    table.add_column("关联性", justify="right")
    table.add_column("已审查", justify="center")

    for e in evidence:
        adm_style = (
            "green"
            if e.admissibility is Admissibility.ADMISSIBLE
            else "red"
            if e.admissibility is Admissibility.INADMISSIBLE
            else "yellow"
        )
        ps_style = {
            ProbativeStrength.HIGH: "green",
            ProbativeStrength.MEDIUM: "yellow",
            ProbativeStrength.LOW: "red",
            ProbativeStrength.INSUFFICIENT: "red",
        }.get(e.probative_strength, "white")
        table.add_row(
            _id_short(e.id),
            _short(e.name, 20),
            e.type.value,
            _short(e.proving_object, 18) or "-",
            Text(e.admissibility.value, style=adm_style),
            Text(e.probative_strength.value, style=ps_style),
            f"{e.relevance_score:.0%}" if e.reviewed else "-",
            "✓" if e.reviewed else "—",
        )
    console.print(table)


@evidence_app.command("review", help="审查单项证据（合法性、关联性、证明力）。")
def evidence_review(
    ctx: typer.Context,
    evidence_id: str = typer.Argument(..., help="证据 ID（支持前缀匹配）"),
) -> None:
    """对单项证据执行全面审查并更新其评估结果。"""

    with _state_session(ctx) as state:
        evidence = _resolve_evidence(state, evidence_id)
        try:
            result = state.reviewer.review(evidence.id)
        except EvidenceError as exc:
            get_console().print(f"[red]✗ 审查失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _audit(
        ctx,
        "judicial.evidence.review",
        {
            "evidence_id": result.evidence_id,
            "admissibility": result.admissibility.value,
            "probative_strength": result.probative_strength.value,
        },
    )

    _print_review_result(evidence, result)


@evidence_app.command("analyze", help="分析案件证据链完整度。")
def evidence_analyze(
    ctx: typer.Context,
    case_id: str = typer.Argument(..., help="案件 ID（支持前缀匹配）"),
) -> None:
    """分析案件证据链的完整度、矛盾与缺口。"""

    with _state_session(ctx, save=False) as state:
        _require_case(state, case_id)
        try:
            result = state.evidence_chain.analyze(case_id)
        except EvidenceError as exc:
            get_console().print(f"[red]✗ 分析失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc
        contradictions = state.evidence_chain.list_relations(
            relation_type=EvidenceRelationType.CONTRADICTS
        )

    _print_chain_analysis(result, contradictions)


def _audit_report_markdown(audit: ChainAuditResult) -> str:
    """Render a :class:`ChainAuditResult` as a markdown report."""
    lines = [
        f"# 证据链审计报告（案件 {audit.case_id}）",
        "",
        f"- **审计结论**: {audit.verdict}",
        f"- **完整度**: {audit.chain.completeness_score:.1%}"
        f"（{audit.chain.admissible_evidence}/{audit.chain.total_evidence} 项可采信）",
        f"- **矛盾**: {len(audit.chain.contradictions)} 处；**缺口**: {len(audit.chain.gaps)} 处",
        "",
    ]
    for title, items in (
        ("保管链条问题", audit.custody_issues),
        ("时间线问题", audit.timeline_issues),
        ("同源佐证警告", audit.independence_warnings),
    ):
        lines.append(f"## {title}（{len(items)}）")
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- 无")
        lines.append("")
    if audit.claim_coverage:
        lines.append(
            f"## 诉讼请求覆盖（{sum(c.covered for c in audit.claim_coverage)}/{len(audit.claim_coverage)} 已覆盖）"
        )
        for c in audit.claim_coverage:
            mark = "✅" if c.covered else "❌"
            lines.append(f"- {mark} {c.claim_description}" + (f"——{c.note}" if c.note else ""))
        lines.append("")
    lines.append(f"> {audit.summary}")
    return "\n".join(lines)


@evidence_app.command("audit", help="全面审计案件证据链（保管/时间线/独立性/诉请覆盖）。")
def evidence_audit(
    ctx: typer.Context,
    case_id: str = typer.Argument(..., help="案件 ID（支持前缀匹配）"),
    fmt: str = typer.Option("rich", "--format", "-f", help="输出格式：rich/json/markdown"),
    output: Path | None = typer.Option(None, "--output", "-o", help="写入文件；省略则输出到终端"),
) -> None:
    """运行确定性证据链审计，无需 LLM。

    覆盖四类检查：保管链条完整性、时间线一致性、同源佐证识别、
    诉讼请求-证据覆盖映射。结论分 通过 / 有瑕疵 / 严重缺陷 三档。
    """
    with _state_session(ctx, save=False) as state:
        case = _require_case(state, case_id)
        filing_date = ""
        for event in case.timeline:
            if getattr(event, "description", "") == "立案":
                filing_date = getattr(event, "date", "")
                break
        auditor = EvidenceAuditor(state.evidence_chain)
        try:
            audit = auditor.audit_case(case.id, claims=list(case.claims), filing_date=filing_date)
        except EvidenceError as exc:
            get_console().print(f"[red]✗ 审计失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _audit(
        ctx,
        "judicial.evidence.audit",
        {"case_id": audit.case_id, "verdict": audit.verdict},
    )

    if output is not None or fmt in ("json", "markdown"):
        payload = (
            audit.model_dump_json(indent=2) if fmt == "json" else _audit_report_markdown(audit)
        )
        if output is not None:
            output.write_text(payload, encoding="utf-8")
            get_console().print(f"[green]✓ 审计报告已写入 {output}[/green]")
        else:
            get_console().print(payload)
        return

    # rich 输出
    console = get_console()
    color = {"通过": "green", "有瑕疵": "yellow"}.get(audit.verdict, "red")
    console.print(
        Panel.fit(
            f"[bold {color}]审计结论：{audit.verdict}[/bold {color}]   "
            f"完整度 {audit.chain.completeness_score:.1%}",
            title=f"证据链审计 · {case.case_number or audit.case_id[:8]}",
        )
    )
    issue_groups = (
        ("保管链条", audit.custody_issues),
        ("时间线", audit.timeline_issues),
        ("同源佐证", audit.independence_warnings),
    )
    for title, items in issue_groups:
        console.print(f"\n[bold]{title}（{len(items)}）[/bold]")
        for item in items:
            console.print(f"  • {item}")
    if audit.claim_coverage:
        covered = sum(c.covered for c in audit.claim_coverage)
        console.print(f"\n[bold]诉讼请求覆盖（{covered}/{len(audit.claim_coverage)}）[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("状态")
        table.add_column("诉讼请求")
        table.add_column("支持证据数")
        for c in audit.claim_coverage:
            table.add_row(
                "✅" if c.covered else "❌",
                c.claim_description,
                str(len(c.supporting_evidence_ids)),
            )
        console.print(table)
    console.print(f"\n[dim]{audit.summary}[/dim]")


# ---------------------------------------------------------------------------
# Document commands
# ---------------------------------------------------------------------------


@doc_app.command("generate", help="为案件生成法律文书。")
def doc_generate(
    ctx: typer.Context,
    case_id: str = typer.Argument(..., help="案件 ID（支持前缀匹配）"),
    doc_type: str = typer.Argument(
        ...,
        help="文书类型（indictment/statement_of_defense/judgment/ruling/"
        "mediation_agreement/agency_opinion/legal_opinion/evidence_list/"
        "cross_examination_opinion）",
    ),
    title: str | None = typer.Option(None, "--title", help="文书标题"),
    output: Path | None = typer.Option(None, "--output", "-o", help="将生成的文书文本写入指定文件"),
    no_verify: bool = typer.Option(False, "--no-verify", help="跳过法条引用校验"),
) -> None:
    """根据案件上下文与模板生成法律文书。

    未配置 LLM 网关时使用模板填充模式；配置网关后可进行 LLM 辅助起草。
    生成的文书会自动校验其中引用的法条是否存在于法律知识库中。
    """

    dtype = _parse_enum(doc_type, LegalDocumentType, "文书类型")

    with _state_session(ctx, save=False) as state:
        _require_case(state, case_id)
        generator = LegalDocumentGenerator(
            state.case_manager,
            evidence_chain=state.evidence_chain,
            knowledge_base=state.knowledge_base,
        )
        try:
            document = generator.generate(
                case_id,
                dtype,
                title=title,
                verify=not no_verify,
            )
        except DocumentGenerationError as exc:
            get_console().print(f"[red]✗ 文书生成失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _audit(
        ctx,
        "judicial.doc.generate",
        {
            "case_id": case_id,
            "doc_type": dtype.value,
            "doc_id": document.id,
            "citations": len(document.citations),
            "all_valid": document.all_citations_valid,
        },
    )

    _print_generated_document(document, output)


@doc_app.command("list-templates", help="列出可用的法律文书模板。")
def doc_list_templates(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """列出所有内置的法律文书模板。"""

    manager = LegalDocumentTemplateManager()
    templates = manager.list_templates()

    if json_output:
        rows = [
            {
                "id": t.id,
                "doc_type": t.doc_type.value,
                "name": t.name,
                "description": t.description,
                "sections": len(t.sections),
                "placeholders": t.placeholders,
            }
            for t in templates
        ]
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    console = get_console()
    table = Table(title=f"法律文书模板（共 {len(templates)} 个）", border_style="cyan")
    table.add_column("文书类型", style="bold")
    table.add_column("名称")
    table.add_column("说明")
    table.add_column("章节数", justify="right")
    table.add_column("占位符", justify="right")

    for t in sorted(templates, key=lambda x: x.doc_type.value):
        table.add_row(
            t.doc_type.value,
            t.name,
            _short(t.description, 36),
            str(len(t.sections)),
            str(len(t.placeholders)),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Law (legal knowledge base) commands
# ---------------------------------------------------------------------------


@law_app.command("add", help="向法律知识库添加一条法条。")
def law_add(
    ctx: typer.Context,
    law_name: str = typer.Option(..., "--law-name", help="法律名称，例如 中华人民共和国民法典"),
    article_number: str = typer.Option(..., "--article-number", help="条号，例如 第143条"),
    content: str | None = typer.Option(None, "--content", help="法条正文"),
    content_file: Path | None = typer.Option(
        None,
        "--content-file",
        exists=True,
        dir_okay=False,
        readable=True,
        help="从文件读取法条正文（与 --content 二选一）",
    ),
    domain: str = typer.Option(
        "civil", "--domain", help="法律领域（civil/criminal/administrative/...）"
    ),
    chapter: str = typer.Option("", "--chapter", help="所属章节"),
    effective_date: str = typer.Option("", "--effective-date", help="生效日期（YYYY-MM-DD）"),
    keywords: str = typer.Option("", "--keywords", help="关键词，逗号分隔"),
) -> None:
    """向法律知识库添加一条法条。

    法条正文通过 ``--content`` 直接给出，或通过 ``--content-file`` 从文件读取。
    """

    if content is None and content_file is None:
        raise typer.BadParameter("必须提供 --content 或 --content-file 之一")
    if content_file is not None:
        try:
            content = content_file.read_text(encoding="utf-8")
        except OSError as exc:
            get_console().print(f"[red]✗ 读取文件失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc
    assert content is not None

    legal_domain = _parse_enum(domain, LegalDomain, "法律领域")
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    dry_run = _get_dry_run(ctx)

    if dry_run:
        get_console().print(
            Panel(
                f"[dry-run] 将添加法条《{law_name}》{article_number}\n"
                f"领域: {legal_domain.value}\n正文: {_short(content, 60)}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    article = LegalArticle(
        law_name=law_name,
        article_number=article_number,
        content=content,
        domain=legal_domain,
        chapter=chapter,
        effective_date=effective_date,
        keywords=keyword_list,
    )

    with _state_session(ctx) as state:
        try:
            state.knowledge_base.add_article(article)
        except LegalKnowledgeError as exc:
            get_console().print(f"[red]✗ {exc}[/red]")
            raise typer.Exit(code=1) from exc

    _audit(
        ctx,
        "judicial.law.add",
        {"article_id": article.id, "citation": article.citation, "domain": legal_domain.value},
    )

    get_console().print(
        f"[green]✓[/green] 已添加法条 [bold]{article.citation}[/bold]（ID: {article.id}）"
    )


@law_app.command("search", help="检索法律法条。")
def law_search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="检索关键词或自然语言查询"),
    top_k: int = typer.Option(10, "--top-k", help="返回结果数量上限"),
    domain: str | None = typer.Option(None, "--domain", help="按法律领域过滤"),
    law_name: str | None = typer.Option(None, "--law-name", help="按法律名称过滤"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """按语义相似度与关键词重叠检索法律法条。"""

    domain_filter = _parse_enum(domain, LegalDomain, "法律领域") if domain is not None else None

    with _state_session(ctx, save=False) as state:
        results = state.knowledge_base.search_articles(
            query,
            top_k=top_k,
            domain=domain_filter,
            law_name=law_name,
        )

    if json_output:
        rows = [
            {
                "id": r.article.id,
                "citation": r.article.citation,
                "law_name": r.article.law_name,
                "article_number": r.article.article_number,
                "domain": r.article.domain.value,
                "status": r.article.status.value,
                "content": r.article.content,
                "score": r.score,
                "match_type": r.match_type,
            }
            for r in results
        ]
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    console = get_console()
    if not results:
        console.print("[dim]未找到匹配的法条。[/dim]")
        return

    table = Table(title=f"法条检索结果（共 {len(results)} 条）", border_style="cyan")
    table.add_column("分数", justify="right", style="bold")
    table.add_column("引用", style="white")
    table.add_column("领域")
    table.add_column("状态", style="bold")
    table.add_column("正文预览")

    for r in results:
        status_style = (
            "green"
            if r.article.is_effective
            else "red"
            if r.article.status is ArticleStatus.REPEALED
            else "yellow"
        )
        table.add_row(
            f"{r.score:.2f}",
            r.article.citation,
            r.article.domain.value,
            Text(r.article.status.value, style=status_style),
            _short(r.article.content, 50),
        )
    console.print(table)


@law_app.command("list", help="列出法律知识库中的法条。")
def law_list(
    ctx: typer.Context,
    domain: str | None = typer.Option(None, "--domain", help="按法律领域过滤"),
    law_name: str | None = typer.Option(None, "--law-name", help="按法律名称过滤"),
    status: str | None = typer.Option(
        None, "--status", help="按状态过滤（effective/repealed/draft）"
    ),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """列出法律知识库中的法条（可按领域/法律名/状态过滤）。"""

    domain_filter = _parse_enum(domain, LegalDomain, "法律领域") if domain is not None else None
    status_filter = _parse_enum(status, ArticleStatus, "法条状态") if status is not None else None

    with _state_session(ctx, save=False) as state:
        articles = state.knowledge_base.list_articles(
            domain=domain_filter, law_name=law_name, status=status_filter
        )

    if json_output:
        rows = [
            {
                "id": a.id,
                "citation": a.citation,
                "law_name": a.law_name,
                "article_number": a.article_number,
                "domain": a.domain.value,
                "status": a.status.value,
                "content": a.content,
            }
            for a in articles
        ]
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    console = get_console()
    if not articles:
        console.print("[dim]法律库为空。用 `justagent judicial law add` 添加法条。[/dim]")
        return

    table = Table(title=f"法律库（共 {len(articles)} 条法条）", border_style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("引用", style="white")
    table.add_column("领域")
    table.add_column("状态", style="bold")
    table.add_column("正文预览")
    for a in articles:
        status_style = (
            "green" if a.is_effective else "red" if a.status is ArticleStatus.REPEALED else "yellow"
        )
        table.add_row(
            a.id[:8],
            a.citation,
            a.domain.value,
            Text(a.status.value, style=status_style),
            _short(a.content, 50),
        )
    console.print(table)


@law_app.command("show", help="查看单条法条的完整信息。")
def law_show(
    ctx: typer.Context,
    article: str = typer.Argument(..., help="法条 ID 或引用（支持前缀匹配）"),
) -> None:
    """查看单条法条的完整信息。"""

    with _state_session(ctx, save=False) as state:
        target = state.knowledge_base.get_article(article)
        if target is None:
            matches = [
                a
                for a in state.knowledge_base.list_articles()
                if a.id.startswith(article) or article in a.citation
            ]
            if len(matches) == 1:
                target = matches[0]
            elif len(matches) > 1:
                raise typer.BadParameter(
                    f"匹配到多条法条：{', '.join(m.citation for m in matches)}"
                )
        if target is None:
            raise typer.BadParameter(f"未找到法条：{article}")

    console = get_console()
    status_style = (
        "green"
        if target.is_effective
        else "red"
        if target.status is ArticleStatus.REPEALED
        else "yellow"
    )
    console.print(
        Panel(
            f"[bold]{target.citation}[/bold]\n"
            f"法律名称: {target.law_name}\n"
            f"条号: {target.article_number}\n"
            f"领域: {target.domain.value} | 状态: [{status_style}]{target.status.value}[/{status_style}]\n"
            f"章节: {target.chapter or '-'} | 生效日期: {target.effective_date or '-'}\n"
            f"关键词: {', '.join(target.keywords) if target.keywords else '-'}\n\n"
            f"{target.content}",
            title="法条详情",
            border_style="cyan",
        )
    )


@case_app.command("summary", help="生成案件摘要与时间轴。")
def case_summary(
    ctx: typer.Context,
    case_id: str = typer.Argument(..., help="案件 ID（支持前缀匹配）"),
) -> None:
    """生成案件摘要与按时间排序的时间轴，便于快速掌握案情全貌。"""

    with _state_session(ctx, save=False) as state:
        case = _require_case(state, case_id)
        materials = state.case_manager.list_materials(case.id)
        evidence = state.evidence_chain.list_evidence(case_id=case.id)

    console = get_console()
    parties_txt = "、".join(f"{p.role.value} {p.name}" for p in case.parties) or "（暂无当事人）"
    claims_txt = (
        "；".join(f"{c.description}（金额 {c.amount}）" for c in case.claims) or "（暂无诉讼请求）"
    )

    overview = (
        f"[bold]案号:[/bold] {case.case_number or '-'}\n"
        f"[bold]案由:[/bold] {case.cause_of_action or '-'}\n"
        f"[bold]法院:[/bold] {case.court or '-'}\n"
        f"[bold]领域/状态:[/bold] {case.domain or '-'} / {case.status.value}\n"
        f"[bold]当事人:[/bold] {parties_txt}\n"
        f"[bold]诉讼请求:[/bold] {claims_txt}\n"
        f"[bold]材料/证据:[/bold] {len(materials)} 份材料 / {len(evidence)} 项证据\n"
        f"[bold]时间线事件:[/bold] {len(case.timeline)} 个"
    )
    console.print(
        Panel(overview, title=f"案件摘要 · {case.case_number or case.id[:8]}", border_style="cyan")
    )

    if case.timeline:
        ttable = Table(title="时间轴（按时间排序）", border_style="green")
        ttable.add_column("日期", style="dim")
        ttable.add_column("类别")
        ttable.add_column("事件描述")
        ordered = sorted(case.timeline, key=lambda ev: (ev.timestamp, ev.date))
        for ev in ordered:
            ttable.add_row(ev.date or "-", ev.category or "-", _short(ev.description, 60))
        console.print(ttable)
    else:
        console.print("[dim]暂无时间线事件（可用 `judicial case import` 从材料抽取）。[/dim]")


@evidence_app.command("export", help="导出证据清单与证据链分析。")
def evidence_export(
    ctx: typer.Context,
    case_id: str = typer.Option("", "--case-id", help="按案件 ID 过滤"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="写入文件（markdown）；省略则输出到终端"
    ),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """导出证据清单与证据链分析（适合归档/共享/提交）。"""

    with _state_session(ctx, save=False) as state:
        evidence = state.evidence_chain.list_evidence(case_id=case_id or "")
        analysis = None
        if evidence:
            try:
                analysis = state.evidence_chain.analyze(case_id=case_id or "")
            except Exception:  # noqa: BLE001
                analysis = None

    if json_output:
        payload = {
            "evidence": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type.value,
                    "proving_object": e.proving_object or "",
                    "admissibility": e.admissibility.value,
                    "probative_strength": e.probative_strength.value,
                    "case_id": e.case_id or "",
                }
                for e in evidence
            ],
            "analysis": (
                {
                    "completeness_score": analysis.completeness_score,
                    "contradiction_count": analysis.contradiction_count,
                    "summary": analysis.summary,
                }
                if analysis
                else None
            ),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        lines = ["# 证据清单", "", f"共 {len(evidence)} 项证据", ""]
        for e in evidence:
            lines.append(f"- **{e.name}**（{e.type.value}）证明对象: {e.proving_object or '-'}")
            lines.append(
                f"  可采性: {e.admissibility.value} | 证明力: {e.probative_strength.value}"
            )
        if analysis:
            lines += ["", "## 证据链分析", "", analysis.summary or ""]
        text = "\n".join(lines)

    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            get_console().print(f"[green]✓ 已导出 {len(evidence)} 项证据到 {output}[/green]")
        except OSError as exc:
            get_console().print(f"[red]✗ 导出失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc
    else:
        typer.echo(text)


@app.command("research", help="进行法律研究并生成研究备忘录。")
def judicial_research(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="研究主题或法律问题"),
    top_k: int = typer.Option(5, "--top-k", help="检索法条数量"),
    draft: bool = typer.Option(True, "--draft/--no-draft", help="是否尝试用 LLM 撰写备忘录"),
) -> None:
    """针对法律问题检索法条并生成研究备忘录（替代人工法律研究）。"""

    console = get_console()
    with _state_session(ctx, save=False) as state:
        results = state.knowledge_base.search_articles(query, top_k=top_k)

    if not results:
        console.print("[red]✗ 未找到相关法条。请先用 `judicial law add` 充实法律库。[/red]")
        raise typer.Exit(code=1)

    # 1) 检索到的法条
    citations = [(r.article, r.score) for r in results]
    lines = [f"# 法律研究备忘录 · {query}", "", f"## 检索到 {len(citations)} 条相关法条", ""]
    for article, score in citations:
        lines.append(f"- **{article.citation}**（领域 {article.domain.value}，匹配度 {score:.2f}）")
        lines.append(f"  正文：{article.content}")

    # 2) LLM 撰写分析（尽力而为；无模型则给出引用式分析）
    analysis = ""
    if draft:
        try:
            config: AppConfig = ctx.obj["config"]
            from justagent.adapters.model_gateway import ChatMessage
            from justagent.cli.commands.fix import _model_router

            router = _model_router(config)
            evidence_text = "\n".join(f"- {a.citation}: {a.content}" for a, _ in citations)
            prompt = (
                f"请针对以下法律问题写一份简明研究备忘录（300字内）：{query}\n"
                f"可依据的法条：\n{evidence_text}\n"
                "结构：一、问题；二、法律依据；三、分析；四、结论。"
            )
            resp = router.chat([ChatMessage(role="user", content=prompt)], "research")
            analysis = resp
            lines.append("", "## 分析（LLM 撰写）", "", analysis)
        except Exception as exc:  # noqa: BLE001 - 无模型时退化为引用式
            lines.append(
                "\n## 分析（无 LLM，给出检索提示）\n\n请人工结合上述法条进行论证。"
                "可用 `justagent judicial doc generate` 生成正式文书。"
            )
            if ctx.obj.get("verbose"):
                console.print(f"[dim]LLM 撰写不可用：{exc}[/dim]")

    text = "\n".join(lines)
    console.print(text)


@law_app.command("export", help="导出法律知识库。")
def law_export(
    ctx: typer.Context,
    output: Path | None = typer.Option(
        None, "--output", "-o", help="写入文件（markdown）；省略则输出到终端"
    ),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """导出法律知识库中的全部法条（适合备份/共享/培训）。"""

    with _state_session(ctx, save=False) as state:
        articles = state.knowledge_base.list_articles()

    if json_output:
        text = json.dumps(
            [
                {
                    "id": a.id,
                    "citation": a.citation,
                    "law_name": a.law_name,
                    "article_number": a.article_number,
                    "domain": a.domain.value,
                    "status": a.status.value,
                    "chapter": a.chapter,
                    "effective_date": a.effective_date,
                    "content": a.content,
                }
                for a in articles
            ],
            ensure_ascii=False,
            indent=2,
        )
    else:
        lines = ["# 法律知识库", "", f"共 {len(articles)} 条法条", ""]
        for a in articles:
            lines.append(f"## {a.citation}")
            lines.append(f"- 领域: {a.domain.value} | 状态: {a.status.value}")
            lines.append(f"- 生效日期: {a.effective_date or '-'} | 章节: {a.chapter or '-'}")
            lines.append(f"- 正文: {a.content}")
            lines.append("")
        text = "\n".join(lines)

    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            get_console().print(f"[green]✓ 已导出 {len(articles)} 条法条到 {output}[/green]")
        except OSError as exc:
            get_console().print(f"[red]✗ 导出失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc
    else:
        typer.echo(text)


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------


def _resolve_evidence(state: _JudicialState, evidence_id: str) -> Evidence:
    """Return an evidence item or raise a Typer error."""

    evidence = state.evidence_chain.get_evidence(evidence_id)
    if evidence is None:
        matches = [e for e in state.evidence_chain.list_evidence() if e.id.startswith(evidence_id)]
        if len(matches) == 1:
            return matches[0]
        raise typer.BadParameter(f"未找到证据：{evidence_id}")
    return evidence


def _print_review_result(evidence: Evidence, result: ReviewResult) -> None:
    """Render an evidence review result as a Rich panel + detail lines."""

    console = get_console()
    adm_style = (
        "green"
        if result.admissibility is Admissibility.ADMISSIBLE
        else "red"
        if result.admissibility is Admissibility.INADMISSIBLE
        else "yellow"
    )
    ps_style = {
        ProbativeStrength.HIGH: "green",
        ProbativeStrength.MEDIUM: "yellow",
        ProbativeStrength.LOW: "red",
        ProbativeStrength.INSUFFICIENT: "red",
    }.get(result.probative_strength, "white")

    body = (
        f"[bold]证据名称:[/bold] {evidence.name}（ID: {evidence.id}）\n"
        f"[bold]证据类型:[/bold] {evidence.type.value}\n"
        f"[bold]证明对象:[/bold] {evidence.proving_object or '-'}\n"
        f"[bold]合法性:[/bold]   {'通过' if result.is_legal else '不通过'}\n"
        f"[bold]可采性:[/bold]     [{adm_style}]{result.admissibility.value}[/{adm_style}]\n"
        f"[bold]关联性得分:[/bold] {result.relevance_score:.0%}\n"
        f"[bold]关联性说明:[/bold] {result.relevance_reasoning}\n"
        f"[bold]证明力:[/bold]     [{ps_style}]{result.probative_strength.value}[/{ps_style}]"
        f"（得分 {result.probative_score:.2f}）\n"
        f"[bold]证明力说明:[/bold] {result.probative_reasoning}"
    )
    if result.legality_issues:
        body += "\n[bold]合法性问题:[/bold]\n" + "\n".join(
            f"  - {issue}" for issue in result.legality_issues
        )
    if result.recommendations:
        body += "\n[bold]建议:[/bold]\n" + "\n".join(f"  - {rec}" for rec in result.recommendations)
    console.print(Panel(body, title=f"证据审查结果 — {evidence.name}", border_style="cyan"))


def _print_chain_analysis(
    result: ChainAnalysisResult,
    contradictions: list[EvidenceRelation],
) -> None:
    """Render an evidence-chain analysis result."""

    console = get_console()
    score_style = (
        "green"
        if result.completeness_score >= 0.7
        else "yellow"
        if result.completeness_score >= 0.4
        else "red"
    )
    summary = (
        f"[bold]案件 ID:[/bold] {result.case_id or '(全部)'}\n"
        f"[bold]证据总数:[/bold] {result.total_evidence}\n"
        f"[bold]可采信证据:[/bold] {result.admissible_evidence}\n"
        f"[bold]支持/印证关系:[/bold] {result.supporting_relations}\n"
        f"[bold]矛盾数量:[/bold] {len(contradictions)}\n"
        f"[bold]证据缺口:[/bold] {len(result.gaps)}\n"
        f"[bold]完整度评分:[/bold] [{score_style}]{result.completeness_score:.0%}[/{score_style}]\n\n"
        f"{result.summary}"
    )
    console.print(Panel(summary, title="证据链分析", border_style="cyan"))

    if contradictions:
        ctable = Table(title="证据矛盾", border_style="red")
        ctable.add_column("证据 A", style="white")
        ctable.add_column("证据 B", style="white")
        ctable.add_column("说明")
        for rel in contradictions:
            ctable.add_row(
                _id_short(rel.evidence_a_id),
                _id_short(rel.evidence_b_id),
                rel.description or "-",
            )
        console.print(ctable)

    if result.gaps:
        console.print("\n[bold red]证据缺口：[/bold red]")
        for gap in result.gaps:
            console.print(f"  - {gap}")


def _print_generated_document(document: GeneratedDocument, output: Path | None) -> None:
    """Render a generated legal document and optionally write it to a file."""

    console = get_console()
    cit_color = "green" if document.all_citations_valid else "yellow"
    header = (
        f"[bold]文书 ID:[/bold] {document.id}\n"
        f"[bold]类型:[/bold]     {document.doc_type.value}\n"
        f"[bold]标题:[/bold]     {document.title or '-'}\n"
        f"[bold]引用法条:[/bold] {len(document.citations)} 条 "
        f"[{cit_color}]（{'全部有效' if document.all_citations_valid else '存在校验问题'}）"
        f"[/{cit_color}]\n"
        f"[bold]生成时间:[/bold] {_format_ts(document.created_at)}"
    )
    console.print(Panel(header, title="法律文书生成结果", border_style="cyan"))
    console.print(
        Panel(
            document.content or "(空文书)", title=document.title or "文书正文", border_style="blue"
        )
    )

    if document.citation_verifications:
        vtable = Table(title="法条引用校验", border_style="magenta")
        vtable.add_column("引用", style="white")
        vtable.add_column("是否有效", style="bold")
        vtable.add_column("匹配法条")
        vtable.add_column("问题")
        for v in document.citation_verifications:
            vtable.add_row(
                v.citation,
                Text("✓ 有效" if v.is_valid else "✗ 无效", style="green" if v.is_valid else "red"),
                v.matched_law_name or "-",
                "; ".join(v.issues) or "-",
            )
        console.print(vtable)

    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(document.content, encoding="utf-8")
            console.print(f"[green]✓[/green] 文书已写入 [bold]{output}[/bold]")
        except OSError as exc:
            console.print(f"[red]✗ 写入文件失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc


__all__ = [
    "app",
    "case_app",
    "doc_app",
    "evidence_app",
    "law_app",
    "register",
]
