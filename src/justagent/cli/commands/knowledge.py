"""``justagent knowledge`` command — document management, vector search,
knowledge graph, and RAG.

Exposes the :mod:`justagent.knowledge` package through the CLI:

* ``justagent knowledge doc``    — add / list / show / search documents.
* ``justagent knowledge graph``  — build / query the knowledge graph.
* ``justagent knowledge rag``    — index / query the RAG pipeline.

The knowledge subsystem components (:class:`DocumentLifecycleManager`,
:class:`FileVectorStore`, :class:`KnowledgeGraph`) are in-memory by design.
To make the CLI usable across invocations, this module transparently
persists state to a directory (``<project_root>/.justagent/knowledge/`` by
default, overridable via the ``JUSTAGENT_KNOWLEDGE_STATE`` environment
variable). Documents are serialised to ``documents.json``, vector records
to ``vectors.json`` (via :class:`FileVectorStore`), and the knowledge graph
to ``graph.json`` (via :class:`KnowledgeGraph.save`). Each mutating command
loads the state, performs its operation, and writes the state back;
read-only commands skip the write.

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
from justagent.knowledge import (
    Document,
    DocumentLifecycleManager,
    DocumentParser,
    DocumentStatus,
    DocumentType,
    FileVectorStore,
    KnowledgeGraph,
    RAGPipeline,
    TextChunker,
    create_default_embedder,
    index_document_chunks,
)
from justagent.models.config import AppConfig

# ---------------------------------------------------------------------------
# Typer sub-apps
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="knowledge",
    help="知识库管理：文档解析、向量检索、知识图谱与 RAG 问答。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

doc_app = typer.Typer(
    name="doc",
    help="文档管理（添加 / 列出 / 查看 / 语义检索）。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

graph_app = typer.Typer(
    name="graph",
    help="知识图谱（构建 / 查询实体与关系）。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

rag_app = typer.Typer(
    name="rag",
    help="RAG 检索增强生成（索引文档 / 问答）。",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def register(parent: typer.Typer) -> None:
    """Register the ``knowledge`` command group and its sub-groups."""

    app.add_typer(doc_app, name="doc")
    app.add_typer(graph_app, name="graph")
    app.add_typer(rag_app, name="rag")
    parent.add_typer(app, name="knowledge")


# ---------------------------------------------------------------------------
# Context accessors — defensive like the other command modules (build fallbacks when ctx.obj
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


def _state_dir(ctx: typer.Context) -> Path:
    """Resolve the knowledge state directory.

    Priority: ``JUSTAGENT_KNOWLEDGE_STATE`` (or legacy ``MYAGENT_KNOWLEDGE_STATE``)
    env var > ``<project_root>/.justagent/knowledge/`` > ``./.justagent/knowledge/``.
    """

    env = os.environ.get("JUSTAGENT_KNOWLEDGE_STATE") or os.environ.get("MYAGENT_KNOWLEDGE_STATE")
    if env:
        return Path(env).expanduser()
    config = _get_config(ctx)
    root = Path(getattr(config, "project_root", ".") or ".")
    return root / ".justagent" / "knowledge"


class _KnowledgeState:
    """Container holding the knowledge subsystem components with JSON persistence.

    The underlying components (:class:`DocumentLifecycleManager`,
    :class:`FileVectorStore`, :class:`KnowledgeGraph`) only offer incremental
    mutation APIs. To round-trip persisted state we restore the internal
    registries directly — this is the integration layer's concern and is
    clearly isolated here rather than spread across commands.

    State files within *base_dir*:

    * ``documents.json`` — serialised :class:`Document` registry.
    * ``vectors.json``   — :class:`FileVectorStore` records (own save/load).
    * ``graph.json``     — :class:`KnowledgeGraph` entities + relations.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.doc_manager = DocumentLifecycleManager()
        self.embedder = create_default_embedder()
        self.parser = DocumentParser()
        self.vector_store = FileVectorStore(base_dir / "vectors.json")
        self.vector_store.load()
        self.graph = KnowledgeGraph.load(base_dir / "graph.json")
        self._docs_path = base_dir / "documents.json"
        self._load_documents()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def make_pipeline(self, gateway: Any = None) -> RAGPipeline:
        """Create a :class:`RAGPipeline` wired to this state's components.

        The pipeline shares the same vector store, embedder, and parser so
        that ingested chunks are immediately visible to subsequent
        ``doc search`` commands and persist on :meth:`save`.
        """

        return RAGPipeline(
            vector_store=self.vector_store,
            embedder=self.embedder,
            gateway=gateway,
            parser=self.parser,
        )

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, base_dir: Path) -> _KnowledgeState:
        """Load state from *base_dir*, or return an empty state on any error."""

        return cls(base_dir)

    def save(self) -> None:
        """Persist all registries to the state directory."""

        # Documents.
        docs = [d.model_dump(mode="json") for d in self.doc_manager._documents.values()]
        try:
            self._docs_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "saved_at": time.time(),
                        "documents": docs,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            get_console().print(
                f"[red]✗ 无法保存文档状态文件 {self._docs_path}：{exc}[/red]",
                style="red",
            )

        # Vector store.
        self.vector_store.save()

        # Knowledge graph.
        self.graph.save(self.base_dir / "graph.json")

    # ------------------------------------------------------------------
    # Internal restore
    # ------------------------------------------------------------------

    def _load_documents(self) -> None:
        """Rebuild the document manager's registry from ``documents.json``."""

        if not self._docs_path.exists():
            return
        try:
            data = json.loads(self._docs_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            get_console().print(
                f"[yellow]⚠ 无法读取文档状态文件 {self._docs_path}：{exc}"
                f"（将以空状态启动）[/yellow]"
            )
            return
        for raw in data.get("documents", []):
            try:
                doc = Document.model_validate(raw)
                self.doc_manager._documents[doc.id] = doc
            except Exception as exc:  # noqa: BLE001 - skip corrupt entries
                get_console().print(f"[yellow]⚠ 跳过无效文档记录：{exc}[/yellow]")


@contextmanager
def _state_session(ctx: typer.Context, *, save: bool = True) -> Iterator[_KnowledgeState]:
    """Load state, yield it, and persist on exit (unless dry-run or read-only).

    Args:
        ctx: The Typer context (for path resolution and dry-run flag).
        save: When ``True`` (default) the state is written back on a clean
            exit. Read-only commands pass ``False`` to avoid needless writes.
    """

    state = _KnowledgeState.load(_state_dir(ctx))
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


def _require_document(state: _KnowledgeState, doc_id: str) -> Document:
    """Return a document or raise a Typer error (exit 1).

    Supports partial / short ID prefix matching against known document IDs.
    """

    doc = state.doc_manager.get(doc_id)
    if doc is None:
        matches = [d for d in state.doc_manager.list_documents() if d.id.startswith(doc_id)]
        if len(matches) == 1:
            return matches[0]
        raise typer.BadParameter(f"未找到文档：{doc_id}")
    return doc


def _parse_enum(value: str, enum_cls: type[Any], label: str) -> Any:
    """Parse *value* into an enum member, raising BadParameter on failure."""

    try:
        return enum_cls(value)
    except ValueError:
        valid = ", ".join(str(m.value) for m in enum_cls)
        raise typer.BadParameter(f"无效的 {label}：{value!r}（可选值：{valid}）") from None


def _get_gateway(ctx: typer.Context) -> Any:
    """Best-effort: resolve a :class:`ModelGateway` from config.

    Returns ``None`` when no model backend is configured. The RAG pipeline
    handles ``None`` gracefully by returning retrieved context snippets
    without LLM-generated answers.
    """

    config = _get_config(ctx)
    try:
        from justagent.core.model_router import ModelRouter

        router = ModelRouter(config)
        return router.select_backend()
    except Exception:  # noqa: BLE001 - gateway is optional
        return None


# ---------------------------------------------------------------------------
# Document commands
# ---------------------------------------------------------------------------


@doc_app.command("add", help="添加文档到知识库（解析并索引）。")
def doc_add(
    ctx: typer.Context,
    file_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="要添加的文件路径（支持 PDF/Word/Excel/PPT/Markdown/HTML/纯文本）",
    ),
    title: str | None = typer.Option(None, "--title", "-t", help="文档标题（默认取文件名）"),
    no_index: bool = typer.Option(False, "--no-index", help="仅解析注册，不索引到向量库"),
    chunk_size: int = typer.Option(1000, "--chunk-size", help="分块大小（字符数）"),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap", help="分块重叠（字符数）"),
) -> None:
    """解析文件、注册文档并索引到向量库。

    支持多种文档格式（PDF、Word、Excel、PPT、Markdown、HTML、纯文本），
    自动检测格式并提取文本。文档内容会被分块并嵌入到向量库中，以便
    后续的语义检索和 RAG 问答。指定 ``--no-index`` 可仅注册文档而不索引。
    """

    verbose = _get_verbose(ctx)
    dry_run = _get_dry_run(ctx)
    console = get_console()

    if dry_run:
        console.print(
            Panel(
                f"[dry-run] 将添加文档\n"
                f"文件: {file_path}\n"
                f"标题: {title or file_path.name}\n"
                f"索引: {'否' if no_index else '是'}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    # Use a chunker with the user-specified parameters.
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    parser = DocumentParser(chunker=chunker)

    try:
        doc = parser.parse_file(file_path, title=title)
    except FileNotFoundError as exc:
        console.print(f"[red]✗ 文件不存在：{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001 - parsing may fail on binary formats
        console.print(f"[red]✗ 解析失败：{exc}[/red]")
        raise typer.Exit(code=1) from exc

    indexed_count = 0
    with _state_session(ctx) as state:
        try:
            state.doc_manager.register(doc)
        except ValueError as exc:
            console.print(f"[red]✗ {exc}[/red]")
            raise typer.Exit(code=1) from exc

        if not no_index and doc.chunks:
            records = index_document_chunks(
                state.vector_store,
                doc.chunks,
                state.embedder,
                document_title=doc.title,
            )
            indexed_count = len(records)

    _audit(
        ctx,
        "knowledge.doc.add",
        {
            "doc_id": doc.id,
            "title": doc.title,
            "source": doc.source,
            "type": doc.type.value,
            "chunks": len(doc.chunks),
            "indexed": indexed_count,
        },
    )

    if verbose:
        console.print(
            Panel(
                f"文档 ID:    {doc.id}\n"
                f"标题:       {doc.title}\n"
                f"来源:       {doc.source}\n"
                f"类型:       {doc.type.value}\n"
                f"内容长度:   {len(doc.content)} 字符\n"
                f"分块数:     {len(doc.chunks)}\n"
                f"已索引:     {indexed_count} 块\n"
                f"状态:       {doc.status.value}\n"
                f"创建时间:   {_format_ts(doc.created_at)}",
                title=f"已添加文档 {doc.title}",
                border_style="green",
            )
        )
    else:
        index_part = f"，已索引 {indexed_count} 块" if indexed_count else "，未索引"
        console.print(
            f"[green]✓[/green] 已添加文档 [bold]{doc.title}[/bold]"
            f"（类型: {doc.type.value}，分块: {len(doc.chunks)}"
            f"{index_part}，ID: {doc.id}）"
        )


@doc_app.command("list", help="列出知识库中的文档。")
def doc_list(
    ctx: typer.Context,
    status: str | None = typer.Option(
        None, "--status", help="按状态过滤（active/archived/deleted）"
    ),
    doc_type: str | None = typer.Option(
        None, "--type", help="按文档类型过滤（pdf/word/excel/ppt/markdown/html/plain_text）"
    ),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """列出知识库中的所有文档，可按状态或类型过滤。"""

    status_filter = _parse_enum(status, DocumentStatus, "状态") if status is not None else None
    type_filter = _parse_enum(doc_type, DocumentType, "文档类型") if doc_type is not None else None

    with _state_session(ctx, save=False) as state:
        documents = state.doc_manager.list_documents(status=status_filter, doc_type=type_filter)
        vector_count = state.vector_store.count()

    if json_output:
        rows = [
            {
                "id": d.id,
                "title": d.title,
                "source": d.source,
                "type": d.type.value,
                "status": d.status.value,
                "version": d.version,
                "chunks": len(d.chunks),
                "token_count": d.token_count,
                "content_length": len(d.content),
                "created_at": d.created_at,
                "updated_at": d.updated_at,
            }
            for d in documents
        ]
        typer.echo(
            json.dumps(
                {"documents": rows, "total": len(rows), "vector_records": vector_count},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    console = get_console()
    if not documents:
        console.print("[dim]暂无文档。使用 `knowledge doc add <文件>` 添加一个。[/dim]")
        return

    table = Table(
        title=f"文档列表（共 {len(documents)} 篇，向量记录 {vector_count} 条）", border_style="cyan"
    )
    table.add_column("ID", style="dim", width=8)
    table.add_column("标题", style="white")
    table.add_column("类型")
    table.add_column("状态", style="bold")
    table.add_column("分块", justify="right")
    table.add_column("版本", justify="right")
    table.add_column("来源", style="dim")
    table.add_column("更新时间", style="dim")

    for d in documents:
        status_style = {
            DocumentStatus.ACTIVE: "green",
            DocumentStatus.ARCHIVED: "yellow",
            DocumentStatus.DELETED: "red",
        }.get(d.status, "white")
        table.add_row(
            _id_short(d.id),
            _short(d.title, 30),
            d.type.value,
            Text(d.status.value, style=status_style),
            str(len(d.chunks)),
            str(d.version),
            _short(d.source, 24),
            _format_ts(d.updated_at),
        )
    console.print(table)


@doc_app.command("show", help="查看文档详情。")
def doc_show(
    ctx: typer.Context,
    doc_id: str = typer.Argument(..., help="文档 ID（支持前缀匹配）"),
) -> None:
    """查看指定文档的详细信息，包括元数据、内容预览、分块与版本历史。"""

    with _state_session(ctx, save=False) as state:
        doc = _require_document(state, doc_id)

    console = get_console()

    header = (
        f"[bold]文档 ID:[/bold]     {doc.id}\n"
        f"[bold]标题:[/bold]        {doc.title}\n"
        f"[bold]来源:[/bold]        {doc.source or '-'}\n"
        f"[bold]类型:[/bold]        {doc.type.value}\n"
        f"[bold]状态:[/bold]        {doc.status.value}\n"
        f"[bold]版本:[/bold]        {doc.version}\n"
        f"[bold]内容长度:[/bold]    {len(doc.content)} 字符\n"
        f"[bold]分块数:[/bold]      {len(doc.chunks)}\n"
        f"[bold]Token 估算:[/bold]  {doc.token_count}\n"
        f"[bold]内容哈希:[/bold]    {doc.content_hash[:16]}...\n"
        f"[bold]创建时间:[/bold]    {_format_ts(doc.created_at)}\n"
        f"[bold]更新时间:[/bold]    {_format_ts(doc.updated_at)}"
    )
    console.print(Panel(header, title=f"文档 {doc.title}", border_style="cyan"))

    # Metadata.
    if doc.metadata:
        meta_lines = "\n".join(f"  {k}: {_short(str(v), 60)}" for k, v in doc.metadata.items())
        console.print(Panel(meta_lines, title="元数据", border_style="blue"))
    else:
        console.print("[dim]元数据：（暂无）[/dim]")

    # Content preview.
    preview = _short(doc.content, 500) if doc.content else "(空)"
    console.print(Panel(preview, title="内容预览", border_style="blue"))

    # Chunks.
    if doc.chunks:
        ctable = Table(title=f"分块（共 {len(doc.chunks)} 块）", border_style="blue")
        ctable.add_column("序号", justify="right", style="dim")
        ctable.add_column("内容预览")
        ctable.add_column("Token", justify="right")
        for chunk in doc.chunks[:20]:  # show first 20 chunks
            ctable.add_row(
                str(chunk.index),
                _short(chunk.content, 60),
                str(chunk.token_count),
            )
        if len(doc.chunks) > 20:
            ctable.add_row("...", f"（还有 {len(doc.chunks) - 20} 块未显示）", "")
        console.print(ctable)
    else:
        console.print("[dim]分块：（暂无）[/dim]")

    # Version history.
    if doc.versions:
        vtable = Table(title=f"版本历史（共 {len(doc.versions)} 个版本）", border_style="magenta")
        vtable.add_column("版本", justify="right", style="bold")
        vtable.add_column("内容哈希", style="dim")
        vtable.add_column("创建时间", style="dim")
        vtable.add_column("内容长度", justify="right")
        for ver in doc.versions:
            vtable.add_row(
                str(ver.version),
                ver.content_hash[:16] + "...",
                _format_ts(ver.created_at),
                str(len(ver.content)),
            )
        console.print(vtable)


@doc_app.command("search", help="语义检索文档。")
def doc_search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="检索查询（自然语言或关键词）"),
    top_k: int = typer.Option(5, "--top-k", help="返回结果数量上限"),
    min_score: float = typer.Option(0.0, "--min-score", help="最低相似度分数（0~1）"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """对已索引的文档分块执行语义检索。

    使用嵌入模型将查询向量化，然后在向量库中执行余弦相似度搜索，
    返回最相关的文档分块及其相似度分数。
    """

    with _state_session(ctx, save=False) as state:
        if state.vector_store.count() == 0:
            get_console().print(
                "[dim]向量库为空。请先使用 `knowledge doc add` 或 `knowledge rag index`"
                " 索引文档。[/dim]"
            )
            return

        query_embedding = state.embedder.embed(query)
        results = state.vector_store.search(
            query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

    if json_output:
        rows = [
            {
                "rank": r.rank,
                "score": r.score,
                "document_id": r.document_id,
                "document_title": r.document_title,
                "chunk_index": r.chunk.index,
                "chunk_id": r.chunk.id,
                "content": r.chunk.content,
                "token_count": r.chunk.token_count,
            }
            for r in results
        ]
        typer.echo(
            json.dumps(
                {"query": query, "results": rows, "total": len(rows)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    console = get_console()
    if not results:
        console.print(f"[dim]未找到与「{query}」匹配的文档分块。[/dim]")
        return

    table = Table(
        title=f"语义检索结果（查询：「{query}」，共 {len(results)} 条）",
        border_style="cyan",
    )
    table.add_column("排名", justify="right", style="bold")
    table.add_column("分数", justify="right", style="bold")
    table.add_column("文档", style="white")
    table.add_column("分块", justify="right")
    table.add_column("内容预览")

    for r in results:
        score_style = "green" if r.score >= 0.5 else "yellow" if r.score >= 0.2 else "dim"
        table.add_row(
            str(r.rank),
            Text(f"{r.score:.3f}", style=score_style),
            _short(r.document_title or r.document_id, 24),
            str(r.chunk.index),
            _short(r.chunk.content, 50),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Knowledge graph commands
# ---------------------------------------------------------------------------


@graph_app.command("build", help="从文档构建知识图谱。")
def graph_build(
    ctx: typer.Context,
    doc_id: str | None = typer.Option(None, "--doc-id", help="从指定文档构建（支持前缀匹配）"),
    all_docs: bool = typer.Option(False, "--all", help="从所有活动文档构建"),
    no_capitalized: bool = typer.Option(
        False, "--no-capitalized", help="跳过大写短语实体抽取（仅抽取邮箱/URL/日期等）"
    ),
) -> None:
    """从文档内容中抽取实体与关系，构建知识图谱。

    使用基于规则的模式抽取实体（邮箱、URL、日期、电话、金额、大写短语）
    和关系（动词模式如 "works at"、"located in" 等）。指定 ``--doc-id``
    从单个文档构建，或指定 ``--all`` 从所有活动文档构建。
    """

    if doc_id is None and not all_docs:
        raise typer.BadParameter("请指定 --doc-id <文档ID> 或 --all 从所有文档构建")

    dry_run = _get_dry_run(ctx)
    console = get_console()

    with _state_session(ctx) as state:
        if doc_id is not None:
            doc = _require_document(state, doc_id)
            documents = [doc]
        else:
            documents = state.doc_manager.list_documents(status=DocumentStatus.ACTIVE)

        if not documents:
            console.print("[dim]没有可用的活动文档。请先使用 `knowledge doc add` 添加文档。[/dim]")
            return

        if dry_run:
            titles = ", ".join(d.title for d in documents)
            console.print(
                Panel(
                    f"[dry-run] 将从以下文档构建知识图谱\n"
                    f"文档: {titles}\n"
                    f"抽取大写短语: {'否' if no_capitalized else '是'}",
                    title="Dry Run",
                    border_style="yellow",
                )
            )
            return

        total_entities = 0
        total_relations = 0
        for doc in documents:
            if not doc.content.strip():
                continue
            entities, relations = state.graph.extract_from_text(
                doc.content,
                document_id=doc.id,
                include_capitalized=not no_capitalized,
            )
            total_entities += len(entities)
            total_relations += len(relations)

    _audit(
        ctx,
        "knowledge.graph.build",
        {
            "doc_ids": [d.id for d in documents],
            "entities_added": total_entities,
            "relations_added": total_relations,
            "total_entities": state.graph.entity_count,
            "total_relations": state.graph.relation_count,
        },
    )

    console.print(
        f"[green]✓[/green] 知识图谱构建完成："
        f"抽取 [bold]{total_entities}[/bold] 个实体、"
        f"[bold]{total_relations}[/bold] 条关系"
        f"（图谱总计：{state.graph.entity_count} 实体 / "
        f"{state.graph.relation_count} 关系）"
    )


@graph_app.command("query", help="查询实体与关系。")
def graph_query(
    ctx: typer.Context,
    entity_type: str | None = typer.Option(
        None,
        "--entity-type",
        help="按实体类型过滤（person/organization/location/date/email/url/phone/money/concept）",
    ),
    name_contains: str | None = typer.Option(None, "--name", help="按名称子串过滤（不区分大小写）"),
    relation_type: str | None = typer.Option(
        None,
        "--relation-type",
        help="按关系类型过滤（works_at/located_in/is_a/founded_by/related_to/...）",
    ),
    entity_id: str | None = typer.Option(
        None, "--entity-id", help="查询与指定实体相关的关系（支持前缀匹配）"
    ),
    direction: str = typer.Option(
        "both", "--direction", help="关系方向（both/outgoing/incoming，仅与 --entity-id 搭配使用）"
    ),
    limit: int = typer.Option(20, "--limit", help="结果数量上限"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """查询知识图谱中的实体和关系。

    支持多种查询模式：

    * ``--entity-type`` / ``--name`` — 查询实体。
    * ``--relation-type`` / ``--entity-id`` — 查询关系。
    * 无过滤参数 — 显示图谱摘要与度数最高的实体。
    """

    with _state_session(ctx, save=False) as state:
        graph = state.graph

        if graph.entity_count == 0 and graph.relation_count == 0:
            get_console().print(
                "[dim]知识图谱为空。请先使用 `knowledge graph build` 构建图谱。[/dim]"
            )
            return

        # Resolve entity_id prefix.
        resolved_entity_id: str | None = None
        if entity_id is not None:
            ent = graph.get_entity(entity_id)
            if ent is None:
                matches = [e for e in graph.list_entities() if e.id.startswith(entity_id)]
                if len(matches) == 1:
                    resolved_entity_id = matches[0].id
                else:
                    raise typer.BadParameter(f"未找到实体：{entity_id}")
            else:
                resolved_entity_id = ent.id

        # Determine query mode.
        query_entities = entity_type is not None or name_contains is not None
        query_relations = relation_type is not None or resolved_entity_id is not None

        entities = graph.query_entities(
            entity_type=entity_type,
            name_contains=name_contains,
            limit=limit if query_entities else None,
        )

        relations = graph.query_relations(
            relation_type=relation_type,
            entity_id=resolved_entity_id,
            direction=direction,
            limit=limit if query_relations else None,
        )

        if json_output:
            ent_rows = [
                {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "aliases": e.aliases,
                    "source_documents": e.source_documents,
                    "degree": graph.degree(e.id),
                }
                for e in entities
            ]
            rel_rows = [
                {
                    "id": r.id,
                    "source_entity_id": r.source_entity_id,
                    "source_name": (
                        src.name if (src := graph.get_entity(r.source_entity_id)) else "-"
                    ),
                    "target_entity_id": r.target_entity_id,
                    "target_name": (
                        dst.name if (dst := graph.get_entity(r.target_entity_id)) else "-"
                    ),
                    "relation_type": r.relation_type,
                    "weight": r.weight,
                    "source_documents": r.source_documents,
                }
                for r in relations
            ]
            typer.echo(
                json.dumps(
                    {
                        "summary": {
                            "total_entities": graph.entity_count,
                            "total_relations": graph.relation_count,
                        },
                        "entities": ent_rows,
                        "relations": rel_rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        console = get_console()

        # Summary.
        console.print(
            Panel(
                f"[bold]实体总数:[/bold] {graph.entity_count}\n"
                f"[bold]关系总数:[/bold] {graph.relation_count}",
                title="知识图谱摘要",
                border_style="cyan",
            )
        )

        # Entities.
        if entities:
            etable = Table(title=f"实体（共 {len(entities)} 个）", border_style="blue")
            etable.add_column("ID", style="dim", width=8)
            etable.add_column("名称", style="white")
            etable.add_column("类型", style="bold")
            etable.add_column("别名")
            etable.add_column("度数", justify="right")
            etable.add_column("来源文档", justify="right")

            for e in entities:
                etable.add_row(
                    _id_short(e.id),
                    _short(e.name, 24),
                    e.entity_type,
                    _short(", ".join(e.aliases), 20) or "-",
                    str(graph.degree(e.id)),
                    str(len(e.source_documents)),
                )
            console.print(etable)
        elif query_entities:
            console.print("[dim]未找到匹配的实体。[/dim]")

        # Relations.
        if relations:
            rtable = Table(title=f"关系（共 {len(relations)} 条）", border_style="blue")
            rtable.add_column("ID", style="dim", width=8)
            rtable.add_column("源实体", style="white")
            rtable.add_column("关系类型", style="bold")
            rtable.add_column("目标实体", style="white")
            rtable.add_column("权重", justify="right")
            rtable.add_column("来源文档", justify="right")

            for r in relations:
                src = graph.get_entity(r.source_entity_id)
                tgt = graph.get_entity(r.target_entity_id)
                src_name = src.name if src else r.source_entity_id[:8]
                tgt_name = tgt.name if tgt else r.target_entity_id[:8]
                rtable.add_row(
                    _id_short(r.id),
                    _short(src_name, 18),
                    r.relation_type,
                    _short(tgt_name, 18),
                    f"{r.weight:.2f}",
                    str(len(r.source_documents)),
                )
            console.print(rtable)
        elif query_relations:
            console.print("[dim]未找到匹配的关系。[/dim]")

        # Default mode: show top entities by degree.
        if not query_entities and not query_relations and graph.entity_count > 0:
            all_entities = graph.list_entities()
            all_entities.sort(key=lambda e: graph.degree(e.id), reverse=True)
            top = all_entities[:limit]
            if top:
                ttable = Table(title=f"度数最高的实体（前 {len(top)} 个）", border_style="magenta")
                ttable.add_column("排名", justify="right", style="bold")
                ttable.add_column("名称", style="white")
                ttable.add_column("类型", style="bold")
                ttable.add_column("度数", justify="right")
                for i, e in enumerate(top, start=1):
                    ttable.add_row(
                        str(i),
                        _short(e.name, 30),
                        e.entity_type,
                        str(graph.degree(e.id)),
                    )
                console.print(ttable)


# ---------------------------------------------------------------------------
# RAG commands
# ---------------------------------------------------------------------------


@rag_app.command("index", help="索引文档以供 RAG 检索。")
def rag_index(
    ctx: typer.Context,
    file_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="要索引的文件路径（支持 PDF/Word/Excel/PPT/Markdown/HTML/纯文本）",
    ),
    title: str | None = typer.Option(None, "--title", "-t", help="文档标题（默认取文件名）"),
) -> None:
    """解析文件并索引到 RAG 向量库。

    将文件解析为文档、分块、嵌入并索引到向量库中，以便后续的 RAG 问答。
    如果同一文档已索引过，旧的分块会被自动清除后重新索引。
    """

    verbose = _get_verbose(ctx)
    dry_run = _get_dry_run(ctx)
    console = get_console()

    if dry_run:
        console.print(
            Panel(
                f"[dry-run] 将索引文件\n文件: {file_path}\n标题: {title or file_path.name}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    with _state_session(ctx) as state:
        try:
            doc = state.parser.parse_file(file_path, title=title)
        except FileNotFoundError as exc:
            console.print(f"[red]✗ 文件不存在：{exc}[/red]")
            raise typer.Exit(code=1) from exc
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]✗ 解析失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc

        # Register the document for lifecycle tracking.
        with suppress(ValueError):
            state.doc_manager.register(doc)

        pipeline = state.make_pipeline()
        try:
            chunk_count = pipeline.ingest_document(doc)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]✗ 索引失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _audit(
        ctx,
        "knowledge.rag.index",
        {
            "doc_id": doc.id,
            "title": doc.title,
            "source": doc.source,
            "chunks_indexed": chunk_count,
            "vector_records": state.vector_store.count(),
        },
    )

    if verbose:
        console.print(
            Panel(
                f"文档 ID:       {doc.id}\n"
                f"标题:          {doc.title}\n"
                f"来源:          {doc.source}\n"
                f"类型:          {doc.type.value}\n"
                f"分块数:        {len(doc.chunks)}\n"
                f"已索引块数:    {chunk_count}\n"
                f"向量库总数:    {state.vector_store.count()}",
                title=f"RAG 索引完成 — {doc.title}",
                border_style="green",
            )
        )
    else:
        console.print(
            f"[green]✓[/green] 已索引 [bold]{doc.title}[/bold]"
            f"（{chunk_count} 块，向量库总计 {state.vector_store.count()} 条，"
            f"ID: {doc.id}）"
        )


@rag_app.command("query", help="RAG 问答（带引用）。")
def rag_query(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="问题"),
    top_k: int = typer.Option(5, "--top-k", help="检索块数量"),
    min_score: float = typer.Option(0.01, "--min-score", help="最低相似度分数（0~1）"),
    no_llm: bool = typer.Option(False, "--no-llm", help="不调用 LLM，仅返回检索到的上下文片段"),
    json_output: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """使用 RAG（检索增强生成）回答问题。

    检索与问题最相关的文档分块，构建上下文提示，然后调用 LLM 生成
    带来源引用的回答。如果未配置 LLM 后端或指定 ``--no-llm``，则仅
    返回检索到的上下文片段。
    """

    verbose = _get_verbose(ctx)
    dry_run = _get_dry_run(ctx)
    console = get_console()

    if dry_run:
        console.print(
            Panel(
                f"[dry-run] 将执行 RAG 问答\n"
                f"问题: {question}\n"
                f"top_k: {top_k}\n"
                f"使用 LLM: {'否' if no_llm else '是（如可用）'}",
                title="Dry Run",
                border_style="yellow",
            )
        )
        return

    with _state_session(ctx, save=False) as state:
        if state.vector_store.count() == 0:
            console.print("[dim]向量库为空。请先使用 `knowledge rag index <文件>` 索引文档。[/dim]")
            return

        gateway = None if no_llm else _get_gateway(ctx)
        pipeline = state.make_pipeline(gateway=gateway)

        try:
            answer = pipeline.query(
                question,
                top_k=top_k,
                min_score=min_score,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]✗ RAG 问答失败：{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _audit(
        ctx,
        "knowledge.rag.query",
        {
            "question": question,
            "answer_length": len(answer.answer),
            "citations": len(answer.citations),
            "latency_ms": answer.latency_ms,
            "no_llm": gateway is None,
        },
    )

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "query": answer.query,
                    "answer": answer.answer,
                    "citations": [
                        {
                            "document_id": c.document_id,
                            "document_title": c.document_title,
                            "chunk_index": c.chunk_index,
                            "chunk_id": c.chunk_id,
                            "score": c.score,
                            "snippet": c.snippet,
                        }
                        for c in answer.citations
                    ],
                    "latency_ms": answer.latency_ms,
                    "num_sources": answer.num_sources,
                    "metadata": answer.metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    # Answer panel.
    answer_panel = Panel(
        answer.answer or "(无回答)",
        title=f"RAG 回答（耗时 {answer.latency_ms:.0f}ms，{answer.num_sources} 个来源）",
        border_style="green",
    )
    console.print(answer_panel)

    # Citations table.
    if answer.citations:
        ctable = Table(title="来源引用", border_style="cyan")
        ctable.add_column("序号", justify="right", style="bold")
        ctable.add_column("分数", justify="right", style="bold")
        ctable.add_column("文档", style="white")
        ctable.add_column("分块", justify="right")
        ctable.add_column("片段预览")

        for i, c in enumerate(answer.citations, start=1):
            score_style = "green" if c.score >= 0.5 else "yellow" if c.score >= 0.2 else "dim"
            ctable.add_row(
                str(i),
                Text(f"{c.score:.3f}", style=score_style),
                _short(c.document_title or c.document_id, 24),
                str(c.chunk_index),
                _short(c.snippet, 50),
            )
        console.print(ctable)
    else:
        console.print("[dim]无来源引用。[/dim]")

    # Verbose metadata.
    if verbose and answer.metadata:
        meta_lines = "\n".join(f"  {k}: {v}" for k, v in answer.metadata.items())
        console.print(Panel(meta_lines, title="元数据", border_style="dim"))


__all__ = [
    "app",
    "doc_app",
    "graph_app",
    "rag_app",
    "register",
]
