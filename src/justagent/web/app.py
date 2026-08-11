"""JustAgent Web — a browser console for the conversational agent.

Provides a Web backend with 1:1 coverage of JustAgent's operational
capabilities: chat, judicial (cases/evidence/laws/documents), knowledge RAG,
config, models, metrics, audit, sessions, plugins and diagnostics — all
manageable from the browser without the CLI.

Run with ``justagent web``.
"""

from __future__ import annotations

import asyncio
import json
import platform
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from justagent.agent.runtime import AgentRuntime, AgentRuntimeConfig, LLMClient
from justagent.agent.tools.builtin import make_default_tools
from justagent.models.config import AppConfig

_HTML_DIR = Path(__file__).resolve().parent / "static"
_REMOTE_PROVIDERS = {"openai", "openrouter", "azure_openai"}


def _judicial_state_path(config: AppConfig) -> Path:
    return config.project_root / ".justagent" / "judicial_state.json"


def _load_judicial(config: AppConfig) -> dict:
    from justagent.cli.commands.judicial import _JudicialState

    state = _JudicialState.load(_judicial_state_path(config))
    return {
        "cases": [
            {
                "id": c.id,
                "case_number": c.case_number or "",
                "cause": c.cause_of_action or "",
                "court": c.court or "",
                "status": c.status.value,
                "parties": len(c.parties),
                "timeline": len(c.timeline),
                "claims": [cc.description for cc in c.claims],
            }
            for c in state.case_manager.list_cases()
        ],
        "evidence": [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type.value,
                "proving_object": e.proving_object or "",
                "admissible": e.admissibility.value,
                "strength": e.probative_strength.value,
            }
            for e in state.evidence_chain.list_evidence()
        ],
        "laws": [
            {
                "id": a.id,
                "citation": a.citation,
                "law_name": a.law_name,
                "article_number": a.article_number,
                "domain": a.domain.value,
                "status": a.status.value,
                "content": a.content,
            }
            for a in state.knowledge_base.list_articles()
        ],
    }


def _build_llm(config: AppConfig) -> LLMClient | None:
    provider, model, api_key, base_url = "", "", "", ""
    api_version, timeout = None, 30.0
    if config.model.backends:
        b = config.model.backends[0]
        provider, model = b.provider.value, b.model or ""
        api_key, base_url = b.api_key or "", str(b.base_url)
        api_version, timeout = b.api_version, b.timeout
    else:
        llm = config.llm
        provider, model = llm.provider.value, llm.model
        api_key, base_url = llm.api_key or "", str(llm.base_url or "")
        api_version, timeout = llm.api_version, llm.timeout
    if not model:
        return None
    if provider in _REMOTE_PROVIDERS and not api_key:
        return None
    return LLMClient(
        model=model, api_key=api_key, base_url=base_url,
        api_version=api_version, timeout=timeout, provider=provider,
    )


def _read_audit(config: AppConfig, limit: int = 100) -> list[dict]:
    from justagent.core.audit_logger import AuditLogger

    lg = AuditLogger(config)
    entries: list[dict] = []
    for f in sorted(lg.log_dir.glob("audit.*.jsonl")):
        try:
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
    return entries[-limit:]


def _get_metrics(config: AppConfig) -> dict:
    # Aggregate request counters from the audit log (the in-process registry is
    # empty in a fresh server process).
    counts: dict[str, int] = {}
    for entry in _read_audit(config, limit=500):
        ev = entry.get("event") or entry.get("action") or "unknown"
        counts[ev] = counts.get(ev, 0) + 1
    return {"events": counts, "total": sum(counts.values())}


def create_app(config: AppConfig) -> FastAPI:
    """Create the FastAPI application bound to a project config."""

    app = FastAPI(title="JustAgent Web", version="1.0.0")
    state_path = _judicial_state_path(config)
    # Per-session runtimes so the web chat keeps multi-turn conversation memory.
    _runtimes: dict[str, Any] = {}
    _runtimes_lock = asyncio.Lock()

    # -- pages --------------------------------------------------------------
    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_HTML_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/state")
    async def get_state() -> dict:
        return _load_judicial(config)

    # -- system / diagnostics ----------------------------------------------
    @app.get("/api/system")
    async def system() -> dict:
        return {
            "version": _pkg("justagent"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "project_root": str(config.project_root),
            "model_backends": len(config.model.backends),
        }

    @app.post("/api/doctor")
    async def doctor() -> list[dict]:
        checks = []
        checks.append({"name": "python", "status": "ok", "detail": platform.python_version()})
        checks.append({"name": "model-backend", "status": "ok" if _build_llm(config) else "warning",
                       "detail": "configured" if _build_llm(config) else "none"})
        sp = _judicial_state_path(config)
        checks.append({"name": "judicial-state", "status": "ok" if sp.exists() else "warning",
                       "detail": str(sp)})
        checks.append({"name": "audit", "status": "ok",
                       "detail": f"{len(_read_audit(config))} entries"})
        return checks

    # -- config -------------------------------------------------------------
    @app.get("/api/config")
    async def get_config() -> dict:
        data = config.model_dump(mode="json")
        _redact(data)
        return data

    # -- models -------------------------------------------------------------
    @app.get("/api/models")
    async def models() -> dict:
        backends = []
        for b in config.model.backends:
            backends.append({
                "provider": b.provider.value,
                "model": b.model,
                "base_url": str(b.base_url),
                "tier": b.tier,
                "key": "set" if b.api_key else "none",
            })
        if not backends and config.llm.model:
            backends.append({
                "provider": config.llm.provider.value,
                "model": config.llm.model,
                "base_url": str(config.llm.base_url or ""),
                "tier": 2,
                "key": "set" if config.llm.api_key else "none",
            })
        return {"items": backends}

    # -- metrics / audit ----------------------------------------------------
    @app.get("/api/metrics")
    async def metrics() -> dict:
        return _get_metrics(config)

    @app.get("/api/audit")
    async def audit(limit: int = 100) -> dict:
        return {"items": _read_audit(config, limit=limit)}

    # -- sessions -----------------------------------------------------------
    @app.get("/api/sessions")
    async def sessions() -> dict:
        from justagent.agent.session import get_session_store

        store = get_session_store()
        metas = store.list_sessions()
        return {"items": [
            {"id": m.session_id, "title": m.title, "created_at": m.created_at,
             "updated_at": m.updated_at, "status": m.status.value}
            for m in metas
        ]}

    # -- plugins ------------------------------------------------------------
    @app.get("/api/plugins")
    async def plugins() -> dict:
        from justagent.core.plugin_registry import PluginRegistry

        return {"items": [
            {"name": p.name, "version": p.version, "trust": p.trust_level.value,
             "source": p.source}
            for p in PluginRegistry().list()
        ]}

    # -- knowledge RAG ------------------------------------------------------
    @app.post("/api/knowledge/search")
    async def knowledge_search(payload: dict) -> dict:
        from justagent.cli.commands.judicial import _JudicialState

        state = _JudicialState.load(state_path)
        query = (payload.get("query") or "").strip()
        top_k = int(payload.get("top_k", 5))
        results = state.knowledge_base.search_articles(query, top_k=top_k)
        return {"hits": [
            {"citation": r.article.citation, "law_name": r.article.law_name,
             "article_number": r.article.article_number, "score": r.score,
             "content": r.article.content}
            for r in results
        ]}

    # -- chat ---------------------------------------------------------------
    @app.post("/api/chat")
    async def chat(payload: dict) -> JSONResponse:
        message = (payload.get("message") or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")
        llm = _build_llm(config)
        if llm is None:
            return JSONResponse(status_code=200, content={
                "reply": (
                    "No LLM backend is configured, so the agent cannot chat. "
                    "Configure a model backend (e.g. in .justagent.toml) or use "
                    "the judicial panel below, which works without an LLM."
                ),
                "error": "no_llm",
            })
        session_id = (payload.get("session_id") or "").strip() or "default"
        async with _runtimes_lock:
            runtime = _runtimes.get(session_id)
        if runtime is None:
            runtime = AgentRuntime(
                client=llm,
                tools=make_default_tools(str(state_path)),
                config=AgentRuntimeConfig(system_prompt=(
                    "You are JustAgent, an assistant for judicial work. Use the "
                    "judicial tool to manage cases, evidence, legal knowledge and "
                    "documents. Be concise and accurate."
                )),
                cwd=str(config.project_root),
            )
            async with _runtimes_lock:
                _runtimes[session_id] = runtime
            try:
                result = await runtime.run(message)
            except Exception as exc:  # noqa: BLE001
                return JSONResponse(status_code=200,
                                    content={"reply": f"Agent error: {exc}", "error": "agent_error"})
        else:
            try:
                result = await runtime.continue_run(message)
            except Exception as exc:  # noqa: BLE001
                return JSONResponse(status_code=200,
                                    content={"reply": f"Agent error: {exc}", "error": "agent_error"})
        return JSONResponse(status_code=200, content={
            "reply": result.final_content or "(no output)",
            "status": result.status,
            "session_id": session_id,
            "error": result.error or "",
        })

    # -- judicial -----------------------------------------------------------
    @app.get("/api/judicial/cases")
    async def list_cases() -> dict:
        return {"items": _load_judicial(config)["cases"]}

    @app.post("/api/judicial/case")
    async def create_case(payload: dict) -> dict:
        from justagent.cli.commands.judicial import _JudicialState

        state = _JudicialState.load(state_path)
        case = state.case_manager.create_case(
            case_number=payload.get("case_number") or "",
            cause_of_action=payload.get("cause") or "",
            court=payload.get("court") or "",
            domain=payload.get("domain") or "",
        )
        state.save()
        return {"ok": True, "id": case.id, "case_number": case.case_number}

    @app.get("/api/judicial/case/{case_id}")
    async def case_detail(case_id: str) -> dict:
        from justagent.cli.commands.judicial import _JudicialState

        state = _JudicialState.load(state_path)
        case = _find_case(state, case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")
        evidence = state.evidence_chain.list_evidence(case_id=case.id)
        return {
            "id": case.id, "case_number": case.case_number,
            "cause": case.cause_of_action, "court": case.court,
            "status": case.status.value, "domain": case.domain,
            "parties": [p.model_dump() for p in case.parties],
            "claims": [c.model_dump() for c in case.claims],
            "timeline": [ev.model_dump() for ev in sorted(case.timeline, key=lambda e: e.timestamp)],
            "evidence": [
                {"id": e.id, "name": e.name, "type": e.type.value,
                 "admissible": e.admissibility.value, "strength": e.probative_strength.value}
                for e in evidence
            ],
        }

    @app.get("/api/judicial/evidence")
    async def list_evidence() -> dict:
        return {"items": _load_judicial(config)["evidence"]}

    @app.post("/api/judicial/evidence/analyze")
    async def analyze_evidence(payload: dict) -> dict:
        from justagent.cli.commands.judicial import _JudicialState

        state = _JudicialState.load(state_path)
        try:
            result = state.evidence_chain.analyze(case_id=payload.get("case_id") or "")
            return {
                "completeness_score": result.completeness_score,
                "total_evidence": result.total_evidence,
                "contradiction_count": len(result.contradictions),
                "gaps": result.gaps,
                "summary": result.summary or "",
            }
        except Exception as exc:  # noqa: BLE001 - graceful when no evidence
            return {"completeness_score": 0.0, "total_evidence": 0,
                    "contradiction_count": 0, "gaps": [], "summary": f"分析不可用: {exc}"}

    @app.get("/api/judicial/laws")
    async def list_laws() -> dict:
        return {"items": _load_judicial(config)["laws"]}

    @app.post("/api/judicial/law")
    async def add_law(payload: dict) -> dict:
        from justagent.cli.commands.judicial import _JudicialState
        from justagent.judicial.legal_knowledge import LegalArticle, LegalDomain

        state = _JudicialState.load(state_path)
        article = LegalArticle(
            law_name=payload.get("law_name") or "未命名法律",
            article_number=payload.get("article_number") or "",
            content=payload.get("content") or "",
            domain=LegalDomain(payload.get("domain", "civil")),
        )
        state.knowledge_base.add_article(article)
        state.save()
        return {"ok": True, "id": article.id, "citation": article.citation}

    @app.post("/api/judicial/doc")
    async def generate_doc(payload: dict) -> dict:
        from justagent.cli.commands.judicial import _JudicialState
        from justagent.judicial.document_generator import LegalDocumentGenerator

        state = _JudicialState.load(state_path)
        case = _find_case(state, payload.get("case_id") or "")
        if case is None:
            return {"ok": False, "error": "case not found"}
        doc_type = payload.get("doc_type") or "evidence_list"
        generator = LegalDocumentGenerator(
            state.case_manager,
            evidence_chain=state.evidence_chain,
            knowledge_base=state.knowledge_base,
        )
        doc = generator.generate(case.id, doc_type, verify=True)
        return {"ok": True, "content": doc.content, "doc_type": doc_type}

    return app


def _find_case(state, case_id: str):
    case = state.case_manager.get_case(case_id)
    if case is not None:
        return case
    for c in state.case_manager.list_cases():
        if c.id.startswith(case_id):
            return c
    return None


def _pkg(name: str) -> str:
    try:
        from importlib.metadata import version as v

        return v(name)
    except Exception:  # noqa: BLE001
        return "?"


def _redact(data: Any) -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                _redact(v)
            elif isinstance(v, str) and any(s in k.lower() for s in ("key", "token", "secret", "password")):
                data[k] = "***"
    elif isinstance(data, list):
        for item in data:
            _redact(item)


def run(config: AppConfig, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the JustAgent web server."""
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)
