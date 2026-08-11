"""JustAgent Web — a browser chat interface for the conversational agent.

Serves a chat UI where users can talk to the JustAgent agent (which has the
``judicial`` tool) and also browse/manage the judicial subsystem directly
(cases / evidence / legal knowledge / documents) without the CLI.

Run with ``justagent web``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from justagent.agent.runtime import AgentRuntime, AgentRuntimeConfig, LLMClient
from justagent.agent.tools.builtin import make_default_tools
from justagent.models.config import AppConfig

_HTML_DIR = Path(__file__).resolve().parent / "static"


def _judicial_state_path(config: AppConfig) -> Path:
    return config.project_root / ".justagent" / "judicial_state.json"


_REMOTE_PROVIDERS = {"openai", "openrouter", "azure_openai"}


def _build_llm(config: AppConfig) -> LLMClient | None:
    """Build an LLM client from config; returns None if no usable backend."""
    provider = ""
    model = ""
    api_key = ""
    base_url = ""
    api_version = None
    timeout = 30.0

    if config.model.backends:
        b = config.model.backends[0]
        provider = b.provider.value
        model = b.model or ""
        api_key = b.api_key or ""
        base_url = str(b.base_url)
        api_version = b.api_version
        timeout = b.timeout
    else:
        llm = config.llm
        provider = llm.provider.value
        model = llm.model
        api_key = llm.api_key or ""
        base_url = str(llm.base_url or "")
        api_version = llm.api_version
        timeout = llm.timeout

    if not model:
        return None
    # Remote providers need an API key; fail fast instead of hanging.
    if provider in _REMOTE_PROVIDERS and not api_key:
        return None
    return LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        timeout=timeout,
        provider=provider,
    )


def _load_judicial(config: AppConfig) -> dict:
    """Return a serialisable snapshot of the judicial subsystem."""
    from justagent.cli.commands.judicial import _JudicialState

    state = _JudicialState.load(_judicial_state_path(config))
    cases = [
        {
            "id": c.id,
            "case_number": c.case_number or "",
            "cause": c.cause_of_action or "",
            "court": c.court or "",
            "status": c.status.value,
            "parties": len(c.parties),
            "timeline": len(c.timeline),
        }
        for c in state.case_manager.list_cases()
    ]
    evidence = [
        {
            "id": e.id,
            "name": e.name,
            "type": e.type.value,
            "proving_object": e.proving_object or "",
            "admissible": e.admissibility.value,
            "strength": e.probative_strength.value,
        }
        for e in state.evidence_chain.list_evidence()
    ]
    laws = [
        {
            "id": a.id,
            "citation": a.citation,
            "law_name": a.law_name,
            "domain": a.domain.value,
            "status": a.status.value,
        }
        for a in state.knowledge_base.list_articles()
    ]
    return {"cases": cases, "evidence": evidence, "laws": laws}


def create_app(config: AppConfig) -> FastAPI:
    """Create the FastAPI application bound to a project config."""

    app = FastAPI(title="JustAgent Web", version="1.0.0")
    state_path = _judicial_state_path(config)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_HTML_DIR / "index.html")

    @app.get("/api/state")
    async def get_state() -> dict:
        return _load_judicial(config)

    @app.post("/api/chat")
    async def chat(payload: dict) -> JSONResponse:
        message = (payload.get("message") or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")
        llm = _build_llm(config)
        if llm is None:
            return JSONResponse(
                status_code=200,
                content={
                    "reply": (
                        "No LLM backend is configured, so the agent cannot chat. "
                        "Configure a model backend (e.g. in .justagent.toml) or use "
                        "the judicial panel below, which works without an LLM."
                    ),
                    "error": "no_llm",
                },
            )
        tools = make_default_tools(str(state_path))
        runtime = AgentRuntime(
            client=llm,
            tools=tools,
            config=AgentRuntimeConfig(system_prompt=(
                "You are JustAgent, an assistant for judicial work. Use the "
                "judicial tool to manage cases, evidence, legal knowledge and "
                "documents. Be concise and accurate."
            )),
            cwd=str(config.project_root),
        )
        try:
            result = await runtime.run(message)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=200,
                content={"reply": f"Agent error: {exc}", "error": "agent_error"},
            )
        return JSONResponse(
            status_code=200,
            content={
                "reply": result.final_content or "(no output)",
                "status": result.status,
                "error": result.error or "",
            },
        )

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

    @app.post("/api/judicial/law")
    async def add_law(payload: dict) -> dict:
        from justagent.cli.commands.judicial import _JudicialState
        from justagent.judicial.legal_knowledge import (
            ArticleStatus,
            LegalArticle,
            LegalDomain,
        )

        state = _JudicialState.load(state_path)
        article = LegalArticle(
            law_name=payload.get("law_name") or "未命名法律",
            article_number=payload.get("article_number") or "",
            content=payload.get("content") or "",
            domain=LegalDomain(payload.get("domain", "civil")),
            status=ArticleStatus.EFFECTIVE,
        )
        state.knowledge_base.add_article(article)
        state.save()
        return {"ok": True, "id": article.id, "citation": article.citation}

    @app.post("/api/judicial/doc")
    async def generate_doc(payload: dict) -> dict:
        from justagent.cli.commands.judicial import _JudicialState
        from justagent.judicial.document_generator import LegalDocumentGenerator

        state = _JudicialState.load(state_path)
        case_id = payload.get("case_id") or ""
        case = state.case_manager.get_case(case_id)
        if case is None:
            for c in state.case_manager.list_cases():
                if c.id.startswith(case_id):
                    case = c
                    break
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

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


def run(config: AppConfig, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the JustAgent web server."""
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)
