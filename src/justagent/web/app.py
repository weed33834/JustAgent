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
import os
import platform
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from justagent.agent.runtime import AgentRuntime, AgentRuntimeConfig, LLMClient
from justagent.agent.tools.builtin import make_default_tools
from justagent.models.config import AppConfig

_HTML_DIR = Path(__file__).resolve().parent / "static"
_REMOTE_PROVIDERS = {"openai", "openrouter", "azure_openai"}


def _judicial_state_path(config: AppConfig) -> Path:
    return config.project_root / ".justagent" / "judicial_state.json"


def _state_path_for(root: Path) -> Path:
    return root / ".justagent" / "judicial_state.json"


def _load_judicial(config: AppConfig) -> dict:
    return _load_judicial_for(config.project_root)


def _load_judicial_for(root: Path) -> dict:
    from justagent.verticals.legal.cli import _JudicialState

    state = _JudicialState.load(_state_path_for(root))
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
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        timeout=timeout,
        provider=provider,
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


def create_app(config: AppConfig, *, no_auth: bool = False) -> FastAPI:
    """Create the FastAPI application bound to a project config."""

    app = FastAPI(title="JustAgent Web", version="1.0.0")
    state_path = _judicial_state_path(config)

    def _resolve_project(request: Request) -> Path:
        """Resolve the active project root from the X-JustAgent-Project header."""
        name = request.headers.get("x-justagent-project", "").strip()
        if name:
            try:
                from justagent.core.project_store import ProjectStore

                proj = ProjectStore().get(name)
                if proj is not None:
                    return Path(proj.path)
            except Exception:  # noqa: BLE001 - fall back to default
                pass
        return config.project_root

    def _project_state(request: Request) -> Path:
        return _state_path_for(_resolve_project(request))

    from justagent.web.users import ADMIN_ROLES, WRITE_ROLES, TokenManager, UserStore

    user_store = UserStore()
    user_store.ensure_admin()
    tokens = TokenManager()
    shared_token = os.environ.get("JUSTAGENT_WEB_TOKEN", "")
    if not no_auth:
        no_auth = os.environ.get("JUSTAGENT_WEB_NO_AUTH", "") not in ("", "0")

    def _resolve_user(request: Request) -> dict | None:
        """Resolve a session user from the Authorization header (if any)."""
        header = request.headers.get("authorization", "")
        if header.startswith("Bearer "):
            return tokens.resolve(header[7:])
        return None

    def _require_write(request: Request) -> None:
        """403 for session users without write role.

        Anonymous requests never reach endpoints (middleware default-deny);
        shared-token and --no-auth requests carry full access.
        """
        user = getattr(request.state, "user", None)
        if user is not None and user.get("role") not in WRITE_ROLES:
            raise HTTPException(status_code=403, detail=f"{user.get('role')} cannot write")

    @app.post("/api/auth/login")
    async def login(payload: dict) -> dict:
        username = (payload.get("username") or "").strip()
        password = payload.get("password") or ""
        user = user_store.authenticate(username, password)
        if user is None:
            raise HTTPException(status_code=401, detail="invalid credentials")
        _notify("auth", f"用户 {user.username} 登录")
        return {"token": tokens.issue(user), "username": user.username, "role": user.role}

    @app.get("/api/auth/users")
    async def list_users(request: Request) -> dict:
        user = _resolve_user(request)
        if not user or user.get("role") not in ADMIN_ROLES:
            raise HTTPException(status_code=403, detail="admin required")
        return {"items": user_store.list_users()}

    @app.post("/api/auth/users/role")
    async def set_user_role(request: Request, payload: dict) -> dict:
        user = _resolve_user(request)
        if not user or user.get("role") not in ADMIN_ROLES:
            raise HTTPException(status_code=403, detail="admin required")
        ok = user_store.set_role(payload.get("username", ""), payload.get("role", ""), user["role"])
        return {"ok": ok}

    @app.middleware("http")
    async def _auth(request: Request, call_next: Any) -> Any:
        path = request.url.path
        if no_auth or path in ("/", "/api/health", "/api/auth/login"):
            return await call_next(request)
        header = request.headers.get("authorization", "")
        # Shared deployment token grants full access (operator-level credential).
        if shared_token and header == f"Bearer {shared_token}":
            return await call_next(request)
        # Otherwise a valid session issued by /api/auth/login is required.
        session = tokens.resolve(header[7:]) if header.startswith("Bearer ") else None
        if session is None:
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        request.state.user = session
        return await call_next(request)

    # Per-session runtimes so the web chat keeps multi-turn conversation memory.
    _runtimes: dict[str, Any] = {}
    _runtimes_lock = asyncio.Lock()

    # -- notifications ------------------------------------------------------
    _notifications: list[dict] = []
    _webhook = os.environ.get("JUSTAGENT_WEBHOOK_URL", "")
    _smtp: dict[str, Any] = {
        "host": os.environ.get("JUSTAGENT_SMTP_HOST", ""),
        "port": int(os.environ.get("JUSTAGENT_SMTP_PORT", "465") or 465),
        "user": os.environ.get("JUSTAGENT_SMTP_USER", ""),
        "password": os.environ.get("JUSTAGENT_SMTP_PASSWORD", ""),
        "to": os.environ.get("JUSTAGENT_SMTP_TO", ""),
    }

    def _notify_email(kind: str, message: str) -> None:
        if not _smtp["host"] or not _smtp["to"]:
            return
        try:
            import smtplib
            from email.mime.text import MIMEText

            body = f"JustAgent 通知 [{kind}]\n\n{message}"
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = f"JustAgent 通知: {kind}"
            msg["From"] = _smtp["user"] or _smtp["to"]
            msg["To"] = _smtp["to"]
            with smtplib.SMTP_SSL(_smtp["host"], _smtp["port"], timeout=10) as s:
                if _smtp["user"]:
                    s.login(_smtp["user"], _smtp["password"])
                s.sendmail(msg["From"], [_smtp["to"]], msg.as_string())
        except Exception:  # noqa: BLE001 - email is best-effort
            pass

    def _notify(kind: str, message: str) -> None:
        entry = {"ts": time.time(), "kind": kind, "message": message}
        _notifications.append(entry)
        del _notifications[:-200]
        if _webhook:
            try:
                import httpx

                httpx.post(_webhook, json=entry, timeout=5)
            except Exception:  # noqa: BLE001 - webhook is best-effort
                pass
        _notify_email(kind, message)

    @app.get("/api/notifications")
    async def notifications() -> dict:
        return {"items": list(reversed(_notifications))}

    @app.post("/api/notifications/test")
    async def notify_test(request: Request) -> dict:
        _require_write(request)
        _notify("info", "测试通知")
        return {"ok": True}

    # -- uploads ------------------------------------------------------------
    @app.post("/api/upload")
    async def upload_file(request: Request) -> dict:
        from fastapi import UploadFile

        _require_write(request)
        form = await request.form()
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(status_code=400, detail="file is required")
        up_dir = config.project_root / ".justagent" / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        safe = (file.filename or "file").replace("/", "_").replace("\\", "_")
        dest = up_dir / f"{int(time.time())}_{safe}"
        dest.write_bytes(await file.read())
        _notify("upload", f"上传附件 {safe}")
        return {"ok": True, "name": safe, "path": str(dest), "size": dest.stat().st_size}

    # -- pages --------------------------------------------------------------
    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_HTML_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/state")
    async def get_state(request: Request) -> dict:
        return _load_judicial_for(_resolve_project(request))

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
        checks.append(
            {
                "name": "model-backend",
                "status": "ok" if _build_llm(config) else "warning",
                "detail": "configured" if _build_llm(config) else "none",
            }
        )
        sp = _judicial_state_path(config)
        checks.append(
            {
                "name": "judicial-state",
                "status": "ok" if sp.exists() else "warning",
                "detail": str(sp),
            }
        )
        checks.append(
            {"name": "audit", "status": "ok", "detail": f"{len(_read_audit(config))} entries"}
        )
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
            backends.append(
                {
                    "provider": b.provider.value,
                    "model": b.model,
                    "base_url": str(b.base_url),
                    "tier": b.tier,
                    "key": "set" if b.api_key else "none",
                }
            )
        if not backends and config.llm.model:
            backends.append(
                {
                    "provider": config.llm.provider.value,
                    "model": config.llm.model,
                    "base_url": str(config.llm.base_url or ""),
                    "tier": 2,
                    "key": "set" if config.llm.api_key else "none",
                }
            )
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
        return {
            "items": [
                {
                    "id": m.id,
                    "title": (m.prompt_preview or "")[:40],
                    "created_at": m.created_at,
                    "updated_at": m.updated_at,
                    "status": m.status.value,
                }
                for m in metas
            ]
        }

    @app.delete("/api/sessions/{session_id}")
    async def session_delete(request: Request, session_id: str) -> dict:
        from justagent.agent.session import get_session_store

        _require_write(request)
        store = get_session_store()
        ok = store.delete(session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="session not found")
        _notify("session", f"删除会话 {session_id[:8]}")
        return {"ok": True}

    # -- plugins ------------------------------------------------------------
    @app.get("/api/plugins")
    async def plugins() -> dict:
        from justagent.core.plugin_registry import PluginRegistry

        return {
            "items": [
                {
                    "name": p.name,
                    "version": p.version,
                    "trust": p.trust_level.value,
                    "source": p.source,
                }
                for p in PluginRegistry().list()
            ]
        }

    # -- schedule ------------------------------------------------------------
    @app.get("/api/schedule")
    async def schedule() -> dict:
        from justagent.core.project_store import ProjectStore
        from justagent.core.scheduler import Scheduler, ScheduleStore

        scheduler = Scheduler(store=ScheduleStore(), project_store=ProjectStore())
        return {
            "items": [
                {
                    "name": t.name,
                    "schedule": t.schedule,
                    "enabled": t.enabled,
                    "last_run": t.last_run,
                    "next_run": t.next_run,
                    "created_at": t.created_at,
                }
                for t in scheduler.list_tasks()
            ]
        }

    @app.post("/api/schedule")
    async def schedule_add(request: Request, payload: dict) -> dict:
        from justagent.core.project_store import ProjectStore
        from justagent.core.scheduler import Scheduler, ScheduleStore

        _require_write(request)
        name = (payload.get("name") or "").strip()
        schedule_expr = (payload.get("schedule") or "").strip() or "0 9 * * *"
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        scheduler = Scheduler(store=ScheduleStore(), project_store=ProjectStore())
        task = scheduler.add_task(
            name=name,
            schedule=schedule_expr,
            command=str(payload.get("action") or payload.get("command") or ""),
            enabled=bool(payload.get("enabled", True)),
        )
        _notify("schedule", f"新建定时任务 {task.name}")
        return {"ok": True, "name": task.name, "schedule": task.schedule}

    # -- knowledge RAG ------------------------------------------------------
    @app.post("/api/knowledge/search")
    async def knowledge_search(payload: dict) -> dict:
        from justagent.verticals.legal.cli import _JudicialState

        state = _JudicialState.load(state_path)
        query = (payload.get("query") or "").strip()
        top_k = int(payload.get("top_k", 5))
        results = state.knowledge_base.search_articles(query, top_k=top_k)
        return {
            "hits": [
                {
                    "citation": r.article.citation,
                    "law_name": r.article.law_name,
                    "article_number": r.article.article_number,
                    "score": r.score,
                    "content": r.article.content,
                }
                for r in results
            ]
        }

    # -- chat ---------------------------------------------------------------
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
        session_id = (payload.get("session_id") or "").strip() or "default"
        async with _runtimes_lock:
            runtime = _runtimes.get(session_id)
        if runtime is None:
            runtime = AgentRuntime(
                client=llm,
                tools=make_default_tools(str(config.project_root)),
                config=AgentRuntimeConfig(
                    system_prompt=(
                        "You are JustAgent, an assistant for judicial work. Use the "
                        "judicial tool to manage cases, evidence, legal knowledge and "
                        "documents. Be concise and accurate."
                    )
                ),
                cwd=str(config.project_root),
            )
            async with _runtimes_lock:
                _runtimes[session_id] = runtime
            try:
                result = await runtime.run(message)
            except Exception as exc:  # noqa: BLE001
                return JSONResponse(
                    status_code=200,
                    content={"reply": f"Agent error: {exc}", "error": "agent_error"},
                )
        else:
            try:
                result = await runtime.continue_run(message)
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
                "session_id": session_id,
                "error": result.error or "",
            },
        )

    # -- chat (SSE streaming) -------------------------------------------------
    @app.post("/api/chat/stream", response_model=None)
    async def chat_stream(payload: dict) -> StreamingResponse | JSONResponse:
        message = (payload.get("message") or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")
        llm = _build_llm(config)
        if llm is None:
            return JSONResponse(
                status_code=200,
                content={
                    "reply": "No LLM backend is configured, so the agent cannot chat.",
                    "error": "no_llm",
                },
            )
        history = payload.get("history") or []
        sys_prompt = (
            "You are JustAgent, an assistant for judicial work. Use the judicial "
            "tool to manage cases, evidence, legal knowledge and documents. "
            "Be concise and accurate."
        )
        if history:
            turns = []
            for h in history[-20:]:
                role = "用户" if h.get("role") == "user" else "助手"
                turns.append(f"{role}: {h.get('content', '')}")
            sys_prompt += "\n\n以下为最近对话上下文：\n" + "\n".join(turns)

        queue: asyncio.Queue = asyncio.Queue()

        async def _emit(event: Any) -> None:
            await queue.put(event)

        runtime = AgentRuntime(
            client=llm,
            tools=make_default_tools(str(config.project_root)),
            config=AgentRuntimeConfig(system_prompt=sys_prompt),
            cwd=str(config.project_root),
            emit=_emit,
        )
        task = asyncio.create_task(runtime.run(message))

        async def _stream() -> AsyncIterator[str]:
            try:
                while True:
                    event = await queue.get()
                    etype = getattr(event, "type", "")
                    if etype == "assistant_message":
                        yield f"data: {json.dumps({'type': 'delta', 'content': event.content}, ensure_ascii=False)}\n\n"
                    elif etype == "tool_started":
                        yield f"data: {json.dumps({'type': 'tool_start', 'tool': getattr(event, 'tool', '')}, ensure_ascii=False)}\n\n"
                    elif etype == "tool_finished":
                        yield f"data: {json.dumps({'type': 'tool_end', 'tool': getattr(event, 'tool', '')}, ensure_ascii=False)}\n\n"
                    elif etype in ("run_completed", "run_failed", "run_aborted"):
                        break
            except Exception:  # noqa: BLE001
                pass
            result = await task
            yield f"data: {json.dumps({'type': 'done', 'content': result.final_content or '(no output)', 'status': result.status, 'error': result.error or ''}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- projects -----------------------------------------------------------
    @app.get("/api/projects")
    async def projects() -> dict:
        from justagent.core.project_store import ProjectStore

        store = ProjectStore()
        return {
            "current": str(config.project_root),
            "items": [{"name": p.name, "path": str(p.path)} for p in store.list_all()],
        }

    @app.post("/api/projects")
    async def project_add(request: Request, payload: dict) -> dict:
        from justagent.core.project_store import ProjectStore

        _require_write(request)
        name = (payload.get("name") or "").strip()
        path = (payload.get("path") or "").strip()
        if not name or not path:
            raise HTTPException(status_code=400, detail="name and path are required")
        root = Path(path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        from justagent.core.project_store import ManagedProject

        ProjectStore().add(ManagedProject(name=name, path=str(root), added_at=time.time()))
        _notify("project", f"添加项目 {name}")
        return {"ok": True, "name": name, "path": str(root)}

    # -- report (printable HTML) --------------------------------------------
    @app.get("/api/report", response_class=HTMLResponse)
    async def report() -> str:
        data = _load_judicial(config)
        cases = (
            "".join(
                f"<h3>{c['case_number'] or c['id'][:8]}</h3>"
                f"<p>案由：{c['cause']}｜法院：{c['court']}｜状态：{c['status']}｜当事人：{c['parties']}｜时间线：{c['timeline']}</p>"
                for c in data["cases"]
            )
            or "<p>（暂无案件）</p>"
        )
        evidence = (
            "".join(
                f"<li>{e['name']}（{e['type']}）可采性：{e['admissible']} 证明力：{e['strength']}</li>"
                for e in data["evidence"]
            )
            or "<li>（暂无证据）</li>"
        )
        laws = (
            "".join(f"<li>{law['citation']}（{law['domain']}）</li>" for law in data["laws"])
            or "<li>（暂无法条）</li>"
        )
        return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>JustAgent 司法报表</title>
        <style>body{{font-family:sans-serif;padding:24px;color:#1b2333}} h1{{color:#6366f1}}
        h2{{border-bottom:1px solid #e6e8f2;padding-bottom:4px}} li{{margin:4px 0}}
        .meta{{color:#96a0b4;font-size:12px}} </style></head><body>
        <h1>JustAgent 司法报表</h1>
        <p class="meta">项目：{config.project_root}｜生成于 {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <h2>案件（{len(data["cases"])}）</h2>{cases}
        <h2>证据（{len(data["evidence"])}）</h2><ul>{evidence}</ul>
        <h2>法条（{len(data["laws"])}）</h2><ul>{laws}</ul>
        </body></html>"""

    # -- vision (multimodal image analysis) ----------------------------------
    @app.post("/api/vision")
    async def vision(payload: dict) -> dict:
        llm = _build_llm(config)
        if llm is None:
            return {"ok": False, "error": "no_llm", "reply": "未配置模型后端，无法进行图像分析。"}
        prompt = (payload.get("prompt") or "").strip() or "请描述这张图片并提取其中的关键信息。"
        image = payload.get("image") or ""  # data URL or base64
        if not image:
            return {"ok": False, "error": "no_image", "reply": "未提供图片。"}
        from openai import OpenAI

        client = OpenAI(
            api_key=llm._api_key or "placeholder",
            base_url=llm._base_url or None,
            max_retries=2,
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image.startswith("data:"):
            content.append({"type": "image_url", "image_url": {"url": image}})
        else:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
            )
        try:
            resp = client.chat.completions.create(
                model=llm._model,
                messages=cast(Any, [{"role": "user", "content": content}]),
                max_tokens=512,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": "vision_error", "reply": f"图像分析失败：{exc}"}
        return {"ok": True, "reply": resp.choices[0].message.content or ""}

    # -- judicial -----------------------------------------------------------
    @app.get("/api/judicial/doc/types")
    async def doc_types() -> dict:
        from justagent.verticals.legal.document_generator import LegalDocumentType

        return {"items": [{"id": t.value, "name": t.name} for t in LegalDocumentType]}

    @app.get("/api/judicial/cases")
    async def list_cases(request: Request) -> dict:
        return {"items": _load_judicial_for(_resolve_project(request))["cases"]}

    @app.post("/api/judicial/case")
    async def create_case(request: Request, payload: dict) -> dict:
        from justagent.verticals.legal.cli import _JudicialState

        _require_write(request)
        state = _JudicialState.load(_project_state(request))
        case = state.case_manager.create_case(
            case_number=payload.get("case_number") or "",
            cause_of_action=payload.get("cause") or "",
            court=payload.get("court") or "",
            domain=payload.get("domain") or "",
        )
        state.save()
        _notify("judicial", f"创建案件 {case.case_number or case.id[:8]}")
        return {"ok": True, "id": case.id, "case_number": case.case_number}

    @app.get("/api/judicial/case/{case_id}")
    async def case_detail(request: Request, case_id: str) -> dict:
        from justagent.verticals.legal.cli import _JudicialState

        state = _JudicialState.load(_project_state(request))
        case = _find_case(state, case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")
        evidence = state.evidence_chain.list_evidence(case_id=case.id)
        return {
            "id": case.id,
            "case_number": case.case_number,
            "cause": case.cause_of_action,
            "court": case.court,
            "status": case.status.value,
            "domain": case.domain,
            "parties": [p.model_dump() for p in case.parties],
            "claims": [c.model_dump() for c in case.claims],
            "timeline": [
                ev.model_dump() for ev in sorted(case.timeline, key=lambda e: e.timestamp)
            ],
            "evidence": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type.value,
                    "admissible": e.admissibility.value,
                    "strength": e.probative_strength.value,
                }
                for e in evidence
            ],
        }

    @app.get("/api/judicial/evidence")
    async def list_evidence(request: Request) -> dict:
        return {"items": _load_judicial_for(_resolve_project(request))["evidence"]}

    @app.post("/api/judicial/evidence/analyze")
    async def analyze_evidence(request: Request, payload: dict) -> dict:
        from justagent.verticals.legal.cli import _JudicialState

        state = _JudicialState.load(_project_state(request))
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
            return {
                "completeness_score": 0.0,
                "total_evidence": 0,
                "contradiction_count": 0,
                "gaps": [],
                "summary": f"分析不可用: {exc}",
            }

    @app.post("/api/judicial/evidence/audit")
    async def audit_evidence(request: Request, payload: dict) -> dict:
        """Deterministic full chain audit — no LLM required.

        Checks: custody-chain integrity, timeline consistency,
        same-source corroboration, and claim-evidence coverage.
        """
        from justagent.verticals.legal.cli import _JudicialState
        from justagent.verticals.legal.evidence import EvidenceAuditor

        state = _JudicialState.load(_project_state(request))
        case = None
        wanted = payload.get("case_id") or ""
        for candidate in state.case_manager.list_cases():
            if not wanted or candidate.id.startswith(wanted) or candidate.case_number == wanted:
                case = candidate
                break
        if case is None:
            raise HTTPException(status_code=404, detail=f"case not found: {wanted}")
        filing_date = ""
        for event in case.timeline:
            if getattr(event, "description", "") == "立案":
                filing_date = str(getattr(event, "timestamp", "") or "")
                break
        try:
            audit = EvidenceAuditor(state.evidence_chain).audit_case(
                case.id, claims=list(case.claims), filing_date=filing_date
            )
        except Exception as exc:  # noqa: BLE001
            return {"verdict": "严重缺陷", "summary": f"审计不可用: {exc}"}
        data = audit.model_dump()
        data["chain"]["contradiction_count"] = len(audit.chain.contradictions)
        return data

    @app.get("/api/judicial/laws")
    async def list_laws(request: Request) -> dict:
        return {"items": _load_judicial_for(_resolve_project(request))["laws"]}

    @app.post("/api/judicial/law")
    async def add_law(request: Request, payload: dict) -> dict:
        from justagent.verticals.legal.cli import _JudicialState
        from justagent.verticals.legal.legal_knowledge import LegalArticle, LegalDomain

        _require_write(request)
        state = _JudicialState.load(_project_state(request))
        article = LegalArticle(
            law_name=payload.get("law_name") or "未命名法律",
            article_number=payload.get("article_number") or "",
            content=payload.get("content") or "",
            domain=LegalDomain(payload.get("domain", "civil")),
        )
        state.knowledge_base.add_article(article)
        state.save()
        _notify("judicial", f"添加法条 {article.citation}")
        return {"ok": True, "id": article.id, "citation": article.citation}

    @app.post("/api/judicial/doc")
    async def generate_doc(request: Request, payload: dict) -> dict:
        from justagent.verticals.legal.cli import _JudicialState
        from justagent.verticals.legal.document_generator import LegalDocumentGenerator

        _require_write(request)
        state = _JudicialState.load(_project_state(request))
        case = _find_case(state, payload.get("case_id") or "")
        if case is None:
            return {"ok": False, "error": "case not found"}
        doc_type = payload.get("doc_type") or "evidence_list"
        generator = LegalDocumentGenerator(
            state.case_manager,
            evidence_chain=state.evidence_chain,
            knowledge_base=state.knowledge_base,
        )
        from justagent.verticals.legal.document_generator import LegalDocumentType

        doc = generator.generate(case.id, LegalDocumentType(doc_type), verify=True)
        return {"ok": True, "content": doc.content, "doc_type": doc_type}

    return app


def _find_case(state: Any, case_id: str) -> Any:
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
            elif isinstance(v, str) and any(
                s in k.lower() for s in ("key", "token", "secret", "password")
            ):
                data[k] = "***"
    elif isinstance(data, list):
        for item in data:
            _redact(item)


def run(
    config: AppConfig, host: str = "127.0.0.1", port: int = 8000, *, no_auth: bool = False
) -> None:
    """Start the JustAgent web server."""
    import uvicorn

    app = create_app(config, no_auth=no_auth)
    uvicorn.run(app, host=host, port=port)
