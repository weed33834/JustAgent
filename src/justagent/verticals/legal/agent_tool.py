"""``judicial`` tool — let the conversational agent drive judicial operations.

Gives the agent access to the judicial subsystem (cases, evidence, legal
knowledge, document generation) so users can manage everything through the
chat interface instead of remembering CLI commands. Loads the same persisted
judicial state the CLI uses (``.justagent/judicial_state.json``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from justagent.agent.tools.base import Tool, ToolContext, ToolResult

JudicialAction = Literal[
    "list_cases",
    "case_summary",
    "list_evidence",
    "analyze_evidence",
    "list_laws",
    "search_laws",
    "generate_doc",
]


class JudicialInput(BaseModel):
    """Input for the ``judicial`` tool."""

    action: JudicialAction = Field(..., description=(
        "Which judicial operation to run: list_cases, case_summary, "
        "list_evidence, analyze_evidence, list_laws, search_laws, generate_doc."
    ))
    case_id: str | None = Field(None, description="Case id (prefix match allowed).")
    query: str | None = Field(None, description="Search query (for search_laws).")
    doc_type: str | None = Field(
        None, description="Legal document type for generate_doc "
        "(indictment/judgment/ruling/evidence_list/...)."
    )
    domain: str | None = Field(None, description="Legal domain filter (civil/criminal/...).")


_JUDICIAL_DESCRIPTION = """\
Manage the judicial subsystem through the chat: list cases, summarize a case,
list/analyze evidence, browse or search the legal knowledge base, and generate
legal documents. This is the conversational entry point to JustAgent's
judicial features (same persisted state as the CLI).
"""


def _load_state(state_path: Path) -> Any:
    from justagent.verticals.legal.cli import _JudicialState

    return _JudicialState.load(state_path)


def _run(action: str, args: JudicialInput, state_path: Path) -> str:
    state = _load_state(state_path)
    case_id = args.case_id or ""

    if action == "list_cases":
        cases = state.case_manager.list_cases()
        if not cases:
            return "No cases."
        lines = [f"{len(cases)} case(s):"]
        for c in cases:
            lines.append(
                f"- {c.case_number or c.id[:8]}: {c.cause_of_action or '-'} "
                f"[{c.status.value}] {len(c.parties)} parties, {len(c.timeline)} timeline events"
            )
        return "\n".join(lines)

    if action == "case_summary":
        case = _find_case(state, case_id)
        if case is None:
            return f"Case not found: {case_id}"
        parties = "、".join(f"{p.role.value} {p.name}" for p in case.parties) or "none"
        claims = "；".join(f"{c.description}" for c in case.claims) or "none"
        lines = [
            f"Case {case.case_number or case.id[:8]}",
            f"  cause: {case.cause_of_action or '-'} | court: {case.court or '-'} | status: {case.status.value}",
            f"  parties: {parties}",
            f"  claims: {claims}",
            f"  evidence: {len(case.evidence_ids)} items | materials: {len(case.material_ids)}",
        ]
        if case.timeline:
            lines.append("  timeline:")
            for ev in sorted(case.timeline, key=lambda e: (e.timestamp, e.date)):
                lines.append(f"    - {ev.date or '-'} [{ev.category or '-'}] {ev.description}")
        return "\n".join(lines)

    if action == "list_evidence":
        evidence = state.evidence_chain.list_evidence(case_id=case_id)
        if not evidence:
            return "No evidence."
        lines = [f"{len(evidence)} evidence item(s):"]
        for e in evidence:
            lines.append(
                f"- {e.name} ({e.type.value}) proving: {e.proving_object or '-'} | "
                f"admissible: {e.admissibility.value} | strength: {e.probative_strength.value}"
            )
        return "\n".join(lines)

    if action == "analyze_evidence":
        result = state.evidence_chain.analyze(case_id=case_id)
        return (
            f"Completeness: {result.completeness_score:.2f} | "
            f"contradictions: {result.contradiction_count}\n{result.summary or ''}"
        )

    if action == "list_laws":
        articles = state.knowledge_base.list_articles(
            domain=_parse_domain(args.domain), law_name=None, status=None
        )
        if not articles:
            return "Legal library is empty."
        return "\n".join(f"- {a.citation} [{a.domain.value}] {a.content[:80]}" for a in articles)

    if action == "search_laws":
        results = state.knowledge_base.search_articles(args.query or "", top_k=5)
        if not results:
            return "No matching articles."
        return "\n".join(
            f"- {r.article.citation} (score {r.score:.2f}): {r.article.content[:100]}"
            for r in results
        )

    if action == "generate_doc":
        from justagent.verticals.legal.document_generator import (
            LegalDocumentGenerator,
            LegalDocumentType,
        )

        case = _find_case(state, case_id)
        if case is None:
            return f"Case not found: {case_id}"
        doc_type = args.doc_type or "evidence_list"
        generator = LegalDocumentGenerator(
            state.case_manager,
            evidence_chain=state.evidence_chain,
            knowledge_base=state.knowledge_base,
        )
        doc = generator.generate(case.id, LegalDocumentType(doc_type), verify=True)
        return f"Generated {doc_type} for {case.case_number or case.id[:8]}:\n{doc.content[:2000]}"

    return f"Unknown action: {action}"


def _find_case(state: Any, case_id: str) -> Any:
    if not case_id:
        return None
    case = state.case_manager.get_case(case_id)
    if case is not None:
        return case
    for c in state.case_manager.list_cases():
        if c.id.startswith(case_id):
            return c
    return None


def _parse_domain(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        from justagent.verticals.legal.legal_knowledge import LegalDomain

        return LegalDomain(value)
    except Exception:  # noqa: BLE001
        return None


def make_judicial_tool(state_path: Path | None = None) -> Tool:
    """Construct the ``judicial`` tool (needs a persisted judicial state path)."""

    async def _exec(args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, JudicialInput)
        if state_path is None:
            return ToolResult.failure(
                "Judicial tool is not configured (no state path). Run inside a JustAgent project."
            )
        try:
            output = _run(args.action, args, Path(state_path))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Judicial operation failed: {exc}")
        return ToolResult.success(output)

    return Tool(
        id="judicial",
        description=_JUDICIAL_DESCRIPTION,
        parameters=JudicialInput,
        execute=_exec,
        timeout_ms=60_000,
    )


__all__ = ["JudicialInput", "make_judicial_tool"]

def make_legal_tool(state_root: Path | None = None) -> Tool:
    """Entry-point factory for the ``justagent.tools`` group.

    Receives the project root and derives the persisted state path from it.
    """
    resolved = Path(state_root) / ".justagent" / "judicial_state.json" if state_root else None
    return make_judicial_tool(resolved)
