"""The ``justagent fix`` command — LLM-powered fix proposal."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import typer
from pydantic import HttpUrl

from justagent.adapters.model_gateway import ChatMessage
from justagent.core.i18n import I18n, get_i18n_from_ctx
from justagent.core.model_router import ModelRouter
from justagent.exceptions import ModelGatewayError
from justagent.models.config import AppConfig, LlmProvider, ModelBackendConfig, Provider, ToolsConfig
from justagent.utils.hashing import ToolVerifier
from justagent.utils.patch import apply_patch, patch_paths_are_safe
from justagent.utils.redaction import redact_paths, redact_text

app = typer.Typer()

ERROR_LOG_PATH = Path.home() / ".local" / "state" / "justagent" / "last_error.txt"

ALLOWED_EXTENSIONS = {".py", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json"}
MAX_FILE_SIZE = 50 * 1024
MAX_RELEVANT_FILES = 5

SYSTEM_PROMPT = (
    "You are an expert software engineer. A verification command failed. "
    "Analyze the error output and project context, then propose a concrete fix. "
    "Respond with a brief explanation followed by a unified diff patch that can "
    "be applied with `git apply` or `patch -p1`. "
    "Fix the implementation/source code, NEVER the tests: do not modify any file "
    "whose path contains 'tests/' or 'test_'. If the error is clearly caused by a "
    "bug in a test, explain that instead of producing a patch. "
    "If you cannot produce a patch, explain what the user should check manually."
)

_TRACEBACK_FRAME_RE = re.compile(r'^\s*File\s+"([^"]+)"\s*,\s*line\s+\d+', re.MULTILINE)

_LLM_PROVIDER_TO_BACKEND: dict[LlmProvider, Provider] = {
    LlmProvider.OPENAI: Provider.OPENAI,
    LlmProvider.OPENROUTER: Provider.OPENROUTER,
    LlmProvider.OLLAMA: Provider.OLLAMA,
}

_DEFAULT_BASE_URLS: dict[Provider, str] = {
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.OPENROUTER: "https://openrouter.ai/api/v1",
    Provider.OLLAMA: "http://127.0.0.1:11434/v1",
}


def _model_router(config: AppConfig) -> ModelRouter:
    if not config.model.backends and config.llm.provider in _LLM_PROVIDER_TO_BACKEND:
        backend_provider = _LLM_PROVIDER_TO_BACKEND[config.llm.provider]
        base_url = config.llm.base_url or cast(HttpUrl, _DEFAULT_BASE_URLS[backend_provider])
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


def register(parent: typer.Typer) -> None:
    parent.command(name="fix")(fix)


@app.command(name="fix")
def fix(
    ctx: typer.Context,
    error_file: Path | None = typer.Argument(
        None, help="Path to error log (defaults to last verify output)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmations"),
) -> None:
    """Ask an LLM to propose a fix for the last verification failure."""
    config: AppConfig = ctx.obj["config"]
    i18n = get_i18n_from_ctx(ctx)
    dry_run: bool = ctx.obj.get("dry_run", False)
    yes = yes or ctx.obj.get("yes", False)

    source = error_file or ERROR_LOG_PATH
    if not source.exists():
        typer.echo(i18n._("fix.no_error_log", path=source), err=True)
        raise typer.Exit(code=1)

    error_context = source.read_text(encoding="utf-8")
    error_context = redact_text(error_context)
    error_context = redact_paths(error_context, config.project_root)
    if not error_context.strip():
        typer.echo(i18n._("fix.empty_error_log", path=source), err=True)
        raise typer.Exit(code=1)

    user_prompt, read_paths = _build_prompt(error_context, config.project_root)
    if read_paths:
        typer.echo(f"Reading {len(read_paths)} relevant file(s): {', '.join(read_paths)}")

    if dry_run:
        typer.echo(
            "Dry run — would send prompt to LLM" if read_paths else "No relevant files found"
        )
        return

    typer.echo("Thinking...")
    router = _model_router(config)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]
    try:
        response = router.chat(messages, "fix")
    except ModelGatewayError as exc:
        raise typer.BadParameter(f"Model backend unavailable: {exc}") from exc

    typer.echo("\n" + response)

    patch = _extract_patch(response)
    applied = False
    if patch and (yes or typer.confirm("Apply this patch?")):
        applied = _apply_patch(config.project_root, patch, i18n, config.tools)

    if not applied:
        if not patch:
            typer.secho("No patch found in LLM response", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)


def _build_prompt(error_context: str, project_root: Path) -> tuple[str, list[str]]:
    relevant_files, read_paths = _collect_relevant_files(project_root, error_context)
    files_section = ""
    if relevant_files:
        files_section = "\n\nRelevant project files:\n" + "\n".join(
            f"--- {path} ---\n{content[:2000]}" for path, content in relevant_files.items()
        )
    prompt = f"Verification failed with the following output:\n\n{error_context}{files_section}"
    return prompt, read_paths


def _collect_relevant_files(
    project_root: Path, error_context: str
) -> tuple[dict[str, str], list[str]]:
    """Extract file paths from traceback frames and error tokens, read up to MAX_RELEVANT_FILES.

    Implementation (non-test) files take all slots first; test files are
    demoted and only fill the remaining budget so the real fix target is
    never crowded out by pytest collection noise.
    """
    root = project_root.resolve()
    files: dict[str, str] = {}
    read_paths: list[str] = []

    # Collect candidate paths: traceback frames first, then token slices
    candidates: list[Path] = []
    for match in _TRACEBACK_FRAME_RE.finditer(error_context):
        p = Path(match.group(1))
        candidates.append(p if p.is_absolute() else project_root / str(p))
    for token in error_context.split():
        token = token.strip("'\":(),")
        if token and not token.startswith("http"):
            p = Path(token)
            candidates.append(p if p.is_absolute() else project_root / token)

    # Split into implementation and test candidates so implementation files
    # take all slots first; test files fill the remaining budget.
    impl_candidates: list[tuple[str, Path]] = []
    test_candidates: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        if not resolved.is_file() or resolved.stat().st_size > MAX_FILE_SIZE:
            continue
        try:
            rel = str(resolved.relative_to(root))
        except ValueError:
            continue
        if rel in seen:
            continue
        seen.add(rel)
        if _is_test_path(rel):
            test_candidates.append((rel, resolved))
        else:
            impl_candidates.append((rel, resolved))

    for rel, resolved in impl_candidates + test_candidates:
        try:
            files[rel] = resolved.read_text(encoding="utf-8")
            read_paths.append(rel)
        except OSError:
            continue
        if len(files) >= MAX_RELEVANT_FILES:
            break

    return files, read_paths


def _extract_patch(response: str) -> str | None:
    for prefix, offset in [("```diff", 7), ("```patch", 8)]:
        idx = response.find(prefix)
        if idx != -1:
            content = response[idx + offset :]
            end = content.find("```")
            if end != -1:
                content = content[:end]
            lines = content.splitlines()
            while lines and not lines[0].strip():
                lines.pop(0)
            patch = "\n".join(lines)
            return (patch + "\n") if patch else None

    plain_start = response.find("--- ")
    if plain_start != -1:
        patch = response[plain_start:].rstrip()
        return (patch + "\n") if patch else None
    return None


def _is_test_path(path: str) -> bool:
    """Return True if ``path`` looks like a test file.

    Classifies a path as a test file when any directory component is ``tests``
    or ``test``, or when the filename matches ``test_*.py``, ``*_test.py``,
    or is exactly ``test.py``.
    """
    from pathlib import PurePosixPath

    p = PurePosixPath(path)
    name = p.name
    if any(part in ("tests", "test") for part in p.parts[:-1]):
        return True
    if name == "test.py":
        return True
    if name.startswith("test_") and name.endswith(".py"):
        return True
    return bool(name.endswith("_test.py"))


# Delegate to the shared path-safety guard from ``utils.patch`` so that
# ``fix`` and ``verify --fix`` use identical traversal / test-file checks.
_patch_paths_are_safe = patch_paths_are_safe


def _apply_patch(
    project_root: Path, patch: str, i18n: I18n, tools: ToolsConfig | None = None
) -> bool:
    if not _patch_paths_are_safe(project_root, patch):
        typer.secho(
            i18n._("fix.patch_unsafe_paths"), fg=typer.colors.YELLOW, err=True
        )
        return False

    verifier = ToolVerifier(tools) if tools else ToolVerifier()
    applied, reason = apply_patch(project_root, patch, verifier)
    if applied:
        typer.echo(i18n._("fix.patch_applied"))
        return True
    if reason:
        typer.secho(i18n._("fix.patch_failed", reason=reason), fg=typer.colors.YELLOW, err=True)
    else:
        typer.secho(i18n._("fix.patch_no_tool"), fg=typer.colors.YELLOW, err=True)
    return False
