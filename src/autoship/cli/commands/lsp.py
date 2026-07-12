"""The ``autoship lsp`` command: LSP server over stdio, powered by pygls."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import structlog
import typer
from lsprotocol.types import (
    TEXT_DOCUMENT_DIAGNOSTIC,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    Diagnostic,
    DiagnosticOptions,
    DiagnosticSeverity,
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    Position,
    Range,
)
from pygls.server import LanguageServer

from autoship import __version__

app = typer.Typer()
logger = structlog.get_logger("autoship")


def register(parent: typer.Typer) -> None:
    parent.command(name="lsp")(lsp)


# ---------------------------------------------------------------------------
# Diagnostic computation
# ---------------------------------------------------------------------------


def _uri_to_path(uri: str) -> Path | None:
    """Convert a ``file://`` URI to a local :class:`Path`, or None."""
    if not uri.startswith("file:"):
        return None
    rest = uri[5:]
    if rest.startswith("//"):
        after_authority = rest[2:]
        slash = after_authority.find("/")
        if slash == -1:
            return None
        authority = after_authority[:slash]
        path = after_authority[slash:]
        if authority not in ("", "localhost"):
            return None
        stripped = path
    elif rest.startswith("/"):
        stripped = rest
    else:
        stripped = "/" + rest
    return Path(stripped)


def _resolve_verify_command(config: Any) -> str:
    for hook in config.hooks.on_save:
        if hook.command == "verify" and hook.verify_command:
            return hook.verify_command
    return "pytest"


def _run_verify(project_root: Path, file_uri: str, config: Any | None = None) -> list[Diagnostic]:
    """Run the configured verify command and return LSP diagnostics."""
    file_path = _uri_to_path(file_uri)
    if file_path is None:
        return []

    from autoship.cli.commands.verify import validate_verify_command
    from autoship.core.config_center import load_config
    from autoship.exceptions import VerifyError

    if config is None:
        try:
            config = load_config(config_path=None, project_root=project_root)
        except Exception as exc:
            logger.warning("LSP config load failed: %s", exc)
            return [
                Diagnostic(
                    range=Range(start=Position(0, 0), end=Position(0, 0)),
                    severity=DiagnosticSeverity.Warning,
                    source="autoship",
                    code="config-load-failed",
                    message=f"autoship config load failed: {exc}",
                )
            ]

    verify_command = _resolve_verify_command(config)
    try:
        cmd_parts = validate_verify_command(verify_command, config.verify, None)
    except VerifyError:
        logger.warning(
            "LSP verify command %r rejected by allowlist; skipping diagnostics", verify_command
        )
        return [
            Diagnostic(
                range=Range(start=Position(0, 0), end=Position(0, 0)),
                severity=DiagnosticSeverity.Warning,
                source="autoship",
                code="allowlist-rejected",
                message=(
                    f"verify command {verify_command!r} rejected by allowlist; "
                    "update [verify] allowed_commands in .autoship.toml"
                ),
            )
        ]

    try:
        result = subprocess.run(
            cmd_parts,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    if result.returncode == 0:
        return []

    first_line = (result.stderr or result.stdout or "verify reported issues").splitlines()[0]
    return [
        Diagnostic(
            range=Range(start=Position(0, 0), end=Position(0, 0)),
            severity=DiagnosticSeverity.Error,
            source="autoship",
            code="verify-failed",
            message=first_line[:200],
        )
    ]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class _AutoShipServer:
    """LSP server wrapper holding mutable state alongside the pygls server."""

    def __init__(self, project_root: Path, ls: LanguageServer):
        self.project_root = project_root
        self.ls = ls
        self.documents: dict[str, str] = {}
        self._config: Any = None
        self._runner: Any = None

    def _get_config(self) -> Any:
        if self._config is not None:
            return self._config
        from autoship.core.config_center import load_config

        self._config = load_config(config_path=None, project_root=self.project_root)
        return self._config

    def _invalidate_config(self) -> None:
        self._config = None
        self._runner = None

    def run_on_save_hooks(self, file_uri: str) -> None:
        file_path = _uri_to_path(file_uri)
        if file_path is None:
            return
        try:
            if self._runner is None:
                from autoship.core.hooks import OnSaveHookRunner

                self._runner = OnSaveHookRunner(self._get_config(), audit_logger=None)
            self._runner.run_for_path(file_path)
        except Exception:
            return

    def compute_diagnostics(self, uri: str) -> list[Diagnostic]:
        return _run_verify(self.project_root, uri, self._get_config())


def lsp(
    ctx: typer.Context,
    project_root: Path | None = typer.Option(
        None, "--project-root", help="Project root to run verify against (defaults to cwd)."
    ),
) -> None:
    """Run the AutoShip LSP server over stdio."""
    root = project_root or Path.cwd()
    ls = LanguageServer("autoship-lsp", __version__)
    state = _AutoShipServer(root, ls)

    @ls.feature(TEXT_DOCUMENT_DID_OPEN)
    def did_open(ls: LanguageServer, params: DidOpenTextDocumentParams):
        state.documents[params.text_document.uri] = params.text_document.text
        _publish_diagnostics(ls, state, params.text_document.uri)

    @ls.feature(TEXT_DOCUMENT_DID_CHANGE)
    def did_change(ls: LanguageServer, params: DidChangeTextDocumentParams):
        if params.content_changes:
            state.documents[params.text_document.uri] = params.content_changes[-1].text
        _publish_diagnostics(ls, state, params.text_document.uri)

    @ls.feature(TEXT_DOCUMENT_DID_SAVE)
    def did_save(ls: LanguageServer, params: DidSaveTextDocumentParams):
        state._invalidate_config()
        state.run_on_save_hooks(params.text_document.uri)
        _publish_diagnostics(ls, state, params.text_document.uri)

    @ls.feature(
        TEXT_DOCUMENT_DIAGNOSTIC,
        DiagnosticOptions(inter_file_dependencies=False, workspace_diagnostics=False),
    )
    def diagnostic(ls: LanguageServer, params: Any):
        uri = params.text_document.uri if hasattr(params, "text_document") else ""
        return state.compute_diagnostics(uri)

    ls.start_io()


def _publish_diagnostics(ls: LanguageServer, state: _AutoShipServer, uri: str) -> None:
    """Proactively publish diagnostics for a document."""
    diags = state.compute_diagnostics(uri)
    ls.publish_diagnostics(uri, diags)
