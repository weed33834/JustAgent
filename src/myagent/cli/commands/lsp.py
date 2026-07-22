"""The ``myagent lsp`` command: a minimal LSP server over stdio.

Exposes MyAgent's ``verify`` diagnostics to any LSP-capable editor
(Neovim, VS Code, Emacs lsp-mode, Helix, …). The implementation is
intentionally minimal and uses only the Python standard library so the
LSP surface adds zero dependencies; ``pygls`` would be overkill for the
two methods we actually answer.

Wire format
-----------
JSON-RPC 2.0 over stdio with ``Content-Length`` framing, exactly as
specified by `base protocol`_.

.. _base protocol: https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#baseProtocol

The server handles the standard lifecycle (``initialize``, ``initialized``,
``shutdown``), document sync (``textDocument/didOpen``, ``didChange``,
``didSave``), and ``textDocument/diagnostic`` (pull model, LSP 3.17). On
``didSave`` it runs the configured ``[hooks] on_save`` entries for the saved
file before republishing diagnostics. Any other method returns
``method_not_found``.

Diagnostics source
------------------
The diagnostic payload comes from running ``myagent verify`` on the
file's project root. The command's exit status maps to LSP severity
(``Error`` on non-zero exit, ``Hint`` otherwise) and the first line of
stderr is surfaced as the diagnostic message. This is conservative: we
never invent diagnostics that MyAgent itself did not produce.

Run from an editor with ``myagent lsp`` as the server command. MyAgent
does not need to be on ``$PATH``; the command can be invoked as
``python -m myagent lsp`` from a checkout.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from myagent import __version__
from myagent.core.i18n import I18n, get_i18n_from_ctx

if TYPE_CHECKING:
    from myagent.models.config import AppConfig

app = typer.Typer()

logger = logging.getLogger("myagent")


def register(parent: typer.Typer) -> None:
    parent.command(name="lsp")(lsp)


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------


def _read_message(stdin: Any) -> bytes | None:
    """Read one LSP base-protocol message from ``stdin``.

    The LSP base protocol is binary: a header block terminated by an empty
    line, then a body of ``Content-Length`` bytes. Reading from ``stdin``
    in text mode would corrupt multibyte UTF-8 (a single ``read(length)``
    could split a surrogate pair), so we read the underlying binary buffer
    when available. ``stdin`` is ``Any`` so we fall back to direct reads
    for tests that pass a BytesIO-like stand-in without a ``.buffer``
    attribute.

    Returns ``None`` on EOF.
    """
    binary = getattr(stdin, "buffer", stdin)

    # ``readline`` on the binary buffer returns ``b""`` only at EOF.
    headers: dict[str, str] = {}
    while True:
        line = binary.readline()
        if not line:
            return None  # EOF
        stripped = line.rstrip(b"\r\n")
        if not stripped:
            break
        if b":" in stripped:
            key_b, _, value_b = stripped.partition(b":")
            headers[key_b.strip().lower().decode("ascii", "replace")] = value_b.strip().decode(
                "utf-8", "replace"
            )

    length_str = headers.get("content-length")
    if not length_str:
        return None
    try:
        length = int(length_str)
    except ValueError:
        return None
    if length <= 0:
        return b""

    body = binary.read(length)
    if len(body) < length:
        return None  # truncated
    return cast(bytes, body)


def _write_message(stdout: Any, payload: dict[str, Any]) -> None:
    """Write one LSP base-protocol message to ``stdout``."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    stdout.buffer.write(header)
    stdout.buffer.write(body)
    stdout.buffer.flush()


def _coerce_dict(value: object) -> dict[str, Any]:
    """Return ``value`` as a dict, or an empty dict if it isn't one."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _coerce_str(value: object) -> str:
    """Return ``value`` as a string, or empty string if it isn't one."""
    return value if isinstance(value, str) else ""


# ---------------------------------------------------------------------------
# Diagnostic computation
# ---------------------------------------------------------------------------


def _resolve_verify_command(config: AppConfig) -> str:
    """Return the verify command configured for ``config``.

    Looks up the first ``verify`` entry in ``[hooks] on_save`` and returns
    its ``verify_command``. Falls back to ``pytest`` when no verify hook is
    configured so the LSP diagnostic provider still has a command to run.
    """
    for hook in config.hooks.on_save:
        if hook.command == "verify" and hook.verify_command:
            return hook.verify_command
    return "pytest"


def _run_verify(
    project_root: Path,
    file_uri: str,
    config: Any | None = None,
) -> list[dict[str, Any]]:
    """Run the configured verify command against ``project_root`` and return LSP diagnostics.

    The verify command's per-file output is intentionally coarse (it returns
    a process exit code and a stdout/stderr blob, not structured diagnostics).
    The LSP layer therefore returns a single diagnostic whose message is the
    first line of stderr (or 'verify reported issues' if stderr is empty)
    and whose range covers the whole document. Future work can teach verify
    to emit JSONL diagnostics that this function would forward verbatim.

    The verify command is resolved from the project's ``[hooks] on_save``
    ``verify`` entries (defaulting to ``pytest``). Before executing it, the
    command is validated against the ``myagent verify`` subcommand's
    allowlist (``verify.allowed_commands``) so the LSP cannot be used to
    bypass the command-allowlist policy.

    A rejected command or a config-load failure does NOT return an empty
    diagnostics list — that would be a false negative, leaving the editor
    to show "no problems" while the LSP silently skipped verification.
    Instead a single Warning-severity diagnostic is returned whose message
    tells the user exactly what to fix. Only subprocess execution errors
    and timeouts still return ``[]`` (no actionable message to surface).

    When ``config`` is provided (e.g. from the :class:`_LSPServer` cache)
    it is used verbatim and ``load_config`` is not called. When ``config``
    is ``None`` the config is loaded from disk; a load failure yields the
    warning diagnostic described above.
    """
    file_path = _uri_to_path(file_uri)
    if file_path is None:
        return []

    # Local imports keep the LSP module importable without the full
    # config/audit stack at module load time.
    from myagent.cli.commands.verify import validate_verify_command
    from myagent.core.config_center import load_config
    from myagent.core.i18n import get_i18n
    from myagent.exceptions import VerifyError

    if config is None:
        try:
            config = load_config(config_path=None, project_root=project_root)
        except Exception as exc:  # noqa: BLE001 - LSP must not die on config errors
            logger.warning("LSP config load failed: %s", exc)
            return [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "severity": 2,  # Warning
                    "source": "myagent",
                    "code": "config-load-failed",
                    "message": f"myagent config load failed: {exc}",
                }
            ]

    verify_command = _resolve_verify_command(config)

    try:
        cmd_parts = validate_verify_command(verify_command, config.verify, get_i18n())
    except VerifyError:
        logger.warning(
            "LSP verify command %r rejected by allowlist; skipping diagnostics",
            verify_command,
        )
        return [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 0},
                },
                "severity": 2,  # Warning
                "source": "myagent",
                "code": "allowlist-rejected",
                "message": (
                    f"verify command {verify_command!r} rejected by allowlist; "
                    "update [verify] allowed_commands in .myagent.toml"
                ),
            }
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
        {
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 0},
            },
            "severity": 1,  # Error
            "source": "myagent",
            "code": "verify-failed",
            "message": first_line[:200],
        }
    ]


def _uri_to_path(uri: str) -> Path | None:
    """Convert a ``file://`` URI to a local :class:`Path`, or None.

    The file URI grammar is ``file:[//authority]/path``. We accept the three
    common forms:

    * ``file:/abs/path``: no authority (legal but rare).
    * ``file:///abs/path``: empty authority, the standard form for local
      files. The triple slash collapses to ``/abs/path``.
    * ``file://localhost/abs/path``: explicit ``localhost`` authority,
      stripped before path extraction.

    UNC paths (``file://server/share/...``) and any authority other than
    empty/localhost are out of scope; we return ``None`` so the caller
    skips diagnostics for them rather than guessing.
    """
    if not uri.startswith("file:"):
        return None
    rest = uri[5:]  # strip leading 'file:'
    if rest.startswith("//"):
        # Has an authority component. Strip leading '//' then look at the
        # host segment up to the next '/'.
        after_authority = rest[2:]
        slash = after_authority.find("/")
        if slash == -1:
            # No path component: malformed.
            return None
        authority = after_authority[:slash]
        path = after_authority[slash:]
        if authority not in ("", "localhost"):
            # UNC or remote authority: out of scope.
            return None
        stripped = path
    elif rest.startswith("/"):
        # No authority, absolute path: file:/abs/path.
        stripped = rest
    else:
        # Windows-style: file:C:/path or file:/C:/path with drive letter.
        stripped = "/" + rest
    return Path(stripped)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class _LSPServer:
    """Stateful LSP server: tracks open documents and dispatches requests."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.documents: dict[str, str] = {}
        self.shutdown_requested = False
        self._lock = threading.Lock()
        # Lazily-built on-save hook runner. Built on first didSave so
        # the LSP server stays zero-config until a save actually arrives.
        self._runner: Any = None
        # Cached AppConfig. Built lazily by ``_get_config`` and dropped by
        # ``_invalidate_config`` on every didSave so edits to
        # ``.myagent.toml`` are picked up without restarting the server.
        self._config: Any = None

    def _get_config(self) -> Any:
        """Lazily load and cache the project's :class:`AppConfig`.

        Returning the cached value avoids re-reading ``.myagent.toml`` on
        every diagnostic request. The cache is invalidated on didSave so
        on-save hooks and the next diagnostic pass observe the user's
        latest config edits.
        """
        if self._config is not None:
            return self._config
        # Local import keeps the LSP module importable without the full
        # config/audit stack at module load time.
        from myagent.core.config_center import load_config

        self._config = load_config(config_path=None, project_root=self.project_root)
        return self._config

    def _invalidate_config(self) -> None:
        """Drop cached config and runner so the next access re-reads disk."""
        self._config = None
        self._runner = None

    def _get_runner(self) -> Any:
        """Lazily build the :class:`OnSaveHookRunner` from project config."""
        if self._runner is not None:
            return self._runner
        # Local import keeps the LSP module importable without the full
        # config/audit stack at module load time.
        from myagent.core.hooks import OnSaveHookRunner

        config = self._get_config()
        self._runner = OnSaveHookRunner(config, audit_logger=None)
        return self._runner

    def run_on_save_hooks(self, file_uri: str) -> None:
        """Run matching on-save hooks for ``file_uri``.

        Failures are swallowed: a hook must never crash the LSP server. The
        runner's debounce protects against rapid save storms.
        """
        file_path = _uri_to_path(file_uri)
        if file_path is None:
            return
        try:
            runner = self._get_runner()
            runner.run_for_path(file_path)
        except Exception:  # noqa: BLE001 - LSP must not die on hook errors
            return

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one parsed JSON-RPC message. Returns the response or None."""
        method = request.get("method", "")
        if not isinstance(method, str):
            method = ""
        raw_params: object = request.get("params", {})
        params: dict[str, Any] = _coerce_dict(raw_params)
        request_id = request.get("id")

        if method == "initialize":
            return self._response(
                request_id,
                {
                    "capabilities": {
                        "textDocumentSync": {
                            "openClose": True,
                            "change": 1,  # full sync
                            "save": {"includeText": False},
                        },
                        "diagnosticProvider": {
                            "interFileDependencies": False,
                            "workspaceDiagnostics": False,
                        },
                        "positionEncoding": "utf-16",
                    },
                    "serverInfo": {"name": "myagent-lsp", "version": __version__},
                },
            )

        if method == "initialized":
            return None  # notification, no response

        if method == "shutdown":
            self.shutdown_requested = True
            return self._response(request_id, None)

        if method == "textDocument/didOpen":
            doc = _coerce_dict(params.get("textDocument"))
            uri = _coerce_str(doc.get("uri"))
            text = _coerce_str(doc.get("text"))
            with self._lock:
                self.documents[uri] = text
            return None  # notification

        if method == "textDocument/didChange":
            doc = _coerce_dict(params.get("textDocument"))
            uri = _coerce_str(doc.get("uri"))
            raw_changes = params.get("contentChanges")
            if isinstance(raw_changes, list):
                change_list = cast(list[object], raw_changes)
                changes: list[dict[str, Any]] = [
                    cast(dict[str, Any], c) for c in change_list if isinstance(c, dict)
                ]
            else:
                changes = []
            if changes:
                # Full sync (change=1): the last change carries the whole text.
                with self._lock:
                    self.documents[uri] = _coerce_str(changes[-1].get("text"))
            return None

        if method == "textDocument/didSave":
            doc = _coerce_dict(params.get("textDocument"))
            uri = _coerce_str(doc.get("uri"))
            # Drop cached config so on-save hooks (and the subsequent
            # diagnostics pass) pick up any ``.myagent.toml`` changes the
            # user just wrote. The save itself is the natural invalidation
            # point: editors send didSave after writing the file to disk.
            self._invalidate_config()
            # Run configured on-save hooks before diagnostics are republished
            # by the serve loop. Hooks are best-effort: a hook failure must
            # never take down the LSP server.
            if uri:
                self.run_on_save_hooks(uri)
            return None

        if method == "textDocument/diagnostic":
            doc = _coerce_dict(params.get("textDocument"))
            uri = _coerce_str(doc.get("uri"))
            diagnostics = _run_verify(self.project_root, uri)
            return self._response(
                request_id,
                {"kind": "full", "items": diagnostics},
            )

        if method == "textDocument/publishDiagnostics":
            # Server-originated notification, but we also proactively push
            # diagnostics on didOpen/didChange. The actual push happens in
            # run() after handle() returns.
            return None

        # Unknown method
        if request_id is not None:
            return self._error(request_id, -32601, f"method not found: {method}")
        return None

    def diagnostics_for_open_docs(self) -> list[tuple[str, list[dict[str, Any]]]]:
        """Compute diagnostics for every open document (for proactive push)."""
        results: list[tuple[str, list[dict[str, Any]]]] = []
        with self._lock:
            uris = list(self.documents.keys())
        for uri in uris:
            diagnostics = _run_verify(self.project_root, uri)
            results.append((uri, diagnostics))
        return results

    @staticmethod
    def _response(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }


def _serve(stdin: Any, stdout: Any, project_root: Path) -> int:
    """Run the LSP read/dispatch loop until ``exit`` or EOF."""
    server = _LSPServer(project_root)
    while True:
        body = _read_message(stdin)
        if body is None:
            return 0
        try:
            parsed: object = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        request: dict[str, Any] = cast(dict[str, Any], parsed)

        response = server.handle(request)
        if response is not None:
            _write_message(stdout, response)

        # Proactive diagnostics push on didOpen / didChange / didSave.
        # didSave runs on-save hooks inside handle() first, so by the time
        # we reach here clean hooks may have reformatted the file and the
        # fresh verify diagnostics reflect the post-save state.
        raw_method: object = request.get("method", "")
        method = raw_method if isinstance(raw_method, str) else ""
        if method in ("textDocument/didOpen", "textDocument/didChange", "textDocument/didSave"):
            for uri, diagnostics in server.diagnostics_for_open_docs():
                _write_message(
                    stdout,
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/publishDiagnostics",
                        "params": {"uri": uri, "diagnostics": diagnostics},
                    },
                )

        if method == "exit":
            return 0 if server.shutdown_requested else 1


def lsp(
    ctx: typer.Context,
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        help="Project root to run verify against (defaults to cwd).",
    ),
) -> None:
    """Run the MyAgent LSP server over stdio."""
    # Reference i18n so the command still binds the global context, even
    # though the LSP protocol speaks JSON-RPC, not human-readable strings.
    _ = get_i18n_from_ctx(ctx)
    _ = I18n  # keep the import live for type checkers

    root = project_root or Path.cwd()
    code = _serve(sys.stdin, sys.stdout, root)
    raise typer.Exit(code=code)
