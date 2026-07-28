"""Tests for the LSP shim.

The LSP server speaks JSON-RPC 2.0 over stdio with Content-Length framing.
These tests drive the protocol end-to-end against an in-memory stdin/stdout
pair, asserting that initialize, didOpen, and diagnostic messages conform
to the LSP 3.17 spec.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from justagent.cli.commands.lsp import _LSPServer, _read_message, _uri_to_path, _write_message


class _BytesStdin:
    """Minimal stdin stand-in exposing a ``.buffer`` over a bytes payload.

    The real ``sys.stdin`` is a text stream wrapping a binary buffer, and the
    LSP base protocol is fundamentally binary (a ``Content-Length`` byte
    count followed by that many raw bytes — UTF-8 inside, but the framing
    counts bytes, not characters). ``_read_message`` therefore reads from
    ``stdin.buffer``. This stand-in exposes the same shape: a ``.buffer``
    attribute whose ``readline`` / ``read`` return ``bytes``.
    """

    def __init__(self, data: bytes) -> None:
        self.buffer = io.BytesIO(data)

    def readline(self) -> str:
        # Text-mode view kept for parity with sys.stdin; not used by the
        # binary-aware reader but harmless to expose.
        line = self.buffer.readline()
        return line.decode("utf-8") if line else ""

    def read(self, n: int) -> bytes:
        return self.buffer.read(n)


class _BytesStdout:
    """Minimal stdout stand-in that captures everything written to .buffer."""

    def __init__(self) -> None:
        self.buffer = io.BytesIO()


def _frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def test_read_message_parses_framed_payload() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    stdin = _BytesStdin(_frame(payload))
    body = _read_message(stdin)
    assert body is not None
    assert json.loads(body.decode("utf-8")) == payload


def test_read_message_returns_none_on_eof() -> None:
    stdin = _BytesStdin(b"")
    assert _read_message(stdin) is None


def test_write_message_emits_content_length_header() -> None:
    stdout = _BytesStdout()
    _write_message(stdout, {"jsonrpc": "2.0", "id": 1, "result": {}})
    written = stdout.buffer.getvalue()
    assert written.startswith(b"Content-Length: ")
    assert b"\r\n\r\n" in written
    body_start = written.index(b"\r\n\r\n") + 4
    parsed = json.loads(written[body_start:].decode("utf-8"))
    assert parsed == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_initialize_returns_server_capabilities(tmp_path: Path) -> None:
    server = _LSPServer(tmp_path)
    request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    response = server.handle(request)
    assert response is not None
    assert response["id"] == 1
    capabilities = response["result"]["capabilities"]
    assert "textDocumentSync" in capabilities
    assert "diagnosticProvider" in capabilities
    assert response["result"]["serverInfo"]["name"] == "justagent-lsp"


def test_initialized_notification_returns_none(tmp_path: Path) -> None:
    server = _LSPServer(tmp_path)
    request = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
    response = server.handle(request)
    assert response is None


def test_shutdown_returns_null_result_and_sets_flag(tmp_path: Path) -> None:
    server = _LSPServer(tmp_path)
    request = {"jsonrpc": "2.0", "id": 7, "method": "shutdown"}
    response = server.handle(request)
    assert response is not None
    assert response["result"] is None
    assert server.shutdown_requested is True


def test_unknown_method_returns_method_not_found(tmp_path: Path) -> None:
    server = _LSPServer(tmp_path)
    request = {"jsonrpc": "2.0", "id": 2, "method": "textDocument/hover", "params": {}}
    response = server.handle(request)
    assert response is not None
    assert "error" in response
    assert response["error"]["code"] == -32601
    assert "method not found" in response["error"]["message"]


def test_unknown_notification_returns_none_no_error(tmp_path: Path) -> None:
    server = _LSPServer(tmp_path)
    request = {"jsonrpc": "2.0", "method": "workspace/didChangeConfiguration", "params": {}}
    response = server.handle(request)
    # Notifications (no id) with unknown method get no response.
    assert response is None


def test_did_open_stores_document(tmp_path: Path) -> None:
    server = _LSPServer(tmp_path)
    request = {
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {
            "textDocument": {
                "uri": "file:///tmp/example.py",
                "languageId": "python",
                "version": 1,
                "text": "import os\n",
            }
        },
    }
    response = server.handle(request)
    assert response is None  # notification
    assert "file:///tmp/example.py" in server.documents
    assert server.documents["file:///tmp/example.py"] == "import os\n"


def test_did_change_updates_document_full_sync(tmp_path: Path) -> None:
    server = _LSPServer(tmp_path)
    server.documents["file:///tmp/example.py"] = "old"
    request = {
        "jsonrpc": "2.0",
        "method": "textDocument/didChange",
        "params": {
            "textDocument": {"uri": "file:///tmp/example.py", "version": 2},
            "contentChanges": [{"text": "new content"}],
        },
    }
    response = server.handle(request)
    assert response is None
    assert server.documents["file:///tmp/example.py"] == "new content"


def test_uri_to_path_unix() -> None:
    path = _uri_to_path("file:///tmp/example.py")
    assert path is not None
    assert path.as_posix() == "/tmp/example.py"


def test_uri_to_path_localhost() -> None:
    path = _uri_to_path("file://localhost/tmp/example.py")
    assert path is not None
    assert path.as_posix() == "/tmp/example.py"


def test_uri_to_path_non_file_returns_none() -> None:
    assert _uri_to_path("untitled:Untitled-1") is None
    assert _uri_to_path("git://example.com/repo.git") is None


def test_diagnostics_for_open_docs_returns_per_doc_diagnostics(tmp_path: Path) -> None:
    """When no documents are open, the result list is empty."""
    server = _LSPServer(tmp_path)
    assert server.diagnostics_for_open_docs() == []


def test_diagnostics_for_open_docs_includes_open_uri(tmp_path: Path) -> None:
    """An open document triggers a verify subprocess call; the result is a (uri, list) pair."""
    server = _LSPServer(tmp_path)
    # Use a URI that points to a non-existent file so verify fails fast.
    server.documents["file:///nonexistent/example.py"] = "import os\n"
    results = server.diagnostics_for_open_docs()
    assert len(results) == 1
    uri, diagnostics = results[0]
    assert uri == "file:///nonexistent/example.py"
    # We don't assert on the diagnostic content — verify may produce
    # different output depending on the host — but the shape must be a list.
    assert isinstance(diagnostics, list)


def test_serve_returns_zero_on_eof(tmp_path: Path) -> None:
    """An empty stdin (immediate EOF) causes _serve to return 0."""
    from justagent.cli.commands.lsp import _serve

    stdin = _BytesStdin(b"")
    stdout = _BytesStdout()
    code = _serve(stdin, stdout, tmp_path)
    assert code == 0
    # Nothing should have been written.
    assert stdout.buffer.getvalue() == b""


def test_serve_handles_initialize_and_exit(tmp_path: Path) -> None:
    """A full initialize -> shutdown -> exit round-trip through _serve."""
    from justagent.cli.commands.lsp import _serve

    init_msg = _frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    shutdown_msg = _frame({"jsonrpc": "2.0", "id": 2, "method": "shutdown"})
    exit_msg = _frame({"jsonrpc": "2.0", "method": "exit"})
    stdin = _BytesStdin(init_msg + shutdown_msg + exit_msg)
    stdout = _BytesStdout()

    code = _serve(stdin, stdout, tmp_path)
    assert code == 0
    written = stdout.buffer.getvalue()
    # Two responses: initialize result + shutdown result.
    assert written.count(b"Content-Length:") == 2
    # The first response should contain capabilities.
    first_body_start = written.index(b"\r\n\r\n") + 4
    first_body = json.loads(
        written[first_body_start : written.index(b"Content-Length:", 4)].decode("utf-8")
    )
    assert "capabilities" in first_body["result"]


def test_serve_skips_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON bodies are silently skipped, not crashed."""
    from justagent.cli.commands.lsp import _serve

    bad_body = b"{not valid json"
    bad_msg = f"Content-Length: {len(bad_body)}\r\n\r\n".encode("ascii") + bad_body
    exit_msg = _frame({"jsonrpc": "2.0", "method": "exit"})
    stdin = _BytesStdin(bad_msg + exit_msg)
    stdout = _BytesStdout()

    code = _serve(stdin, stdout, tmp_path)
    assert code == 1  # exit without shutdown = 1


def test_uri_to_path_unc_returns_none() -> None:
    """UNC paths (file://server/share) are out of scope."""
    assert _uri_to_path("file://server/share/file.py") is None


def test_uri_to_path_no_authority() -> None:
    """file:/abs/path (no authority) should resolve."""
    path = _uri_to_path("file:/tmp/example.py")
    assert path is not None
    assert path.as_posix() == "/tmp/example.py"


# ---------------------------------------------------------------------------
# didSave + on-save hooks
# ---------------------------------------------------------------------------


def test_initialize_advertises_save_capability(tmp_path: Path) -> None:
    server = _LSPServer(tmp_path)
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert response is not None
    sync = response["result"]["capabilities"]["textDocumentSync"]
    assert "save" in sync
    assert sync["save"] == {"includeText": False}


def test_did_save_runs_on_save_hooks(tmp_path: Path) -> None:
    server = _LSPServer(tmp_path)
    called: list[str] = []
    server.run_on_save_hooks = lambda uri: called.append(uri)  # type: ignore[assignment]
    request = {
        "jsonrpc": "2.0",
        "method": "textDocument/didSave",
        "params": {"textDocument": {"uri": "file:///tmp/example.py"}},
    }
    response = server.handle(request)
    assert response is None  # notification
    assert called == ["file:///tmp/example.py"]


def test_did_save_without_uri_does_not_run_hooks(tmp_path: Path) -> None:
    server = _LSPServer(tmp_path)
    called: list[str] = []
    server.run_on_save_hooks = lambda uri: called.append(uri)  # type: ignore[assignment]
    request = {
        "jsonrpc": "2.0",
        "method": "textDocument/didSave",
        "params": {"textDocument": {}},
    }
    response = server.handle(request)
    assert response is None
    assert called == []


def test_run_on_save_hooks_swallows_errors(tmp_path: Path) -> None:
    """A hook failure must never crash the LSP server."""

    class _BoomRunner:
        def run_for_path(self, path):  # type: ignore[no-untyped-def]
            raise RuntimeError("hook exploded")

    server = _LSPServer(tmp_path)
    # Bypass the lazy builder with a runner that always raises.
    server._runner = _BoomRunner()  # type: ignore[assignment]
    # Should not raise.
    server.run_on_save_hooks("file:///tmp/example.py")


def test_serve_republishes_diagnostics_on_did_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """didSave triggers a fresh publishDiagnostics push."""
    from justagent.cli.commands.lsp import _serve

    # Stub run_on_save_hooks so no real subprocess is spawned.
    monkeypatch.setattr(_LSPServer, "run_on_save_hooks", lambda self, uri: None)
    # Stub _run_verify so diagnostics computation does not spawn a subprocess.
    import justagent.cli.commands.lsp as lsp_mod

    monkeypatch.setattr(lsp_mod, "_run_verify", lambda root, uri: [])

    open_msg = _frame(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {"textDocument": {"uri": "file:///tmp/example.py", "text": "x = 1\n"}},
        }
    )
    save_msg = _frame(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/didSave",
            "params": {"textDocument": {"uri": "file:///tmp/example.py"}},
        }
    )
    exit_msg = _frame({"jsonrpc": "2.0", "method": "exit"})
    stdin = _BytesStdin(open_msg + save_msg + exit_msg)
    stdout = _BytesStdout()
    _serve(stdin, stdout, tmp_path)
    out = stdout.buffer.getvalue()
    # Count publishDiagnostics notifications: one for didOpen, one for didSave.
    assert out.count(b'"method":"textDocument/publishDiagnostics"') >= 2


# ---------------------------------------------------------------------------
# _read_message edge cases (Content-Length framing)
# ---------------------------------------------------------------------------


def test_read_message_returns_none_when_content_length_missing() -> None:
    """Headers without a Content-Length header yield None."""
    stdin = _BytesStdin(b"Some-Header: value\r\n\r\n")
    assert _read_message(stdin) is None


def test_read_message_returns_none_for_non_integer_content_length() -> None:
    """A non-integer Content-Length yields None."""
    msg = b"Content-Length: not-a-number\r\n\r\n{}"
    stdin = _BytesStdin(msg)
    assert _read_message(stdin) is None


def test_read_message_returns_empty_bytes_for_zero_content_length() -> None:
    """Content-Length: 0 returns b'' (distinguished from None/EOF)."""
    stdin = _BytesStdin(b"Content-Length: 0\r\n\r\n")
    assert _read_message(stdin) == b""


def test_read_message_returns_none_for_truncated_body() -> None:
    """When fewer bytes than Content-Length are available, return None."""
    msg = b"Content-Length: 100\r\n\r\n{short}"
    stdin = _BytesStdin(msg)
    assert _read_message(stdin) is None


def test_read_message_skips_header_lines_without_colon() -> None:
    """Header lines without a colon are ignored; Content-Length still parsed."""
    body = b"{}"
    msg = b"NoColonHere\r\nContent-Length: 2\r\n\r\n" + body
    stdin = _BytesStdin(msg)
    assert _read_message(stdin) == body


# ---------------------------------------------------------------------------
# _run_verify: subprocess execution and exit-code -> severity mapping
# ---------------------------------------------------------------------------


def _fake_completed_process(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    return type(
        "CompletedProcess",
        (),
        {"returncode": returncode, "stdout": stdout, "stderr": stderr},
    )()


def test_run_verify_returns_empty_list_for_non_file_uri(tmp_path: Path) -> None:
    """A non-file:// URI means file_path is None — no diagnostics."""
    from justagent.cli.commands.lsp import _run_verify

    assert _run_verify(tmp_path, "untitled:Untitled-1") == []


def test_run_verify_returns_empty_on_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When verify exits 0, no diagnostics are produced."""
    from justagent.cli.commands.lsp import _run_verify

    file_path = tmp_path / "example.py"
    file_path.write_text("x = 1\n")

    def fake_run(*_args: object, **_kwargs: object) -> Any:
        return _fake_completed_process(0)

    monkeypatch.setattr("justagent.cli.commands.lsp.subprocess.run", fake_run)
    assert _run_verify(tmp_path, file_path.as_uri()) == []


def test_run_verify_maps_nonzero_exit_to_error_severity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero verify exit produces a single Error-severity diagnostic."""
    from justagent.cli.commands.lsp import _run_verify

    file_path = tmp_path / "example.py"
    file_path.write_text("x = 1\n")

    def fake_run(*_args: object, **_kwargs: object) -> Any:
        return _fake_completed_process(1, stderr="syntax error on line 1\ntraceback")

    monkeypatch.setattr("justagent.cli.commands.lsp.subprocess.run", fake_run)
    diagnostics = _run_verify(tmp_path, file_path.as_uri())
    assert len(diagnostics) == 1
    diag = diagnostics[0]
    assert diag["severity"] == 1  # Error
    assert diag["source"] == "justagent"
    assert diag["message"] == "syntax error on line 1"
    assert diag["range"]["start"] == {"line": 0, "character": 0}
    assert diag["range"]["end"] == {"line": 0, "character": 0}


def test_run_verify_falls_back_to_stdout_when_stderr_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When stderr is empty, the first line of stdout is the diagnostic message."""
    from justagent.cli.commands.lsp import _run_verify

    file_path = tmp_path / "example.py"
    file_path.write_text("x = 1\n")

    def fake_run(*_args: object, **_kwargs: object) -> Any:
        return _fake_completed_process(2, stdout="issue found\nmore")

    monkeypatch.setattr("justagent.cli.commands.lsp.subprocess.run", fake_run)
    diagnostics = _run_verify(tmp_path, file_path.as_uri())
    assert len(diagnostics) == 1
    assert diagnostics[0]["message"] == "issue found"


def test_run_verify_falls_back_to_default_message_when_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both stderr and stdout are empty, a default message is used."""
    from justagent.cli.commands.lsp import _run_verify

    file_path = tmp_path / "example.py"
    file_path.write_text("x = 1\n")

    def fake_run(*_args: object, **_kwargs: object) -> Any:
        return _fake_completed_process(1)

    monkeypatch.setattr("justagent.cli.commands.lsp.subprocess.run", fake_run)
    diagnostics = _run_verify(tmp_path, file_path.as_uri())
    assert len(diagnostics) == 1
    assert diagnostics[0]["message"] == "verify reported issues"


def test_run_verify_returns_empty_on_subprocess_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If subprocess.run raises SubprocessError/OSError, return an empty list."""
    from justagent.cli.commands.lsp import _run_verify

    file_path = tmp_path / "example.py"
    file_path.write_text("x = 1\n")

    def boom(*_args: object, **_kwargs: object) -> Any:
        raise OSError("interpreter gone")

    monkeypatch.setattr("justagent.cli.commands.lsp.subprocess.run", boom)
    assert _run_verify(tmp_path, file_path.as_uri()) == []


# ---------------------------------------------------------------------------
# textDocument/diagnostic (pull) and textDocument/publishDiagnostics
# ---------------------------------------------------------------------------


def test_text_document_diagnostic_returns_full_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """textDocument/diagnostic (pull mode) returns {kind: full, items: [...]}."""
    import justagent.cli.commands.lsp as lsp_mod

    monkeypatch.setattr(lsp_mod, "_run_verify", lambda root, uri: [])
    server = _LSPServer(tmp_path)
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "textDocument/diagnostic",
        "params": {"textDocument": {"uri": "file:///tmp/example.py"}},
    }
    response = server.handle(request)
    assert response is not None
    assert response["id"] == 5
    assert response["result"]["kind"] == "full"
    assert response["result"]["items"] == []


def test_text_document_publish_diagnostics_returns_none(tmp_path: Path) -> None:
    """publishDiagnostics is server-originated; handle() returns None."""
    server = _LSPServer(tmp_path)
    request = {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {"uri": "file:///tmp/example.py", "diagnostics": []},
    }
    assert server.handle(request) is None


# ---------------------------------------------------------------------------
# didChange and dispatch edge cases
# ---------------------------------------------------------------------------


def test_did_change_with_non_list_content_changes_is_noop(tmp_path: Path) -> None:
    """contentChanges that isn't a list is treated as no changes."""
    server = _LSPServer(tmp_path)
    server.documents["file:///tmp/example.py"] = "original"
    request = {
        "jsonrpc": "2.0",
        "method": "textDocument/didChange",
        "params": {
            "textDocument": {"uri": "file:///tmp/example.py", "version": 2},
            "contentChanges": "not a list",
        },
    }
    response = server.handle(request)
    assert response is None
    # Document content unchanged because no valid changes were applied.
    assert server.documents["file:///tmp/example.py"] == "original"


def test_uri_to_path_authority_without_path_returns_none() -> None:
    """file://localhost (authority but no path) is malformed."""
    assert _uri_to_path("file://localhost") is None


def test_uri_to_path_windows_drive_letter() -> None:
    """file:C:/path (no authority, no leading slash) maps to /C:/path."""
    path = _uri_to_path("file:C:/Users/test/example.py")
    assert path is not None
    assert path.as_posix() == "/C:/Users/test/example.py"


def test_run_on_save_hooks_ignores_non_file_uri(tmp_path: Path) -> None:
    """run_on_save_hooks is a no-op for non-file:// URIs (no runner built)."""
    server = _LSPServer(tmp_path)
    server.run_on_save_hooks("untitled:Untitled-1")
    assert server._runner is None


def test_handle_with_non_string_method_returns_method_not_found(tmp_path: Path) -> None:
    """A non-string method is coerced to '' and treated as unknown."""
    server = _LSPServer(tmp_path)
    request = {"jsonrpc": "2.0", "id": 1, "method": 123, "params": {}}
    response = server.handle(request)
    assert response is not None
    assert response["error"]["code"] == -32601


def test_serve_skips_non_dict_json_body(tmp_path: Path) -> None:
    """A JSON body that parses to a non-dict (e.g. a list) is skipped."""
    from justagent.cli.commands.lsp import _serve

    body = b"[1, 2, 3]"
    msg = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    exit_msg = _frame({"jsonrpc": "2.0", "method": "exit"})
    stdin = _BytesStdin(msg + exit_msg)
    stdout = _BytesStdout()
    code = _serve(stdin, stdout, tmp_path)
    assert code == 1  # exit without prior shutdown


# ---------------------------------------------------------------------------
# lsp typer command entry
# ---------------------------------------------------------------------------


def test_lsp_command_entry_invokes_serve_and_exits_with_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lsp typer command invokes _serve and exits with its return code."""
    from typer.testing import CliRunner

    from justagent.cli.main import app

    captured: dict[str, object] = {}

    def fake_serve(_stdin: object, _stdout: object, root: Path) -> int:
        captured["root"] = root
        return 0

    monkeypatch.setattr("justagent.cli.commands.lsp._serve", fake_serve)

    runner = CliRunner()
    result = runner.invoke(app, ["lsp", "--project-root", str(tmp_path)])
    assert result.exit_code == 0
    assert captured["root"] == tmp_path


def test_lsp_command_entry_propagates_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero _serve return code propagates as a non-zero CLI exit code."""
    from typer.testing import CliRunner

    from justagent.cli.main import app

    def fake_serve(_stdin: object, _stdout: object, _root: Path) -> int:
        return 1

    monkeypatch.setattr("justagent.cli.commands.lsp._serve", fake_serve)

    runner = CliRunner()
    result = runner.invoke(app, ["lsp"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# _resolve_verify_command (M5)
# ---------------------------------------------------------------------------


def test_resolve_verify_command_returns_pytest_when_no_verify_hook() -> None:
    """With no on-save hooks configured, the default ``pytest`` is returned."""
    from justagent.cli.commands.lsp import _resolve_verify_command
    from justagent.models.config import AppConfig

    config = AppConfig()
    assert _resolve_verify_command(config) == "pytest"


def test_resolve_verify_command_returns_hook_verify_command() -> None:
    """The first ``verify`` hook's ``verify_command`` is returned."""
    from justagent.cli.commands.lsp import _resolve_verify_command
    from justagent.models.config import AppConfig, HookConfig, HooksConfig

    config = AppConfig(
        hooks=HooksConfig(on_save=[HookConfig(command="verify", verify_command="ruff check .")])
    )
    assert _resolve_verify_command(config) == "ruff check ."


def test_resolve_verify_command_skips_clean_hooks() -> None:
    """``clean`` hooks are skipped; the first ``verify`` hook wins."""
    from justagent.cli.commands.lsp import _resolve_verify_command
    from justagent.models.config import AppConfig, HookConfig, HooksConfig

    config = AppConfig(
        hooks=HooksConfig(
            on_save=[
                HookConfig(command="clean"),
                HookConfig(command="verify", verify_command="mypy src"),
            ]
        )
    )
    assert _resolve_verify_command(config) == "mypy src"


# ---------------------------------------------------------------------------
# _run_verify message truncation and timeout (M6)
# ---------------------------------------------------------------------------


def test_run_verify_truncates_long_message_to_200_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The diagnostic message is capped at 200 characters."""
    from justagent.cli.commands.lsp import _run_verify

    file_path = tmp_path / "example.py"
    file_path.write_text("x = 1\n")

    long_message = "E" * 500
    monkeypatch.setattr(
        "justagent.cli.commands.lsp.subprocess.run",
        lambda *_a, **_k: _fake_completed_process(1, stderr=long_message),
    )
    diagnostics = _run_verify(tmp_path, file_path.as_uri())
    assert len(diagnostics) == 1
    assert len(diagnostics[0]["message"]) == 200
    assert diagnostics[0]["message"] == "E" * 200


def test_run_verify_returns_empty_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subprocess timeout yields an empty diagnostics list, not a crash."""
    from justagent.cli.commands.lsp import _run_verify

    file_path = tmp_path / "example.py"
    file_path.write_text("x = 1\n")

    def _timeout(*_args: object, **_kwargs: object) -> Any:
        raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=30)

    monkeypatch.setattr("justagent.cli.commands.lsp.subprocess.run", _timeout)
    assert _run_verify(tmp_path, file_path.as_uri()) == []
