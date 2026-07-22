"""Self-hosted, localhost-only model relay for MyAgent-CLI.

Exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint and forwards
requests to a local model backend (Ollama by default). Binds to ``127.0.0.1``
exclusively and rejects any upstream host not on :data:`ALLOWED_HOSTS` with
HTTP 403, so prompts and code never leave the machine. See README.md for the
full security model. Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger("myagent.model-relay")

# Hosts the relay is willing to proxy to. Any other upstream host is rejected
# with HTTP 403 to enforce the local-first promise.
ALLOWED_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1"})

# Headers that must never appear in audit or error output. They are still
# forwarded to the upstream when present (the upstream may need them).
SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"}
)

# RFC 7230 hop-by-hop headers that must not be forwarded verbatim.
_HOP_BY_HOP: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

_DEFAULT_PORT = 8787
_DEFAULT_UPSTREAM = "http://localhost:11434"
_DEFAULT_TIMEOUT = 60.0


def redact_headers(headers: Any) -> dict[str, str]:
    """Return ``headers`` with sensitive values replaced by ``"***REDACTED***"``.

    Accepts a mapping or a sequence of ``(key, value)`` pairs. Use before
    writing any header value to a log so bearer tokens never leak.
    """
    pairs = headers.items() if hasattr(headers, "items") else headers
    return {
        key: ("***REDACTED***" if key.lower() in SENSITIVE_HEADERS else value)
        for key, value in pairs
    }


def build_upstream_url(upstream: str, path: str) -> str:
    """Compose ``upstream + path``, rejecting non-allowlisted hosts.

    Strips any ``userinfo`` from ``upstream`` so URL-embedded credentials are
    never forwarded.

    Raises:
        PermissionError: If the upstream host is not in :data:`ALLOWED_HOSTS`.
    """
    parsed = urlparse(upstream)
    host = parsed.hostname or ""
    if host not in ALLOWED_HOSTS:
        raise PermissionError(
            f"upstream host {host!r} is not in the allowlist {sorted(ALLOWED_HOSTS)}"
        )
    netloc = host if not parsed.port else f"{host}:{parsed.port}"
    scheme = parsed.scheme or "http"
    base = urlunparse((scheme, netloc, parsed.path.rstrip("/"), "", "", ""))
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _scrub_exception(exc: BaseException) -> str:
    """Stringify ``exc`` while hiding any embedded URL credentials."""
    text = repr(exc)
    if "://" in text:
        head, tail = text.split("://", 1)
        text = head + "://***" + tail
    return text


class RelayHandler(BaseHTTPRequestHandler):
    """HTTP handler that proxies chat completions to a local upstream."""

    server_version = "MyAgentModelRelay/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """Route access logs through ``logging`` instead of writing stderr directly."""
        logger.debug("access: " + format, *args)

    def do_GET(self) -> None:
        """Serve ``/healthz`` and a minimal root index."""
        path = self.path.rstrip("/")
        if path in ("", "/"):
            self._send_json(
                200,
                {
                    "service": "myagent-model-relay",
                    "endpoints": ["/v1/chat/completions", "/healthz"],
                },
            )
            return
        if path == "/healthz":
            self._send_json(200, {"status": "ok", "upstream": self.server.upstream})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        """Forward ``/v1/chat/completions`` to the local upstream."""
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        try:
            target = build_upstream_url(self.server.upstream, "/v1/chat/completions")
        except PermissionError as exc:
            logger.error("rejected upstream: %s", exc)
            self._send_json(403, {"error": "upstream host not allowed"})
            return
        self._proxy(target, body)

    def _proxy(self, target: str, body: bytes) -> None:
        """Forward ``body`` to ``target`` and stream the response back."""
        forwarded = self._collect_forwarded_headers()
        logger.debug("forwarding to %s headers=%s", target, redact_headers(forwarded))
        req = urllib.request.Request(target, data=body or None, method="POST")
        for key, value in forwarded.items():
            req.add_header(key, value)

        try:
            with urllib.request.urlopen(req, timeout=self.server.timeout) as resp:
                resp_body = resp.read()
                self._send_raw(resp.status, resp.getheaders(), resp_body)
        except urllib.error.HTTPError as exc:
            resp_body = exc.read()
            self._send_raw(exc.code, exc.headers.items(), resp_body)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.error(
                "upstream unreachable (host=%s): %s",
                self._upstream_host(),
                _scrub_exception(exc),
            )
            self._send_json(502, {"error": "upstream unreachable"})

    def _collect_forwarded_headers(self) -> dict[str, str]:
        """Build the header set to forward, dropping hop-by-hop entries."""
        forwarded: dict[str, str] = {}
        for key, value in self.headers.items():
            if key.lower() in _HOP_BY_HOP:
                continue
            forwarded[key] = value
        if not any(k.lower() == "content-type" for k in forwarded):
            forwarded["Content-Type"] = "application/json"
        return forwarded

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_raw(
            status,
            [("Content-Type", "application/json"), ("Content-Length", str(len(body)))],
            body,
        )

    def _send_raw(self, status: int, headers: Any, body: bytes) -> None:
        self.send_response(status)
        for key, value in headers:
            if key.lower() in _HOP_BY_HOP:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _upstream_host(self) -> str:
        return urlparse(self.server.upstream).hostname or ""


class RelayServer(ThreadingHTTPServer):
    """Threading HTTP server bound to ``127.0.0.1`` only."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int, upstream: str, timeout: float = _DEFAULT_TIMEOUT) -> None:
        super().__init__(("127.0.0.1", port), RelayHandler)
        self.upstream = upstream
        self.timeout = timeout


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python relay.py``. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Local-first model relay for MyAgent-CLI (OpenAI-compatible).",
    )
    parser.add_argument(
        "--port", type=int, default=_DEFAULT_PORT, help="Port to listen on (127.0.0.1 only)."
    )
    parser.add_argument(
        "--upstream",
        default=_DEFAULT_UPSTREAM,
        help="Local upstream base URL (Ollama default: http://localhost:11434).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT,
        help="Upstream request timeout in seconds.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    host = urlparse(args.upstream).hostname or ""
    if host not in ALLOWED_HOSTS:
        logger.warning(
            "upstream %r not in allowlist %s; /v1/chat/completions will return 403",
            host,
            sorted(ALLOWED_HOSTS),
        )

    server = RelayServer(args.port, args.upstream, timeout=args.timeout)
    logger.info("listening on http://127.0.0.1:%s (upstream=%s)", args.port, args.upstream)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
