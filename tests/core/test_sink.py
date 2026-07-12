"""Tests for the self-hosted sink server."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx

from autoship.core.sink import SinkServer


def _start_server(
    tmp_path: Path, *, token: str | None = None, bind: str = "127.0.0.1"
) -> tuple[SinkServer, int, threading.Thread]:
    """Start a SinkServer on an ephemeral port; return (server, port, thread)."""
    server = SinkServer(tmp_path / "store", bind=bind, port=0, token=token, retention_days=7)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, thread


def _stop(server: SinkServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _url(port: int, path: str) -> str:
    return f"http://127.0.0.1:{port}{path}"


# ---------------------------------------------------------------------------
# SinkServer core
# ---------------------------------------------------------------------------


def test_store_dir_created_with_restrictive_permissions(tmp_path: Path) -> None:
    store = tmp_path / "store"
    SinkServer(store, port=0, retention_days=7)
    assert store.exists()
    import stat

    mode = stat.S_IMODE(store.stat().st_mode)
    assert mode == 0o700


def test_status_returns_zero_counts_initially(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path)
    try:
        resp = httpx.get(_url(port, "/status"), timeout=3.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["audit_records"] == 0
        assert data["telemetry_records"] == 0
        assert data["latest_ts"] is None
        assert data["retention_days"] == 7
    finally:
        _stop(server, thread)


def test_healthz_returns_ok(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path)
    try:
        resp = httpx.get(_url(port, "/healthz"), timeout=3.0)
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
    finally:
        _stop(server, thread)


def test_root_returns_service_info(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path)
    try:
        resp = httpx.get(_url(port, "/"), timeout=3.0)
        assert resp.status_code == 200
        info = resp.json()
        assert info["service"] == "autoship-sink"
        assert "/audit" in info["endpoints"]
        assert "/telemetry" in info["endpoints"]
    finally:
        _stop(server, thread)


# ---------------------------------------------------------------------------
# POST /audit
# ---------------------------------------------------------------------------


def test_post_audit_single_object_persists_record(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path)
    try:
        record = {
            "ts": "2026-07-04T00:00:00Z",
            "event": "cli.invoked",
            "payload": {"command": "clean"},
        }
        resp = httpx.post(_url(port, "/audit"), json=record, timeout=3.0)
        assert resp.status_code == 200
        assert resp.json() == {"accepted": 1}
        # The store file should contain exactly one JSONL line.
        files = list((tmp_path / "store").glob("sink-audit.*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        persisted = json.loads(lines[0])
        assert persisted["event"] == "cli.invoked"
    finally:
        _stop(server, thread)


def test_post_audit_array_persists_all(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path)
    try:
        records = [
            {"ts": "2026-07-04T00:00:01Z", "event": "a"},
            {"ts": "2026-07-04T00:00:02Z", "event": "b"},
            {"ts": "2026-07-04T00:00:03Z", "event": "c"},
        ]
        resp = httpx.post(_url(port, "/audit"), json=records, timeout=3.0)
        assert resp.status_code == 200
        assert resp.json() == {"accepted": 3}
        # /status reflects the count
        status = httpx.get(_url(port, "/status"), timeout=3.0).json()
        assert status["audit_records"] == 3
        assert status["latest_ts"] == "2026-07-04T00:00:03Z"
    finally:
        _stop(server, thread)


def test_post_telemetry_persists_to_separate_file(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path)
    try:
        resp = httpx.post(
            _url(port, "/telemetry"), json={"command": "clean", "duration_ms": 12.3}, timeout=3.0
        )
        assert resp.status_code == 200
        assert resp.json() == {"accepted": 1}
        audit_files = list((tmp_path / "store").glob("sink-audit.*.jsonl"))
        telemetry_files = list((tmp_path / "store").glob("sink-telemetry.*.jsonl"))
        assert audit_files == []
        assert len(telemetry_files) == 1
        status = httpx.get(_url(port, "/status"), timeout=3.0).json()
        assert status["audit_records"] == 0
        assert status["telemetry_records"] == 1
    finally:
        _stop(server, thread)


def test_post_unknown_path_returns_404(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path)
    try:
        resp = httpx.post(_url(port, "/unknown"), json={}, timeout=3.0)
        assert resp.status_code == 404
    finally:
        _stop(server, thread)


def test_post_invalid_json_returns_400(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path)
    try:
        resp = httpx.post(
            _url(port, "/audit"),
            content=b"{not valid json",
            headers={"Content-Type": "application/json"},
            timeout=3.0,
        )
        assert resp.status_code == 400
    finally:
        _stop(server, thread)


def test_post_non_object_json_returns_400(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path)
    try:
        resp = httpx.post(
            _url(port, "/audit"),
            content=b"42",
            headers={"Content-Type": "application/json"},
            timeout=3.0,
        )
        assert resp.status_code == 400
    finally:
        _stop(server, thread)


# ---------------------------------------------------------------------------
# Token auth
# ---------------------------------------------------------------------------


def test_token_required_when_set_unauthorized_without(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path, token="s3cret")
    try:
        resp = httpx.post(_url(port, "/audit"), json={"event": "x"}, timeout=3.0)
        assert resp.status_code == 401
    finally:
        _stop(server, thread)


def test_token_required_when_set_accepted_with(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path, token="s3cret")
    try:
        resp = httpx.post(
            _url(port, "/audit"),
            json={"event": "x"},
            headers={"Authorization": "Bearer s3cret"},
            timeout=3.0,
        )
        assert resp.status_code == 200
        assert resp.json() == {"accepted": 1}
    finally:
        _stop(server, thread)


def test_no_token_accepts_all_when_loopback(tmp_path: Path) -> None:
    """Without a token configured, all requests are accepted (loopback default)."""
    server, port, thread = _start_server(tmp_path, token=None)
    try:
        resp = httpx.post(_url(port, "/audit"), json={"event": "x"}, timeout=3.0)
        assert resp.status_code == 200
    finally:
        _stop(server, thread)


# ---------------------------------------------------------------------------
# Redaction (defense in depth)
# ---------------------------------------------------------------------------


def test_incoming_secret_payload_is_redacted(tmp_path: Path) -> None:
    server, port, thread = _start_server(tmp_path)
    try:
        record = {
            "ts": "2026-07-04T00:00:00Z",
            "event": "cli.invoked",
            "payload": {"api_key": "ghp_aBcDeF1234567890z"},
        }
        resp = httpx.post(_url(port, "/audit"), json=record, timeout=3.0)
        assert resp.status_code == 200
        files = list((tmp_path / "store").glob("sink-audit.*.jsonl"))
        persisted = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
        # The api_key value should be masked, not stored verbatim.
        assert persisted["payload"]["api_key"] != "ghp_aBcDeF1234567890z"
        assert "***" in str(persisted["payload"]["api_key"])
    finally:
        _stop(server, thread)


# ---------------------------------------------------------------------------
# Retention cleanup
# ---------------------------------------------------------------------------


def test_retention_deletes_old_files(tmp_path: Path) -> None:
    """Files older than retention_days are removed on startup."""
    store = tmp_path / "store"
    store.mkdir()
    # Plant an old audit file dated 100 days ago via its filename + mtime.
    import datetime as dt
    import os

    old_file = store / "sink-audit.2020-01-01.jsonl"
    old_file.write_text('{"event":"old"}\n', encoding="utf-8")
    old_ts = time.time() - 100 * 86400
    os.utime(old_file, (old_ts, old_ts))

    fresh_file = store / (
        "sink-audit." + dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d") + ".jsonl"
    )
    fresh_file.write_text('{"event":"fresh"}\n', encoding="utf-8")

    SinkServer(store, port=0, retention_days=7)
    assert not old_file.exists()
    assert fresh_file.exists()


# ---------------------------------------------------------------------------
# Content-Length validation (H2)
# ---------------------------------------------------------------------------


def _post_with_raw_content_length(
    port: int, path: str, content_length: str, body: bytes = b""
) -> int:
    """Send a POST with an explicit (possibly bogus) Content-Length header.

    Uses :mod:`http.client` so we can set a Content-Length that does not match
    the body — the server validates Content-Length *before* reading the body,
    so no hang occurs.
    """
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3.0)
    conn.putrequest("POST", path)
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", content_length)
    conn.endheaders()
    if body:
        conn.send(body)
    resp = conn.getresponse()
    status = resp.status
    resp.read()
    conn.close()
    return status


def test_post_oversized_body_returns_413(tmp_path: Path) -> None:
    """A Content-Length exceeding MAX_BODY_BYTES yields 413."""
    from autoship.core.sink import SinkHandler

    server, port, thread = _start_server(tmp_path)
    try:
        oversized = SinkHandler.MAX_BODY_BYTES + 1
        status = _post_with_raw_content_length(port, "/audit", str(oversized))
        assert status == 413
    finally:
        _stop(server, thread)


def test_post_negative_content_length_returns_400(tmp_path: Path) -> None:
    """A negative Content-Length yields 400."""
    server, port, thread = _start_server(tmp_path)
    try:
        status = _post_with_raw_content_length(port, "/audit", "-1")
        assert status == 400
    finally:
        _stop(server, thread)


def test_post_non_numeric_content_length_returns_400(tmp_path: Path) -> None:
    """A non-numeric Content-Length yields 400."""
    server, port, thread = _start_server(tmp_path)
    try:
        status = _post_with_raw_content_length(port, "/audit", "not-a-number")
        assert status == 400
    finally:
        _stop(server, thread)


# ---------------------------------------------------------------------------
# check_token edge cases (H3)
# ---------------------------------------------------------------------------


def test_check_token_with_non_ascii_authorization_does_not_raise(tmp_path: Path) -> None:
    """A non-ASCII Authorization header must not raise TypeError from compare_digest."""
    server = SinkServer(tmp_path / "store", port=0, token="s3cret", retention_days=7)
    # The non-ASCII 'é' in s3crét would crash compare_digest if it received a
    # raw str; the implementation encodes to bytes first.
    result = server.check_token("Bearer s3crét")
    assert result is False


def test_check_token_constant_time_with_empty_token(tmp_path: Path) -> None:
    """check_token covers the constant-time compare_digest path and short-circuits."""
    server = SinkServer(tmp_path / "store", port=0, token="s3cret", retention_days=7)
    # No authorization header -> False (short-circuit before compare_digest).
    assert server.check_token(None) is False
    # Empty authorization -> False.
    assert server.check_token("") is False
    # Wrong token -> False via constant-time compare_digest.
    assert server.check_token("Bearer wrong") is False
    # Correct token -> True.
    assert server.check_token("Bearer s3cret") is True


# ---------------------------------------------------------------------------
# Concurrency (L1)
# ---------------------------------------------------------------------------


def test_concurrent_append_never_loses_records(tmp_path: Path) -> None:
    """N threads POSTing /audit concurrently must all be counted."""
    import concurrent.futures

    server, port, thread = _start_server(tmp_path)
    n = 20
    try:
        record = {"event": "concurrent", "ts": "2026-07-04T00:00:00Z"}

        def _post(_i: int) -> int:
            # Retry on transient connection resets (common when many threads
            # hit the ThreadingHTTPServer simultaneously).
            for _attempt in range(3):
                try:
                    resp = httpx.post(_url(port, "/audit"), json=record, timeout=5.0)
                    if resp.status_code == 200:
                        return 1
                except httpx.TransportError:
                    continue
            return 0

        accepted = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            for result in pool.map(_post, range(n)):
                accepted += result

        assert accepted == n
        status = httpx.get(_url(port, "/status"), timeout=3.0).json()
        assert status["audit_records"] == n
    finally:
        _stop(server, thread)
