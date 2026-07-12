"""Tests for the ``autoship sink`` command."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from typer.testing import CliRunner

from autoship.cli.main import app
from autoship.core.sink import SinkServer

runner = CliRunner()


def _write_config(
    tmp_path: Path, *, sink_url: str | None = None, server_token: str | None = None
) -> Path:
    config_path = tmp_path / ".autoship.toml"
    lines = [
        f'schema_version = 1\nproject_root = "{tmp_path}"\n',
        '\n[audit]\nlog_dir = "' + str(tmp_path / "logs") + '"\n',
    ]
    if sink_url or server_token:
        lines.append("\n[sink]\n")
        if sink_url:
            lines.append(f'url = "{sink_url}"\n')
        if server_token:
            lines.append(f'server_token = "{server_token}"\n')
    config_path.write_text("".join(lines), encoding="utf-8")
    return config_path


def _start_sink(
    tmp_path: Path, *, token: str | None = None
) -> tuple[SinkServer, int, threading.Thread]:
    server = SinkServer(tmp_path / "store", port=0, token=token, retention_days=7)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, thread


def _stop(server: SinkServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_sink_status_no_url_exits_two(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["--config", str(config_path), "sink", "status"])
    assert result.exit_code == 2
    assert "sink URL" in result.output.lower() or "no" in result.output.lower()


def test_sink_status_prints_counters(tmp_path: Path) -> None:
    server, port, thread = _start_sink(tmp_path)
    try:
        # Seed one audit record so the counter is non-zero.
        httpx.post(f"http://127.0.0.1:{port}/audit", json={"event": "x"}, timeout=3.0)
        config_path = _write_config(tmp_path, sink_url=f"http://127.0.0.1:{port}")
        result = runner.invoke(app, ["--config", str(config_path), "sink", "status"])
        assert result.exit_code == 0
        assert "audit_records" in result.output
        assert "1" in result.output
        assert "Sink status" in result.output
    finally:
        _stop(server, thread)


def test_sink_status_unreachable_exits_one(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, sink_url="http://127.0.0.1:1")
    result = runner.invoke(app, ["--config", str(config_path), "sink", "status"])
    assert result.exit_code == 1


def test_sink_serve_non_loopback_without_token_exits_two(tmp_path: Path) -> None:
    """Binding to a non-loopback address without a token must refuse to start."""
    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["--config", str(config_path), "sink", "serve", "--bind", "0.0.0.0", "--port", "0"],
    )
    assert result.exit_code == 2
    assert "token" in result.output.lower()


def test_sink_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["sink", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.output
    assert "status" in result.output


# ---------------------------------------------------------------------------
# sink serve (KeyboardInterrupt path with a mocked server)
# ---------------------------------------------------------------------------


def test_sink_serve_keyboard_interrupt_shuts_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """serve_forever → KeyboardInterrupt → shutdown + server_close run."""
    config_path = _write_config(tmp_path)

    fake_server = MagicMock()
    fake_server.serve_forever.side_effect = KeyboardInterrupt()

    import autoship.cli.commands.sink as sink_mod

    monkeypatch.setattr(sink_mod, "SinkServer", lambda *a, **kw: fake_server)

    result = runner.invoke(
        app,
        [
            "--config",
            str(config_path),
            "sink",
            "serve",
            "--port",
            "0",
            "--token",
            "secret",
        ],
    )
    assert result.exit_code == 0
    fake_server.serve_forever.assert_called_once()
    fake_server.shutdown.assert_called_once()
    fake_server.server_close.assert_called_once()
    assert "stopped" in result.output.lower()
    assert "token" in result.output.lower()


# ---------------------------------------------------------------------------
# sink status (mocked httpx responses)
# ---------------------------------------------------------------------------


def _mock_response(
    *, status_code: int = 200, json_data: Any = None, json_error: Exception | None = None
) -> MagicMock:
    """Build a fake ``httpx.Response`` for the status command."""
    resp = MagicMock()
    resp.status_code = status_code
    if json_error is not None:
        resp.json.side_effect = json_error
    else:
        resp.json.return_value = json_data or {}
    return resp


def test_sink_status_non_200_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-200 status code prints ``status_failed`` and exits 1."""
    config_path = _write_config(tmp_path, sink_url="http://example.com")
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _mock_response(status_code=403))
    result = runner.invoke(app, ["--config", str(config_path), "sink", "status"])
    assert result.exit_code == 1
    assert "403" in result.output


def test_sink_status_invalid_json_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200 response with a non-JSON body triggers ``status_invalid``."""
    config_path = _write_config(tmp_path, sink_url="http://example.com")
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: _mock_response(
            status_code=200,
            json_error=json.JSONDecodeError("msg", "doc", 0),
        ),
    )
    result = runner.invoke(app, ["--config", str(config_path), "sink", "status"])
    assert result.exit_code == 1
    assert "non-json" in result.output.lower()


def test_sink_status_with_token_sends_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--token`` adds an ``Authorization: Bearer`` header to the request."""
    config_path = _write_config(tmp_path, sink_url="http://example.com")

    captured: dict[str, Any] = {}

    def fake_get(url: str, headers: dict[str, str] | None = None, timeout: float = 5.0) -> Any:
        captured["url"] = url
        captured["headers"] = headers
        return _mock_response(status_code=200, json_data={"status": "ok"})

    monkeypatch.setattr(httpx, "get", fake_get)

    result = runner.invoke(
        app,
        ["--config", str(config_path), "sink", "status", "--token", "secret"],
    )
    assert result.exit_code == 0
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["url"] == "http://example.com/status"


# ---------------------------------------------------------------------------
# _resolve_store_dir
# ---------------------------------------------------------------------------


def test_resolve_store_dir_uses_config_server_store_dir(tmp_path: Path) -> None:
    """When ``server_store_dir`` is set in config, it wins over the default."""
    from autoship.cli.commands.sink import _resolve_store_dir
    from autoship.models.config import AppConfig, SinkConfig

    custom = tmp_path / "custom_store"
    config = AppConfig(project_root=tmp_path, sink=SinkConfig(server_store_dir=custom))

    assert _resolve_store_dir(config) == custom
