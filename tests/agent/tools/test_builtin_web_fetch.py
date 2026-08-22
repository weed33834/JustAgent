"""Tests for the ``web_fetch`` built-in tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from justagent.agent.tools.base import InvalidArgumentsError, ToolContext
from justagent.agent.tools.builtin.web_fetch import (
    DEFAULT_TIMEOUT_MS,
    WebFetchInput,
    make_web_fetch_tool,
)


def _make_ctx() -> ToolContext:
    return ToolContext(
        tool_call_id="call-1",
        iteration=1,
        cwd="/tmp",
    )


@pytest.mark.asyncio
async def test_web_fetch_rejects_non_http_url() -> None:
    tool = make_web_fetch_tool()
    result = await tool.invoke({"url": "ftp://example.com"}, _make_ctx())
    assert result.is_error
    assert "http" in result.error.lower()


@pytest.mark.asyncio
async def test_web_fetch_upgrades_http_to_https() -> None:
    """http:// URLs should be silently upgraded to https://."""

    tool = make_web_fetch_tool()
    captured_urls: list[str] = []

    class _FakeResponse:
        status_code = 200
        reason_phrase = "OK"
        headers = {"content-type": "text/plain"}
        text = "plain text response"

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, **kwargs: object) -> _FakeResponse:
            captured_urls.append(url)
            return _FakeResponse()

    with patch("httpx.AsyncClient", _FakeClient):
        result = await tool.invoke({"url": "http://example.com"}, _make_ctx())
    assert not result.is_error
    assert captured_urls == ["https://example.com"]
    assert "plain text response" in result.output


@pytest.mark.asyncio
async def test_web_fetch_invalid_format() -> None:
    tool = make_web_fetch_tool()
    result = await tool.invoke({"url": "https://example.com", "format": "bogus"}, _make_ctx())
    assert result.is_error
    assert "format" in result.error.lower()


@pytest.mark.asyncio
async def test_web_fetch_html_converted_to_markdown() -> None:
    tool = make_web_fetch_tool()

    html_body = (
        "<!DOCTYPE html><html><head><title>T</title></head>"
        "<body><h1>Title</h1><p>Para text</p>"
        "<pre>code block</pre></body></html>"
    )

    class _FakeResponse:
        status_code = 200
        reason_phrase = "OK"
        headers = {"content-type": "text/html"}
        text = html_body

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, **kwargs: object) -> _FakeResponse:
            return _FakeResponse()

    with patch("httpx.AsyncClient", _FakeClient):
        result = await tool.invoke({"url": "https://example.com"}, _make_ctx())
    assert not result.is_error
    # Markdown heading prefix.
    assert "# Title" in result.output
    # Code block preserved.
    assert "```" in result.output
    assert "code block" in result.output


@pytest.mark.asyncio
async def test_web_fetch_raw_format() -> None:
    tool = make_web_fetch_tool()

    class _FakeResponse:
        status_code = 200
        reason_phrase = "OK"
        headers = {"content-type": "text/html"}
        text = "<html><body>raw</body></html>"

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, **kwargs: object) -> _FakeResponse:
            return _FakeResponse()

    with patch("httpx.AsyncClient", _FakeClient):
        result = await tool.invoke({"url": "https://example.com", "format": "raw"}, _make_ctx())
    assert not result.is_error
    # Raw mode should return the original body unchanged.
    assert "<html>" in result.output


@pytest.mark.asyncio
async def test_web_fetch_http_error() -> None:
    tool = make_web_fetch_tool()

    import httpx

    class _FakeResponse:
        status_code = 404
        reason_phrase = "Not Found"
        headers: dict[str, str] = {}
        text = ""

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("GET", "https://example.com"),
                response=httpx.Response(404),
            )

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, **kwargs: object) -> _FakeResponse:
            return _FakeResponse()

    with patch("httpx.AsyncClient", _FakeClient):
        result = await tool.invoke({"url": "https://example.com"}, _make_ctx())
    assert result.is_error
    assert "404" in result.error


@pytest.mark.asyncio
async def test_web_fetch_input_validation() -> None:
    tool = make_web_fetch_tool()
    with pytest.raises(InvalidArgumentsError):
        await tool.invoke({}, _make_ctx())


def test_web_fetch_input_model() -> None:
    inp = WebFetchInput(url="https://example.com")
    assert inp.url == "https://example.com"
    assert inp.format == "markdown"
    assert inp.timeout_ms is None


def test_make_web_fetch_tool_metadata() -> None:
    tool = make_web_fetch_tool()
    assert tool.id == "web_fetch"
    assert tool.timeout_ms == 0  # uses per-call timeout
    assert tool.completes_run is False


def test_default_timeout_constant() -> None:
    assert DEFAULT_TIMEOUT_MS == 30_000
