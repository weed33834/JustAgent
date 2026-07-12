"""Tests for the web search adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from autoship.adapters.web_search import WebSearchAdapter, WebSearchError, format_results

SAMPLE_HTML = """
<div class="result results_links results_links_deep web-result">
    <a class="result__a" href="https://example.com/fix">How to fix it</a>
    <a class="result__snippet">This is a helpful snippet.</a>
</div>
<div class="result results_links results_links_deep web-result">
    <a class="result__a" href="https://example.com/other">Another page</a>
    <a class="result__snippet">Another snippet.</a>
</div>
"""


def test_search_parses_results() -> None:
    adapter = WebSearchAdapter()
    response = MagicMock()
    response.text = SAMPLE_HTML
    response.raise_for_status.return_value = None

    with patch.object(httpx, "get", return_value=response):
        results = adapter.search("pytest error", max_results=2)

    assert len(results) == 2
    assert results[0].title == "How to fix it"
    assert results[0].url == "https://example.com/fix"
    assert "helpful snippet" in results[0].snippet


def test_search_limits_results() -> None:
    adapter = WebSearchAdapter()
    response = MagicMock()
    response.text = SAMPLE_HTML
    response.raise_for_status.return_value = None

    with patch.object(httpx, "get", return_value=response):
        results = adapter.search("pytest error", max_results=1)

    assert len(results) == 1


def test_search_http_error_raises() -> None:
    adapter = WebSearchAdapter()
    with (
        patch.object(httpx, "get", side_effect=httpx.ConnectError("no network")),
        pytest.raises(WebSearchError, match="DuckDuckGo search request failed"),
    ):
        adapter.search("pytest error")


def test_format_results_empty() -> None:
    assert "No web search results" in format_results([])
