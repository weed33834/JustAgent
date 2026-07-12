"""Tests for the built-in web-search plugin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from autoship.adapters.web_search import (
    BraveSearchAdapter,
    GoogleSearchAdapter,
    SearxngSearchAdapter,
    WebSearchError,
    WebSearchResult,
)
from autoship.core.context import CommandContext
from autoship.core.fix import FixSuggestion
from autoship.exceptions import VerifyError
from autoship.models.config import AppConfig, WebSearchConfig, WebSearchProvider
from autoship.plugins import web_search


@pytest.fixture
def web_context(app_config: AppConfig) -> CommandContext:
    """Return a CommandContext with web search enabled."""
    app_config.web_search = WebSearchConfig(enabled=True, max_results=2)
    return CommandContext(
        command="verify",
        project_root=app_config.project_root,
        config=app_config,
        extras={"fix": True, "verify_command": "pytest"},
    )


def test_web_search_disabled_by_default(app_config: AppConfig) -> None:
    ctx = CommandContext(
        command="verify",
        project_root=app_config.project_root,
        config=app_config,
        extras={"fix": True},
    )
    assert web_search.plugin.on_error(ctx, VerifyError("fail")) is None


def test_web_search_returns_suggestion(web_context: CommandContext) -> None:
    results = [
        MagicMock(title="Fix 1", url="https://example.com/1", snippet="do this"),
    ]

    with (
        patch.object(web_search.WebSearchAdapter, "search", return_value=results),
        patch.object(web_search, "ModelRouter") as mock_router,
    ):
        mock_router.return_value.__enter__.return_value = mock_router.return_value
        mock_router.return_value.__exit__.return_value = False
        mock_router.return_value.chat.return_value = "Try updating pytest."
        suggestion = web_search.plugin.on_error(
            web_context,
            VerifyError("failed", details={"command": "pytest", "stderr": "error"}),
        )

    assert isinstance(suggestion, FixSuggestion)
    assert "Try updating pytest" in suggestion.description


def test_web_search_no_results_returns_none(web_context: CommandContext) -> None:
    with patch.object(web_search.WebSearchAdapter, "search", return_value=[]):
        assert web_search.plugin.on_error(web_context, VerifyError("fail")) is None


def test_web_search_without_fix_flag_returns_none(web_context: CommandContext) -> None:
    web_context.extras["fix"] = False
    assert web_search.plugin.on_error(web_context, VerifyError("fail")) is None


def test_web_search_default_provider_is_duckduckgo(app_config: AppConfig) -> None:
    assert app_config.web_search.provider == WebSearchProvider.DUCKDUCKGO


def test_web_search_routes_to_duckduckgo(web_context: CommandContext) -> None:
    results = [
        MagicMock(title="Fix 1", url="https://example.com/1", snippet="do this"),
    ]

    with (
        patch.object(web_search.WebSearchAdapter, "search", return_value=results) as mock_search,
        patch.object(web_search, "ModelRouter") as mock_router,
    ):
        mock_router.return_value.__enter__.return_value = mock_router.return_value
        mock_router.return_value.__exit__.return_value = False
        mock_router.return_value.chat.return_value = "Try updating pytest."
        suggestion = web_search.plugin.on_error(
            web_context,
            VerifyError("failed", details={"command": "pytest", "stderr": "error"}),
        )

    assert isinstance(suggestion, FixSuggestion)
    assert "Try updating pytest" in suggestion.description
    mock_search.assert_called_once()


def test_web_search_brave_missing_api_key_raises(web_context: CommandContext) -> None:
    web_context.config.web_search.provider = WebSearchProvider.BRAVE
    with pytest.raises(ValueError, match="Brave web search provider requires an API key"):
        web_search.plugin.on_error(
            web_context,
            VerifyError("failed", details={"command": "pytest", "stderr": "error"}),
        )


def test_search_routes_to_duckduckgo() -> None:
    with patch.object(web_search.WebSearchAdapter, "search", return_value=[]) as mock_search:
        web_search._search(
            "pytest error",
            WebSearchProvider.DUCKDUCKGO,
            api_key=None,
            cx=None,
            instance_url=None,
            max_results=3,
            timeout=10.0,
        )
    mock_search.assert_called_once_with("pytest error", max_results=3)


def test_search_routes_to_brave() -> None:
    with patch.object(web_search.BraveSearchAdapter, "search", return_value=[]) as mock_search:
        web_search._search(
            "pytest error",
            WebSearchProvider.BRAVE,
            api_key="test-key",
            cx=None,
            instance_url=None,
            max_results=3,
            timeout=10.0,
        )
    mock_search.assert_called_once_with("pytest error", max_results=3)


def test_search_brave_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="Brave web search provider requires an API key"):
        web_search._search(
            "pytest error",
            WebSearchProvider.BRAVE,
            api_key=None,
            cx=None,
            instance_url=None,
            max_results=3,
            timeout=10.0,
        )


def test_search_unsupported_provider_raises() -> None:
    with pytest.raises(NotImplementedError, match="Unsupported"):
        web_search._search(
            "pytest error",
            MagicMock(value="unknown"),  # type: ignore[arg-type]
            api_key=None,
            cx=None,
            instance_url=None,
            max_results=3,
            timeout=10.0,
        )


def test_brave_search_returns_results() -> None:
    brave_response = {
        "query": {"original": "pytest error"},
        "web": {
            "results": [
                {
                    "title": "Fix 1",
                    "url": "https://example.com/1",
                    "description": "Do this.",
                },
                {
                    "title": "Fix 2",
                    "url": "https://example.com/2",
                    "description": "Do that.",
                },
            ]
        },
    }

    with respx.mock:
        route = respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            return_value=httpx.Response(200, json=brave_response)
        )
        adapter = BraveSearchAdapter(api_key="test-key", timeout=5.0)
        results = adapter.search("pytest error", max_results=2)

    assert route.called
    request = route.calls.last.request
    assert request.headers["X-Subscription-Token"] == "test-key"
    assert request.url.params["q"] == "pytest error"
    assert request.url.params["count"] == "2"

    assert results == [
        WebSearchResult(title="Fix 1", url="https://example.com/1", snippet="Do this."),
        WebSearchResult(title="Fix 2", url="https://example.com/2", snippet="Do that."),
    ]


def test_brave_search_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="Brave web search provider requires an API key"):
        web_search._search(
            "pytest error",
            WebSearchProvider.BRAVE,
            api_key=None,
            cx=None,
            instance_url=None,
            max_results=3,
            timeout=10.0,
        )


def test_brave_search_http_error_raises() -> None:
    with respx.mock:
        route = respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )
        adapter = BraveSearchAdapter(api_key="bad-key")
        with pytest.raises(WebSearchError, match="Brave search returned HTTP 401"):
            adapter.search("pytest error", max_results=2)

    assert route.called


def test_search_routes_to_google() -> None:
    with patch.object(web_search.GoogleSearchAdapter, "search", return_value=[]) as mock_search:
        web_search._search(
            "pytest error",
            WebSearchProvider.GOOGLE,
            api_key="test-key",
            cx="test-cx",
            instance_url=None,
            max_results=3,
            timeout=10.0,
        )
    mock_search.assert_called_once_with("pytest error", max_results=3)


def test_search_google_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="Google web search provider requires an API key"):
        web_search._search(
            "pytest error",
            WebSearchProvider.GOOGLE,
            api_key=None,
            cx="test-cx",
            instance_url=None,
            max_results=3,
            timeout=10.0,
        )


def test_search_google_missing_cx_raises() -> None:
    with pytest.raises(ValueError, match="Google web search provider requires a search engine ID"):
        web_search._search(
            "pytest error",
            WebSearchProvider.GOOGLE,
            api_key="test-key",
            cx=None,
            instance_url=None,
            max_results=3,
            timeout=10.0,
        )


def test_google_search_returns_results() -> None:
    google_response = {
        "items": [
            {
                "title": "Fix 1",
                "link": "https://example.com/1",
                "snippet": "Do this.",
            },
            {
                "title": "Fix 2",
                "link": "https://example.com/2",
                "snippet": "Do that.",
            },
        ]
    }

    with respx.mock:
        route = respx.get("https://www.googleapis.com/customsearch/v1").mock(
            return_value=httpx.Response(200, json=google_response)
        )
        adapter = GoogleSearchAdapter(api_key="test-key", cx="test-cx", timeout=5.0)
        results = adapter.search("pytest error", max_results=2)

    assert route.called
    request = route.calls.last.request
    assert request.url.params["key"] == "test-key"
    assert request.url.params["cx"] == "test-cx"
    assert request.url.params["q"] == "pytest error"
    assert request.url.params["num"] == "2"

    assert results == [
        WebSearchResult(title="Fix 1", url="https://example.com/1", snippet="Do this."),
        WebSearchResult(title="Fix 2", url="https://example.com/2", snippet="Do that."),
    ]


def test_google_search_http_error_raises() -> None:
    with respx.mock:
        route = respx.get("https://www.googleapis.com/customsearch/v1").mock(
            return_value=httpx.Response(403, text="Forbidden")
        )
        adapter = GoogleSearchAdapter(api_key="bad-key", cx="bad-cx")
        with pytest.raises(WebSearchError, match="Google search returned HTTP 403"):
            adapter.search("pytest error", max_results=2)

    assert route.called


def test_search_routes_to_searxng() -> None:
    with patch.object(web_search.SearxngSearchAdapter, "search", return_value=[]) as mock_search:
        web_search._search(
            "pytest error",
            WebSearchProvider.SEARXNG,
            api_key=None,
            cx=None,
            instance_url="https://searx.example.com",
            max_results=3,
            timeout=10.0,
        )
    mock_search.assert_called_once_with("pytest error", max_results=3)


def test_search_searxng_missing_instance_url_raises() -> None:
    with pytest.raises(ValueError, match="SearXNG web search provider requires an instance URL"):
        web_search._search(
            "pytest error",
            WebSearchProvider.SEARXNG,
            api_key=None,
            cx=None,
            instance_url=None,
            max_results=3,
            timeout=10.0,
        )


def test_searxng_search_returns_results() -> None:
    searxng_response = {
        "results": [
            {
                "title": "Fix 1",
                "url": "https://example.com/1",
                "content": "Do this.",
            },
            {
                "title": "Fix 2",
                "url": "https://example.com/2",
                "content": "Do that.",
            },
        ]
    }

    with respx.mock:
        route = respx.get("https://searx.example.com/search").mock(
            return_value=httpx.Response(200, json=searxng_response)
        )
        adapter = SearxngSearchAdapter(instance_url="https://searx.example.com", timeout=5.0)
        results = adapter.search("pytest error", max_results=2)

    assert route.called
    request = route.calls.last.request
    assert request.url.params["q"] == "pytest error"
    assert request.url.params["format"] == "json"
    assert request.url.params["pageno"] == "1"

    assert results == [
        WebSearchResult(title="Fix 1", url="https://example.com/1", snippet="Do this."),
        WebSearchResult(title="Fix 2", url="https://example.com/2", snippet="Do that."),
    ]


def test_searxng_search_http_error_raises() -> None:
    with respx.mock:
        route = respx.get("https://searx.example.com/search").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        adapter = SearxngSearchAdapter(instance_url="https://searx.example.com")
        with pytest.raises(WebSearchError, match="SearXNG search returned HTTP 500"):
            adapter.search("pytest error", max_results=2)

    assert route.called
