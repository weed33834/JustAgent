"""Lightweight web search adapter for AI-assisted debugging.

The default provider is DuckDuckGo Lite, which does not require an API key.
Because this sends error snippets to a public service, it is disabled by default
and must be explicitly enabled via configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import httpx
from duckduckgo_search import DDGS

from autoship.core.cache import DiskCache


@dataclass
class WebSearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str


class WebSearchError(Exception):
    """Raised when a web search request fails."""


def _format_search_error(provider: str, exc: Exception) -> str:
    """Return a redacted, provider-agnostic error message for *exc*.

    ``httpx.HTTPError`` string representations may contain request URLs
    that include API keys or search engine IDs, so this helper avoids
    interpolating the raw exception text into user-facing messages.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{provider} search returned HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return f"{provider} search request timed out"
    if isinstance(exc, httpx.RequestError):
        return f"{provider} search request failed"
    if isinstance(exc, ValueError):
        return f"{provider} search returned invalid JSON"
    return f"{provider} search request failed"


def _search_cache_key(provider: str, query: str, max_results: int) -> str:
    """Return a SHA256 cache key for a search query."""
    payload = json.dumps(
        {"provider": provider, "query": query, "max_results": max_results},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _result_to_dict(result: WebSearchResult) -> dict[str, str]:
    return {
        "title": result.title,
        "url": result.url,
        "snippet": result.snippet,
    }


def _results_to_dicts(results: list[WebSearchResult]) -> list[dict[str, str]]:
    return [_result_to_dict(result) for result in results]


def _dicts_to_results(dicts: Any) -> list[WebSearchResult]:
    return [
        WebSearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("snippet", ""),
        )
        for item in dicts
    ]


class _BraveWebResult(TypedDict, total=False):
    title: str
    url: str
    description: str


class _BraveWebSection(TypedDict, total=False):
    results: list[_BraveWebResult]


class _BraveSearchResponse(TypedDict, total=False):
    query: dict[str, Any]
    web: _BraveWebSection


class _GoogleSearchItem(TypedDict, total=False):
    title: str
    link: str
    snippet: str


class _GoogleSearchResponse(TypedDict, total=False):
    items: list[_GoogleSearchItem]


class _SearxngSearchResult(TypedDict, total=False):
    title: str
    url: str
    content: str


class _SearxngSearchResponse(TypedDict, total=False):
    results: list[_SearxngSearchResult]


class WebSearchAdapter:
    """Search the web and return a list of results."""

    _PROVIDER = "duckduckgo"

    def __init__(self, timeout: float = 10.0, cache: DiskCache | None = None) -> None:
        self.timeout = timeout
        self.cache = cache

    def _cache_key(self, query: str, max_results: int) -> str:
        return _search_cache_key(self._PROVIDER, query, max_results)

    def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        """Search for ``query`` and return up to ``max_results`` items."""
        cache_key = self._cache_key(query, max_results)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return _dicts_to_results(cached)

        try:
            raw = DDGS().text(query, max_results=max_results)
        except Exception as exc:
            raise WebSearchError(_format_search_error("DuckDuckGo", exc)) from exc

        results = [
            WebSearchResult(
                title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", "")
            )
            for r in raw
        ]
        if self.cache is not None:
            self.cache.set(cache_key, _results_to_dicts(results))
        return results


class BraveSearchAdapter:
    """Search the web using the Brave Search API."""

    _BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
    _PROVIDER = "brave"

    def __init__(self, api_key: str, timeout: float = 10.0, cache: DiskCache | None = None) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.cache = cache

    def _cache_key(self, query: str, max_results: int) -> str:
        return _search_cache_key(self._PROVIDER, query, max_results)

    def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        """Search for ``query`` via Brave and return up to ``max_results`` items."""
        cache_key = self._cache_key(query, max_results)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return _dicts_to_results(cached)

        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        params: dict[str, str | int] = {"q": query, "count": max_results}
        try:
            response = httpx.get(
                self._BRAVE_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(_format_search_error("Brave", exc)) from exc

        try:
            payload = cast(_BraveSearchResponse, response.json())
        except ValueError as exc:
            raise WebSearchError(_format_search_error("Brave", exc)) from exc
        results = self._parse_brave(payload, max_results)
        if self.cache is not None:
            self.cache.set(cache_key, _results_to_dicts(results))
        return results

    def _parse_brave(
        self, payload: _BraveSearchResponse, max_results: int
    ) -> list[WebSearchResult]:
        """Parse Brave Search JSON response."""
        results: list[WebSearchResult] = []
        web = payload.get("web")
        if web is None:
            return results

        items = web.get("results")
        if items is None:
            return results

        for item in items:
            if len(results) >= max_results:
                break
            title = item.get("title")
            url = item.get("url")
            snippet = item.get("description")
            if title and url:
                results.append(
                    WebSearchResult(
                        title=title,
                        url=url,
                        snippet=snippet or "",
                    )
                )

        return results


class GoogleSearchAdapter:
    """Search the web using the Google Custom Search JSON API."""

    _GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, cx: str, timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.cx = cx
        self.timeout = timeout

    def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        """Search for ``query`` via Google and return up to ``max_results`` items."""
        params: dict[str, str | int] = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": max_results,
        }
        try:
            response = httpx.get(
                self._GOOGLE_SEARCH_URL,
                params=params,
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(_format_search_error("Google", exc)) from exc

        try:
            payload = cast(_GoogleSearchResponse, response.json())
        except ValueError as exc:
            raise WebSearchError(_format_search_error("Google", exc)) from exc
        return self._parse_google(payload, max_results)

    def _parse_google(
        self, payload: _GoogleSearchResponse, max_results: int
    ) -> list[WebSearchResult]:
        """Parse Google Custom Search JSON response."""
        results: list[WebSearchResult] = []
        items = payload.get("items")
        if items is None:
            return results

        for item in items:
            if len(results) >= max_results:
                break
            title = item.get("title")
            url = item.get("link")
            snippet = item.get("snippet")
            if title and url:
                results.append(
                    WebSearchResult(
                        title=title,
                        url=url,
                        snippet=snippet or "",
                    )
                )

        return results


class SearxngSearchAdapter:
    """Search the web using a SearXNG instance."""

    def __init__(self, instance_url: str, timeout: float = 10.0) -> None:
        self.instance_url = instance_url.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        """Search for ``query`` via SearXNG and return up to ``max_results`` items."""
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        try:
            response = httpx.get(
                f"{self.instance_url}/search",
                params=params,
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebSearchError(_format_search_error("SearXNG", exc)) from exc

        try:
            payload = cast(_SearxngSearchResponse, response.json())
        except ValueError as exc:
            raise WebSearchError(_format_search_error("SearXNG", exc)) from exc
        return self._parse_searxng(payload, max_results)

    def _parse_searxng(
        self, payload: _SearxngSearchResponse, max_results: int
    ) -> list[WebSearchResult]:
        """Parse SearXNG JSON response."""
        results: list[WebSearchResult] = []
        items = payload.get("results")
        if items is None:
            return results

        for item in items:
            if len(results) >= max_results:
                break
            title = item.get("title")
            url = item.get("url")
            snippet = item.get("content")
            if title and url:
                results.append(
                    WebSearchResult(
                        title=title,
                        url=url,
                        snippet=snippet or "",
                    )
                )

        return results


def format_results(results: list[WebSearchResult]) -> str:
    """Format search results for inclusion in a model prompt."""
    if not results:
        return "No web search results."
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}\n   {result.url}\n   {result.snippet}")
    return "\n\n".join(lines)
