"""Official web-search plugin for MyAgent-CLI.

When enabled, this plugin searches the web for context about a verification
failure and includes the top results in the AI fix prompt. Web search is
disabled by default and must be explicitly turned on via configuration.
"""

from __future__ import annotations

import logging
import time

from myagent.adapters.model_gateway import ChatMessage
from myagent.adapters.web_search import (
    BraveSearchAdapter,
    GoogleSearchAdapter,
    SearxngSearchAdapter,
    WebSearchAdapter,
    WebSearchResult,
    format_results,
)
from myagent.core.audit_logger import redact_text
from myagent.core.context import CommandContext
from myagent.core.fix import FixSuggestion
from myagent.core.metrics import get_registry
from myagent.core.model_router import ModelRouter
from myagent.exceptions import ModelGatewayError, VerifyError
from myagent.hookspec import hookimpl
from myagent.models.config import WebSearchProvider

logger = logging.getLogger("myagent")


class WebSearchPlugin:
    """Provide web-augmented fix suggestions via ``on_error``."""

    @hookimpl
    def on_error(self, context: CommandContext, error: Exception) -> FixSuggestion | None:
        """Search the web for error context and return an augmented fix suggestion."""
        if not context.extras.get("fix"):
            return None

        config = context.config.web_search
        if not config.enabled:
            return None

        query = _build_query(error)
        if not query:
            return None

        try:
            results = _search(
                query,
                config.provider,
                api_key=config.api_key,
                cx=config.cx,
                instance_url=config.instance_url,
                max_results=config.max_results,
                timeout=config.timeout,
            )
        except (NotImplementedError, ValueError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Web search failed: %s", exc)
            return None

        if not results:
            return None

        return _suggest_with_search(context, error, results)


def _search(
    query: str,
    provider: WebSearchProvider,
    *,
    api_key: str | None,
    cx: str | None,
    instance_url: str | None,
    max_results: int,
    timeout: float,
) -> list[WebSearchResult]:
    """Route the search query to the configured provider implementation."""
    registry = get_registry()
    registry.inc("web_search_requests", description="Total web search requests")
    start = time.perf_counter()
    try:
        if provider == WebSearchProvider.DUCKDUCKGO:
            results = WebSearchAdapter(timeout=timeout).search(query, max_results=max_results)
        elif provider == WebSearchProvider.BRAVE:
            if not api_key:
                raise ValueError(
                    "Brave web search provider requires an API key. "
                    "Set web_search.api_key in your configuration."
                )
            results = BraveSearchAdapter(api_key=api_key, timeout=timeout).search(
                query, max_results=max_results
            )
        elif provider == WebSearchProvider.GOOGLE:
            if not api_key:
                raise ValueError(
                    "Google web search provider requires an API key. "
                    "Set web_search.api_key in your configuration."
                )
            if not cx:
                raise ValueError(
                    "Google web search provider requires a search engine ID (cx). "
                    "Set web_search.cx in your configuration."
                )
            results = GoogleSearchAdapter(api_key=api_key, cx=cx, timeout=timeout).search(
                query, max_results=max_results
            )
        elif provider == WebSearchProvider.SEARXNG:
            if not instance_url:
                raise ValueError(
                    "SearXNG web search provider requires an instance URL. "
                    "Set web_search.instance_url in your configuration."
                )
            results = SearxngSearchAdapter(instance_url=instance_url, timeout=timeout).search(
                query, max_results=max_results
            )
        else:
            raise NotImplementedError(f"Unsupported web search provider: {provider.value}")
    except Exception:
        registry.inc("web_search_errors", description="Web search errors")
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        registry.record("web_search_latency_ms", elapsed_ms, description="Web search latency")

    registry.set(
        "web_search_results_last", len(results), description="Last web search result count"
    )
    return results


def _build_query(error: Exception) -> str:
    """Create a concise search query from an error."""
    if isinstance(error, VerifyError):
        command = error.details.get("command", "")
        stderr_raw = error.details.get("stderr", "")
        stderr = redact_text(stderr_raw)[:200]
        return f"{command} {stderr}".strip()
    return str(error)[:200]


def _suggest_with_search(
    context: CommandContext,
    error: Exception,
    results: list[WebSearchResult],
) -> FixSuggestion | None:
    """Ask the model to suggest a fix using web search results as context."""
    search_context = format_results(results)
    prompt = (
        "The following verification command failed. Use the web search results "
        "below to suggest a concise fix in one or two sentences.\n\n"
        f"Error: {error}\n\n"
        f"Web search results:\n{search_context}"
    )

    with ModelRouter(context.config) as router:
        try:
            suggestion = router.chat(
                [
                    ChatMessage(role="system", content="You are a helpful debugging assistant."),
                    ChatMessage(role="user", content=prompt),
                ],
                "verify-fix",
            )
        except ModelGatewayError:
            return None

        return FixSuggestion(description=suggestion.strip())


plugin = WebSearchPlugin()
