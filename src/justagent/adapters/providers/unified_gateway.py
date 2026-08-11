"""Unified model gateway backed by the OpenAI SDK.

All supported providers (Ollama, vLLM, LM Studio, llama.cpp, OpenAI,
OpenRouter, Azure) expose OpenAI-compatible endpoints, so a single official
``openai`` client with a per-provider ``base_url`` covers every backend —
no LiteLLM indirection needed. This is lighter, installs cleanly, and keeps
the full tool-call / usage / streaming surface.

To use a custom gateway, set provider=openai with a custom base_url.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, cast

from openai import OpenAI

from justagent.adapters.model_gateway import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelGateway,
)
from justagent.exceptions import ModelGatewayError
from justagent.models.config import ModelBackendConfig, Provider

logger = logging.getLogger("justagent.adapters.providers.unified_gateway")

# Map internal providers to LiteLLM model prefix / custom_llm_provider.
_PROVIDER_MAP: dict[Provider, str] = {
    Provider.OLLAMA: "ollama",
    Provider.LM_STUDIO: "openai",  # LM Studio exposes OpenAI-compatible API
    Provider.LLAMA_CPP: "openai",  # llama.cpp server exposes OpenAI-compatible API
    Provider.VLLM: "openai",  # vLLM exposes OpenAI-compatible API
    Provider.OPENAI: "openai",
    Provider.OPENROUTER: "openrouter",
    Provider.AZURE_OPENAI: "azure",
}

# Default base URL for each local provider.
#
# This is the single canonical source of provider -> base URL defaults for
# the whole project. The CLI command modules (``agent``, ``fix``, ``skill``)
# import this mapping instead of redefining their own so the values stay in
# sync. Note that ``UnifiedGateway.__init__`` runs these through :func:`_normalize_base_url` (which appends ``/v1`` for bare roots), so the
# values here are intentionally unnormalized.
_PROVIDER_BASE_URLS: dict[Provider, str] = {
    Provider.OLLAMA: "http://localhost:11434",
    Provider.LM_STUDIO: "http://localhost:1234/v1",
    Provider.LLAMA_CPP: "http://localhost:8080/v1",
    Provider.VLLM: "http://localhost:8000/v1",
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.OPENROUTER: "https://openrouter.ai/api/v1",
}

# Error patterns that indicate upstream service unavailability.
# Each (pattern, category) tuple produces a human-readable label.
_ERROR_PATTERNS: list[tuple[str, str]] = [
    ("no available accounts", "No available upstream account"),
    ("insufficient_quota", "API quota exhausted"),
    ("rate_limit", "Rate limited by upstream provider"),
    ("invalid_api_key", "Invalid API key"),
    ("api_key_required", "API key required but not provided"),
    ("context_length_exceeded", "Prompt exceeds model context window"),
    ("connection refused", "Backend connection refused"),
    ("dns", "DNS resolution failed for backend"),
    ("timeout", "Backend request timed out"),
    ("not found", "Model or endpoint not found"),
]


def _normalize_base_url(raw: str) -> str:
    """Normalize a user-supplied base URL for LiteLLM consumption.

    - Strip trailing slashes.
    - Auto-upgrade ``http://`` to ``https://`` for non-localhost hosts
      (many API gateways redirect HTTP -> HTTPS anyway).
    - Ensure the path ends with ``/v1`` if it looks like a bare root
      without an API version segment.
    """
    url = raw.rstrip("/")
    # Upgrade HTTP -> HTTPS for remote hosts (not localhost / 127.x / LAN).
    if url.startswith("http://") and not re.match(
        r"http://(localhost|127\.\d+\.\d+\.\d+|"
        r"10\.\d+\.\d+\.\d+|"
        r"192\.168\.\d+\.\d+|"
        r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)",
        url,
    ):
        url = "https://" + url[len("http://"):]
    # Append /v1 if the URL is a bare root without an API version segment.
    if not re.search(r"/(v\d+|api)/?$", url):
        url += "/v1"
    return url


def _classify_error(exc: Exception) -> str:
    """Extract a categorized, human-readable error label from an exception."""
    msg = str(exc).lower()
    for pattern, label in _ERROR_PATTERNS:
        if pattern in msg:
            return label
    return "Upstream error"


# Providers that authenticate via an API key. Local/self-hosted backends
# (Ollama, vLLM, LM Studio, llama.cpp) accept requests without one.
_KEY_REQUIRED_PROVIDERS: frozenset[Provider] = frozenset(
    {Provider.OPENAI, Provider.OPENROUTER, Provider.AZURE_OPENAI}
)

# Placeholder used when no real key is configured.
_PLACEHOLDER_KEY = "placeholder"


class UnifiedGateway(ModelGateway):
    """Single gateway implementation covering all providers via LiteLLM.

    LiteLLM handles: unified API format, automatic retries, cost tracking,
    load balancing, and fallback across 100+ providers.
    """

    def __init__(self, cfg: ModelBackendConfig) -> None:
        super().__init__(cfg)
        self._provider = cfg.provider
        self._model = cfg.model or "gpt-4o-mini"
        self._base_url = _normalize_base_url(
            str(cfg.base_url) if cfg.base_url
            else _PROVIDER_BASE_URLS.get(cfg.provider, "")
        )
        self._api_key = cfg.api_key or _PLACEHOLDER_KEY
        self._api_version = cfg.api_version
        self._timeout = cfg.timeout
        # Remote providers must have a real key; fail fast instead of hanging
        # on a doomed network call (e.g. ``justagent fix`` with no key).
        self._key_required = cfg.provider in _KEY_REQUIRED_PROVIDERS
        # One official OpenAI client; base_url pins it to the provider gateway.
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url or None,
            timeout=self._timeout,
            max_retries=2,
            default_headers={"api-version": self._api_version} if self._api_version else None,
        )

    def _fails_fast(self) -> bool:
        """True when this backend cannot possibly succeed and should not be
        queried (remote provider missing an API key)."""
        return self._key_required and self._api_key == _PLACEHOLDER_KEY

    def close(self) -> None:
        """The OpenAI client holds no resources requiring explicit close."""

    def health(self) -> bool:
        if self._fails_fast():
            logger.info("Skipping health check for %s: no API key set", self._provider.value)
            return False
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return bool(response and response.choices)
        except Exception:
            logger.debug(
                "Health check failed for provider %s", self._provider.value, exc_info=True
            )
            return False

    def list_models(self) -> list[str]:
        if self._fails_fast():
            return [self._model]
        try:
            listed = self._client.models.list()
            ids = [m.id for m in getattr(listed, "data", []) or []]
            if not ids:
                # Some OpenAI-compatible servers lack GET /models; fall back to the
                # configured model so callers still have a usable list.
                return [self._model]
            return cast(list[str], ids)
        except Exception as exc:
            raise ModelGatewayError(
                f"Failed to list models from {self._provider.value}: "
                f"{_classify_error(exc)} — {exc}"
            ) from exc

    def chat(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        if self._fails_fast():
            raise ModelGatewayError(
                f"{self._provider.value} backend requires an API key, "
                "but none is configured. Set one in the config or via the API key "
                "environment variable."
            )
        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in req.messages],
                temperature=req.temperature if req.temperature is not None else 0.7,
                max_tokens=req.max_tokens if req.max_tokens is not None else 512,
            )
        except Exception as exc:
            raise ModelGatewayError(
                f"Chat request failed [{self._provider.value}/{self._model}]: "
                f"{_classify_error(exc)} — {exc}"
            ) from exc

        content = response.choices[0].message.content or ""
        usage_dict: dict[str, Any] | None = None
        if response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return ChatCompletionResponse(
            content=content,
            model=response.model or self._model,
            usage=usage_dict,
            latency_ms=(time.time() - start) * 1000,
        )


def format_provider_error(provider_name: str, exc: Exception) -> str:
    """Return a user-facing, redacted error message for the given exception."""
    msg = str(exc)
    return f"{provider_name} error: {msg[:200]}"
