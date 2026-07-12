"""Tests that provider error messages do not leak sensitive information."""

from __future__ import annotations

import httpx
import pytest
import respx

from autoship.adapters.model_gateway import ChatCompletionRequest, ChatMessage
from autoship.adapters.providers.azure_openai import AzureOpenAIGateway
from autoship.adapters.providers.llama_cpp import LlamaCppGateway
from autoship.adapters.providers.lm_studio import LmStudioGateway
from autoship.adapters.providers.ollama import OllamaGateway
from autoship.adapters.providers.openai import OpenAIGateway
from autoship.adapters.providers.openrouter import OpenRouterGateway
from autoship.adapters.providers.vllm import VllmGateway
from autoship.exceptions import ModelGatewayError
from autoship.models.config import ModelBackendConfig, Provider

OPENAI_BASE_URL = "https://api.openai.com/v1"
AZURE_BASE_URL = "https://my-resource.openai.azure.com/openai/deployments/my-deployment"
AZURE_API_VERSION = "2024-02-01"
OLLAMA_BASE_URL = "http://localhost:11434"
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080/v1"
ALL_BASE_URLS = (
    OPENAI_BASE_URL,
    AZURE_BASE_URL,
    OLLAMA_BASE_URL,
    LM_STUDIO_BASE_URL,
    OPENROUTER_BASE_URL,
    VLLM_BASE_URL,
    LLAMA_CPP_BASE_URL,
)


class _LeakyConnectError(httpx.ConnectError):
    """A connect error whose string representation embeds the request URL."""

    def __str__(self) -> str:
        return f"connection refused for {self.request.url}"


def _openai_gateway() -> OpenAIGateway:
    cfg = ModelBackendConfig(
        provider=Provider.OPENAI,
        base_url=OPENAI_BASE_URL,
        api_key="test-api-key",
        model="gpt-4o-mini",
        timeout=5.0,
    )
    return OpenAIGateway(cfg)


def _azure_gateway() -> AzureOpenAIGateway:
    cfg = ModelBackendConfig(
        provider=Provider.AZURE_OPENAI,
        base_url=AZURE_BASE_URL,
        api_key="test-api-key",
        api_version=AZURE_API_VERSION,
        model="gpt-4o",
        timeout=5.0,
    )
    return AzureOpenAIGateway(cfg)


def _ollama_gateway() -> OllamaGateway:
    cfg = ModelBackendConfig(
        provider=Provider.OLLAMA,
        base_url=OLLAMA_BASE_URL,
        model="llama3",
        timeout=5.0,
    )
    return OllamaGateway(cfg)


def _lm_studio_gateway() -> LmStudioGateway:
    cfg = ModelBackendConfig(
        provider=Provider.LM_STUDIO,
        base_url=LM_STUDIO_BASE_URL,
        model="qwen2.5",
        timeout=5.0,
    )
    return LmStudioGateway(cfg)


def _openrouter_gateway() -> OpenRouterGateway:
    cfg = ModelBackendConfig(
        provider=Provider.OPENROUTER,
        base_url=OPENROUTER_BASE_URL,
        api_key="test-api-key",
        model="auto",
        timeout=5.0,
    )
    return OpenRouterGateway(cfg)


def _vllm_gateway() -> VllmGateway:
    cfg = ModelBackendConfig(
        provider=Provider.VLLM,
        base_url=VLLM_BASE_URL,
        model="qwen2.5",
        timeout=5.0,
    )
    return VllmGateway(cfg)


def _llama_cpp_gateway() -> LlamaCppGateway:
    cfg = ModelBackendConfig(
        provider=Provider.LLAMA_CPP,
        base_url=LLAMA_CPP_BASE_URL,
        model="qwen2.5",
        timeout=5.0,
    )
    return LlamaCppGateway(cfg)


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])


# (gateway factory, chat-completions endpoint, provider name) for every provider.
_PROVIDER_CASES = [
    (_openai_gateway(), f"{OPENAI_BASE_URL}/chat/completions", "OpenAI"),
    (_azure_gateway(), f"{AZURE_BASE_URL}/chat/completions", "Azure OpenAI"),
    (_ollama_gateway(), f"{OLLAMA_BASE_URL}/chat/completions", "Ollama"),
    (_lm_studio_gateway(), f"{LM_STUDIO_BASE_URL}/chat/completions", "LM Studio"),
    (_openrouter_gateway(), f"{OPENROUTER_BASE_URL}/chat/completions", "OpenRouter"),
    (_vllm_gateway(), f"{VLLM_BASE_URL}/chat/completions", "vLLM"),
    (_llama_cpp_gateway(), f"{LLAMA_CPP_BASE_URL}/chat/completions", "llama.cpp"),
]


@pytest.mark.parametrize(("gateway", "endpoint"), [(c[0], c[1]) for c in _PROVIDER_CASES])
def test_chat_request_error_does_not_leak_base_url(gateway, endpoint: str) -> None:
    """Connection errors must not embed the backend URL in the exception text."""
    with respx.mock:
        respx.post(endpoint).mock(
            side_effect=_LeakyConnectError(
                "refused",
                request=httpx.Request("POST", endpoint),
            )
        )
        with pytest.raises(ModelGatewayError) as exc_info:
            gateway.chat(_request())

    message = str(exc_info.value)
    for base_url in ALL_BASE_URLS:
        assert base_url not in message
    assert "test-api-key" not in message


@pytest.mark.parametrize(
    ("gateway", "endpoint", "provider_name"),
    _PROVIDER_CASES,
)
def test_chat_timeout_error_message(gateway, endpoint: str, provider_name: str) -> None:
    with respx.mock:
        respx.post(endpoint).mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(ModelGatewayError, match=f"{provider_name} request timed out"):
            gateway.chat(_request())


@pytest.mark.parametrize(
    ("gateway", "endpoint", "provider_name"),
    _PROVIDER_CASES,
)
def test_chat_http_status_error_message(gateway, endpoint: str, provider_name: str) -> None:
    with respx.mock:
        respx.post(endpoint).respond(500)
        with pytest.raises(ModelGatewayError, match=f"{provider_name} returned HTTP 500"):
            gateway.chat(_request())


@pytest.mark.parametrize(
    ("gateway", "endpoint", "provider_name"),
    _PROVIDER_CASES,
)
def test_chat_invalid_json_error_message(gateway, endpoint: str, provider_name: str) -> None:
    with respx.mock:
        respx.post(endpoint).respond(200, text="not json")
        with pytest.raises(ModelGatewayError, match=f"{provider_name} returned invalid JSON"):
            gateway.chat(_request())


@pytest.mark.asyncio
async def test_achat_request_error_does_not_leak_base_url() -> None:
    with respx.mock:
        respx.post(f"{OPENAI_BASE_URL}/chat/completions").mock(
            side_effect=_LeakyConnectError(
                "refused",
                request=httpx.Request("POST", f"{OPENAI_BASE_URL}/chat/completions"),
            )
        )
        with pytest.raises(ModelGatewayError) as exc_info:
            await _openai_gateway().achat(_request())

    message = str(exc_info.value)
    assert OPENAI_BASE_URL not in message
    assert "test-api-key" not in message
