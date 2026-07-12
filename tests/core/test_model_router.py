"""Tests for the model router."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoship.adapters.model_gateway import ChatCompletionResponse, ChatMessage
from autoship.adapters.providers import (
    AzureOpenAIGateway,
    LlamaCppGateway,
    LmStudioGateway,
    OpenAIGateway,
    OpenRouterGateway,
    VllmGateway,
)
from autoship.core.model_router import ModelRouter
from autoship.exceptions import ModelGatewayError
from autoship.models.config import AppConfig, ModelBackendConfig, Provider


def _backend(
    provider: Provider = Provider.OLLAMA, base_url: str = "http://localhost:11434"
) -> ModelBackendConfig:
    return ModelBackendConfig(provider=provider, base_url=base_url, model="llama3")


def test_chat_raises_when_no_backends(project_root: Path) -> None:
    config = AppConfig(project_root=project_root)
    router = ModelRouter(config)
    with pytest.raises(ModelGatewayError, match="No model backends"):
        router.chat([ChatMessage(role="user", content="hi")], "test")


def test_chat_uses_healthy_backend(project_root: Path) -> None:
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config)
    gateway = MagicMock()
    gateway.health.return_value = True
    gateway.chat.return_value = ChatCompletionResponse(content="hello", model="llama3")
    with patch.object(router, "_gateways", return_value=[gateway]):
        result = router.chat([ChatMessage(role="user", content="hi")], "test")
    assert result == "hello"


def test_chat_falls_back_to_second_backend(project_root: Path) -> None:
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config)
    first = MagicMock()
    first.health.return_value = False
    second = MagicMock()
    second.health.return_value = True
    second.chat.return_value = ChatCompletionResponse(content="ok", model="llama3")
    with patch.object(router, "_gateways", return_value=[first, second]):
        result = router.chat([ChatMessage(role="user", content="hi")], "test")
    assert result == "ok"


def test_chat_raises_when_all_backends_unhealthy(project_root: Path) -> None:
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config)
    gateway = MagicMock()
    gateway.health.return_value = False
    with (
        patch.object(router, "_gateways", return_value=[gateway]),
        pytest.raises(ModelGatewayError, match="All model backends are unhealthy"),
    ):
        router.chat([ChatMessage(role="user", content="hi")], "test")


def test_generate_commit_message(project_root: Path) -> None:
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config)
    with patch.object(router, "chat", return_value="feat(scope): add feature") as mock_chat:
        result = router.generate_commit_message(diff="diff", stats="stats")
    assert result == "feat(scope): add feature"
    args = mock_chat.call_args[0]
    messages: list[ChatMessage] = args[0]
    assert any(m.role == "system" for m in messages)
    assert any("Diff" in m.content for m in messages)


def test_router_instantiates_lm_studio_backend(project_root: Path) -> None:
    config = AppConfig(
        project_root=project_root,
        model={"backends": [_backend(Provider.LM_STUDIO, "http://localhost:1234/v1")]},
    )
    router = ModelRouter(config)
    gateways = router._gateways()
    assert len(gateways) == 1
    assert isinstance(gateways[0], LmStudioGateway)


def test_router_instantiates_llama_cpp_backend(project_root: Path) -> None:
    config = AppConfig(
        project_root=project_root,
        model={"backends": [_backend(Provider.LLAMA_CPP, "http://localhost:8080/v1")]},
    )
    router = ModelRouter(config)
    gateways = router._gateways()
    assert len(gateways) == 1
    assert isinstance(gateways[0], LlamaCppGateway)


def test_router_instantiates_vllm_backend(project_root: Path) -> None:
    config = AppConfig(
        project_root=project_root,
        model={"backends": [_backend(Provider.VLLM, "http://localhost:8000/v1")]},
    )
    router = ModelRouter(config)
    gateways = router._gateways()
    assert len(gateways) == 1
    assert isinstance(gateways[0], VllmGateway)


def test_router_instantiates_openai_backend(project_root: Path) -> None:
    config = AppConfig(
        project_root=project_root,
        model={
            "backends": [
                ModelBackendConfig(
                    provider=Provider.OPENAI,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                    model="gpt-4o-mini",
                )
            ]
        },
    )
    router = ModelRouter(config)
    gateways = router._gateways()
    assert len(gateways) == 1
    assert isinstance(gateways[0], OpenAIGateway)


def test_router_instantiates_azure_openai_backend(project_root: Path) -> None:
    config = AppConfig(
        project_root=project_root,
        model={
            "backends": [
                ModelBackendConfig(
                    provider=Provider.AZURE_OPENAI,
                    base_url="https://my-resource.openai.azure.com/openai/deployments/d",
                    api_key="azure-test",
                    api_version="2024-02-01",
                    model="gpt-4o",
                )
            ]
        },
    )
    router = ModelRouter(config)
    gateways = router._gateways()
    assert len(gateways) == 1
    assert isinstance(gateways[0], AzureOpenAIGateway)


def test_router_instantiates_openrouter_backend(project_root: Path) -> None:
    config = AppConfig(
        project_root=project_root,
        model={
            "backends": [
                ModelBackendConfig(
                    provider=Provider.OPENROUTER,
                    base_url="https://openrouter.ai/api/v1",
                    api_key="sk-or-test",
                    model="anthropic/claude-3.5-sonnet",
                )
            ]
        },
    )
    router = ModelRouter(config)
    gateways = router._gateways()
    assert len(gateways) == 1
    assert isinstance(gateways[0], OpenRouterGateway)


def test_router_raises_for_unsupported_provider(project_root: Path) -> None:
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config)
    with (
        patch.dict(
            "autoship.core.model_router._PROVIDER_GATEWAYS",
            {},
            clear=True,
        ),
        pytest.raises(ModelGatewayError, match="Unsupported model backend provider"),
    ):
        router._gateways()


def test_select_backend_returns_first_healthy_backend(project_root: Path) -> None:
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config)
    gateway = MagicMock()
    gateway.health.return_value = True
    with patch.object(router, "_gateways", return_value=[gateway]):
        result = router.select_backend()
    assert result is gateway


def test_select_backend_filters_by_tier(project_root: Path) -> None:
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config)
    low = MagicMock()
    low.cfg.tier = 1
    low.health.return_value = True
    high = MagicMock()
    high.cfg.tier = 3
    high.health.return_value = True
    with patch.object(router, "_gateways", return_value=[low, high]):
        assert router.select_backend(tier=3) is high
        assert router.select_backend(tier=1) is low


def test_select_backend_falls_back_to_other_tiers(project_root: Path) -> None:
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config)
    low = MagicMock()
    low.cfg.tier = 1
    low.health.return_value = False
    high = MagicMock()
    high.cfg.tier = 3
    high.health.return_value = True
    with patch.object(router, "_gateways", return_value=[low, high]):
        assert router.select_backend(tier=1) is high


def test_select_backend_honors_fallback_disabled(project_root: Path) -> None:
    config = AppConfig(
        project_root=project_root, model={"backends": [_backend()], "fallback": False}
    )
    router = ModelRouter(config)
    low = MagicMock()
    low.cfg.tier = 1
    low.health.return_value = False
    high = MagicMock()
    high.cfg.tier = 3
    high.health.return_value = True
    with patch.object(router, "_gateways", return_value=[low, high]):
        assert router.select_backend(tier=1) is None


def test_health_check_caches_within_ttl(project_root: Path) -> None:
    """Health() should be called at most once per TTL window."""
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config, health_ttl=60.0)
    gateway = MagicMock()
    gateway.health.return_value = True
    gateway.chat.return_value = ChatCompletionResponse(content="ok", model="llama3")
    with patch.object(router, "_gateways", return_value=[gateway]):
        router.chat([ChatMessage(role="user", content="hi")], "test")
        router.chat([ChatMessage(role="user", content="hi2")], "test")
    # health() should be called once (cached for the second chat)
    assert gateway.health.call_count == 1


def test_health_check_rechecks_after_ttl_expires(project_root: Path) -> None:
    """Health() should be re-checked after the TTL expires."""
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config, health_ttl=0.0)
    gateway = MagicMock()
    gateway.health.return_value = True
    gateway.chat.return_value = ChatCompletionResponse(content="ok", model="llama3")
    with patch.object(router, "_gateways", return_value=[gateway]):
        router.chat([ChatMessage(role="user", content="hi")], "test")
        router.chat([ChatMessage(role="user", content="hi2")], "test")
    # With TTL=0, each call should re-probe
    assert gateway.health.call_count == 2


def test_invalidate_health_cache_clears_cache(project_root: Path) -> None:
    """invalidate_health_cache() forces a fresh health probe."""
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config, health_ttl=60.0)
    gateway = MagicMock()
    gateway.health.return_value = True
    gateway.chat.return_value = ChatCompletionResponse(content="ok", model="llama3")
    with patch.object(router, "_gateways", return_value=[gateway]):
        router.chat([ChatMessage(role="user", content="hi")], "test")
        router.invalidate_health_cache()
        router.chat([ChatMessage(role="user", content="hi2")], "test")
    assert gateway.health.call_count == 2


def test_chat_invalidates_cache_on_error(project_root: Path) -> None:
    """Cache entry is dropped when chat() raises, so next call re-probes."""
    config = AppConfig(project_root=project_root, model={"backends": [_backend()]})
    router = ModelRouter(config, health_ttl=60.0)
    gateway = MagicMock()
    gateway.health.return_value = True
    gateway.chat.side_effect = ModelGatewayError("boom")
    with (
        patch.object(router, "_gateways", return_value=[gateway]),
        pytest.raises(ModelGatewayError, match="All model backends unhealthy"),
    ):
        router.chat([ChatMessage(role="user", content="hi")], "test")
    # Cache should have been invalidated for this gateway
    assert router._health_cache == {}
