"""Integration tests for the Ollama model backend.

These tests require a running Ollama instance on ``OLLAMA_BASE_URL``
(default ``http://localhost:11434``) and the model configured by
``OLLAMA_MODEL`` (default ``llama3``) to be available. When the backend
is unreachable, every test is skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoship.adapters.providers import OllamaGateway
from autoship.core.model_router import ModelRouter
from autoship.exceptions import ModelGatewayError
from autoship.models.config import Provider

from .conftest import (
    app_config_with_backend,
    backend_config,
    ollama_available,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def ollama_backend(ollama_base_url: str, ollama_model: str):
    """Return a configured Ollama backend config."""
    return backend_config(
        Provider.OLLAMA,
        ollama_base_url,
        ollama_model,
        timeout=30.0,
    )


@pytest.fixture
def ollama_gateway(ollama_backend):
    """Return a live OllamaGateway instance."""
    return OllamaGateway(ollama_backend)


@pytest.fixture(autouse=True)
def _skip_if_ollama_missing(ollama_base_url: str):
    """Skip tests when Ollama is not reachable."""
    if not ollama_available(ollama_base_url):
        pytest.skip(f"Ollama is not reachable at {ollama_base_url}")


def test_ollama_backend_is_healthy(ollama_gateway: OllamaGateway) -> None:
    """A live Ollama instance reports healthy."""
    assert ollama_gateway.health() is True


def test_ollama_list_models_includes_configured_model(
    ollama_gateway: OllamaGateway, ollama_model: str
) -> None:
    """``list_models`` returns the configured model if it is pulled."""
    models = ollama_gateway.list_models()
    assert ollama_model in models


def test_ollama_chat_returns_content(ollama_gateway: OllamaGateway, chat_messages) -> None:
    """A simple chat request returns non-empty content."""
    from autoship.adapters.model_gateway import ChatCompletionRequest

    request = ChatCompletionRequest(messages=chat_messages, max_tokens=64)
    response = ollama_gateway.chat(request)

    assert response.content
    assert isinstance(response.content, str)
    assert response.model


def test_ollama_chat_missing_model_raises(
    ollama_base_url: str,
) -> None:
    """Requesting a model that does not exist raises ``ModelGatewayError``."""
    cfg = backend_config(
        Provider.OLLAMA,
        ollama_base_url,
        "definitely-not-a-real-model-12345",
        timeout=10.0,
    )
    gateway = OllamaGateway(cfg)

    from autoship.adapters.model_gateway import ChatCompletionRequest, ChatMessage

    with pytest.raises(ModelGatewayError):
        gateway.chat(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hello")],
                max_tokens=16,
            )
        )


def test_model_router_selects_ollama_backend(project_root: Path, ollama_backend) -> None:
    """``ModelRouter.select_backend`` returns the Ollama gateway when healthy."""
    config = app_config_with_backend(project_root, ollama_backend)
    router = ModelRouter(config)

    selected = router.select_backend()

    assert selected is not None
    assert isinstance(selected, OllamaGateway)


def test_model_router_chat_through_ollama(
    project_root: Path, ollama_backend, chat_messages
) -> None:
    """The full router chat pipeline works against a live Ollama backend."""
    config = app_config_with_backend(project_root, ollama_backend)
    router = ModelRouter(config)

    content = router.chat(chat_messages, "integration-test")

    assert isinstance(content, str)
    assert content.strip()


def test_unreachable_ollama_reports_unhealthy(project_root: Path) -> None:
    """A deliberately wrong URL reports unhealthy."""
    cfg = backend_config(
        Provider.OLLAMA,
        "http://127.0.0.1:1",
        "llama3",
        timeout=1.0,
    )
    gateway = OllamaGateway(cfg)

    assert gateway.health() is False


def test_unreachable_ollama_router_raises(project_root: Path, chat_messages) -> None:
    """Router surfaces an error when Ollama is unreachable and fallback is off."""
    cfg = backend_config(
        Provider.OLLAMA,
        "http://127.0.0.1:1",
        "llama3",
        timeout=1.0,
    )
    config = app_config_with_backend(project_root, cfg, fallback=False)
    router = ModelRouter(config)

    with pytest.raises(ModelGatewayError, match="All model backends"):
        router.chat(chat_messages, "integration-test")
