"""Integration tests for the LM Studio OpenAI-compatible backend.

These tests require a running LM Studio server on ``LM_STUDIO_BASE_URL``
(default ``http://localhost:1234/v1``) and the model configured by
``LM_STUDIO_MODEL`` (default ``local-model``) to be loaded. When the
backend is unreachable, every test is skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoship.adapters.model_gateway import ChatCompletionRequest, ChatMessage
from autoship.adapters.providers import LmStudioGateway
from autoship.core.model_router import ModelRouter
from autoship.exceptions import ModelGatewayError
from autoship.models.config import Provider

from .conftest import (
    app_config_with_backend,
    backend_config,
    lm_studio_available,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def lm_studio_backend(lm_studio_base_url: str, lm_studio_model: str, lm_studio_api_key: str | None):
    """Return a configured LM Studio backend config."""
    return backend_config(
        Provider.LM_STUDIO,
        lm_studio_base_url,
        lm_studio_model,
        api_key=lm_studio_api_key,
        timeout=30.0,
    )


@pytest.fixture
def lm_studio_gateway(lm_studio_backend):
    """Return a live LmStudioGateway instance."""
    return LmStudioGateway(lm_studio_backend)


@pytest.fixture(autouse=True)
def _skip_if_lm_studio_missing(lm_studio_base_url: str):
    """Skip tests when LM Studio is not reachable."""
    if not lm_studio_available(lm_studio_base_url):
        pytest.skip(f"LM Studio is not reachable at {lm_studio_base_url}")


def test_lm_studio_backend_is_healthy(lm_studio_gateway: LmStudioGateway) -> None:
    """A live LM Studio server reports healthy."""
    assert lm_studio_gateway.health() is True


def test_lm_studio_list_models_includes_configured_model(
    lm_studio_gateway: LmStudioGateway, lm_studio_model: str
) -> None:
    """``list_models`` returns the configured model if it is loaded."""
    models = lm_studio_gateway.list_models()
    assert lm_studio_model in models


def test_lm_studio_chat_returns_content(lm_studio_gateway: LmStudioGateway, chat_messages) -> None:
    """A simple chat request returns non-empty content."""
    request = ChatCompletionRequest(messages=chat_messages, max_tokens=64)
    response = lm_studio_gateway.chat(request)

    assert response.content
    assert isinstance(response.content, str)
    assert response.model


def test_lm_studio_chat_missing_model_raises(
    lm_studio_base_url: str, lm_studio_api_key: str | None
) -> None:
    """Requesting a model that is not loaded raises ``ModelGatewayError``."""
    cfg = backend_config(
        Provider.LM_STUDIO,
        lm_studio_base_url,
        "definitely-not-a-real-model-12345",
        api_key=lm_studio_api_key,
        timeout=10.0,
    )
    gateway = LmStudioGateway(cfg)

    with pytest.raises(ModelGatewayError):
        gateway.chat(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hello")],
                max_tokens=16,
            )
        )


def test_model_router_selects_lm_studio_backend(project_root: Path, lm_studio_backend) -> None:
    """``ModelRouter.select_backend`` returns the LM Studio gateway when healthy."""
    config = app_config_with_backend(project_root, lm_studio_backend)
    router = ModelRouter(config)

    selected = router.select_backend()

    assert selected is not None
    assert isinstance(selected, LmStudioGateway)


def test_model_router_chat_through_lm_studio(
    project_root: Path, lm_studio_backend, chat_messages
) -> None:
    """The full router chat pipeline works against a live LM Studio backend."""
    config = app_config_with_backend(project_root, lm_studio_backend)
    router = ModelRouter(config)

    content = router.chat(chat_messages, "integration-test")

    assert isinstance(content, str)
    assert content.strip()


def test_unreachable_lm_studio_reports_unhealthy(project_root: Path) -> None:
    """A deliberately wrong URL reports unhealthy."""
    cfg = backend_config(
        Provider.LM_STUDIO,
        "http://localhost:59999/v1",
        "local-model",
        timeout=1.0,
    )
    gateway = LmStudioGateway(cfg)

    assert gateway.health() is False


def test_unreachable_lm_studio_router_raises(project_root: Path, chat_messages) -> None:
    """Router surfaces an error when LM Studio is unreachable and fallback is off."""
    cfg = backend_config(
        Provider.LM_STUDIO,
        "http://localhost:59999/v1",
        "local-model",
        timeout=1.0,
    )
    config = app_config_with_backend(project_root, cfg, fallback=False)
    router = ModelRouter(config)

    with pytest.raises(ModelGatewayError, match="All model backends"):
        router.chat(chat_messages, "integration-test")
