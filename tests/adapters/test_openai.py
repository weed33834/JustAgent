"""Tests for the OpenAI model gateway adapter."""

from __future__ import annotations

import httpx
import pytest
import respx

from autoship.adapters.model_gateway import ChatCompletionRequest, ChatMessage
from autoship.adapters.providers.openai import OpenAIGateway
from autoship.exceptions import ModelGatewayError
from autoship.models.config import ModelBackendConfig, Provider

BASE_URL = "https://api.openai.com/v1"


def _gateway(api_key: str | None = "sk-test") -> OpenAIGateway:
    cfg = ModelBackendConfig(
        provider=Provider.OPENAI,
        base_url=BASE_URL,
        api_key=api_key,
        model="gpt-4o-mini",
        timeout=5.0,
    )
    return OpenAIGateway(cfg)


def test_default_base_url() -> None:
    assert OpenAIGateway.DEFAULT_BASE_URL == "https://api.openai.com/v1"


def test_health_returns_true_when_models_endpoint_ok() -> None:
    with respx.mock:
        route = respx.get(f"{BASE_URL}/models").respond(200)
        assert _gateway().health() is True
        assert route.called


def test_health_returns_false_on_error() -> None:
    with respx.mock:
        respx.get(f"{BASE_URL}/models").respond(500)
        assert _gateway().health() is False


def test_health_returns_false_on_connection_error() -> None:
    with respx.mock:
        respx.get(f"{BASE_URL}/models").mock(side_effect=httpx.ConnectError("refused"))
        assert _gateway().health() is False


def test_health_raises_when_api_key_missing() -> None:
    with pytest.raises(ModelGatewayError, match="requires an API key"):
        _gateway(api_key=None).health()


def test_chat_raises_when_api_key_missing() -> None:
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ModelGatewayError, match="requires an API key"):
        _gateway(api_key=None).chat(req)


def test_chat() -> None:
    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").respond(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.5,
            max_tokens=100,
        )
        resp = _gateway().chat(req)
    assert resp.content == "hello"
    assert resp.model == "gpt-4o-mini"
    assert resp.usage == {"prompt_tokens": 5, "completion_tokens": 2}


def test_chat_request_payload() -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/chat/completions").respond(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": "ok"}}],
            },
        )
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.5,
            max_tokens=100,
        )
        _gateway().chat(req)
        sent = route.calls.last.request.content
        assert b'"temperature":0.5' in sent
        assert b'"max_tokens":100' in sent
        assert b'"stream":false' in sent


def test_api_key_header() -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/chat/completions").respond(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": "ok"}}],
            },
        )
        _gateway(api_key="secret").chat(
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        )
        assert route.calls.last.request.headers["Authorization"] == "Bearer secret"
