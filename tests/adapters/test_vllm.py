"""Tests for the vLLM model gateway adapter."""

from __future__ import annotations

import httpx
import pytest
import respx

from autoship.adapters.model_gateway import ChatCompletionRequest, ChatMessage
from autoship.adapters.providers.vllm import VllmGateway
from autoship.exceptions import ModelGatewayError
from autoship.models.config import ModelBackendConfig, Provider

BASE_URL = "http://localhost:8000/v1"
HEALTH_URL = "http://localhost:8000/health"


def _gateway() -> VllmGateway:
    cfg = ModelBackendConfig(
        provider=Provider.VLLM,
        base_url=BASE_URL,
        model="qwen2.5:7b",
        timeout=5.0,
    )
    return VllmGateway(cfg)


def test_default_base_url() -> None:
    assert VllmGateway.DEFAULT_BASE_URL == "http://127.0.0.1:8000/v1"


def test_health_uses_root_health_endpoint() -> None:
    with respx.mock:
        route = respx.get(HEALTH_URL).respond(200)
        assert _gateway().health() is True
        assert route.called


def test_health_returns_false_on_root_health_error() -> None:
    with respx.mock:
        respx.get(HEALTH_URL).respond(503)
        assert _gateway().health() is False


def test_health_returns_false_on_connection_error() -> None:
    with respx.mock:
        respx.get(HEALTH_URL).mock(side_effect=httpx.ConnectError("refused"))
        assert _gateway().health() is False


def test_list_models() -> None:
    with respx.mock:
        respx.get(f"{BASE_URL}/models").respond(
            200, json={"data": [{"id": "qwen2.5:7b"}, {"id": "llama3"}]}
        )
        models = _gateway().list_models()
    assert models == ["qwen2.5:7b", "llama3"]


def test_chat() -> None:
    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").respond(
            200,
            json={
                "model": "qwen2.5:7b",
                "choices": [{"message": {"content": "hello"}}],
            },
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        resp = _gateway().chat(req)
    assert resp.content == "hello"
    assert resp.model == "qwen2.5:7b"


def test_chat_request_payload() -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/chat/completions").respond(
            200,
            json={
                "model": "qwen2.5:7b",
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


def test_chat_raises_on_server_error() -> None:
    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").respond(500)
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError):
            _gateway().chat(req)
