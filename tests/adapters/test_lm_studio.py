"""Tests for the LM Studio model gateway adapter."""

from __future__ import annotations

import httpx
import pytest
import respx

from autoship.adapters.model_gateway import ChatCompletionRequest, ChatMessage
from autoship.adapters.providers.lm_studio import LmStudioGateway
from autoship.models.config import ModelBackendConfig, Provider

BASE_URL = "http://localhost:1234/v1"


def _gateway(api_key: str | None = None) -> LmStudioGateway:
    cfg = ModelBackendConfig(
        provider=Provider.LM_STUDIO,
        base_url=BASE_URL,
        model="qwen2.5:7b",
        timeout=5.0,
        api_key=api_key,
    )
    return LmStudioGateway(cfg)


def test_default_base_url() -> None:
    assert LmStudioGateway.DEFAULT_BASE_URL == "http://127.0.0.1:1234/v1"


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


def test_list_models() -> None:
    with respx.mock:
        respx.get(f"{BASE_URL}/models").respond(
            200, json={"data": [{"id": "qwen2.5:7b"}, {"id": "phi3"}]}
        )
        models = _gateway().list_models()
    assert models == ["qwen2.5:7b", "phi3"]


def test_chat() -> None:
    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").respond(
            200,
            json={
                "model": "qwen2.5:7b",
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
    assert resp.model == "qwen2.5:7b"
    assert resp.usage == {"prompt_tokens": 5, "completion_tokens": 2}


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


def test_api_key_header() -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/chat/completions").respond(
            200,
            json={
                "model": "qwen2.5:7b",
                "choices": [{"message": {"content": "ok"}}],
            },
        )
        _gateway(api_key="secret").chat(
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        )
        assert route.calls.last.request.headers["Authorization"] == "Bearer secret"


def test_chat_raises_on_empty_choices() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").respond(
            200, json={"model": "qwen2.5:7b", "choices": []}
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError, match="Unexpected response structure"):
            _gateway().chat(req)


def test_chat_raises_on_missing_choices() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").respond(200, json={"model": "qwen2.5:7b"})
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError, match="Unexpected response structure"):
            _gateway().chat(req)


def test_chat_raises_on_malformed_message() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").respond(
            200, json={"model": "qwen2.5:7b", "choices": [{"message": {}}]}
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError, match="Unexpected response structure"):
            _gateway().chat(req)


def test_chat_raises_on_invalid_json() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").respond(200, text="not json")
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError):
            _gateway().chat(req)


def test_chat_raises_on_server_error() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").respond(500)
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError):
            _gateway().chat(req)


def test_chat_raises_on_timeout() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError):
            _gateway().chat(req)


def test_list_models_raises_on_invalid_structure() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.get(f"{BASE_URL}/models").respond(200, json={"models": []})
        with pytest.raises(ModelGatewayError, match="Unexpected response structure"):
            _gateway().list_models()


@pytest.mark.asyncio
async def test_achat() -> None:
    with respx.mock:
        respx.post(f"{BASE_URL}/chat/completions").respond(
            200,
            json={
                "model": "qwen2.5:7b",
                "choices": [{"message": {"content": "hello async"}}],
            },
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        resp = await _gateway().achat(req)
    assert resp.content == "hello async"
    assert resp.model == "qwen2.5:7b"
