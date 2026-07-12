"""Tests for the Ollama model gateway adapter."""

from __future__ import annotations

import httpx
import pytest
import respx

from autoship.adapters.model_gateway import ChatCompletionRequest, ChatMessage
from autoship.adapters.providers.ollama import OllamaGateway
from autoship.models.config import ModelBackendConfig, Provider


def _gateway() -> OllamaGateway:
    cfg = ModelBackendConfig(
        provider=Provider.OLLAMA,
        base_url="http://localhost:11434/v1",
        model="llama3",
        timeout=5.0,
    )
    return OllamaGateway(cfg)


def test_health_returns_true_when_models_endpoint_ok() -> None:
    with respx.mock:
        route = respx.get("http://localhost:11434/v1/models").respond(200)
        assert _gateway().health() is True
        assert route.called


def test_health_returns_false_on_error() -> None:
    with respx.mock:
        respx.get("http://localhost:11434/v1/models").respond(500)
        assert _gateway().health() is False


def test_health_returns_false_on_connection_error() -> None:
    with respx.mock:
        respx.get("http://localhost:11434/v1/models").mock(
            side_effect=httpx.ConnectError("refused")
        )
        assert _gateway().health() is False


def test_list_models() -> None:
    with respx.mock:
        respx.get("http://localhost:11434/v1/models").respond(
            200, json={"data": [{"id": "llama3"}, {"id": "phi3"}]}
        )
        models = _gateway().list_models()
    assert models == ["llama3", "phi3"]


def test_chat() -> None:
    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").respond(
            200,
            json={
                "model": "llama3",
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
    assert resp.model == "llama3"
    assert resp.usage == {"prompt_tokens": 5, "completion_tokens": 2}


def test_chat_request_payload() -> None:
    with respx.mock:
        route = respx.post("http://localhost:11434/v1/chat/completions").respond(
            200,
            json={
                "model": "llama3",
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


def test_chat_raises_on_empty_choices() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").respond(
            200, json={"model": "llama3", "choices": []}
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError, match="Unexpected response structure"):
            _gateway().chat(req)


def test_chat_raises_on_missing_choices() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").respond(
            200, json={"model": "llama3"}
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError, match="Unexpected response structure"):
            _gateway().chat(req)


def test_chat_raises_on_malformed_message() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").respond(
            200, json={"model": "llama3", "choices": [{"message": {}}]}
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError, match="Unexpected response structure"):
            _gateway().chat(req)


def test_chat_raises_on_invalid_json() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").respond(200, text="not json")
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError):
            _gateway().chat(req)


def test_chat_raises_on_server_error() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").respond(500)
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError):
            _gateway().chat(req)


def test_chat_raises_on_timeout() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ModelGatewayError):
            _gateway().chat(req)


def test_list_models_raises_on_invalid_structure() -> None:
    from autoship.exceptions import ModelGatewayError

    with respx.mock:
        respx.get("http://localhost:11434/v1/models").respond(200, json={"models": []})
        with pytest.raises(ModelGatewayError, match="Unexpected response structure"):
            _gateway().list_models()
