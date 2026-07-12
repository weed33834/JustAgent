"""Route tasks to local model backends with fallback — uses unified openai SDK gateway."""

from __future__ import annotations

import contextlib
import threading
import time

import httpx

from autoship.adapters.model_gateway import ChatCompletionRequest, ChatMessage, ModelGateway
from autoship.adapters.providers import UnifiedGateway
from autoship.core.metrics import get_registry
from autoship.exceptions import ModelGatewayError
from autoship.models.config import AppConfig

_DEFAULT_HEALTH_TTL: float = 30.0


class ModelRouter:
    """Select a model backend and tier for a given task."""

    def __init__(self, config: AppConfig, *, health_ttl: float = _DEFAULT_HEALTH_TTL) -> None:
        self.config = config
        self._health_ttl = health_ttl
        self._health_cache: dict[tuple[str, str], tuple[bool, float]] = {}
        self._health_lock: threading.Lock = threading.Lock()
        self._gateway_instances: list[ModelGateway] | None = None

    def _build_gateways(self) -> list[ModelGateway]:
        return [UnifiedGateway(backend) for backend in self.config.model.backends]

    def _gateways(self) -> list[ModelGateway]:
        if self._gateway_instances is None:
            with self._health_lock:
                if self._gateway_instances is None:
                    self._gateway_instances = self._build_gateways()
        return self._gateway_instances

    def close(self) -> None:
        if self._gateway_instances is None:
            return
        for gateway in self._gateway_instances:
            with contextlib.suppress(Exception):
                gateway.close()
        self._gateway_instances = None

    def __enter__(self) -> ModelRouter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _health_key(self, gateway: ModelGateway) -> tuple[str, str]:
        return (gateway.cfg.provider.value, str(gateway.cfg.base_url))

    def _check_health(self, gateway: ModelGateway) -> bool:
        key = self._health_key(gateway)
        now = time.monotonic()
        cached = self._health_cache.get(key)
        if cached is not None:
            is_healthy, checked_at = cached
            ttl = self._health_ttl if is_healthy else self._health_ttl / 3
            if now - checked_at < ttl:
                return is_healthy
        with self._health_lock:
            cached = self._health_cache.get(key)
            if cached is not None:
                is_healthy, checked_at = cached
                ttl = self._health_ttl if is_healthy else self._health_ttl / 3
                if time.monotonic() - checked_at < ttl:
                    return is_healthy
            try:
                healthy = gateway.health()
            except (ModelGatewayError, httpx.RequestError, httpx.TimeoutException):
                healthy = False
            self._health_cache[key] = (healthy, time.monotonic())
            return healthy

    def invalidate_health_cache(self) -> None:
        with self._health_lock:
            self._health_cache.clear()

    def _chat(self, messages: list[ChatMessage], task_type: str) -> str:
        registry = get_registry()
        gateways = self._gateways()
        if not gateways:
            raise ModelGatewayError("No model backends configured")

        last_error: Exception | None = None
        attempts = 0
        for gateway in gateways:
            attempts += 1
            try:
                if self._check_health(gateway):
                    req = ChatCompletionRequest(messages=messages)
                    resp = gateway.chat(req)
                    registry.inc("model_backend_success")
                    if attempts > 1:
                        registry.inc("model_backend_fallbacks")
                    return resp.content
            except (ModelGatewayError, httpx.RequestError, httpx.TimeoutException) as exc:
                registry.inc("model_backend_errors")
                last_error = exc
                with self._health_lock:
                    self._health_cache.pop(self._health_key(gateway), None)
                if not self.config.model.fallback:
                    break
                continue

        if last_error:
            raise ModelGatewayError(f"All model backends unhealthy: {last_error}") from last_error
        raise ModelGatewayError("All model backends are unhealthy")

    def select_backend(self, tier: int | None = None) -> ModelGateway | None:
        gateways = self._gateways()
        if tier is not None:
            tier_gateways = [g for g in gateways if g.cfg.tier == tier]
            for gateway in tier_gateways:
                if self._check_health(gateway):
                    return gateway
            if not self.config.model.fallback:
                return None
            gateways = [g for g in gateways if g.cfg.tier != tier]
        for gateway in gateways:
            if self._check_health(gateway):
                return gateway
        return None

    def chat(self, messages: list[ChatMessage], task_type: str) -> str:
        return self._chat(messages, task_type)

    def generate_commit_message(self, diff: str, stats: str) -> str:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are an expert software engineer writing concise Git commit messages. "
                    "Use conventional commits format: type(scope): subject. "
                    "Types: feat, fix, refactor, docs, test, chore. "
                    "Keep the subject under 72 characters."
                ),
            ),
            ChatMessage(role="user", content=f"Git stats:\n{stats}\n\nDiff:\n{diff[:8000]}"),
        ]
        return self.chat(messages, "commit").strip()
