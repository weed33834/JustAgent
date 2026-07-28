"""Model backend provider — unified LiteLLM gateway."""

from __future__ import annotations

from justagent.adapters.providers.unified_gateway import (
    UnifiedGateway,
    format_provider_error,
)

__all__ = [
    "UnifiedGateway",
    "format_provider_error",
]
