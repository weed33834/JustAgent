"""Tests for the official plugin registry index."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from autoship.core.registry_index import RegistryIndex

SAMPLE_REGISTRY = {
    "version": 1,
    "plugins": [
        {
            "name": "security-scan",
            "package": "autoship",
            "version": "0.2.0",
            "description": "Security scanning plugin.",
            "trust_level": "builtin",
        },
        {
            "name": "docker-ship",
            "package": "autoship",
            "version": "0.2.0",
            "description": "Docker build and push plugin.",
            "trust_level": "builtin",
        },
    ],
}


@pytest.fixture(autouse=True)
def _clear_process_cache() -> None:
    """Ensure each test starts with an empty process cache."""
    RegistryIndex.invalidate_process_cache()
    yield
    RegistryIndex.invalidate_process_cache()


def test_registry_index_loads_plugins() -> None:
    with patch("autoship.core.registry_index.RegistryClient") as mock_client:
        mock_client.return_value.get.return_value = SAMPLE_REGISTRY
        index = RegistryIndex()
        plugins = index.list_plugins()
    assert len(plugins) == 2


def test_registry_index_search_by_name() -> None:
    with patch("autoship.core.registry_index.RegistryClient") as mock_client:
        mock_client.return_value.get.return_value = SAMPLE_REGISTRY
        index = RegistryIndex()
        results = index.search("docker")
    assert len(results) == 1
    assert results[0]["name"] == "docker-ship"


def test_registry_index_search_by_description() -> None:
    with patch("autoship.core.registry_index.RegistryClient") as mock_client:
        mock_client.return_value.get.return_value = SAMPLE_REGISTRY
        index = RegistryIndex()
        results = index.search("scanning")
    assert len(results) == 1
    assert results[0]["name"] == "security-scan"


def test_registry_index_get_by_name() -> None:
    with patch("autoship.core.registry_index.RegistryClient") as mock_client:
        mock_client.return_value.get.return_value = SAMPLE_REGISTRY
        index = RegistryIndex()
        plugin = index.get("security-scan")
    assert plugin is not None
    assert plugin["package"] == "autoship"


def test_registry_index_get_missing() -> None:
    with patch("autoship.core.registry_index.RegistryClient") as mock_client:
        mock_client.return_value.get.return_value = SAMPLE_REGISTRY
        index = RegistryIndex()
    assert index.get("not-found") is None


def test_registry_index_empty() -> None:
    with patch("autoship.core.registry_index.RegistryClient") as mock_client:
        mock_client.return_value.get.return_value = {"version": 1, "plugins": []}
        index = RegistryIndex()
        plugins = index.list_plugins()
    assert plugins == []


def test_process_cache_shares_data_across_instances() -> None:
    """A second RegistryIndex instance should reuse cached data without re-fetching."""
    with patch("autoship.core.registry_index.RegistryClient") as mock_client:
        mock_client.return_value.get.return_value = SAMPLE_REGISTRY
        mock_client.return_value.config.url = "https://example.com/registry.json"
        mock_client.return_value.config.cache_ttl_seconds = 300

        index1 = RegistryIndex()
        index1.load()
        assert mock_client.return_value.get.call_count == 1

        # Second instance with same config should hit the process cache
        index2 = RegistryIndex()
        index2.load()
        assert mock_client.return_value.get.call_count == 1  # still 1, not 2


def test_no_cache_bypasses_process_cache() -> None:
    """no_cache=True should bypass the process cache and re-fetch."""
    with patch("autoship.core.registry_index.RegistryClient") as mock_client:
        mock_client.return_value.get.return_value = SAMPLE_REGISTRY
        mock_client.return_value.config.url = "https://example.com/registry.json"
        mock_client.return_value.config.cache_ttl_seconds = 300

        index1 = RegistryIndex()
        index1.load()
        assert mock_client.return_value.get.call_count == 1

        index2 = RegistryIndex()
        index2.load(no_cache=True)
        assert mock_client.return_value.get.call_count == 2


def test_invalidate_process_cache_forces_refetch() -> None:
    """invalidate_process_cache() should force the next load to re-fetch."""
    with patch("autoship.core.registry_index.RegistryClient") as mock_client:
        mock_client.return_value.get.return_value = SAMPLE_REGISTRY
        mock_client.return_value.config.url = "https://example.com/registry.json"
        mock_client.return_value.config.cache_ttl_seconds = 300

        index1 = RegistryIndex()
        index1.load()
        assert mock_client.return_value.get.call_count == 1

        RegistryIndex.invalidate_process_cache()

        index2 = RegistryIndex()
        index2.load()
        assert mock_client.return_value.get.call_count == 2
