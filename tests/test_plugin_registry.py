"""Tests for PluginRegistry."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from autoship.core.plugin_registry import PluginRegistry, PluginSpec, TrustLevel


@pytest.fixture
def registry(tmp_path: Path) -> PluginRegistry:
    """Return a PluginRegistry backed by a temporary directory."""
    return PluginRegistry(registry_dir=tmp_path)


def test_add_and_list(registry: PluginRegistry) -> None:
    spec = PluginSpec(name="my-plugin", source="pypi")
    registry.add(spec)
    plugins = registry.list()
    assert len(plugins) == 1
    assert plugins[0].name == "my-plugin"


def test_get_and_trust(registry: PluginRegistry) -> None:
    registry.add(PluginSpec(name="a", source="local"))
    assert registry.trust("a", TrustLevel.VERIFIED) is True
    plugin = registry.get("a")
    assert plugin is not None
    assert plugin.trust_level == TrustLevel.VERIFIED


def test_remove(registry: PluginRegistry) -> None:
    registry.add(PluginSpec(name="b", source="local"))
    assert registry.remove("b") is True
    assert registry.remove("b") is False


def test_persistence(tmp_path: Path) -> None:
    registry = PluginRegistry(registry_dir=tmp_path)
    registry.add(PluginSpec(name="persist", source="git"))

    registry2 = PluginRegistry(registry_dir=tmp_path)
    assert registry2.get("persist") is not None


def test_registry_has_restrictive_permissions(tmp_path: Path) -> None:
    """Plugin registry directory and file are only owner-readable/writable."""
    registry = PluginRegistry(registry_dir=tmp_path)
    registry.add(PluginSpec(name="secure", source="local"))

    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert registry.registry_file.exists()
    assert stat.S_IMODE(registry.registry_file.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_trust_unknown_plugin_returns_false(registry: PluginRegistry) -> None:
    assert registry.trust("does-not-exist", TrustLevel.VERIFIED) is False


def test_capability_summary_with_network_and_shell() -> None:
    from autoship.core.plugin_registry import CapabilityManifest

    cap = CapabilityManifest(network=True, shell=True)
    summary = cap.summary()
    assert "network=yes" in summary
    assert "shell=yes" in summary


def test_capability_summary_without_network_and_shell() -> None:
    from autoship.core.plugin_registry import CapabilityManifest

    cap = CapabilityManifest(network=False, shell=False)
    summary = cap.summary()
    assert "network=yes" not in summary
    assert "shell=yes" not in summary


def test_capability_summary_with_env() -> None:
    from autoship.core.plugin_registry import CapabilityManifest

    cap = CapabilityManifest(env=["HOME", "PATH"])
    summary = cap.summary()
    assert "env=HOME,PATH" in summary


def test_load_corrupted_registry_file(tmp_path: Path) -> None:
    """Registry gracefully handles a file with invalid JSON."""
    registry_dir = tmp_path / "corrupt_registry"
    registry_dir.mkdir(parents=True)
    (registry_dir / "registry.json").write_text("this is not valid json {{{")

    registry = PluginRegistry(registry_dir=registry_dir)
    # Should not crash; the registry should be empty.
    assert registry.list() == []


def test_load_skips_invalid_plugin_entries(tmp_path: Path) -> None:
    """Registry skips entries that fail validation, keeps valid ones."""
    registry_dir = tmp_path / "partial_registry"
    registry_dir.mkdir(parents=True)
    (registry_dir / "registry.json").write_text(
        '{"plugins": [{"name": "good", "source": "pypi"}, {"bad": true}]}'
    )

    registry = PluginRegistry(registry_dir=registry_dir)
    plugins = registry.list()
    assert len(plugins) == 1
    assert plugins[0].name == "good"
