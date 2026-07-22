"""Tests for the per-entry plugin validator (scripts/validate_plugin_entry.py)."""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any, cast

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "validate_plugin_entry.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("validate_plugin_entry", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> Any:
    return _load_module()


def _valid_builtin() -> dict[str, Any]:
    return {
        "name": "my-plugin",
        "package": "myagent-my-plugin",
        "module": "myagent_my_plugin.plugin",
        "version": "1.0.0",
        "description": "A test plugin.",
        "trust_level": "builtin",
        "entry_point": "myagent_my_plugin.plugin:MyPlugin",
        "hooks": ["pre_commit"],
        "publisher": {"id": "myagent-team", "verified": True, "url": "https://myagent.dev"},
        "maintainer": "MyAgent Team",
        "license": "MIT",
        "permissions": {
            "filesystem": "read-only",
            "network": False,
            "shell": False,
            "git": False,
            "env": [],
        },
        "categories": ["quality"],
        "audit_status": "approved",
    }


def _publisher_index() -> dict[str, dict[str, Any]]:
    return {
        "myagent-team": {"id": "myagent-team", "verified": True},
        "alice-chen": {"id": "alice-chen", "verified": True},
    }


def _run(mod: Any, plugin: dict[str, Any]) -> int:
    return mod.validate_plugin(
        cast(dict[str, Any], plugin),
        1,
        set(),
        {},
        _publisher_index(),
    )


def test_valid_builtin_plugin_has_no_errors(mod: Any) -> None:
    assert _run(mod, _valid_builtin()) == 0


def test_verified_plugin_requires_sha256(mod: Any) -> None:
    plugin = _valid_builtin()
    plugin["name"] = "verified-no-sha"
    plugin["trust_level"] = "verified"
    plugin["publisher"] = {"id": "alice-chen", "verified": True}
    plugin.pop("sha256", None)
    plugin.pop("signature", None)
    assert _run(mod, plugin) >= 1


def test_verified_plugin_requires_signature(mod: Any) -> None:
    plugin = _valid_builtin()
    plugin["name"] = "verified-no-sig"
    plugin["trust_level"] = "verified"
    plugin["publisher"] = {"id": "alice-chen", "verified": True}
    plugin["sha256"] = "a" * 64
    plugin.pop("signature", None)
    assert _run(mod, plugin) >= 1


def test_verified_plugin_with_sha256_and_signature_passes(mod: Any) -> None:
    plugin = _valid_builtin()
    plugin["name"] = "verified-ok"
    plugin["trust_level"] = "verified"
    plugin["publisher"] = {"id": "alice-chen", "verified": True}
    plugin["sha256"] = "b" * 64
    plugin["signature"] = "-----BEGIN PGP SIGNATURE-----\nabc\n-----END PGP SIGNATURE-----"
    assert _run(mod, plugin) == 0


def test_unregistered_publisher_fails(mod: Any) -> None:
    plugin = _valid_builtin()
    plugin["publisher"] = {"id": "ghost-org", "verified": False}
    assert _run(mod, plugin) >= 1


def test_verified_trust_level_requires_verified_publisher(mod: Any) -> None:
    plugin = _valid_builtin()
    plugin["name"] = "bad-verified"
    plugin["trust_level"] = "verified"
    plugin["publisher"] = {"id": "alice-chen", "verified": False}
    plugin["sha256"] = "c" * 64
    plugin["signature"] = "sig"
    # alice-chen is verified:true in publishers.json, but the plugin claims
    # verified:false -> disagreement error + verified-needs-verified error.
    assert _run(mod, plugin) >= 1


def test_invalid_name_pattern_fails(mod: Any) -> None:
    plugin = _valid_builtin()
    plugin["name"] = "Bad Name!"
    assert _run(mod, plugin) >= 1


def test_duplicate_name_detected(mod: Any) -> None:
    names: set[str] = set()
    mod.validate_plugin(cast(dict[str, Any], _valid_builtin()), 1, names, {}, _publisher_index())
    # second entry with same name -> duplicate error
    errors = mod.validate_plugin(
        cast(dict[str, Any], _valid_builtin()), 2, names, {}, _publisher_index()
    )
    assert errors >= 1


def test_duplicate_sha256_detected(mod: Any) -> None:
    sha = "d" * 64
    sha_seen: dict[str, str] = {}
    first = _valid_builtin()
    first["name"] = "first"
    first["sha256"] = sha
    mod.validate_plugin(cast(dict[str, Any], first), 1, set(), sha_seen, _publisher_index())
    second = _valid_builtin()
    second["name"] = "second"
    second["sha256"] = sha
    errors = mod.validate_plugin(
        cast(dict[str, Any], second), 2, set(), sha_seen, _publisher_index()
    )
    assert errors >= 1


def test_bad_permissions_shape_fails(mod: Any) -> None:
    plugin = _valid_builtin()
    plugin["permissions"] = {
        "filesystem": "read-only",
        "network": "no",
        "shell": False,
        "git": False,
    }
    assert _run(mod, plugin) >= 1


def test_malformed_sha256_fails(mod: Any) -> None:
    plugin = _valid_builtin()
    plugin["sha256"] = "not-hex"
    assert _run(mod, plugin) >= 1
