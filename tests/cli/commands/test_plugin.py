"""Tests for the plugin CLI command."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from packaging.version import parse as parse_version
from typer.testing import CliRunner

from justagent.cli.main import app
from justagent.core.package_verifier import PackageVerificationError
from justagent.core.plugin_registry import PluginSpec, TrustLevel

runner = CliRunner()


def test_plugin_list_empty() -> None:
    with patch("justagent.cli.commands.plugin.PluginRegistry") as mock_cls:
        mock_cls.return_value.list.return_value = []
        result = runner.invoke(app, ["plugin", "list"])
    assert result.exit_code == 0
    assert "No plugins registered" in result.output


def test_plugin_list_shows_plugins() -> None:
    plugins = [
        PluginSpec(name="a", version="1.0.0", source="pypi", trust_level=TrustLevel.VERIFIED),
        PluginSpec(name="b", version="2.0.0", source="git", trust_level=TrustLevel.COMMUNITY),
    ]
    with patch("justagent.cli.commands.plugin.PluginRegistry") as mock_cls:
        mock_cls.return_value.list.return_value = plugins
        result = runner.invoke(app, ["plugin", "list"])
    assert result.exit_code == 0
    assert "a" in result.output
    assert "verified" in result.output


def test_plugin_trust() -> None:
    with patch("justagent.cli.commands.plugin.PluginRegistry") as mock_cls:
        mock_cls.return_value.trust.return_value = True
        result = runner.invoke(app, ["plugin", "trust", "a", "verified"])
    assert result.exit_code == 0
    assert "Set trust level of a to verified" in result.output


def test_plugin_search_shows_indexed_plugins() -> None:
    result = runner.invoke(app, ["plugin", "search"])
    assert result.exit_code == 0
    assert "security-scan" in result.output


def test_plugin_search_filters_by_keyword() -> None:
    result = runner.invoke(app, ["plugin", "search", "docker"])
    assert result.exit_code == 0
    assert "docker-ship" in result.output
    assert "web-search" not in result.output


def test_plugin_install_dry_run() -> None:
    result = runner.invoke(app, ["plugin", "install", "my-plugin", "--dry-run"])
    assert result.exit_code == 0
    assert "[dry-run] Would install my-plugin" in result.output


def test_plugin_install_from_registry_dry_run() -> None:
    result = runner.invoke(app, ["plugin", "install", "security-scan", "--dry-run"])
    assert result.exit_code == 0
    assert "[dry-run] Would install security-scan" in result.output
    assert "justagent" in result.output


def test_plugin_uninstall_dry_run() -> None:
    with patch("justagent.cli.commands.plugin.PluginRegistry") as mock_cls:
        mock_cls.return_value.get.return_value = PluginSpec(name="x", source="pypi")
        result = runner.invoke(app, ["plugin", "uninstall", "x", "--dry-run"])
    assert result.exit_code == 0
    assert "[dry-run] Would uninstall x" in result.output


def test_plugin_install_pip_failure() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("no pip")):
        result = runner.invoke(app, ["plugin", "install", "bad", "--yes"])
    assert result.exit_code != 0
    assert result.exception is not None
    assert "Failed to install plugin" in str(result.exception)


def test_plugin_install_community_requires_confirmation() -> None:
    result = runner.invoke(app, ["plugin", "install", "jira-link"], input="n\n")
    assert result.exit_code == 0
    assert "community plugin" in result.output


def test_plugin_install_untrusted_requires_confirmation() -> None:
    result = runner.invoke(
        app, ["plugin", "install", "./local-plugin", "--trust", "untrusted"], input="n\n"
    )
    assert result.exit_code == 0
    assert "untrusted" in result.output


def test_plugin_install_skip_trust_check() -> None:
    with patch("justagent.cli.commands.plugin._run_pip_install") as mock_install:
        mock_install.return_value = subprocess.CompletedProcess(
            args=["pip", "install", "jira-link"], returncode=0, stdout="", stderr=""
        )
        result = runner.invoke(
            app, ["plugin", "install", "jira-link", "--skip-trust-check", "--yes"]
        )
    assert result.exit_code == 0
    assert "Installed plugin: jira-link" in result.output
    mock_install.assert_called_once()


def test_plugin_install_verified_without_signature_warns() -> None:
    with patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index:
        mock_index.return_value.get.return_value = {
            "name": "verified-no-sig",
            "package": "verified-no-sig",
            "version": "1.0.0",
            "trust_level": "verified",
            "entry_point": "verified_no_sig.plugin:Plugin",
            "hooks": [],
        }
        result = runner.invoke(app, ["plugin", "install", "verified-no-sig"], input="n\n")
    assert result.exit_code == 0
    assert "no checksum or signature" in result.output


def test_plugin_update_requires_name_or_all() -> None:
    result = runner.invoke(app, ["plugin", "update"])
    assert result.exit_code != 0
    assert result.exception is not None
    assert "name or use --all" in str(result.exception)


def test_plugin_update_no_updates() -> None:
    plugins = [
        PluginSpec(name="a", version="1.0.0", source="pkg-a", trust_level=TrustLevel.VERIFIED),
    ]
    with (
        patch("justagent.cli.commands.plugin.PluginRegistry") as mock_reg,
        patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index,
        patch(
            "justagent.cli.commands.plugin._installed_version", return_value=parse_version("1.0.0")
        ),
    ):
        mock_reg.return_value.list.return_value = plugins
        mock_reg.return_value.get.return_value = plugins[0]
        mock_index.return_value.get.return_value = {"version": "1.0.0"}
        result = runner.invoke(app, ["plugin", "update", "--all"])
    assert result.exit_code == 0
    assert "No plugin updates" in result.output


def test_plugin_update_dry_run() -> None:
    plugins = [
        PluginSpec(name="a", version="1.0.0", source="pkg-a", trust_level=TrustLevel.VERIFIED),
    ]
    with (
        patch("justagent.cli.commands.plugin.PluginRegistry") as mock_reg,
        patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index,
        patch(
            "justagent.cli.commands.plugin._installed_version", return_value=parse_version("1.0.0")
        ),
    ):
        mock_reg.return_value.list.return_value = plugins
        mock_reg.return_value.get.return_value = plugins[0]
        mock_index.return_value.get.return_value = {"version": "2.0.0"}
        result = runner.invoke(app, ["plugin", "update", "--all", "--dry-run"])
    assert result.exit_code == 0
    assert "Would update a" in result.output


def test_plugin_update_skips_builtin() -> None:
    plugins = [
        PluginSpec(
            name="security-scan",
            version="1.0.0",
            source="justagent",
            trust_level=TrustLevel.BUILTIN,
        ),
    ]
    with (
        patch("justagent.cli.commands.plugin.PluginRegistry") as mock_reg,
        patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index,
        patch(
            "justagent.cli.commands.plugin._installed_version", return_value=parse_version("1.0.0")
        ),
    ):
        mock_reg.return_value.list.return_value = plugins
        mock_index.return_value.get.return_value = {"version": "2.0.0"}
        result = runner.invoke(app, ["plugin", "update", "--all"])
    assert result.exit_code == 0
    assert "built-in" in result.output


def test_plugin_update_upgrades_plugin() -> None:
    plugins = [
        PluginSpec(name="a", version="1.0.0", source="pkg-a", trust_level=TrustLevel.VERIFIED),
    ]
    with (
        patch("justagent.cli.commands.plugin.PluginRegistry") as mock_reg,
        patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index,
        patch(
            "justagent.cli.commands.plugin._installed_version", return_value=parse_version("1.0.0")
        ),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["pip", "install", "--upgrade", "pkg-a"], returncode=0, stdout="", stderr=""
        )
        mock_reg.return_value.list.return_value = plugins
        mock_reg.return_value.get.return_value = plugins[0]
        mock_index.return_value.get.return_value = {"version": "2.0.0"}
        result = runner.invoke(app, ["plugin", "update", "--all", "--yes"])
    assert result.exit_code == 0
    assert "Updated plugin: a -> 2.0.0" in result.output
    mock_run.assert_called_once()


def test_plugin_rate_records_rating() -> None:
    with (
        patch("justagent.cli.commands.plugin.PluginRegistry") as mock_reg,
        patch("justagent.cli.commands.plugin.PluginStats") as mock_stats,
    ):
        mock_reg.return_value.get.return_value = PluginSpec(name="a", source="pypi")
        result = runner.invoke(app, ["plugin", "rate", "a", "4.5"])
    assert result.exit_code == 0
    assert "Rated a: 4.5" in result.output
    mock_stats.return_value.record_rate.assert_called_once_with("a", 4.5)


def test_plugin_rate_rejects_out_of_range() -> None:
    with patch("justagent.cli.commands.plugin.PluginRegistry") as mock_reg:
        mock_reg.return_value.get.return_value = PluginSpec(name="a", source="pypi")
        result = runner.invoke(app, ["plugin", "rate", "a", "6"])
    assert result.exit_code != 0


def test_plugin_stats_shows_summary() -> None:
    with patch("justagent.cli.commands.plugin.PluginStats") as mock_stats:
        mock_stats.return_value.summary.return_value = {
            "a": {
                "installs": 2,
                "uninstalls": 1,
                "rating": {"score": 4.5, "count": 2},
            }
        }
        result = runner.invoke(app, ["plugin", "stats"])
    assert result.exit_code == 0
    assert "a" in result.output
    assert "4.5 (2)" in result.output


def test_plugin_info_shows_details() -> None:
    with patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index:
        mock_index.return_value.get.return_value = {
            "name": "commit-sign",
            "version": "0.1.0",
            "trust_level": "verified",
            "description": "Sign commits",
            "publisher": {
                "id": "alice-chen",
                "verified": True,
                "url": "https://github.com/alice-chen",
            },
            "maintainer": "Alice Chen",
            "license": "Apache-2.0",
            "downloads": 42,
        }
        result = runner.invoke(app, ["plugin", "info", "commit-sign"])
    assert result.exit_code == 0
    assert "commit-sign" in result.output
    assert "alice-chen" in result.output
    assert "42" in result.output


def test_plugin_info_shows_permissions() -> None:
    with patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index:
        mock_index.return_value.get.return_value = {
            "name": "jira-link",
            "version": "0.2.1",
            "trust_level": "community",
            "permissions": {
                "filesystem": "read-only",
                "network": False,
                "shell": False,
                "git": True,
                "env": ["JIRA_BASE_URL"],
            },
        }
        result = runner.invoke(app, ["plugin", "info", "jira-link"])
    assert result.exit_code == 0
    assert "Permissions:" in result.output
    assert "git=yes" in result.output
    assert "JIRA_BASE_URL" in result.output


def test_plugin_install_community_shows_permissions() -> None:
    result = runner.invoke(app, ["plugin", "install", "jira-link"], input="n\n")
    assert result.exit_code == 0
    assert "community plugin" in result.output
    assert "Requested permissions" in result.output
    assert "git=yes" in result.output


def test_plugin_install_stores_capabilities() -> None:
    with (
        patch("justagent.cli.commands.plugin._run_pip_install") as mock_install,
        patch("justagent.cli.commands.plugin.PluginRegistry") as mock_reg,
        patch("justagent.cli.commands.plugin.PluginStats"),
    ):
        mock_install.return_value = subprocess.CompletedProcess(
            args=["pip", "install", "jira-link"], returncode=0, stdout="", stderr=""
        )
        result = runner.invoke(
            app,
            ["plugin", "install", "jira-link", "--yes", "--skip-trust-check"],
        )
    assert result.exit_code == 0
    spec = mock_reg.return_value.add.call_args[0][0]
    assert spec.capabilities.git is True
    assert "JIRA_BASE_URL" in spec.capabilities.env


def test_plugin_install_with_sha256_downloads_and_verifies() -> None:
    downloaded = Path("/tmp/justagent-pkg/verified-plugin.whl")
    with (
        patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index,
        patch("justagent.cli.commands.plugin._run_pip_install") as mock_install,
        patch("justagent.cli.commands.plugin.PluginRegistry"),
        patch("justagent.cli.commands.plugin.PluginStats"),
        patch("justagent.cli.commands.plugin.download_and_verify", return_value=downloaded),
    ):
        mock_index.return_value.get.return_value = {
            "name": "verified-plugin",
            "package": "verified-plugin",
            "version": "1.0.0",
            "trust_level": "verified",
            "sha256": "a" * 64,
            "signature": "sig",
        }
        mock_install.return_value = subprocess.CompletedProcess(
            args=["pip", "install", str(downloaded)], returncode=0, stdout="", stderr=""
        )
        result = runner.invoke(
            app,
            ["plugin", "install", "verified-plugin", "--yes", "--skip-trust-check"],
        )
    assert result.exit_code == 0
    assert "Installed plugin: verified-plugin" in result.output
    mock_install.assert_called_once()
    assert mock_install.call_args[0][1] == str(downloaded)


def test_plugin_install_with_sha256_verification_failure() -> None:
    with (
        patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index,
        patch(
            "justagent.cli.commands.plugin.download_and_verify",
            side_effect=PackageVerificationError("sha256 mismatch"),
        ),
    ):
        mock_index.return_value.get.return_value = {
            "name": "bad-plugin",
            "package": "bad-plugin",
            "version": "1.0.0",
            "trust_level": "verified",
            "sha256": "a" * 64,
        }
        result = runner.invoke(
            app,
            ["plugin", "install", "bad-plugin", "--yes", "--skip-trust-check"],
        )
    assert result.exit_code != 0
    assert result.exception is not None
    assert "integrity verification" in str(result.exception)


def test_plugin_update_with_sha256_downloads_and_verifies() -> None:
    downloaded = Path("/tmp/justagent-pkg/pkg-a.whl")
    plugins = [
        PluginSpec(name="a", version="1.0.0", source="pkg-a", trust_level=TrustLevel.VERIFIED),
    ]
    with (
        patch("justagent.cli.commands.plugin.PluginRegistry") as mock_reg,
        patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index,
        patch(
            "justagent.cli.commands.plugin._installed_version", return_value=parse_version("1.0.0")
        ),
        patch("justagent.cli.commands.plugin._run_pip_install") as mock_install,
        patch("justagent.cli.commands.plugin.download_and_verify", return_value=downloaded),
    ):
        mock_reg.return_value.list.return_value = plugins
        mock_reg.return_value.get.return_value = plugins[0]
        mock_index.return_value.get.return_value = {
            "version": "2.0.0",
            "sha256": "a" * 64,
        }
        mock_install.return_value = subprocess.CompletedProcess(
            args=["pip", "install", "--upgrade", str(downloaded)],
            returncode=0,
            stdout="",
            stderr="",
        )
        result = runner.invoke(app, ["plugin", "update", "--all", "--yes"])
    assert result.exit_code == 0
    assert "Updated plugin: a -> 2.0.0" in result.output
    mock_install.assert_called_once()
    assert mock_install.call_args[0][1] == str(downloaded)


def test_plugin_install_no_sandbox_untrusted_is_blocked() -> None:
    """--no-sandbox must be hard-rejected for UNTRUSTED plugins."""
    result = runner.invoke(
        app,
        ["plugin", "install", "./local-plugin", "--trust", "untrusted", "--no-sandbox", "--yes"],
    )
    assert result.exit_code != 0
    assert result.exception is not None
    assert "Refusing to install untrusted plugin" in str(result.exception)


def test_plugin_install_no_sandbox_community_requires_name_confirmation() -> None:
    """--no-sandbox for COMMUNITY plugins requires typing the plugin name."""
    result = runner.invoke(
        app,
        ["plugin", "install", "jira-link", "--no-sandbox", "--yes", "--skip-trust-check"],
        input="wrong-name\n",
    )
    assert result.exit_code == 0
    assert "DANGER" in result.output
    assert "aborted" in result.output.lower()


def test_plugin_install_no_sandbox_community_proceeds_with_correct_name() -> None:
    """--no-sandbox for COMMUNITY plugins proceeds when name is typed correctly."""
    with (
        patch("justagent.cli.commands.plugin._run_pip_install") as mock_install,
        patch("justagent.cli.commands.plugin.PluginRegistry"),
        patch("justagent.cli.commands.plugin.PluginStats"),
    ):
        mock_install.return_value = subprocess.CompletedProcess(
            args=["pip", "install", "jira-link"], returncode=0, stdout="", stderr=""
        )
        result = runner.invoke(
            app,
            ["plugin", "install", "jira-link", "--no-sandbox", "--yes", "--skip-trust-check"],
            input="jira-link\n",
        )
    assert result.exit_code == 0
    assert "DANGER" in result.output
    assert "Installed plugin: jira-link" in result.output
    # Verify sandbox was NOT used
    assert mock_install.call_args.kwargs.get("sandbox") is False


def test_plugin_install_no_sandbox_verified_is_noop() -> None:
    """--no-sandbox for VERIFIED plugins is a no-op (sandbox not applied anyway)."""
    with (
        patch("justagent.cli.commands.plugin.RegistryIndex") as mock_index,
        patch("justagent.cli.commands.plugin._run_pip_install") as mock_install,
        patch("justagent.cli.commands.plugin.PluginRegistry"),
        patch("justagent.cli.commands.plugin.PluginStats"),
    ):
        mock_index.return_value.get.return_value = {
            "name": "verified-plugin",
            "package": "verified-plugin",
            "version": "1.0.0",
            "trust_level": "verified",
        }
        mock_install.return_value = subprocess.CompletedProcess(
            args=["pip", "install", "verified-plugin"], returncode=0, stdout="", stderr=""
        )
        result = runner.invoke(
            app,
            ["plugin", "install", "verified-plugin", "--no-sandbox", "--yes", "--skip-trust-check"],
        )
    assert result.exit_code == 0
    assert "Installed plugin: verified-plugin" in result.output
    # sandbox should be False for verified plugins regardless
    assert mock_install.call_args.kwargs.get("sandbox") is False


def test_plugin_install_no_sandbox_dry_run_shows_warning() -> None:
    """--no-sandbox with --dry-run for COMMUNITY plugins shows danger warning."""
    result = runner.invoke(
        app,
        ["plugin", "install", "jira-link", "--no-sandbox", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "DANGER" in result.output
    assert "[dry-run] Would install jira-link" in result.output


def test_run_pip_install_uv_local_path_uses_no_deps(tmp_path: Path) -> None:
    """uv pip install of a local plugin directory must pass --no-deps.

    Local plugin packages declare ``justagent``/``justagent-sdk`` as
    dependencies, but neither name is published to PyPI. ``uv pip install``
    resolves against the registry and fails with "No solution found" even
    when both packages already exist in the host environment. Skipping
    dependency resolution for local sources under uv lets the install
    succeed; runtime deps are provided by the justagent host environment.
    """
    from justagent.cli.commands.plugin import _run_pip_install

    plugin_dir = tmp_path / "my-plugin"
    plugin_dir.mkdir()
    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    with patch("justagent.cli.commands.plugin.SandboxRunner") as mock_runner_cls:
        mock_runner_cls.return_value.run.side_effect = fake_run
        _run_pip_install(["uv", "pip"], str(plugin_dir), sandbox=True)

    args = captured["args"]
    assert "--no-deps" in args
    # --no-deps must come before the install spec (the resolved local path)
    assert args.index("--no-deps") < args.index(str(plugin_dir.resolve()))


def test_run_pip_install_uv_registry_spec_skips_no_deps() -> None:
    """uv pip install of a non-local (registry) spec must NOT pass --no-deps.

    Registry plugins are real PyPI packages whose dependencies should be
    resolved normally; --no-deps is only for local directory sources.
    """
    from justagent.cli.commands.plugin import _run_pip_install

    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    with patch("justagent.cli.commands.plugin.subprocess.run", side_effect=fake_run):
        _run_pip_install(["uv", "pip"], "some-registry-plugin", sandbox=False)

    assert "--no-deps" not in captured["args"]


def test_run_pip_install_pip_local_path_skips_no_deps(tmp_path: Path) -> None:
    """Plain pip install of a local directory must NOT pass --no-deps.

    Plain pip already satisfies dependencies from the active environment, so
    --no-deps is unnecessary and would silently skip legitimate deps.
    """
    from justagent.cli.commands.plugin import _run_pip_install

    plugin_dir = tmp_path / "my-plugin"
    plugin_dir.mkdir()
    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    with patch("justagent.cli.commands.plugin.SandboxRunner") as mock_runner_cls:
        mock_runner_cls.return_value.run.side_effect = fake_run
        _run_pip_install(["pip"], str(plugin_dir), sandbox=True)

    assert "--no-deps" not in captured["args"]
