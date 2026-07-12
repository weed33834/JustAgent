"""The ``autoship plugin`` command: install, uninstall, search, list, info, update."""

from __future__ import annotations

import contextlib
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import Any, cast

import typer
from packaging.version import Version
from packaging.version import parse as parse_version

from autoship.core.plugin_registry import CapabilityManifest, PluginRegistry, PluginSpec, TrustLevel
from autoship.core.registry_index import RegistryIndex
from autoship.core.sandbox import SandboxRunner
from autoship.exceptions import PluginError
from autoship.utils.hashing import pip_cmd

app = typer.Typer()


def _read_project_name(source: str) -> str | None:
    """Read ``project.name`` from a local ``pyproject.toml`` if present."""
    path = Path(source)
    if not path.is_dir():
        return None
    pyproject = path / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        import tomllib

        with open(pyproject, "rb") as _f:
            data = tomllib.load(_f)
    except (OSError, Exception):
        return None
    project = cast("dict[str, Any] | None", data.get("project"))
    if isinstance(project, dict):
        return project.get("name")
    return None


def _run_pip_install(
    cmd: list[str],
    spec: str,
    *,
    upgrade: bool = False,
    sandbox: bool = True,
    env_whitelist: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run pip install, optionally inside a sandbox for untrusted plugins."""
    spec_path = Path(spec)
    is_local = spec_path.exists()
    install_spec = str(spec_path.resolve()) if is_local else spec

    args = [*cmd, "install", "--quiet"]
    if upgrade:
        args.append("--upgrade")
    if is_local and cmd[:1] == ["uv"]:
        args.append("--no-deps")
    args.append(install_spec)

    if sandbox:
        runner = SandboxRunner(
            network=True,
            env_whitelist=env_whitelist
            or ["PATH", "HOME", "USER", "LANG", "LC_ALL", "PIP_INDEX_URL", "VIRTUAL_ENV"],
        )
        result = runner.run(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    return subprocess.run(args, check=True, capture_output=True, text=True)


def _installed_version(package: str) -> Version | None:
    """Return the installed version of a package, or None if not installed."""
    try:
        raw = _distribution_version(package)
    except PackageNotFoundError:
        return None
    except Exception:
        return None
    try:
        return parse_version(raw)
    except Exception:
        return None


def _public_key_b64(config: Any) -> str | None:
    return config.registry.public_key if config and getattr(config, "registry", None) else None


def _capabilities_from_index(indexed: dict[str, Any] | None) -> dict[str, Any]:
    if not indexed:
        return {}
    permissions = cast(
        dict[str, Any], indexed.get("permissions") or indexed.get("capabilities") or {}
    )
    return {
        "filesystem": permissions.get("filesystem", "read-only"),
        "network": permissions.get("network", False),
        "shell": permissions.get("shell", False),
        "git": permissions.get("git", False),
        "env": permissions.get("env", []),
    }


def _format_capabilities(capabilities: CapabilityManifest) -> str:
    return ", ".join(capabilities.summary()) or "none"


def _publisher_badge(plugin: dict[str, Any]) -> str:
    publisher = plugin.get("publisher")
    if not publisher:
        return ""
    verified = publisher.get("verified")
    badge = "verified" if verified else "unverified"
    return f"{publisher.get('id', '?')} ({badge})"


def _cleanup_downloaded(path: Path | None) -> None:
    if path is None:
        return
    with contextlib.suppress(OSError):
        path.unlink()


def _confirm_trust(
    plugin_name: str,
    plugin_trust: TrustLevel,
    indexed: dict[str, Any] | None,
    capabilities: CapabilityManifest,
    yes: bool,
    skip_trust_check: bool,
) -> None:
    if (
        plugin_trust == TrustLevel.VERIFIED
        and indexed
        and not indexed.get("sha256")
        and not indexed.get("signature")
    ):
        typer.echo(
            f"Warning: VERIFIED plugin '{plugin_name}' has no integrity checksum. "
            "Proceed with caution."
        )
        if not yes and not typer.confirm("Continue?", abort=False):
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    if skip_trust_check or yes:
        return

    if plugin_trust in (TrustLevel.COMMUNITY, TrustLevel.UNTRUSTED):
        level_name = "COMMUNITY" if plugin_trust == TrustLevel.COMMUNITY else "UNTRUSTED"
        typer.echo(
            f"Warning: plugin '{plugin_name}' has trust level {level_name}. "
            "This plugin may have full access to your system."
        )
        typer.echo(f"Requested capabilities: {_format_capabilities(capabilities)}")
        if not typer.confirm("Trust this plugin and continue?", abort=False):
            typer.echo("Aborted.")
            raise typer.Exit(code=0)


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="plugin")


@app.command("list")
def list_plugins(
    ctx: typer.Context,
) -> None:
    """List registered plugins and their trust levels."""
    registry = PluginRegistry()
    plugins = registry.list()
    if not plugins:
        typer.echo("No plugins registered.")
        typer.echo("Use 'autoship plugin search <keyword>' to find plugins.")
        return

    typer.echo(f"{'Name':<30} {'Version':<10} {'Trust':<12} {'Source'}")
    for plugin in plugins:
        typer.echo(
            f"{plugin.name:<30} {plugin.version:<10} {plugin.trust_level.value:<12} {plugin.source}"
        )
    typer.echo("Use 'autoship plugin search <keyword>' to find more.")


@app.command("search")
def search_plugins(
    ctx: typer.Context,
    keyword: str | None = typer.Argument(None, help="Keyword to search in name or description"),
) -> None:
    """Search the plugin registry index."""
    index = RegistryIndex(ctx.obj.get("config"))
    plugins = index.search(keyword)
    if not plugins:
        typer.echo("No matching plugins found.")
        return

    typer.echo(f"{'Name':<30} {'Version':<10} {'Trust':<12} {'Description'}")
    for plugin in plugins:
        typer.echo(
            f"{plugin['name']:<30} {plugin.get('version', '?'):<10} "
            f"{plugin.get('trust_level', 'community'):<12} {plugin.get('description', '')}"
        )


@app.command("info")
def info(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Plugin name"),
) -> None:
    """Show detailed information about a plugin in the registry."""
    index = RegistryIndex(ctx.obj.get("config"))
    plugin = index.get(name)
    if plugin is None:
        raise PluginError(f"Plugin '{name}' not found in registry.")

    field_width = 14
    typer.echo(f"{'Name:':<{field_width}}{plugin['name']}")
    typer.echo(f"{'Version:':<{field_width}}{plugin.get('version', '?')}")
    typer.echo(f"{'Trust:':<{field_width}}{plugin.get('trust_level', 'community')}")
    typer.echo(f"{'Publisher:':<{field_width}}{_publisher_badge(plugin) or 'unknown'}")
    typer.echo(f"{'Maintainer:':<{field_width}}{plugin.get('maintainer', 'unknown')}")
    typer.echo(f"{'License:':<{field_width}}{plugin.get('license', 'unknown')}")
    typer.echo(f"{'Categories:':<{field_width}}{', '.join(plugin.get('categories', [])) or '—'}")
    typer.echo(f"{'Tags:':<{field_width}}{', '.join(plugin.get('tags', [])) or '—'}")
    typer.echo(
        f"{'Permissions:':<{field_width}}"
        f"{_format_capabilities(CapabilityManifest(**_capabilities_from_index(plugin)))}"
    )
    typer.echo(f"{'Downloads:':<{field_width}}{plugin.get('downloads', 0)}")
    rating = plugin.get("rating")
    rating_str = (
        f"{rating['score']:.1f} / 5 ({rating['count']})" if rating and rating.get("count") else "—"
    )
    typer.echo(f"{'Rating:':<{field_width}}{rating_str}")
    if plugin.get("homepage"):
        typer.echo(f"{'Homepage:':<{field_width}}{plugin['homepage']}")
    if plugin.get("source_url"):
        typer.echo(f"{'Source:':<{field_width}}{plugin['source_url']}")
    typer.echo(f"{'Install:':<{field_width}}autoship plugin install {plugin['name']}")


@app.command("install")
def install(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Package spec or plugin name from registry"),
    name: str | None = typer.Option(None, "--name", help="Plugin name to register"),
    version: str | None = typer.Option(None, "--version", help="Plugin version"),
    trust: TrustLevel | None = typer.Option(None, "--trust", help="Initial trust level"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show actions without executing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmations"),
    skip_trust_check: bool = typer.Option(
        False, "--skip-trust-check", help="Skip trust level warnings"
    ),
    no_sandbox: bool = typer.Option(False, "--no-sandbox", help="Run pip install without sandbox"),
) -> None:
    """Install a plugin package and register it locally."""
    registry = PluginRegistry()
    index = RegistryIndex(ctx.obj.get("config"))
    indexed = index.get(source)

    if indexed:
        plugin_name = name or indexed["name"]
        package = indexed["package"]
        plugin_version = version or indexed.get("version", "0.0.0")
        plugin_trust = trust or TrustLevel(indexed.get("trust_level", "verified"))
        source_for_pip = package
        publisher = _publisher_badge(indexed)
        if publisher:
            typer.echo(f"Publisher: {publisher}")
    else:
        plugin_name = name or Path(source).name
        package = _read_project_name(source) or plugin_name
        plugin_version = version or "0.0.0"
        plugin_trust = trust or TrustLevel.COMMUNITY
        source_for_pip = source

    capabilities = CapabilityManifest(**_capabilities_from_index(indexed))

    if not dry_run:
        _confirm_trust(plugin_name, plugin_trust, indexed, capabilities, yes, skip_trust_check)

    if (
        not dry_run
        and not yes
        and not typer.confirm(
            f"Install plugin '{plugin_name}'?",
        )
    ):
        typer.echo("Aborted.")
        raise typer.Exit(code=0)

    if dry_run:
        typer.echo(
            f"[DRY-RUN] Would install plugin '{plugin_name}' ({plugin_version}) "
            f"with trust level {plugin_trust.value}"
        )
        return

    use_sandbox = plugin_trust in (TrustLevel.COMMUNITY, TrustLevel.UNTRUSTED)

    cmd = pip_cmd()

    try:
        result = _run_pip_install(
            cmd,
            source_for_pip,
            sandbox=use_sandbox,
            env_whitelist=[
                "PATH",
                "HOME",
                "USER",
                "LANG",
                "LC_ALL",
                "PIP_INDEX_URL",
                "VIRTUAL_ENV",
                "XDG_CACHE_HOME",
                "UV_CACHE_DIR",
            ],
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, output=result.stdout, stderr=result.stderr
            )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise PluginError(f"Failed to install plugin '{plugin_name}': {exc}") from exc

    registry.add(
        PluginSpec(
            name=plugin_name,
            version=plugin_version,
            source=source_for_pip,
            package=package,
            entry_point=indexed.get("entry_point") if indexed else None,
            hooks=indexed.get("hooks", []) if indexed else [],
            trust_level=plugin_trust,
            capabilities=capabilities,
            sha256=indexed.get("sha256") if indexed else None,
            signature=indexed.get("signature") if indexed else None,
            maintainer=indexed.get("maintainer") if indexed else None,
            license=indexed.get("license") if indexed else None,
        )
    )
    typer.echo(f"Plugin '{plugin_name}' installed successfully.")


@app.command("uninstall")
def uninstall(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name of the plugin to uninstall"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show actions without executing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmations"),
) -> None:
    """Uninstall a plugin package and remove it from the local registry."""
    registry = PluginRegistry()

    spec = registry.get(name)
    if spec is None:
        raise PluginError(f"Plugin '{name}' is not registered.")

    if not dry_run and not yes and not typer.confirm(f"Uninstall plugin '{name}'?"):
        typer.echo("Aborted.")
        raise typer.Exit(code=0)

    if dry_run:
        typer.echo(f"[DRY-RUN] Would uninstall plugin '{name}'")
        return

    package = spec.package or name
    cmd = pip_cmd()
    args = [*cmd, "uninstall", "--quiet", package]
    if cmd[0] != "uv":
        args.append("-y")
    try:
        subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise PluginError(f"Failed to uninstall plugin '{name}': {exc}") from exc

    registry.remove(name)
    typer.echo(f"Plugin '{name}' uninstalled successfully.")


@app.command("update")
def update(
    ctx: typer.Context,
    name: str | None = typer.Argument(None, help="Plugin name to update"),
    all_plugins: bool = typer.Option(False, "--all", help="Update all registered plugins"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show actions without executing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmations"),
    skip_trust_check: bool = typer.Option(
        False, "--skip-trust-check", help="Skip trust level warnings"
    ),
    no_sandbox: bool = typer.Option(False, "--no-sandbox", help="Run pip install without sandbox"),
) -> None:
    """Check for and install plugin updates."""
    registry = PluginRegistry()
    index = RegistryIndex(ctx.obj.get("config"))

    plugins = registry.list()
    if name:
        plugin = registry.get(name)
        if plugin is None:
            raise PluginError(f"Plugin '{name}' is not registered.")
        candidates = [plugin]
    elif all_plugins:
        candidates = plugins
    else:
        raise PluginError("Specify a plugin name or use --all.")

    updatable: list[tuple[PluginSpec, Version, Version]] = []
    skipped: list[tuple[PluginSpec, str]] = []
    for plugin in candidates:
        if plugin.trust_level == TrustLevel.BUILTIN:
            skipped.append((plugin, "builtin — skipped"))
            continue
        if not plugin.source or plugin.source.startswith((".", "/", "~")):
            skipped.append((plugin, "local source — skipped"))
            continue

        installed = _installed_version(plugin.source)
        if installed is None:
            skipped.append((plugin, "not installed — skipped"))
            continue

        indexed = index.get(plugin.name)
        latest_raw = indexed.get("version") if indexed else plugin.version
        try:
            latest = parse_version(latest_raw or plugin.version)
        except Exception:
            latest = parse_version(plugin.version)

        if latest > installed:
            updatable.append((plugin, installed, latest))
        else:
            skipped.append((plugin, "already up to date"))

    if not updatable:
        typer.echo("All plugins are up to date.")
        for plugin, reason in skipped:
            typer.echo(f"  - {plugin.name}: {reason}")
        return

    typer.echo("Available updates:")
    for plugin, installed, latest in updatable:
        typer.echo(f"  - {plugin.name}: {installed} -> {latest}")

    if not dry_run and not yes and not typer.confirm("Apply updates?"):
        typer.echo("Aborted.")
        raise typer.Exit(code=0)

    cmd = pip_cmd()
    for plugin, _installed, latest in updatable:
        if dry_run:
            typer.echo(f"[DRY-RUN] Would update '{plugin.name}' to {latest}")
            continue

        indexed = index.get(plugin.name)
        _confirm_trust(
            plugin.name,
            plugin.trust_level,
            indexed,
            plugin.capabilities,
            yes,
            skip_trust_check,
        )

        use_sandbox = plugin.trust_level in (
            TrustLevel.COMMUNITY,
            TrustLevel.UNTRUSTED,
        )

        try:
            result = _run_pip_install(
                cmd,
                plugin.source,
                upgrade=True,
                sandbox=use_sandbox,
                env_whitelist=plugin.capabilities.env
                + ["PATH", "HOME", "USER", "LANG", "LC_ALL", "PIP_INDEX_URL"],
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, result.args, output=result.stdout, stderr=result.stderr
                )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            raise PluginError(f"Failed to update plugin '{plugin.name}': {exc}") from exc

        plugin.version = str(latest)
        registry.add(plugin)
        typer.echo(f"Updated '{plugin.name}' to {latest}")

    if skipped:
        typer.echo("Skipped:")
        for plugin, reason in skipped:
            typer.echo(f"  - {plugin.name}: {reason}")
