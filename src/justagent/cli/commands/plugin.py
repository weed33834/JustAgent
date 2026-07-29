"""``justagent plugin`` command: install, uninstall, list, search, info, rate, stats, update, trust."""

from __future__ import annotations

import logging
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import Any, cast

import typer
from packaging.version import Version
from packaging.version import parse as parse_version

from justagent.core.i18n import I18n, get_i18n_from_ctx
from justagent.core.package_verifier import PackageVerificationError, download_and_verify
from justagent.core.plugin_registry import CapabilityManifest, PluginRegistry, PluginSpec, TrustLevel
from justagent.core.plugin_stats import PluginStats
from justagent.core.registry_index import RegistryIndex
from justagent.core.sandbox import SandboxRunner
from justagent.exceptions import PluginError
from justagent.utils.hashing import pip_cmd

logger = logging.getLogger("justagent.cli.commands.plugin")

app = typer.Typer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_project_name(source: str) -> str | None:
    """Read ``project.name`` from a local ``pyproject.toml``."""
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
    """Execute ``pip install``, optionally inside a sandbox."""
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
    """Return the installed version of *package*, or ``None`` if not installed."""
    try:
        raw = _distribution_version(package)
    except PackageNotFoundError:
        return None
    except Exception:
        logger.debug(
            "Failed to read installed version for %s", package, exc_info=True
        )
        return None
    try:
        return parse_version(raw)
    except Exception:
        logger.debug(
            "Failed to parse version %r for %s", raw, package, exc_info=True
        )
        return None


def _capabilities_from_permissions(permissions: dict[str, Any] | None) -> CapabilityManifest:
    """Build a :class:`CapabilityManifest` from registry ``permissions`` data."""
    if not permissions or not isinstance(permissions, dict):
        return CapabilityManifest()
    env_raw = permissions.get("env", [])
    if not isinstance(env_raw, list):
        env_raw = []
    return CapabilityManifest(
        filesystem=str(permissions.get("filesystem", "read-only")),
        network=bool(permissions.get("network", False)),
        shell=bool(permissions.get("shell", False)),
        git=bool(permissions.get("git", False)),
        env=[str(e) for e in env_raw],
    )


def _capabilities_default() -> CapabilityManifest:
    return CapabilityManifest(
        filesystem="read-only", network=False, shell=False, git=False, env=[]
    )


def _confirm_trust(
    plugin_name: str,
    plugin_trust: TrustLevel,
    capabilities: CapabilityManifest,
    yes: bool,
    skip_trust_check: bool,
    i18n: I18n,
) -> None:
    """Display trust warnings and ask for confirmation when necessary."""
    if skip_trust_check or yes:
        return

    if plugin_trust in (TrustLevel.COMMUNITY, TrustLevel.UNTRUSTED):
        if plugin_trust == TrustLevel.COMMUNITY:
            typer.echo(
                i18n._("plugin.install_trust_warning_community", plugin_name=plugin_name)
            )
        else:
            typer.echo(
                i18n._("plugin.install_trust_warning_untrusted", plugin_name=plugin_name)
            )
        typer.echo(
            i18n._(
                "plugin.install_permissions",
                permissions=", ".join(capabilities.summary()) or "none",
            )
        )
        if not typer.confirm(i18n._("plugin.install_trust_confirm"), abort=False):
            typer.echo(i18n._("common.aborted"))
            raise typer.Exit(code=0)


def _parse_trust_level(value: str | None, default: TrustLevel) -> TrustLevel:
    """Convert a string trust level to :class:`TrustLevel`, falling back to *default*."""
    if value is None:
        return default
    try:
        return TrustLevel(value)
    except ValueError:
        return default


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="plugin")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command("list")
def list_plugins(ctx: typer.Context) -> None:
    """List registered plugins and their trust levels."""
    i18n: I18n = get_i18n_from_ctx(ctx)
    registry = PluginRegistry()
    plugins = registry.list()
    if not plugins:
        typer.echo(i18n._("plugin.no_plugins"))
        typer.echo(i18n._("plugin.list_search_tip"))
        return

    typer.echo(
        f"{i18n._('plugin.header.name'):<30} {i18n._('plugin.header.version'):<10} "
        f"{i18n._('plugin.header.trust'):<12} {i18n._('plugin.header.source')}"
    )
    for plugin in plugins:
        typer.echo(
            f"{plugin.name:<30} {plugin.version:<10} {plugin.trust_level.value:<12} {plugin.source}"
        )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command("search")
def search_plugins(
    ctx: typer.Context,
    keyword: str | None = typer.Argument(None, help="Search keyword"),
) -> None:
    """Search the official plugin registry index."""
    i18n: I18n = get_i18n_from_ctx(ctx)
    index = RegistryIndex()
    plugins = index.search(keyword)
    if not plugins:
        typer.echo(i18n._("plugin.no_matches"))
        return

    typer.echo(
        f"{i18n._('plugin.header.name'):<30} {i18n._('plugin.header.version'):<10} "
        f"{i18n._('plugin.header.trust'):<12} {i18n._('plugin.header.description')}"
    )
    for plugin in plugins:
        typer.echo(
            f"{plugin.get('name', ''):<30} {plugin.get('version', ''):<10} "
            f"{plugin.get('trust_level', ''):<12} {plugin.get('description', '')}"
        )


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@app.command("info")
def plugin_info(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Plugin name in the registry"),
) -> None:
    """Show detailed information about a plugin in the registry."""
    i18n: I18n = get_i18n_from_ctx(ctx)
    index = RegistryIndex()
    plugin = index.get(name)
    if plugin is None:
        typer.echo(i18n._("plugin.not_found_in_registry", name=name))
        raise typer.Exit(code=1)

    typer.echo(
        f"{i18n._('plugin.info.name')}: {plugin.get('name', i18n._('plugin.info.unknown'))}"
    )
    typer.echo(
        f"{i18n._('plugin.info.version')}: {plugin.get('version', i18n._('plugin.info.unknown'))}"
    )
    typer.echo(
        f"{i18n._('plugin.info.trust')}: {plugin.get('trust_level', i18n._('plugin.info.unknown'))}"
    )

    description = plugin.get("description")
    if description:
        typer.echo(f"{i18n._('plugin.info.description')}: {description}")

    publisher = plugin.get("publisher")
    if isinstance(publisher, dict):
        typer.echo(
            f"{i18n._('plugin.info.publisher')}: "
            f"{publisher.get('id', i18n._('plugin.info.unknown'))}"
        )

    maintainer = plugin.get("maintainer")
    if maintainer:
        typer.echo(f"{i18n._('plugin.info.maintainer')}: {maintainer}")

    license_ = plugin.get("license")
    if license_:
        typer.echo(f"{i18n._('plugin.info.license')}: {license_}")

    downloads = plugin.get("downloads")
    if downloads is not None:
        typer.echo(f"{i18n._('plugin.info.downloads')}: {downloads}")

    rating = plugin.get("rating")
    if isinstance(rating, dict):
        score = rating.get("score", 0.0)
        count = rating.get("count", 0)
        typer.echo(f"{i18n._('plugin.info.rating')}: {score} ({count})")

    homepage = plugin.get("homepage")
    if homepage:
        typer.echo(f"{i18n._('plugin.info.homepage')}: {homepage}")

    categories = plugin.get("categories")
    if categories:
        typer.echo(f"{i18n._('plugin.info.categories')}: {', '.join(categories)}")

    tags = plugin.get("tags")
    if tags:
        typer.echo(f"{i18n._('plugin.info.tags')}: {', '.join(tags)}")

    permissions = plugin.get("permissions")
    if isinstance(permissions, dict):
        typer.echo(f"{i18n._('plugin.info.permissions')}:")
        caps = _capabilities_from_permissions(permissions)
        for item in caps.summary():
            typer.echo(f"  {item}")


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


@app.command("install")
def install(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Package name or local path"),
    name: str | None = typer.Option(None, "--name", help="Plugin registry name"),
    version: str | None = typer.Option(None, "--version", help="Plugin version"),
    trust: TrustLevel | None = typer.Option(None, "--trust", help="Initial trust level"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show actions without executing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmations"),
    skip_trust_check: bool = typer.Option(
        False, "--skip-trust-check", help="Skip trust level warning"
    ),
    no_sandbox: bool = typer.Option(
        False, "--no-sandbox", help="Do not run pip install in sandbox"
    ),
) -> None:
    """Install a plugin package and register it locally."""
    i18n: I18n = get_i18n_from_ctx(ctx)
    registry = PluginRegistry()

    # --- Resolve plugin metadata from the registry index --------------------
    index = RegistryIndex()
    try:
        registry_data: dict[str, Any] | None = index.get(source)
    except Exception:  # noqa: BLE001 - never crash on registry lookup
        registry_data = None

    if registry_data:
        plugin_name = name or registry_data.get("name", source)
        package = registry_data.get("package") or source
        plugin_version = version or registry_data.get("version", "0.0.0")
        plugin_trust = trust or _parse_trust_level(
            registry_data.get("trust_level"), TrustLevel.COMMUNITY
        )
        capabilities = _capabilities_from_permissions(registry_data.get("permissions"))
        sha256 = registry_data.get("sha256")
        signature = registry_data.get("signature")
        source_for_pip = package
    else:
        plugin_name = name or Path(source).name
        package = _read_project_name(source) or plugin_name
        plugin_version = version or "0.0.0"
        plugin_trust = trust or TrustLevel.COMMUNITY
        capabilities = _capabilities_default()
        sha256 = None
        signature = None
        source_for_pip = source

    # --- --no-sandbox handling ---------------------------------------------
    use_sandbox = plugin_trust in (TrustLevel.COMMUNITY, TrustLevel.UNTRUSTED)

    if no_sandbox:
        if plugin_trust == TrustLevel.UNTRUSTED:
            raise PluginError(
                i18n._("plugin.no_sandbox_untrusted_blocked", plugin_name=plugin_name)
            )
        if plugin_trust == TrustLevel.COMMUNITY:
            typer.echo(
                i18n._("plugin.no_sandbox_community_warning", plugin_name=plugin_name)
            )
            if not dry_run:
                confirmation = typer.prompt(i18n._("plugin.no_sandbox_confirm"), prompt_suffix="")
                if confirmation != plugin_name:
                    typer.echo(i18n._("common.aborted"))
                    raise typer.Exit(code=0)
        use_sandbox = False

    # --- Trust confirmation -------------------------------------------------
    if not dry_run and not skip_trust_check:
        if plugin_trust == TrustLevel.VERIFIED and not sha256 and not signature:
            typer.echo(
                i18n._("plugin.install_unverified_signature", plugin_name=plugin_name)
            )
        _confirm_trust(plugin_name, plugin_trust, capabilities, yes, skip_trust_check, i18n)

    # --- General install confirmation ---------------------------------------
    if not dry_run and not yes and not typer.confirm(
        i18n._("plugin.install_confirm", plugin_name=plugin_name, source_for_pip=source_for_pip)
    ):
        typer.echo(i18n._("common.aborted"))
        raise typer.Exit(code=0)

    # --- Dry-run ------------------------------------------------------------
    if dry_run:
        typer.echo(
            i18n._(
                "plugin.install_dry_run",
                plugin_name=plugin_name,
                source_for_pip=source_for_pip,
            )
        )
        return

    # --- Download & verify (when sha256 is available) ----------------------
    install_spec = source_for_pip
    if sha256:
        try:
            downloaded = download_and_verify(
                source_for_pip,
                sha256_hex=sha256,
                signature_b64=signature,
                public_key_b64=None,
            )
        except PackageVerificationError as exc:
            raise PluginError(
                i18n._("plugin.install_verify_failed", plugin_name=plugin_name, exc=exc)
            ) from exc
        install_spec = str(downloaded)

    # --- pip install --------------------------------------------------------
    cmd = pip_cmd()
    try:
        result = _run_pip_install(
            cmd,
            install_spec,
            sandbox=use_sandbox,
            env_whitelist=[
                "PATH", "HOME", "USER", "LANG", "LC_ALL",
                "PIP_INDEX_URL", "VIRTUAL_ENV", "XDG_CACHE_HOME", "UV_CACHE_DIR",
            ],
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, output=result.stdout, stderr=result.stderr
            )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise PluginError(
            i18n._("plugin.install_failed", plugin_name=plugin_name, exc=exc)
        ) from exc

    # --- Register & record stats --------------------------------------------
    registry.add(
        PluginSpec(
            name=plugin_name,
            version=plugin_version,
            source=source,
            package=package,
            trust_level=plugin_trust,
            capabilities=capabilities,
            sha256=sha256,
            signature=signature,
        )
    )

    stats = PluginStats()
    stats.record_install(plugin_name)

    typer.echo(i18n._("plugin.installed", plugin_name=plugin_name))


# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------


@app.command("uninstall")
def uninstall(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Plugin name to uninstall"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show actions without executing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmations"),
) -> None:
    """Uninstall a plugin package and remove it from the local registry."""
    i18n: I18n = get_i18n_from_ctx(ctx)
    registry = PluginRegistry()

    spec = registry.get(name)
    if spec is None:
        raise PluginError(i18n._("plugin.not_registered", name=name))

    if not dry_run and not yes and not typer.confirm(
        i18n._("plugin.uninstall_confirm", name=name)
    ):
        typer.echo(i18n._("common.aborted"))
        raise typer.Exit(code=0)

    if dry_run:
        typer.echo(i18n._("plugin.uninstall_dry_run", name=name))
        return

    package = spec.package or name
    cmd = pip_cmd()
    args = [*cmd, "uninstall", "--quiet", package]
    if cmd[0] != "uv":
        args.append("-y")
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise PluginError(
            i18n._("plugin.uninstall_failed", name=name, exc=exc)
        ) from exc

    registry.remove(name)

    stats = PluginStats()
    stats.record_uninstall(name)

    typer.echo(i18n._("plugin.uninstalled", name=name))


# ---------------------------------------------------------------------------
# trust
# ---------------------------------------------------------------------------


@app.command("trust")
def trust_plugin(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Registered plugin name"),
    level: TrustLevel = typer.Argument(..., help="New trust level"),
) -> None:
    """Update the trust level of a registered plugin."""
    i18n: I18n = get_i18n_from_ctx(ctx)
    registry = PluginRegistry()
    if not registry.trust(name, level):
        raise PluginError(i18n._("plugin.not_registered", name=name))
    typer.echo(i18n._("plugin.trust_set", name=name, level=level))


# ---------------------------------------------------------------------------
# rate
# ---------------------------------------------------------------------------


@app.command("rate")
def rate_plugin(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Registered plugin name"),
    score: float = typer.Argument(..., help="Rating score (1-5)"),
) -> None:
    """Rate a registered plugin."""
    i18n: I18n = get_i18n_from_ctx(ctx)
    registry = PluginRegistry()
    if registry.get(name) is None:
        raise PluginError(i18n._("plugin.not_registered", name=name))

    if not 1 <= score <= 5:
        raise PluginError(i18n._("plugin.rate_invalid"))

    stats = PluginStats()
    stats.record_rate(name, score)

    typer.echo(i18n._("plugin.rated", name=name, score=score))


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@app.command("stats")
def plugin_stats(ctx: typer.Context) -> None:
    """Show local plugin usage statistics."""
    i18n: I18n = get_i18n_from_ctx(ctx)
    stats = PluginStats()
    summary = stats.summary()
    if not summary:
        typer.echo(i18n._("plugin.no_stats"))
        return

    typer.echo(
        f"{i18n._('plugin.stats.header_plugin'):<30} "
        f"{i18n._('plugin.stats.header_installs'):<10} "
        f"{i18n._('plugin.stats.header_uninstalls'):<12} "
        f"{i18n._('plugin.stats.header_rating')}"
    )
    for name, data in summary.items():
        rating = data.get("rating", {})
        score = rating.get("score", 0.0)
        count = rating.get("count", 0)
        typer.echo(
            f"{name:<30} {data.get('installs', 0):<10} "
            f"{data.get('uninstalls', 0):<12} {score} ({count})"
        )


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


@app.command("update")
def update_plugins(
    ctx: typer.Context,
    name: str | None = typer.Argument(None, help="Plugin name to update"),
    all: bool = typer.Option(False, "--all", help="Update all plugins"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show actions without executing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmations"),
) -> None:
    """Check for and install plugin updates."""
    i18n: I18n = get_i18n_from_ctx(ctx)

    if name is None and not all:
        raise PluginError(i18n._("plugin.update_name_or_all"))

    registry = PluginRegistry()
    index = RegistryIndex()

    plugins = registry.list()
    if name:
        spec = registry.get(name)
        if spec is None:
            raise PluginError(i18n._("plugin.not_registered", name=name))
        plugins = [spec]

    updates: list[tuple[PluginSpec, dict[str, Any], str, str | None, str | None]] = []
    skipped: list[str] = []

    for spec in plugins:
        # Skip built-in plugins
        if spec.trust_level == TrustLevel.BUILTIN:
            typer.echo(f"{spec.name}: {i18n._('plugin.update_skip_builtin')}")
            skipped.append(spec.name)
            continue

        # Skip local-source plugins (can't upgrade via pip)
        source_path = Path(spec.source)
        if source_path.exists():
            typer.echo(f"{spec.name}: {i18n._('plugin.update_skip_local')}")
            skipped.append(spec.name)
            continue

        # Look up latest version in the registry
        try:
            latest_data = index.get(spec.name)
        except Exception:  # noqa: BLE001
            latest_data = None

        if latest_data is None:
            typer.echo(f"{spec.name}: {i18n._('plugin.update_skip_not_installed')}")
            skipped.append(spec.name)
            continue

        latest_version_str = latest_data.get("version")
        if not latest_version_str:
            skipped.append(spec.name)
            continue

        latest_version = parse_version(latest_version_str)
        package = spec.package or spec.source
        installed = _installed_version(package)

        if installed is not None and installed >= latest_version:
            typer.echo(
                f"{spec.name}: "
                f"{i18n._('plugin.update_skip_latest', installed=installed, latest=latest_version)}"
            )
            skipped.append(spec.name)
            continue

        sha256 = latest_data.get("sha256")
        signature = latest_data.get("signature")
        updates.append((spec, latest_data, package, sha256, signature))

    if not updates:
        typer.echo(i18n._("plugin.update_none"))
        return

    if dry_run:
        for spec, _latest_data, package, _sha256, _sig in updates:
            typer.echo(
                i18n._(
                    "plugin.update_dry_run",
                    plugin_name=spec.name,
                    source_for_pip=package,
                )
            )
        return

    if not yes and not typer.confirm(i18n._("plugin.update_confirm")):
        typer.echo(i18n._("common.aborted"))
        raise typer.Exit(code=0)

    cmd = pip_cmd()
    for spec, latest_data, package, sha256, signature in updates:
        latest_version_str = latest_data.get("version", "0.0.0")
        use_sandbox = spec.trust_level in (TrustLevel.COMMUNITY, TrustLevel.UNTRUSTED)
        install_spec = package

        if sha256:
            try:
                downloaded = download_and_verify(
                    package,
                    sha256_hex=sha256,
                    signature_b64=signature,
                    public_key_b64=None,
                )
            except PackageVerificationError as exc:
                raise PluginError(
                    i18n._("plugin.update_verify_failed", plugin_name=spec.name, exc=exc)
                ) from exc
            install_spec = str(downloaded)

        try:
            result = _run_pip_install(
                cmd,
                install_spec,
                upgrade=True,
                sandbox=use_sandbox,
                env_whitelist=[
                    "PATH", "HOME", "USER", "LANG", "LC_ALL",
                    "PIP_INDEX_URL", "VIRTUAL_ENV", "XDG_CACHE_HOME", "UV_CACHE_DIR",
                ],
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, result.args, output=result.stdout, stderr=result.stderr
                )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            raise PluginError(
                i18n._("plugin.update_failed", plugin_name=spec.name, exc=exc)
            ) from exc

        registry.add(
            PluginSpec(
                name=spec.name,
                version=latest_version_str,
                source=spec.source,
                package=spec.package,
                trust_level=spec.trust_level,
                capabilities=spec.capabilities,
                sha256=sha256,
                signature=signature,
            )
        )

        typer.echo(
            i18n._("plugin.updated", plugin_name=spec.name, version=latest_version_str)
        )
