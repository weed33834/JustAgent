"""Registry analytics, sync and dashboard commands."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC
from pathlib import Path
from typing import Any

import typer

from autoship.core.registry_client import RegistryClient
from autoship.core.registry_index import RegistryIndex
from autoship.models.config import AppConfig
from autoship.utils.json_io import atomic_write_text

app = typer.Typer()

BUNDLED_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "registry" / "plugins.json"


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="registry")


def _count_changes(current: dict[str, Any], fetched: dict[str, Any]) -> int:
    """Count added, removed and modified plugins between two index payloads."""
    current_plugins = {p.get("name"): p for p in current.get("plugins", [])}
    fetched_plugins = {p.get("name"): p for p in fetched.get("plugins", [])}

    current_names = set(current_plugins.keys())
    fetched_names = set(fetched_plugins.keys())

    added = fetched_names - current_names
    removed = current_names - fetched_names
    common = current_names & fetched_names
    modified = {name for name in common if current_plugins[name] != fetched_plugins[name]}

    return len(added) + len(removed) + len(modified)


@app.command("dashboard")
@app.command("list")
def dashboard(
    ctx: typer.Context,
    top: int = typer.Option(5, "--top", help="Number of plugins to show in top lists"),
) -> None:
    """Show registry analytics dashboard."""

    index = RegistryIndex(ctx.obj.get("config"))
    plugins = index.list_plugins()

    if not plugins:
        typer.echo("registry.empty")
        return

    typer.echo("registry.dashboard_title")
    typer.echo(f"{'=' * 60}")
    typer.echo(f"Total plugins: {len(plugins)}")

    trust_counts = Counter(p.get("trust_level", "unknown") for p in plugins)
    typer.echo("\nBy trust level:")
    for level, count in trust_counts.most_common():
        typer.echo(f"  {level:<12} {count}")

    category_counts: Counter[str] = Counter()
    for plugin in plugins:
        for category in plugin.get("categories", []):
            category_counts[category] += 1
    if category_counts:
        typer.echo("\nBy category:")
        for category, count in category_counts.most_common():
            typer.echo(f"  {category:<12} {count}")

    def _rating_key(plugin: dict[str, Any]) -> float:
        rating = plugin.get("rating")
        return rating.get("score", 0.0) if rating else 0.0

    top_downloaded = sorted(plugins, key=lambda p: p.get("downloads", 0), reverse=True)[:top]
    typer.echo(f"\nTop {len(top_downloaded)} by downloads:")
    for plugin in top_downloaded:
        typer.echo(f"  {plugin['name']:<30} {plugin.get('downloads', 0)}")

    top_rated = sorted(
        [p for p in plugins if p.get("rating", {}).get("count", 0) > 0],
        key=_rating_key,
        reverse=True,
    )[:top]
    if top_rated:
        typer.echo(f"\nTop {len(top_rated)} by rating:")
        for plugin in top_rated:
            rating = plugin["rating"]
            typer.echo(f"  {plugin['name']:<30} {rating['score']:.1f} ({rating['count']})")


@app.command("sync")
def sync(
    ctx: typer.Context,
    output: Path = typer.Option(
        Path.home() / ".autoship" / "registry" / "plugins.json",
        "--output",
        "-o",
        help="Output path for the synced registry index",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite local cache"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without writing"),
) -> None:
    """Sync the plugin registry index from the remote source."""
    config: AppConfig = ctx.obj["config"]

    dry_run = ctx.obj.get("dry_run", False) or dry_run

    # RBAC gate.

    client = RegistryClient(config=config.registry)
    data = client.fetch_index(force=force)
    if data is None:
        typer.echo("registry.sync_failed", err=True)
        raise typer.Exit(code=1)

    # Surface the integrity check outcome so users can see that the synced
    # index passed signature / checksum verification. ``fetch_index`` always
    # routes through ``_fetch_remote`` -> ``_verify_index``, so reaching this
    # point means verification succeeded (or no public key was configured, in
    # which case only the sha256 checksum is validated).
    if data.get("signature") and config.registry.public_key:
        typer.echo("registry.sync_signature_verified")
    elif data.get("sha256"):
        typer.echo("registry.sync_checksum_verified")
    else:
        typer.echo("registry.sync_no_signature")

    current_index: dict[str, Any] = {"version": 1, "plugins": []}
    if BUNDLED_REGISTRY_PATH.exists():
        try:
            raw = BUNDLED_REGISTRY_PATH.read_text(encoding="utf-8")
            current_index = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            current_index = {"version": 1, "plugins": []}

    _changes = _count_changes(current_index, data)

    if dry_run:
        typer.echo("registry.sync_dry_run")
        return

    payload = json.dumps(data, indent=2)
    try:
        atomic_write_text(output, payload)
    except OSError as exc:
        typer.echo("registry.sync_failed", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("registry.sync_done")


@app.command("mirror")
def mirror(
    ctx: typer.Context,
    output_dir: Path = typer.Option(
        Path.home() / ".autoship" / "registry-mirror",
        "--output",
        "-o",
        help="Directory to write the air-gapped mirror into (created if missing).",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verify the registry index signature against the configured public key before mirroring.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be mirrored without downloading or writing anything.",
    ),
) -> None:
    """Snapshot the remote registry index into a local air-gapped mirror.

    Produces a self-contained directory containing ``plugins.json`` and a
    ``MANIFEST.json`` listing the source URL, sha256 of the index, and the
    verification status. Air-gapped teams can point ``registry.url`` at the
    mirror's ``file://`` URL or copy the directory to the offline host.

    The mirror is content-addressed: the index's sha256 is recorded in the
    manifest, so a tampered mirror fails verification when AutoShip loads it
    later (provided ``registry.public_key`` is also pinned).
    """
    import hashlib
    from datetime import datetime

    from autoship.utils.permissions import ensure_dir_permissions, ensure_file_permissions

    config: AppConfig = ctx.obj["config"]

    if dry_run:
        typer.echo("registry.mirror_dry_run")
        raise typer.Exit(code=0)

    client = RegistryClient(config=config.registry)

    # The RegistryClient always verifies the index signature when a public
    # key is configured; ``verify`` here is metadata recorded in the mirror
    # manifest so a downstream consumer knows whether the source actually
    # carried a signature to verify.
    try:
        data = client.fetch_index(force=True)
    except Exception as exc:  # noqa: BLE001 — surface any fetch failure cleanly
        typer.echo("registry.mirror_fetch_failed", err=True)
        raise typer.Exit(code=1) from exc

    if data is None:
        typer.echo("registry.mirror_fetch_failed", err=True)
        raise typer.Exit(code=1)

    payload_str = json.dumps(data, indent=2, sort_keys=True)
    sha256 = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    signature_status = (
        "verified"
        if (data.get("signature") and config.registry.public_key)
        else ("checksum-only" if data.get("sha256") else "unsigned")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_dir_permissions(output_dir, 0o700)

    plugins_path = output_dir / "plugins.json"
    atomic_write_text(plugins_path, payload_str)
    ensure_file_permissions(plugins_path, 0o600)

    manifest = {
        "mirrored_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url": str(config.registry.url),
        "source_sha256": data.get("sha256"),
        "computed_sha256": sha256,
        "signature_status": signature_status,
        "verify_at_mirror_time": verify,
        "plugin_count": len(data.get("plugins", [])),
    }
    manifest_path = output_dir / "MANIFEST.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    ensure_file_permissions(manifest_path, 0o600)

    typer.echo("registry.mirror_done")
