"""``myagent plugin`` 命令：本地插件的安装、卸载、列表。

远程 registry 搜索已移除（企业级功能，偏离本地 AI 编码智能体定位）。
"""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import Any, cast

import typer
from packaging.version import Version
from packaging.version import parse as parse_version

from myagent.core.plugin_registry import CapabilityManifest, PluginRegistry, PluginSpec, TrustLevel
from myagent.core.sandbox import SandboxRunner
from myagent.exceptions import PluginError
from myagent.utils.hashing import pip_cmd

app = typer.Typer()


def _read_project_name(source: str) -> str | None:
    """从本地 pyproject.toml 读取 project.name。"""
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
    """执行 pip install，可选在沙箱中运行。"""
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
    """返回已安装包的版本，未安装则返回 None。"""
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
) -> None:
    if skip_trust_check or yes:
        return

    if plugin_trust in (TrustLevel.COMMUNITY, TrustLevel.UNTRUSTED):
        level_name = "COMMUNITY" if plugin_trust == TrustLevel.COMMUNITY else "UNTRUSTED"
        typer.echo(
            f"警告：插件 '{plugin_name}' 信任级别为 {level_name}。"
            "此插件可能拥有系统完全访问权限。"
        )
        typer.echo(f"请求的权限：{', '.join(capabilities.summary()) or 'none'}")
        if not typer.confirm("信任此插件并继续？", abort=False):
            typer.echo("已中止。")
            raise typer.Exit(code=0)


def register(parent: typer.Typer) -> None:
    parent.add_typer(app, name="plugin")


@app.command("list")
def list_plugins(ctx: typer.Context) -> None:
    """列出已注册的插件及其信任级别。"""
    registry = PluginRegistry()
    plugins = registry.list()
    if not plugins:
        typer.echo("没有已注册的插件。")
        typer.echo("使用 'myagent plugin install <source>' 安装插件。")
        return

    typer.echo(f"{'名称':<30} {'版本':<10} {'信任':<12} {'来源'}")
    for plugin in plugins:
        typer.echo(
            f"{plugin.name:<30} {plugin.version:<10} {plugin.trust_level.value:<12} {plugin.source}"
        )


@app.command("install")
def install(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="包名或本地路径"),
    name: str | None = typer.Option(None, "--name", help="插件注册名"),
    version: str | None = typer.Option(None, "--version", help="插件版本"),
    trust: TrustLevel | None = typer.Option(None, "--trust", help="初始信任级别"),
    dry_run: bool = typer.Option(False, "--dry-run", help="仅显示操作而不执行"),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
    skip_trust_check: bool = typer.Option(False, "--skip-trust-check", help="跳过信任级别警告"),
    no_sandbox: bool = typer.Option(False, "--no-sandbox", help="不在沙箱中运行 pip install"),
) -> None:
    """安装插件包并在本地注册。"""
    registry = PluginRegistry()

    plugin_name = name or Path(source).name
    package = _read_project_name(source) or plugin_name
    plugin_version = version or "0.0.0"
    plugin_trust = trust or TrustLevel.COMMUNITY
    capabilities = _capabilities_default()

    if not dry_run:
        _confirm_trust(plugin_name, plugin_trust, capabilities, yes, skip_trust_check)

    if not dry_run and not yes and not typer.confirm(f"安装插件 '{plugin_name}'？"):
        typer.echo("已中止。")
        raise typer.Exit(code=0)

    if dry_run:
        typer.echo(
            f"[dry-run] 将安装插件 '{plugin_name}' ({plugin_version}) "
            f"信任级别 {plugin_trust.value}"
        )
        return

    use_sandbox = plugin_trust in (TrustLevel.COMMUNITY, TrustLevel.UNTRUSTED)
    cmd = pip_cmd()

    try:
        result = _run_pip_install(
            cmd,
            source,
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
        raise PluginError(f"安装插件 '{plugin_name}' 失败：{exc}") from exc

    registry.add(
        PluginSpec(
            name=plugin_name,
            version=plugin_version,
            source=source,
            package=package,
            trust_level=plugin_trust,
            capabilities=capabilities,
        )
    )
    typer.echo(f"插件 '{plugin_name}' 安装成功。")


@app.command("uninstall")
def uninstall(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="要卸载的插件名"),
    dry_run: bool = typer.Option(False, "--dry-run", help="仅显示操作而不执行"),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
) -> None:
    """卸载插件包并从本地注册表中移除。"""
    registry = PluginRegistry()

    spec = registry.get(name)
    if spec is None:
        raise PluginError(f"插件 '{name}' 未注册。")

    if not dry_run and not yes and not typer.confirm(f"卸载插件 '{name}'？"):
        typer.echo("已中止。")
        raise typer.Exit(code=0)

    if dry_run:
        typer.echo(f"[dry-run] 将卸载插件 '{name}'")
        return

    package = spec.package or name
    cmd = pip_cmd()
    args = [*cmd, "uninstall", "--quiet", package]
    if cmd[0] != "uv":
        args.append("-y")
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise PluginError(f"卸载插件 '{name}' 失败：{exc}") from exc

    registry.remove(name)
    typer.echo(f"插件 '{name}' 已卸载。")
