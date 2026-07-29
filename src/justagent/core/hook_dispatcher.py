"""Plugin hook dispatcher based on pluggy."""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import textwrap
from collections.abc import Sequence
from contextlib import suppress
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pluggy
import structlog

from justagent.core.context import CommandContext
from justagent.core.plugin_registry import PluginRegistry, TrustLevel
from justagent.core.sandbox import SandboxRunner
from justagent.exceptions import PluginError
from justagent.hookspec import JustAgentHookSpec
from justagent.plugins.defaults import FixSuggestion

if TYPE_CHECKING:
    from justagent.core.audit_logger import AuditLogger

logger = structlog.get_logger("justagent")

_SANDBOX_SCRIPT = r"""
import json
import os
import sys
from importlib.metadata import entry_points
from pathlib import Path

from justagent.core.context import CommandContext
from justagent.plugins.defaults import FixSuggestion
from justagent.models.config import AppConfig

def _serialize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, FixSuggestion):
        return {"description": value.description, "patch": value.patch}
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    raise TypeError(f"Unsupported return type: {type(value)}")

def main() -> None:
    payload = json.loads(sys.argv[1])
    plugin_name = payload["plugin_name"]
    hook_name = payload["hook_name"]
    ctx_data = payload["context"]
    kwargs = payload.get("kwargs", {})

    config = AppConfig.model_validate(ctx_data["config"])
    ctx = CommandContext(
        command=ctx_data["command"],
        project_root=Path(ctx_data["project_root"]),
        config=config,
        verbose=ctx_data.get("verbose", False),
        dry_run=ctx_data.get("dry_run", False),
        yes=ctx_data.get("yes", False),
        trace_id=ctx_data.get("trace_id", ""),
        extras=ctx_data.get("extras", {}),
    )

    eps = entry_points()
    group = (
        eps.select(group="justagent.plugins")
        if hasattr(eps, "select")
        else eps.get("justagent.plugins", [])
    )
    ep = next((ep for ep in group if ep.name == plugin_name), None)
    if ep is None:
        print(json.dumps({"error": f"Plugin {plugin_name} not found"}), file=sys.stderr)
        sys.exit(1)

    plugin = ep.load()
    if callable(plugin) and not hasattr(plugin, hook_name):
        plugin = plugin()

    method = getattr(plugin, hook_name, None)
    if method is None:
        print(json.dumps({"result": None}))
        return

    result = method(context=ctx, **kwargs)
    print(json.dumps({"result": _serialize(result)}))

if __name__ == "__main__":
    main()
"""


class HookDispatcher:
    """Manages plugin registration and hook invocation.

    Built-in and verified plugins are invoked directly in the host process.
    Community and untrusted plugins are executed inside a subprocess sandbox
    so that a compromised plugin cannot directly access the user's environment.
    """

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        sandbox_runner_factory: type[SandboxRunner] = SandboxRunner,
        no_sandbox: bool = False,
        load_builtins: bool = True,
        load_entry_points: bool = True,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.pm = pluggy.PluginManager("justagent")
        self.pm.add_hookspecs(JustAgentHookSpec)
        self.registry = registry or PluginRegistry()
        self._sandbox_runner_factory = sandbox_runner_factory
        self._no_sandbox = no_sandbox
        self._audit_logger = audit_logger
        self._builtin_names: set[str] = set()
        if load_builtins:
            self._load_builtin()
        if load_entry_points:
            self._discover_entry_points()

    def _load_builtin(self) -> None:
        """加载内置插件，单个失败时跳过并警告。"""
        # 内置插件列表；web_search 已移除（依赖 duckduckgo-search/diskcache）。
        builtin_modules = [
            "defaults",
            "security_scan",
            "docker_ship",
            "typecheck",
        ]
        for mod_name in builtin_modules:
            try:
                mod = import_module(f"justagent.plugins.{mod_name}")
            except Exception as exc:  # noqa: BLE001 - 不让单个可选插件拖垮整体
                logger.warning("跳过内置插件 %s（导入失败：%s）", mod_name, exc)
                continue
            plugin = getattr(mod, "plugin", None)
            if plugin is None:
                continue
            self.pm.register(plugin)
            name = self.pm.get_name(plugin)
            if name:
                self._builtin_names.add(name)

    def _discover_entry_points(self) -> None:
        """Discover and register external plugins via ``justagent.plugins`` entry points."""
        try:
            eps = entry_points()
            group: Sequence[Any] = eps.select(group="justagent.plugins")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to discover entry-point plugins: %s", exc)
            return

        for ep in group:
            try:
                plugin = ep.load()
                if inspect.isfunction(plugin) or inspect.ismethod(plugin):
                    plugin = plugin()
                self.pm.register(plugin, name=ep.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load plugin %s: %s", ep.name, exc)

    def _trust_level_for(self, plugin_name: str) -> TrustLevel | None:
        """Return the trust level for a registered plugin, or None if unknown."""
        if plugin_name in self._builtin_names:
            return TrustLevel.BUILTIN
        spec = self.registry.get(plugin_name)
        return spec.trust_level if spec is not None else None

    def _serialize_context(self, context: CommandContext) -> dict[str, Any]:
        """Convert a CommandContext to a JSON-serializable dict."""
        config_dict: dict[str, Any] = context.config.model_dump(mode="json", warnings=False)
        return {
            "command": context.command,
            "project_root": str(context.project_root),
            "config": config_dict,
            "verbose": context.verbose,
            "dry_run": context.dry_run,
            "yes": context.yes,
            "trace_id": context.trace_id,
            "extras": context.extras,
        }

    def _call_in_sandbox(
        self,
        plugin_name: str,
        hook_name: str,
        context: CommandContext,
        kwargs: dict[str, Any],
    ) -> Any:
        """Run a plugin hook inside a subprocess sandbox."""
        payload = {
            "plugin_name": plugin_name,
            "hook_name": hook_name,
            "context": self._serialize_context(context),
            "kwargs": kwargs,
        }
        try:
            json.dumps(payload)
        except (TypeError, ValueError) as exc:
            raise PluginError(
                f"Cannot sandbox hook {hook_name} for {plugin_name}: context/kwargs are not serializable"
            ) from exc

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(textwrap.dedent(_SANDBOX_SCRIPT))
            script_path = Path(f.name)

        try:
            # Preserve PYTHONPATH so the subprocess can find justagent and installed plugins.
            env_whitelist = [
                "PATH",
                "HOME",
                "USER",
                "LANG",
                "LC_ALL",
                "PYTHONPATH",
                "VIRTUAL_ENV",
                "XDG_CACHE_HOME",
                "UV_CACHE_DIR",
            ]
            runner = self._sandbox_runner_factory(network=False, env_whitelist=env_whitelist)
            result = runner.run(
                [sys.executable, str(script_path), json.dumps(payload)],
                timeout=60.0,
            )
        finally:
            with suppress(OSError):
                os.unlink(script_path)

        if result.returncode != 0:
            raise PluginError(
                f"Sandboxed hook {hook_name} for {plugin_name} failed: {result.stderr}"
            )

        try:
            data = cast("dict[str, Any]", json.loads(result.stdout))
        except json.JSONDecodeError as exc:
            raise PluginError(
                f"Sandboxed hook {hook_name} for {plugin_name} returned invalid JSON"
            ) from exc

        raw = cast(Any, data.get("result"))
        if raw is None:
            return None
        if isinstance(raw, dict) and "description" in raw:
            raw_dict = cast("dict[str, Any]", raw)
            description = cast(str, raw_dict["description"])
            patch = cast("str | None", raw_dict.get("patch"))
            return cast(Any, FixSuggestion(description=description, patch=patch))
        return cast(Any, raw)

    def call(
        self,
        hook_name: str,
        context: CommandContext,
        *,
        fail_fast: bool = True,
        **kwargs: Any,
    ) -> list[Any]:
        """Invoke the named hook for all registered plugins.

        Args:
            hook_name: Name of the hook to invoke.
            context: The current CommandContext.
            fail_fast: If True, a plugin exception aborts the command.
            **kwargs: Additional keyword arguments passed to hook implementations.

        Returns:
            A list of hook return values.
        """
        hook = getattr(self.pm.hook, hook_name)
        results: list[Any] = []
        for impl in hook.get_hookimpls():
            plugin_name = self.pm.get_name(impl.plugin) or ""
            trust = self._trust_level_for(plugin_name)

            try:
                if not self._no_sandbox and trust in (TrustLevel.COMMUNITY, TrustLevel.UNTRUSTED):
                    result = self._call_in_sandbox(plugin_name, hook_name, context, kwargs)
                else:
                    if (
                        self._no_sandbox
                        and trust is not None
                        and trust in (TrustLevel.COMMUNITY, TrustLevel.UNTRUSTED)
                    ):
                        logger.warning(
                            "Plugin %r (trust=%s) executed in the host process because "
                            "no_sandbox mode is enabled; it is NOT sandbox-isolated.",
                            plugin_name,
                            trust.value,
                        )
                        if self._audit_logger is not None:
                            self._audit_logger.record(
                                "plugin.unsandboxed",
                                {
                                    "plugin": plugin_name,
                                    "trust_level": trust.value,
                                    "reason": "no_sandbox",
                                },
                            )
                    result = impl.function(context=context, **kwargs)
            except Exception as exc:
                logger.warning("Hook %s failed for plugin %s: %s", hook_name, plugin_name, exc)
                if fail_fast:
                    raise PluginError(f"Hook {hook_name} failed: {exc}") from exc
                continue

            if result is not None:
                results.append(result)

        return results
