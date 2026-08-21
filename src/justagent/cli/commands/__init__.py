"""自动发现并注册所有 Typer 子命令。"""

from __future__ import annotations

import sys
import warnings
from importlib import import_module
from importlib.metadata import entry_points
from pkgutil import iter_modules

import typer


def _register_vertical_commands(parent: typer.Typer) -> None:
    """注册垂直包通过 ``justagent.cli`` entry-point 发布的子命令。

    引擎不直接 import 任何垂直包；垂直包以
    ``[project.entry-points."justagent.cli"]`` 声明模块路径，
    模块需暴露 ``register(parent)`` 函数。单个入口失败只告警。
    """
    try:
        eps = entry_points(group="justagent.cli")
    except TypeError:  # pragma: no cover - Python 3.9 fallback signature
        eps = entry_points().get("justagent.cli", [])
    for ep in eps:
        try:
            mod = ep.load()
        except Exception as exc:  # noqa: BLE001 - 不让单个垂直拖垮 CLI 启动
            warnings.warn(f"跳过命令入口 {ep.name}（加载失败：{exc}）。", RuntimeWarning, stacklevel=2)
            continue
        register = getattr(mod, "register", None)
        if callable(register):
            register(parent)


def register_all(parent: typer.Typer) -> None:
    """注册每个模块级的 ``register`` 函数为 Typer 子命令。

    单个模块导入失败时跳过并发出警告，避免一个可选命令的依赖问题
    阻断整个 CLI 启动。
    """
    package = __package__ or "justagent.cli.commands"
    for _, module_name, _ in iter_modules(__path__, prefix=package + "."):
        try:
            mod = import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - 不让单个可选命令拖垮整体
            warnings.warn(
                f"跳过命令模块 {module_name}（导入失败：{exc}）。",
                RuntimeWarning,
                stacklevel=2,
            )
            # 同步打印到 stderr，确保用户在非 Python 警告过滤下也能看到。
            print(f"[justagent] 跳过命令模块 {module_name}：{exc}", file=sys.stderr)
            continue
        register = getattr(mod, "register", None)
        if callable(register):
            register(parent)
    _register_vertical_commands(parent)
