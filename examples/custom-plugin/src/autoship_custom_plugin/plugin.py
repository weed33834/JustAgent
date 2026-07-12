"""AutoShip-CLI 真实插件示例：项目健康检查。

该插件演示了三个常见扩展点的实际用法：

- ``pre_commit``: 在提交前检查常见代码规范问题（TODO/FIXME、行尾空白、
  文件末尾空行）。
- ``pre_verify``: 在运行验证命令前做快速静态检查。
- ``on_error``: 当 ``autoship verify --fix`` 失败时给出可操作建议。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from autoship_sdk import CommandContext, FixSuggestion, Plugin, VerifyError, hook


class ProjectGuardPlugin(Plugin):
    """一个实用的项目健康检查插件。"""

    _TODOLIKE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
    _TRAILING_WHITESPACE = re.compile(r"[ \t]+$")

    @hook
    def pre_commit(self, context: CommandContext) -> None:
        """检查暂存区文件中的常见代码规范问题。"""
        staged = self._staged_python_files(context.project_root)
        issues: list[str] = []

        for path in staged:
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if self._TODOLIKE.search(line):
                    issues.append(f"{path}:{lineno}: {line.strip()}")
                if self._TRAILING_WHITESPACE.search(line):
                    issues.append(f"{path}:{lineno}: trailing whitespace")
            if text and not text.endswith("\n"):
                issues.append(f"{path}: missing trailing newline")

        if issues:
            print("[project-guard] Found potential issues in staged files:")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")

    @hook
    def pre_verify(self, context: CommandContext) -> None:
        """快速语法检查：避免让慢速测试套件去跑有语法错误的代码。"""
        python_files = list(context.project_root.rglob("*.py"))
        if not python_files:
            return

        cmd = ["python", "-m", "py_compile"] + [str(p) for p in python_files[:50]]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                "[project-guard] Syntax error detected; consider fixing before running full tests."
            )

    @hook
    def on_error(self, context: CommandContext, error: Exception) -> FixSuggestion | None:
        """针对验证失败给出可操作建议。"""
        if not context.extras.get("fix"):
            return None

        if isinstance(error, VerifyError):
            return FixSuggestion(
                description="[project-guard] Run `autoship clean` and then re-run `autoship verify`.",
            )

        return None

    @staticmethod
    def _staged_python_files(project_root: Path) -> list[Path]:
        """获取暂存区中的 Python 文件路径。"""
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []
        files: list[Path] = []
        for name in result.stdout.splitlines():
            path = project_root / name
            if path.suffix == ".py" and path.exists():
                files.append(path)
        return files


def register() -> ProjectGuardPlugin:
    """``autoship.plugins`` 入口点使用的工厂函数。"""
    return ProjectGuardPlugin()
