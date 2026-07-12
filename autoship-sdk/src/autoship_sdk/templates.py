"""Plugin project scaffolding utilities."""

from __future__ import annotations

import re
import string
from pathlib import Path


class TemplateError(Exception):
    """Raised when scaffolding fails."""


def _to_package_name(name: str) -> str:
    """Convert a plugin name to a valid Python package name."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        raise TemplateError(f"Invalid plugin name: {name!r}")
    return cleaned


def _to_class_name(name: str) -> str:
    """Convert a plugin name to a valid Python class name."""
    return "".join(part.capitalize() for part in _to_package_name(name).split("_"))


def _render(template: str, mapping: dict[str, str]) -> str:
    """Render a template string with ``string.Template``."""
    return string.Template(template).substitute(mapping)


def create_plugin(
    target_dir: Path,
    plugin_name: str,
    description: str = "",
    repository_url: str = "",
) -> Path:
    """Scaffold a minimal AutoShip plugin project at ``target_dir``.

    Args:
        target_dir: Directory where the plugin project will be created.
        plugin_name: Short identifier for the plugin, e.g. ``security-scan``.
        description: Plugin description.
        repository_url: Optional repository URL.

    Returns:
        The path to the created project root.

    Raises:
        TemplateError: If the plugin name is invalid or the target already exists.
    """
    package_name = _to_package_name(plugin_name)
    class_name = _to_class_name(plugin_name)
    distribution_name = f"autoship-{plugin_name.lower()}"
    project_root = Path(target_dir)

    if project_root.exists() and any(project_root.iterdir()):
        raise TemplateError(f"Target directory is not empty: {project_root}")

    src_dir = project_root / "src" / package_name
    src_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "package_name": package_name,
        "class_name": class_name,
        "plugin_name": plugin_name,
        "distribution_name": distribution_name,
        "description": description or f"AutoShip plugin: {plugin_name}",
        "repository_url": repository_url or "https://github.com/MS33834/autoship-cli",
    }

    template_dir = Path(__file__).parent / "templates" / "plugin"
    files = {
        "pyproject.toml": "pyproject.toml.tpl",
        "README.md": "README.md.tpl",
        f"src/{package_name}/plugin.py": "plugin.py.tpl",
        f"src/{package_name}/__init__.py": None,
    }

    for dest_name, template_name in files.items():
        dest = project_root / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if template_name is None:
            dest.write_text('"""AutoShip plugin package."""\n')
        else:
            template = (template_dir / template_name).read_text(encoding="utf-8")
            dest.write_text(_render(template, mapping), encoding="utf-8")

    return project_root
