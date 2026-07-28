"""Detect the type of project in a directory."""

from __future__ import annotations

from pathlib import Path

PROJECT_MARKERS: dict[str, list[str]] = {
    "python": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"],
    "node": ["package.json"],
    "rust": ["Cargo.toml"],
    "go": ["go.mod"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
}


def detect_project_type(root: Path) -> str:
    """Return the project type based on marker files, or ``generic``."""
    for ptype, markers in PROJECT_MARKERS.items():
        if any((root / marker).exists() for marker in markers):
            return ptype
    return "generic"
