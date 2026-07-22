"""Render default configuration templates."""

from __future__ import annotations


def render_default_config(project_type: str, default_tier: int = 2) -> str:
    """Render a default ``.autoship.toml`` for the given project type."""
    return f'''# AutoShip configuration
schema_version = 1
project_type = "{project_type}"

[model]
default_tier = {default_tier}
fallback = true

# Local model backend example (Ollama).
# Replace with your own endpoint / model as needed.
[[model.backends]]
provider = "ollama"
base_url = "http://127.0.0.1:11434/v1"
model = "qwen2.5:7b"
timeout = 30.0

[clean]
enabled = true
tools = ["autoflake", "black"]

[commit]
enabled = true
max_tokens = 512
conventional_commits = true
'''
