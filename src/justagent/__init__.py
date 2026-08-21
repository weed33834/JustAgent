"""JustAgent — an auditable multi-agent platform.

The engine provides:
  - Iterative tool-calling agent loop with Plan/Act modes
  - Permission engine (allow / deny / ask) and shell safety
  - Checkpoints (shadow git), session persistence, audit logging
  - MCP client, subagents, context engineering, compaction
  - Multi-agent orchestration primitives

Vertical applications (e.g. the bundled legal vertical) live in
``justagent.verticals`` and register via entry points — see
DESIGN.dual-track.zh.md for the layering rules.
"""

__version__ = "3.1.0"

# Optional engine capability modules (feature-detection for integrations).
CAPABILITY_MODULES = [
    "knowledge",
    "communication",
    "resources",
    "security",
    "orchestration",
]
