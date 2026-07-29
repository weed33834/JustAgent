"""Skills system — SKILL.md loader with progressive disclosure.

Skills are markdown files that extend the agent's capabilities with
domain-specific instructions. Each skill lives in its own directory
under ``.justagent/skills/`` and contains a ``SKILL.md`` file with
YAML-like frontmatter (name, description, triggers) and markdown body.

Progressive disclosure: only the name + description are injected into
the system prompt initially. When the LLM invokes the skill (via a
tool call or slash command), the full body is loaded and inserted
into the conversation context.

Reference: Cline's ``.clineskills/`` and Claude Code's skill system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from justagent.exceptions import MyAgentError
from justagent.utils.atomic_write import atomic_write_text


class SkillError(MyAgentError):
    """Raised when a skill operation fails."""


@dataclass(frozen=True)
class SkillTrigger:
    """A single trigger that activates a skill.

    Attributes:
        type: Trigger type — ``"keyword"`` (matched against user text),
            ``"tool"`` (matched against a tool name), or ``"manual"``
            (only activated by an explicit invocation).
        value: The trigger value — e.g. the keyword text or tool name.
    """

    type: str
    value: str


@dataclass(frozen=True)
class Skill:
    """A parsed skill loaded from a ``SKILL.md`` file.

    Attributes:
        name: Unique skill name (from frontmatter).
        description: Short human-readable description (from frontmatter).
        path: Path to the ``SKILL.md`` file.
        body: Full markdown body (everything after the frontmatter).
        triggers: Activation triggers.
        metadata: Extra frontmatter key/value pairs not covered by the
            fields above.
    """

    name: str
    description: str
    path: Path
    body: str
    triggers: list[SkillTrigger] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillSummary:
    """Lightweight skill summary for system-prompt injection.

    Only carries the name and description — the body is loaded
    separately via :meth:`SkillLoader.load_body` when the LLM
    explicitly invokes the skill (progressive disclosure).
    """

    name: str
    description: str


def _default_skills_dirs() -> list[str]:
    return [".justagent/skills"]


@dataclass
class SkillConfig:
    """Configuration for :class:`SkillLoader`.

    Attributes:
        skills_dirs: Directories (relative to project root or absolute)
            to scan for skill directories.
        max_skills: Maximum number of skills to load. Excess skills are
            dropped after sorting by name. ``0`` means unlimited.
        max_body_chars: Maximum number of characters returned by
            :meth:`SkillLoader.load_body`. Longer bodies are truncated.
            ``0`` means unlimited.
        enabled: If False, :meth:`SkillLoader.discover` returns an empty
            list.
    """

    skills_dirs: list[str] = field(default_factory=_default_skills_dirs)
    max_skills: int = 50
    max_body_chars: int = 50000
    enabled: bool = True


# ---------------------------------------------------------------------------
# Frontmatter parsing (line-based; no PyYAML dependency)
# ---------------------------------------------------------------------------


def _split_frontmatter(content: str) -> tuple[list[str], str]:
    """Split ``SKILL.md`` content into frontmatter lines and body.

    Raises :class:`SkillError` if the frontmatter delimiters are missing.
    """
    lines = content.split("\n")
    if not lines or lines[0].strip() != "---":
        raise SkillError("Skill file is missing frontmatter opening '---'")
    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx == -1:
        raise SkillError("Skill file is missing frontmatter closing '---'")
    frontmatter_lines = lines[1:close_idx]
    body = "\n".join(lines[close_idx + 1 :])
    # Conventionally a single blank line follows the closing delimiter;
    # strip it but keep the rest of the body verbatim.
    if body.startswith("\n"):
        body = body[1:]
    return frontmatter_lines, body


def _parse_trigger_block(block: list[str]) -> list[SkillTrigger]:
    """Parse a list of ``- type: ... / value: ...`` items into triggers."""
    triggers: list[SkillTrigger] = []
    current: dict[str, str] = {}
    for raw in block:
        stripped = raw.strip()
        if stripped.startswith("-"):
            # New list item — flush the previous one.
            if current:
                triggers.append(
                    SkillTrigger(
                        type=current.get("type", ""),
                        value=current.get("value", ""),
                    )
                )
                current = {}
            rest = stripped[1:].strip()
            if ":" in rest:
                k, _, v = rest.partition(":")
                current[k.strip()] = v.strip()
        elif ":" in stripped:
            k, _, v = stripped.partition(":")
            current[k.strip()] = v.strip()
    if current:
        triggers.append(
            SkillTrigger(
                type=current.get("type", ""),
                value=current.get("value", ""),
            )
        )
    return triggers


def _parse_frontmatter(
    frontmatter_lines: list[str],
) -> tuple[str, str, list[SkillTrigger], dict[str, str]]:
    """Parse frontmatter lines into ``(name, description, triggers, metadata)``.

    Raises :class:`SkillError` if the required ``name`` field is missing.
    """
    name = ""
    description = ""
    triggers: list[SkillTrigger] = []
    metadata: dict[str, str] = {}

    i = 0
    n = len(frontmatter_lines)
    while i < n:
        line = frontmatter_lines[i]
        if line.strip() == "":
            i += 1
            continue
        # Only top-level (non-indented) keys are parsed here; indented
        # lines belong to the preceding list block.
        if line[0] in (" ", "\t"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        i += 1
        if key == "triggers":
            block: list[str] = []
            while i < n and (
                frontmatter_lines[i].startswith(" ")
                or frontmatter_lines[i].startswith("\t")
                or frontmatter_lines[i].strip() == ""
            ):
                if frontmatter_lines[i].strip() != "":
                    block.append(frontmatter_lines[i])
                i += 1
            triggers = _parse_trigger_block(block)
        elif key == "name":
            name = value
        elif key == "description":
            description = value
        else:
            metadata[key] = value

    if not name:
        raise SkillError("Skill frontmatter is missing required 'name' field")
    return name, description, triggers, metadata


def parse_skill_file(path: Path) -> Skill:
    """Parse a ``SKILL.md`` file into a :class:`Skill`.

    Raises :class:`SkillError` on malformed frontmatter.
    """
    content = path.read_text(encoding="utf-8")
    frontmatter_lines, body = _split_frontmatter(content)
    name, description, triggers, metadata = _parse_frontmatter(frontmatter_lines)
    return Skill(
        name=name,
        description=description,
        path=path,
        body=body,
        triggers=triggers,
        metadata=metadata,
    )


def _serialize_skill(
    name: str,
    description: str,
    body: str,
    triggers: list[SkillTrigger],
) -> str:
    """Serialize a skill into ``SKILL.md`` content with frontmatter."""
    lines = ["---", f"name: {name}", f"description: {description}"]
    if triggers:
        lines.append("triggers:")
        for t in triggers:
            lines.append(f"  - type: {t.type}")
            lines.append(f"    value: {t.value}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class SkillLoader:
    """Discovers and loads ``SKILL.md`` files with progressive disclosure.

    Skills live as ``SKILL.md`` files inside per-skill directories under
    one or more ``skills_dirs`` (default ``.justagent/skills``). Only the
    name and description are surfaced for system-prompt injection; the
    full body is loaded on demand via :meth:`load_body`.

    Example:

        >>> loader = SkillLoader(project_root=Path("/project"))
        >>> for summary in loader.summaries():
        ...     print(summary.name, summary.description)
        >>> body = loader.load_body("migrations")
    """

    def __init__(
        self,
        config: SkillConfig | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._config = config or SkillConfig()
        self._project_root = (
            Path(project_root) if project_root is not None else Path.cwd()
        )

    @property
    def config(self) -> SkillConfig:
        """The active skill configuration."""

        return self._config

    @property
    def project_root(self) -> Path:
        """The project root skills_dirs are resolved against."""

        return self._project_root

    def _resolve_skills_dir(self, skills_dir: str) -> Path:
        p = Path(skills_dir)
        if p.is_absolute():
            return p
        return self._project_root / p

    def discover(self) -> list[Skill]:
        """Scan ``skills_dirs`` for ``SKILL.md`` files and return parsed skills.

        Skills are sorted by name. A single corrupt or unreadable skill
        file is skipped rather than aborting the whole scan. Returns an
        empty list if the config is disabled or no skills are found.
        """
        if not self._config.enabled:
            return []
        skills: list[Skill] = []
        seen_paths: set[Path] = set()
        for skills_dir in self._config.skills_dirs:
            dir_path = self._resolve_skills_dir(skills_dir)
            if not dir_path.is_dir():
                continue
            for skill_file in sorted(dir_path.rglob("SKILL.md")):
                resolved = skill_file.resolve()
                if resolved in seen_paths:
                    continue
                try:
                    skill = parse_skill_file(skill_file)
                except (SkillError, OSError):
                    continue
                seen_paths.add(resolved)
                skills.append(skill)
        skills.sort(key=lambda s: s.name)
        if self._config.max_skills > 0 and len(skills) > self._config.max_skills:
            skills = skills[: self._config.max_skills]
        return skills

    def get(self, name: str) -> Skill | None:
        """Return the skill with the given name, or ``None`` if not found."""

        for skill in self.discover():
            if skill.name == name:
                return skill
        return None

    def summaries(self) -> list[SkillSummary]:
        """Return lightweight summaries for system-prompt injection."""

        return [
            SkillSummary(name=s.name, description=s.description)
            for s in self.discover()
        ]

    def format_summaries_for_prompt(self) -> str:
        """Format summaries as a markdown block for the system prompt.

        Returns an empty string when there are no skills.
        """
        summaries = self.summaries()
        if not summaries:
            return ""
        lines = ["## Available Skills", ""]
        for s in summaries:
            lines.append(f"- **{s.name}**: {s.description}")
        lines.append("")
        return "\n".join(lines)

    def load_body(self, name: str) -> str:
        """Load the full body of a skill, truncated to ``max_body_chars``.

        Raises :class:`SkillError` if the skill does not exist.
        """
        skill = self.get(name)
        if skill is None:
            raise SkillError(f"Skill not found: {name}", details={"name": name})
        body = skill.body
        if self._config.max_body_chars > 0 and len(body) > self._config.max_body_chars:
            body = body[: self._config.max_body_chars]
        return body

    def matches_trigger(self, text: str) -> list[Skill]:
        """Return skills whose keyword triggers match the given text.

        Matching is case-insensitive substring matching against each
        ``keyword`` trigger's value. A skill matches if any of its
        keyword triggers match. Non-keyword triggers are ignored.
        """
        text_lower = text.lower()
        matched: list[Skill] = []
        for skill in self.discover():
            for trigger in skill.triggers:
                if trigger.type == "keyword" and trigger.value.lower() in text_lower:
                    matched.append(skill)
                    break
        return matched

    def create_skill(
        self,
        name: str,
        description: str,
        body: str,
        triggers: list[SkillTrigger] | None = None,
    ) -> Skill:
        """Create a new skill on disk and return the parsed :class:`Skill`.

        Writes to the first configured skills directory under
        ``<name>/SKILL.md``. The directory is created if needed.
        """
        triggers = triggers or []
        if not self._config.skills_dirs:
            raise SkillError("No skills_dirs configured")
        skills_dir = self._resolve_skills_dir(self._config.skills_dirs[0])
        skill_dir = skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        content = _serialize_skill(name, description, body, triggers)
        atomic_write_text(skill_file, content, encoding="utf-8")
        return Skill(
            name=name,
            description=description,
            path=skill_file,
            body=body,
            triggers=triggers,
            metadata={},
        )

    def update_skill(
        self,
        name: str,
        *,
        description: str | None = None,
        body: str | None = None,
        triggers: list[SkillTrigger] | None = None,
    ) -> Skill:
        """Update an existing skill's fields and persist to disk.

        Only the provided fields are updated; others remain unchanged.

        Raises:
            SkillError: If the skill does not exist.
        """

        skill = self.get(name)
        if skill is None:
            raise SkillError(f"Skill not found: {name}", details={"name": name})
        new_description = description if description is not None else skill.description
        new_body = body if body is not None else skill.body
        new_triggers = triggers if triggers is not None else skill.triggers
        content = _serialize_skill(name, new_description, new_body, new_triggers)
        atomic_write_text(skill.path, content, encoding="utf-8")
        return Skill(
            name=name,
            description=new_description,
            path=skill.path,
            body=new_body,
            triggers=new_triggers,
            metadata=skill.metadata,
        )

    def delete_skill(self, name: str) -> bool:
        """Delete a skill and its directory from disk.

        Returns True if the skill was found and deleted, False if not found.
        """

        skill = self.get(name)
        if skill is None:
            return False
        skill_dir = skill.path.parent
        # Remove the SKILL.md file.
        skill.path.unlink(missing_ok=True)
        # Remove the skill directory if it's now empty (or only contains
        # non-skill files that were part of the skill bundle).
        import shutil

        shutil.rmtree(skill_dir, ignore_errors=True)
        return True

    def import_skill(self, source_path: Path, name: str | None = None) -> Skill:
        """Import a skill from an external ``SKILL.md`` file.

        Copies the file into the first configured skills directory. If
        *name* is provided, the skill is renamed; otherwise the original
        directory name is used.

        Raises:
            SkillError: If the source file doesn't exist or is malformed.
        """

        source = Path(source_path)
        if not source.is_file():
            raise SkillError(
                f"Source file not found: {source_path}",
                details={"path": str(source_path)},
            )
        try:
            skill = parse_skill_file(source)
        except SkillError:
            raise
        skill_name = name or skill.name
        if not self._config.skills_dirs:
            raise SkillError("No skills_dirs configured")
        skills_dir = self._resolve_skills_dir(self._config.skills_dirs[0])
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        dest = skill_dir / "SKILL.md"
        content = source.read_text(encoding="utf-8")
        # Re-serialize with the new name if renamed.
        if name and name != skill.name:
            content = _serialize_skill(name, skill.description, skill.body, skill.triggers)
        atomic_write_text(dest, content, encoding="utf-8")
        return Skill(
            name=skill_name,
            description=skill.description,
            path=dest,
            body=skill.body,
            triggers=skill.triggers,
            metadata=skill.metadata,
        )


__all__ = [
    "Skill",
    "SkillConfig",
    "SkillError",
    "SkillLoader",
    "SkillSummary",
    "SkillTrigger",
    "parse_skill_file",
]
