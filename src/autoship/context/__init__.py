"""Instructions auto-discovery for the autoship agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from autoship.context.repo_map import (
    FileSymbols,
    RepoMapConfig,
    RepoMapGenerator,
    Symbol,
    SymbolKind,
)


@dataclass(frozen=True)
class InstructionFile:
    """A discovered instruction file (e.g. AGENTS.md) with location metadata."""

    path: Path
    filename: str
    content: str
    level: int  # 0 = cwd, 1 = parent, 2 = grandparent, ...


@dataclass
class InstructionConfig:
    """Configuration for instruction file discovery."""

    filenames: list[str] = field(
        default_factory=lambda: ["AGENTS.md", "CLAUDE.md", "CONTEXT.md", ".cursorrules"]
    )
    max_levels: int = 5
    max_total_chars: int = 20000


class RepoMap:
    """Repo map generator wrapper.

    Delegates to :class:`RepoMapGenerator` for the actual scanning and
    symbol extraction. Kept as a thin wrapper for backwards
    compatibility with code that imports ``RepoMap`` from
    ``autoship.context``.
    """

    def __init__(self, config: RepoMapConfig | None = None) -> None:
        self._generator = RepoMapGenerator(config)

    def generate(self, root: str | Path) -> str:
        return self._generator.generate(root)


class InstructionDiscovery:
    """Walks up from a start directory to discover instruction files."""

    def __init__(self, config: InstructionConfig | None = None) -> None:
        self.config = config or InstructionConfig()

    def discover(self, start_dir: str | Path) -> list[InstructionFile]:
        start = Path(start_dir).resolve()
        found: list[InstructionFile] = []
        current = start
        for level in range(self.config.max_levels + 1):
            if not current.is_dir():
                break
            found.extend(self.discover_at_level(current, level))
            if current.parent == current:
                break  # Reached filesystem root.
            current = current.parent
        found.sort(key=lambda f: (f.level, self.config.filenames.index(f.filename)))
        return found

    def discover_at_level(self, directory: Path, level: int) -> list[InstructionFile]:
        results: list[InstructionFile] = []
        for filename in self.config.filenames:
            candidate = directory / filename
            if candidate.is_file():
                results.append(
                    InstructionFile(
                        path=candidate,
                        filename=filename,
                        content=candidate.read_text(encoding="utf-8"),
                        level=level,
                    )
                )
        return results

    def load(self, start_dir: str | Path) -> str:
        files = self.discover(start_dir)
        if not files:
            return ""
        parts: list[str] = []
        for f in files:
            display = f"./{f.filename}" if f.level == 0 else f"{'../' * f.level}{f.filename}"
            parts.append(f"# From {display}\n{f.content}\n\n")
        combined = "".join(parts)
        if len(combined) > self.config.max_total_chars:
            combined = combined[: self.config.max_total_chars]
        return combined


from autoship.context.skill import (  # noqa: E402
    Skill,
    SkillConfig,
    SkillError,
    SkillLoader,
    SkillSummary,
    SkillTrigger,
    parse_skill_file,
)

__all__ = [
    "FileSymbols",
    "InstructionConfig",
    "InstructionDiscovery",
    "InstructionFile",
    "RepoMap",
    "RepoMapConfig",
    "RepoMapGenerator",
    "Skill",
    "SkillConfig",
    "SkillError",
    "SkillLoader",
    "SkillSummary",
    "SkillTrigger",
    "Symbol",
    "SymbolKind",
    "parse_skill_file",
]
