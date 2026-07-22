"""Tests for the skills system (SKILL.md loader with progressive disclosure)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from autoship.context.skill import (
    Skill,
    SkillConfig,
    SkillError,
    SkillLoader,
    SkillSummary,
    SkillTrigger,
    _parse_frontmatter,
    _split_frontmatter,
    parse_skill_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill_md(
    skill_dir: Path,
    name: str,
    *,
    description: str = "",
    body: str = "",
    triggers: list[tuple[str, str]] | None = None,
    extra_frontmatter: str = "",
    raw_content: str | None = None,
) -> Path:
    """Write a SKILL.md file inside ``skill_dir`` and return its path."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    if raw_content is not None:
        skill_file.write_text(raw_content, encoding="utf-8")
        return skill_file
    parts = ["---", f"name: {name}"]
    if description:
        parts.append(f"description: {description}")
    if triggers:
        parts.append("triggers:")
        for t_type, t_value in triggers:
            parts.append(f"  - type: {t_type}")
            parts.append(f"    value: {t_value}")
    if extra_frontmatter:
        parts.append(extra_frontmatter.rstrip("\n"))
    parts.append("---")
    parts.append("")
    parts.append(body)
    skill_file.write_text("\n".join(parts), encoding="utf-8")
    return skill_file


# ---------------------------------------------------------------------------
# TestSkillDataclass
# ---------------------------------------------------------------------------


class TestSkillDataclass:
    def test_skill_trigger_construction(self) -> None:
        t = SkillTrigger(type="keyword", value="migration")
        assert t.type == "keyword"
        assert t.value == "migration"

    def test_skill_construction_with_defaults(self) -> None:
        skill = Skill(
            name="x",
            description="d",
            path=Path("/tmp/x"),
            body="body",
        )
        assert skill.name == "x"
        assert skill.triggers == []
        assert skill.metadata == {}

    def test_skill_construction_full(self) -> None:
        t = SkillTrigger(type="keyword", value="db")
        skill = Skill(
            name="x",
            description="d",
            path=Path("/tmp/x"),
            body="body",
            triggers=[t],
            metadata={"author": "me"},
        )
        assert skill.triggers == [t]
        assert skill.metadata == {"author": "me"}

    def test_skill_is_frozen(self) -> None:
        skill = Skill(name="x", description="d", path=Path("/tmp/x"), body="b")
        with pytest.raises(dataclasses.FrozenInstanceError):
            skill.name = "y"  # type: ignore[misc]

    def test_skill_trigger_is_frozen(self) -> None:
        t = SkillTrigger(type="keyword", value="v")
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.value = "other"  # type: ignore[misc]

    def test_skill_summary_construction(self) -> None:
        s = SkillSummary(name="x", description="d")
        assert s.name == "x"
        assert s.description == "d"

    def test_skill_summary_is_frozen(self) -> None:
        s = SkillSummary(name="x", description="d")
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.name = "y"  # type: ignore[misc]

    def test_skill_equality(self) -> None:
        a = Skill(name="x", description="d", path=Path("/tmp/x"), body="b")
        b = Skill(name="x", description="d", path=Path("/tmp/x"), body="b")
        assert a == b


# ---------------------------------------------------------------------------
# TestSkillConfig
# ---------------------------------------------------------------------------


class TestSkillConfig:
    def test_defaults(self) -> None:
        cfg = SkillConfig()
        assert cfg.skills_dirs == [".autoship/skills"]
        assert cfg.max_skills == 50
        assert cfg.max_body_chars == 50000
        assert cfg.enabled is True

    def test_custom_values(self) -> None:
        cfg = SkillConfig(
            skills_dirs=["a", "b"],
            max_skills=5,
            max_body_chars=1000,
            enabled=False,
        )
        assert cfg.skills_dirs == ["a", "b"]
        assert cfg.max_skills == 5
        assert cfg.max_body_chars == 1000
        assert cfg.enabled is False

    def test_default_skills_dirs_is_independent_per_instance(self) -> None:
        a = SkillConfig()
        b = SkillConfig()
        a.skills_dirs.append("extra")
        assert "extra" not in b.skills_dirs


# ---------------------------------------------------------------------------
# TestParsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_split_frontmatter_basic(self) -> None:
        content = "---\nname: x\ndescription: y\n---\n\nbody text"
        fm, body = _split_frontmatter(content)
        assert fm == ["name: x", "description: y"]
        assert body == "body text"

    def test_split_frontmatter_strips_leading_blank_line(self) -> None:
        content = "---\nname: x\n---\n\n# Heading"
        _fm, body = _split_frontmatter(content)
        assert body == "# Heading"

    def test_split_frontmatter_missing_opening(self) -> None:
        with pytest.raises(SkillError, match="opening"):
            _split_frontmatter("name: x\n---\nbody")

    def test_split_frontmatter_missing_closing(self) -> None:
        with pytest.raises(SkillError, match="closing"):
            _split_frontmatter("---\nname: x\nbody")

    def test_parse_frontmatter_with_triggers(self) -> None:
        fm = [
            "name: migrations",
            "description: Helps with database migrations",
            "triggers:",
            "  - type: keyword",
            "    value: migration",
            "  - type: keyword",
            "    value: database",
        ]
        name, desc, triggers, metadata = _parse_frontmatter(fm)
        assert name == "migrations"
        assert desc == "Helps with database migrations"
        assert triggers == [
            SkillTrigger(type="keyword", value="migration"),
            SkillTrigger(type="keyword", value="database"),
        ]
        assert metadata == {}

    def test_parse_frontmatter_without_triggers(self) -> None:
        fm = ["name: plain", "description: A plain skill"]
        name, desc, triggers, metadata = _parse_frontmatter(fm)
        assert name == "plain"
        assert desc == "A plain skill"
        assert triggers == []
        assert metadata == {}

    def test_parse_frontmatter_collects_extra_metadata(self) -> None:
        fm = ["name: x", "description: y", "author: jane", "version: '1.0'"]
        _name, _desc, _triggers, metadata = _parse_frontmatter(fm)
        assert metadata == {"author": "jane", "version": "'1.0'"}

    def test_parse_frontmatter_missing_name_raises(self) -> None:
        with pytest.raises(SkillError, match="name"):
            _parse_frontmatter(["description: no name"])

    def test_parse_frontmatter_empty_name_raises(self) -> None:
        with pytest.raises(SkillError, match="name"):
            _parse_frontmatter(["name: ", "description: y"])

    def test_parse_frontmatter_empty_input_raises(self) -> None:
        with pytest.raises(SkillError, match="name"):
            _parse_frontmatter([])

    def test_parse_frontmatter_tool_trigger(self) -> None:
        fm = [
            "name: tool-skill",
            "triggers:",
            "  - type: tool",
            "    value: run_command",
        ]
        _name, _desc, triggers, _metadata = _parse_frontmatter(fm)
        assert triggers == [SkillTrigger(type="tool", value="run_command")]

    def test_parse_skill_file_full(self, tmp_path: Path) -> None:
        skill_file = _write_skill_md(
            tmp_path / "my-skill",
            "my-skill",
            description="Does a thing",
            body="# My Skill\n\nDetailed instructions",
            triggers=[("keyword", "thing")],
        )
        skill = parse_skill_file(skill_file)
        assert skill.name == "my-skill"
        assert skill.description == "Does a thing"
        assert skill.path == skill_file
        assert "# My Skill" in skill.body
        assert "Detailed instructions" in skill.body
        assert skill.triggers == [SkillTrigger(type="keyword", value="thing")]

    def test_parse_skill_file_empty_body(self, tmp_path: Path) -> None:
        skill_file = _write_skill_md(tmp_path / "empty", "empty", description="d")
        skill = parse_skill_file(skill_file)
        assert skill.body == ""

    def test_parse_skill_file_malformed_no_frontmatter(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "bad" / "SKILL.md"
        skill_file.parent.mkdir(parents=True)
        skill_file.write_text("just some markdown with no frontmatter", encoding="utf-8")
        with pytest.raises(SkillError, match="opening"):
            parse_skill_file(skill_file)

    def test_parse_skill_file_malformed_unclosed_frontmatter(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "bad" / "SKILL.md"
        skill_file.parent.mkdir(parents=True)
        skill_file.write_text("---\nname: x\nbody never closes", encoding="utf-8")
        with pytest.raises(SkillError, match="closing"):
            parse_skill_file(skill_file)

    def test_parse_skill_file_malformed_missing_name(self, tmp_path: Path) -> None:
        skill_file = _write_skill_md(
            tmp_path / "bad",
            "ignored",
            raw_content="---\ndescription: no name\n---\n\nbody",
        )
        with pytest.raises(SkillError, match="name"):
            parse_skill_file(skill_file)


# ---------------------------------------------------------------------------
# TestDiscover
# ---------------------------------------------------------------------------


class TestDiscover:
    def test_discover_finds_skills(self, tmp_path: Path) -> None:
        skills_root = tmp_path / ".autoship" / "skills"
        _write_skill_md(skills_root / "alpha", "alpha", description="a")
        _write_skill_md(skills_root / "beta", "beta", description="b")
        loader = SkillLoader(project_root=tmp_path)
        skills = loader.discover()
        assert [s.name for s in skills] == ["alpha", "beta"]

    def test_discover_returns_empty_when_no_dir(self, tmp_path: Path) -> None:
        loader = SkillLoader(project_root=tmp_path)
        assert loader.discover() == []

    def test_discover_multiple_skill_dirs(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "skills-a"
        dir_b = tmp_path / "skills-b"
        _write_skill_md(dir_a / "alpha", "alpha", description="a")
        _write_skill_md(dir_b / "beta", "beta", description="b")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills-a", "skills-b"]),
            project_root=tmp_path,
        )
        skills = loader.discover()
        assert {s.name for s in skills} == {"alpha", "beta"}

    def test_discover_nested_dirs(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(skills_root / "category" / "deep", "deep", description="d")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        skills = loader.discover()
        assert len(skills) == 1
        assert skills[0].name == "deep"

    def test_discover_sorted_by_name(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        for name in ["zeta", "alpha", "mike"]:
            _write_skill_md(skills_root / name, name, description=name)
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        assert [s.name for s in loader.discover()] == ["alpha", "mike", "zeta"]

    def test_discover_skips_corrupt_file(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(skills_root / "good", "good", description="ok")
        # Corrupt file: no frontmatter at all.
        bad = skills_root / "bad" / "SKILL.md"
        bad.parent.mkdir(parents=True)
        bad.write_text("no frontmatter here", encoding="utf-8")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        skills = loader.discover()
        assert [s.name for s in skills] == ["good"]

    def test_discover_empty_skills_dir(self, tmp_path: Path) -> None:
        (tmp_path / "skills").mkdir()
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        assert loader.discover() == []

    def test_discover_absolute_skills_dir(self, tmp_path: Path) -> None:
        abs_dir = tmp_path / "abs-skills"
        _write_skill_md(abs_dir / "x", "x", description="x")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=[str(abs_dir)]),
            project_root=tmp_path,
        )
        assert [s.name for s in loader.discover()] == ["x"]


# ---------------------------------------------------------------------------
# TestGet
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_existing(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(skills_root / "alpha", "alpha", description="a")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        skill = loader.get("alpha")
        assert skill is not None
        assert skill.name == "alpha"

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        loader = SkillLoader(project_root=tmp_path)
        assert loader.get("nope") is None

    def test_get_returns_skill_with_body(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "alpha",
            "alpha",
            description="a",
            body="the body content",
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        skill = loader.get("alpha")
        assert skill is not None
        assert skill.body == "the body content"


# ---------------------------------------------------------------------------
# TestSummaries
# ---------------------------------------------------------------------------


class TestSummaries:
    def test_summaries_return_correct_fields(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(skills_root / "alpha", "alpha", description="does alpha")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        summaries = loader.summaries()
        assert len(summaries) == 1
        assert summaries[0].name == "alpha"
        assert summaries[0].description == "does alpha"

    def test_summaries_exclude_body(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "alpha",
            "alpha",
            description="d",
            body="secret body",
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        summary = loader.summaries()[0]
        assert not hasattr(summary, "body")

    def test_summaries_empty(self, tmp_path: Path) -> None:
        loader = SkillLoader(project_root=tmp_path)
        assert loader.summaries() == []

    def test_format_summaries_for_prompt_has_header(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(skills_root / "alpha", "alpha", description="does alpha")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        result = loader.format_summaries_for_prompt()
        assert "## Available Skills" in result
        assert "- **alpha**: does alpha" in result

    def test_format_summaries_for_prompt_empty(self, tmp_path: Path) -> None:
        loader = SkillLoader(project_root=tmp_path)
        assert loader.format_summaries_for_prompt() == ""

    def test_format_summaries_lists_all(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(skills_root / "alpha", "alpha", description="a")
        _write_skill_md(skills_root / "beta", "beta", description="b")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        result = loader.format_summaries_for_prompt()
        assert "- **alpha**: a" in result
        assert "- **beta**: b" in result


# ---------------------------------------------------------------------------
# TestLoadBody
# ---------------------------------------------------------------------------


class TestLoadBody:
    def test_load_body(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "alpha",
            "alpha",
            description="a",
            body="full body text",
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        assert loader.load_body("alpha") == "full body text"

    def test_load_body_truncation(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        long_body = "x" * 1000
        _write_skill_md(
            skills_root / "alpha",
            "alpha",
            description="a",
            body=long_body,
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"], max_body_chars=100),
            project_root=tmp_path,
        )
        body = loader.load_body("alpha")
        assert len(body) == 100
        assert body == "x" * 100

    def test_load_body_no_truncation_when_zero(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "alpha",
            "alpha",
            description="a",
            body="x" * 200,
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"], max_body_chars=0),
            project_root=tmp_path,
        )
        assert len(loader.load_body("alpha")) == 200

    def test_load_body_missing_raises(self, tmp_path: Path) -> None:
        loader = SkillLoader(project_root=tmp_path)
        with pytest.raises(SkillError, match="not found"):
            loader.load_body("nope")


# ---------------------------------------------------------------------------
# TestMatchesTrigger
# ---------------------------------------------------------------------------


class TestMatchesTrigger:
    def test_keyword_match(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "migrations",
            "migrations",
            description="m",
            triggers=[("keyword", "migration")],
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        matched = loader.matches_trigger("help me with migration please")
        assert len(matched) == 1
        assert matched[0].name == "migrations"

    def test_keyword_match_case_insensitive(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "migrations",
            "migrations",
            description="m",
            triggers=[("keyword", "Migration")],
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        assert len(loader.matches_trigger("running a migration")) == 1
        assert len(loader.matches_trigger("MIGRATION time")) == 1

    def test_multiple_matches(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "a",
            "a",
            description="a",
            triggers=[("keyword", "deploy")],
        )
        _write_skill_md(
            skills_root / "b",
            "b",
            description="b",
            triggers=[("keyword", "ship")],
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        matched = loader.matches_trigger("deploy and ship it")
        assert {s.name for s in matched} == {"a", "b"}

    def test_no_matches(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "a",
            "a",
            description="a",
            triggers=[("keyword", "deploy")],
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        assert loader.matches_trigger("nothing relevant here") == []

    def test_tool_trigger_ignored_for_text(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "a",
            "a",
            description="a",
            triggers=[("tool", "run_command")],
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        # tool triggers are not matched against free text
        assert loader.matches_trigger("run_command") == []

    def test_skill_with_multiple_keywords_matches_once(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(
            skills_root / "a",
            "a",
            description="a",
            triggers=[("keyword", "db"), ("keyword", "database")],
        )
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        matched = loader.matches_trigger("use the database")
        assert len(matched) == 1
        assert matched[0].name == "a"


# ---------------------------------------------------------------------------
# TestCreateSkill
# ---------------------------------------------------------------------------


class TestCreateSkill:
    def test_create_skill_writes_file(self, tmp_path: Path) -> None:
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        skill = loader.create_skill(
            "new-skill",
            "A new skill",
            "body content",
            triggers=[SkillTrigger(type="keyword", value="new")],
        )
        assert skill.name == "new-skill"
        assert skill.path == tmp_path / "skills" / "new-skill" / "SKILL.md"
        assert skill.path.is_file()

    def test_create_skill_file_content(self, tmp_path: Path) -> None:
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        skill = loader.create_skill(
            "new-skill",
            "A new skill",
            "body content",
            triggers=[SkillTrigger(type="keyword", value="new")],
        )
        written = skill.path.read_text(encoding="utf-8")
        assert written.startswith("---")
        assert "name: new-skill" in written
        assert "description: A new skill" in written
        assert "- type: keyword" in written
        assert "value: new" in written
        assert "body content" in written

    def test_create_skill_without_triggers(self, tmp_path: Path) -> None:
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        skill = loader.create_skill("plain", "desc", "body")
        written = skill.path.read_text(encoding="utf-8")
        assert "triggers:" not in written

    def test_create_skill_can_be_rediscovered(self, tmp_path: Path) -> None:
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        loader.create_skill(
            "rediscover",
            "desc",
            "body",
            triggers=[SkillTrigger(type="keyword", value="x")],
        )
        rediscovered = loader.get("rediscover")
        assert rediscovered is not None
        assert rediscovered.description == "desc"
        assert rediscovered.body == "body"
        assert rediscovered.triggers == [SkillTrigger(type="keyword", value="x")]

    def test_create_skill_creates_nested_dir(self, tmp_path: Path) -> None:
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["deep/nested/skills"]),
            project_root=tmp_path,
        )
        skill = loader.create_skill("x", "d", "b")
        assert skill.path.is_file()
        assert (tmp_path / "deep" / "nested" / "skills" / "x" / "SKILL.md").is_file()


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_disabled_config_discover_returns_empty(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(skills_root / "alpha", "alpha", description="a")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"], enabled=False),
            project_root=tmp_path,
        )
        assert loader.discover() == []
        assert loader.summaries() == []
        assert loader.format_summaries_for_prompt() == ""
        assert loader.matches_trigger("alpha") == []

    def test_max_skills_limit(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        for name in ["a", "b", "c", "d", "e"]:
            _write_skill_md(skills_root / name, name, description=name)
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"], max_skills=2),
            project_root=tmp_path,
        )
        skills = loader.discover()
        assert len(skills) == 2
        # Sorted by name, so first two alphabetically.
        assert [s.name for s in skills] == ["a", "b"]

    def test_max_skills_zero_means_unlimited(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        for name in ["a", "b", "c"]:
            _write_skill_md(skills_root / name, name, description=name)
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"], max_skills=0),
            project_root=tmp_path,
        )
        assert len(loader.discover()) == 3

    def test_missing_skills_dir_handled_gracefully(self, tmp_path: Path) -> None:
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["does-not-exist"]),
            project_root=tmp_path,
        )
        assert loader.discover() == []

    def test_corrupt_file_skipped_does_not_crash(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(skills_root / "good", "good", description="ok")
        bad = skills_root / "broken" / "SKILL.md"
        bad.parent.mkdir(parents=True)
        # Missing closing frontmatter delimiter.
        bad.write_text("---\nname: broken\nno close", encoding="utf-8")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        skills = loader.discover()
        assert [s.name for s in skills] == ["good"]

    def test_default_project_root_is_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        loader = SkillLoader()
        assert loader.project_root.resolve() == tmp_path.resolve()

    def test_get_after_create(self, tmp_path: Path) -> None:
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"]),
            project_root=tmp_path,
        )
        loader.create_skill("fresh", "desc", "body")
        skill = loader.get("fresh")
        assert skill is not None
        assert skill.name == "fresh"

    def test_load_body_missing_when_disabled(self, tmp_path: Path) -> None:
        skills_root = tmp_path / "skills"
        _write_skill_md(skills_root / "alpha", "alpha", description="a", body="b")
        loader = SkillLoader(
            config=SkillConfig(skills_dirs=["skills"], enabled=False),
            project_root=tmp_path,
        )
        # Disabled → discover returns [] → get returns None → load_body raises.
        with pytest.raises(SkillError, match="not found"):
            loader.load_body("alpha")

    def test_context_reexport(self) -> None:
        from autoship.context import (
            Skill as ReexportedSkill,
        )
        from autoship.context import (
            SkillConfig as ReexportedSkillConfig,
        )
        from autoship.context import (
            SkillError as ReexportedSkillError,
        )
        from autoship.context import (
            SkillLoader as ReexportedSkillLoader,
        )
        from autoship.context import (
            SkillSummary as ReexportedSkillSummary,
        )
        from autoship.context import (
            SkillTrigger as ReexportedSkillTrigger,
        )

        assert ReexportedSkill is Skill
        assert ReexportedSkillConfig is SkillConfig
        assert ReexportedSkillError is SkillError
        assert ReexportedSkillLoader is SkillLoader
        assert ReexportedSkillSummary is SkillSummary
        assert ReexportedSkillTrigger is SkillTrigger
