"""Tests for ``autoship.agent.search_replace`` (Aider SEARCH/REPLACE format)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoship.agent.search_replace import (
    SearchReplaceError,
    apply_search_replace,
    parse_search_replace,
)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_single_block_with_explicit_filename() -> None:
    content = "\n".join(
        [
            "src/foo.py",
            "```",
            "<<<<<<< SEARCH",
            "old line",
            "=======",
            "new line",
            ">>>>>>> REPLACE",
            "```",
        ]
    )
    edits = parse_search_replace(content)
    assert len(edits) == 1
    assert edits[0].filename == "src/foo.py"
    assert edits[0].search == "old line\n"
    assert edits[0].replace == "new line\n"


def test_parse_multiple_blocks_share_current_filename() -> None:
    content = "\n".join(
        [
            "foo.py",
            "```",
            "<<<<<<< SEARCH",
            "a",
            "=======",
            "b",
            ">>>>>>> REPLACE",
            "```",
            "<<<<<<< SEARCH",
            "c",
            "=======",
            "d",
            ">>>>>>> REPLACE",
        ]
    )
    edits = parse_search_replace(content)
    assert len(edits) == 2
    assert edits[0].filename == "foo.py"
    assert edits[1].filename == "foo.py"


def test_parse_tolerates_variable_marker_width() -> None:
    """Aider accepts 5–9 char-wide markers (handles typos)."""

    content = "\n".join(
        [
            "foo.py",
            "<<<<<< SEARCH",  # 6 chars
            "x",
            "=======",  # 7 chars
            "y",
            ">>>>>>>>> REPLACE",  # 9 chars
        ]
    )
    edits = parse_search_replace(content)
    assert len(edits) == 1


def test_parse_missing_filename_raises() -> None:
    with pytest.raises(SearchReplaceError, match="Bad/missing filename"):
        parse_search_replace(
            "\n".join(
                [
                    "<<<<<<< SEARCH",
                    "x",
                    "=======",
                    "y",
                    ">>>>>>> REPLACE",
                ]
            )
        )


def test_parse_missing_divider_raises() -> None:
    with pytest.raises(SearchReplaceError, match="======="):
        parse_search_replace(
            "\n".join(
                [
                    "foo.py",
                    "<<<<<<< SEARCH",
                    "x",
                    ">>>>>>> REPLACE",
                ]
            )
        )


def test_parse_missing_terminator_raises() -> None:
    with pytest.raises(SearchReplaceError, match=">>>>>>> REPLACE"):
        parse_search_replace(
            "\n".join(
                [
                    "foo.py",
                    "<<<<<<< SEARCH",
                    "x",
                    "=======",
                    "y",
                ]
            )
        )


# ---------------------------------------------------------------------------
# Apply — basic
# ---------------------------------------------------------------------------


def test_apply_perfect_match(tmp_path: Path) -> None:
    target = tmp_path / "foo.py"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    content = "\n".join(
        [
            "foo.py",
            "<<<<<<< SEARCH",
            "beta",
            "=======",
            "BETA",
            ">>>>>>> REPLACE",
        ]
    )
    result = apply_search_replace(content, tmp_path)
    assert result.touched == ["foo.py"]
    assert not result.failed
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_apply_creates_new_file_with_empty_search(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "new.txt",
            "<<<<<<< SEARCH",
            "=======",
            "hello world",
            ">>>>>>> REPLACE",
        ]
    )
    result = apply_search_replace(content, tmp_path)
    assert result.touched == ["new.txt"]
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello world\n"


def test_apply_appends_to_existing_file_with_empty_search(tmp_path: Path) -> None:
    target = tmp_path / "log.txt"
    target.write_text("line1\n", encoding="utf-8")
    content = "\n".join(
        [
            "log.txt",
            "<<<<<<< SEARCH",
            "=======",
            "line2",
            ">>>>>>> REPLACE",
        ]
    )
    apply_search_replace(content, tmp_path)
    assert target.read_text(encoding="utf-8") == "line1\nline2\n"


def test_apply_multi_line_replace(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n", encoding="utf-8")

    content = "\n".join(
        [
            "mod.py",
            "<<<<<<< SEARCH",
            "def f():",
            "    return 1",
            "=======",
            "def f():",
            "    return 1",
            "    print('called')",
            ">>>>>>> REPLACE",
        ]
    )
    apply_search_replace(content, tmp_path)
    body = target.read_text(encoding="utf-8")
    assert "print('called')" in body
    assert body.count("def f():") == 1
    assert "def g():" in body


# ---------------------------------------------------------------------------
# Apply — fuzzy / tolerant matching
# ---------------------------------------------------------------------------


def test_apply_tolerates_uniform_outdent(tmp_path: Path) -> None:
    """If the LLM uniformly outdents SEARCH & REPLACE, the edit should still apply."""

    target = tmp_path / "indented.py"
    target.write_text("def f():\n    alpha\n    beta\n    gamma\n", encoding="utf-8")

    # SEARCH/REPLACE are missing the 4-space indent.
    content = "\n".join(
        [
            "indented.py",
            "<<<<<<< SEARCH",
            "alpha",
            "beta",
            "=======",
            "alpha",
            "BETA",
            ">>>>>>> REPLACE",
        ]
    )
    apply_search_replace(content, tmp_path)
    assert (
        target.read_text(encoding="utf-8")
        == "def f():\n    alpha\n    BETA\n    gamma\n"
    )


def test_apply_handles_dotdotdot_elision(tmp_path: Path) -> None:
    target = tmp_path / "long.py"
    body = "a\nb\nc\nd\ne\nf\ng\n"
    target.write_text(body, encoding="utf-8")

    content = "\n".join(
        [
            "long.py",
            "<<<<<<< SEARCH",
            "a",
            "b",
            "...",
            "f",
            "g",
            "=======",
            "a",
            "b",
            "...",
            "f",
            "G",
            ">>>>>>> REPLACE",
        ]
    )
    apply_search_replace(content, tmp_path)
    assert target.read_text(encoding="utf-8") == "a\nb\nc\nd\ne\nf\nG\n"


def test_apply_fuzzy_fallback(tmp_path: Path) -> None:
    """Near-miss SEARCH blocks (similarity ≥ 0.8) apply via fuzzy fallback."""

    target = tmp_path / "fuzzy.txt"
    target.write_text("the quick brown fox\njumps over the lazy dog\n", encoding="utf-8")

    # The SEARCH has a tiny typo (`quck` instead of `quick`).
    content = "\n".join(
        [
            "fuzzy.txt",
            "<<<<<<< SEARCH",
            "the quck brown fox",
            "=======",
            "THE QUICK BROWN FOX",
            ">>>>>>> REPLACE",
        ]
    )
    apply_search_replace(content, tmp_path)
    assert (
        target.read_text(encoding="utf-8")
        == "THE QUICK BROWN FOX\njumps over the lazy dog\n"
    )


def test_apply_failed_match_collected_not_raised(tmp_path: Path) -> None:
    """When SEARCH truly doesn't match, the edit goes to ``failed`` not raised."""

    target = tmp_path / "x.txt"
    target.write_text("alpha\n", encoding="utf-8")

    content = "\n".join(
        [
            "x.txt",
            "<<<<<<< SEARCH",
            "completely unrelated content that is long enough to score low",
            "more unrelated material here to push similarity down further",
            "=======",
            "replacement",
            ">>>>>>> REPLACE",
        ]
    )
    result = apply_search_replace(content, tmp_path)
    assert not result.touched
    assert len(result.failed) == 1
    failed_edit, reason = result.failed[0]
    assert failed_edit.filename == "x.txt"
    assert "did not match" in reason
    # File untouched.
    assert target.read_text(encoding="utf-8") == "alpha\n"


# ---------------------------------------------------------------------------
# Apply — multiple edits
# ---------------------------------------------------------------------------


def test_apply_multiple_edits_to_different_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("two\n", encoding="utf-8")

    content = "\n".join(
        [
            "a.py",
            "<<<<<<< SEARCH",
            "one",
            "=======",
            "ONE",
            ">>>>>>> REPLACE",
            "b.py",
            "<<<<<<< SEARCH",
            "two",
            "=======",
            "TWO",
            ">>>>>>> REPLACE",
        ]
    )
    result = apply_search_replace(content, tmp_path)
    assert set(result.touched) == {"a.py", "b.py"}
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "ONE\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "TWO\n"


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def test_apply_rejects_path_traversal(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "../escape.txt",
            "<<<<<<< SEARCH",
            "=======",
            "evil",
            ">>>>>>> REPLACE",
        ]
    )
    with pytest.raises(SearchReplaceError, match="Path must stay within cwd"):
        apply_search_replace(content, tmp_path)


def test_apply_atomic_on_failure(tmp_path: Path) -> None:
    """A failed edit must not corrupt the file."""

    target = tmp_path / "f.txt"
    target.write_text("alpha\n", encoding="utf-8")

    content = "\n".join(
        [
            "f.txt",
            "<<<<<<< SEARCH",
            "this does not match anything in the file at all",
            "=======",
            "BETA",
            ">>>>>>> REPLACE",
        ]
    )
    result = apply_search_replace(content, tmp_path)
    assert result.failed
    assert target.read_text(encoding="utf-8") == "alpha\n"
