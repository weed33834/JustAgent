"""Tests for ``myagent.agent.patch`` (ported from Cline's apply-patch.test.ts)."""

from __future__ import annotations

from pathlib import Path

import pytest

from myagent.agent.patch import (
    DiffError,
    apply_patch_text,
    compute_patch_changes,
    parse_patch,
)

# ---------------------------------------------------------------------------
# Add File
# ---------------------------------------------------------------------------


def test_add_file_without_sentinels(tmp_path: Path) -> None:
    """A raw patch body (no Begin/End) for a new file applies cleanly."""

    result = apply_patch_text(
        "*** Add File: note.txt\n+hello",
        tmp_path,
    )
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"
    assert result[0] == ["note.txt"]


def test_add_file_with_explicit_sentinels(tmp_path: Path) -> None:
    apply_patch_text(
        "*** Begin Patch\n*** Add File: note.txt\n+hello\n*** End Patch",
        tmp_path,
    )
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"


def test_add_file_with_trailing_whitespace_in_end_sentinel(tmp_path: Path) -> None:
    result = apply_patch_text(
        "\n".join(
            [
                "*** Begin Patch",
                "*** Add File: note.txt",
                "+hello",
                "*** End Patch ",
            ]
        ),
        tmp_path,
    )
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"
    assert result[0] == ["note.txt"]


def test_add_file_multi_line(tmp_path: Path) -> None:
    result = apply_patch_text(
        "*** Add File: poem.txt\n+line one\n+line two\n+line three",
        tmp_path,
    )
    body = (tmp_path / "poem.txt").read_text(encoding="utf-8")
    assert body == "line one\nline two\nline three"
    assert result[0] == ["poem.txt"]


def test_add_file_rejects_existing_file(tmp_path: Path) -> None:
    (tmp_path / "exists.txt").write_text("data", encoding="utf-8")
    with pytest.raises(DiffError, match="File already exists"):
        apply_patch_text(
            "*** Add File: exists.txt\n+overwrite",
            tmp_path,
        )


def test_add_file_rejects_missing_plus_prefix(tmp_path: Path) -> None:
    with pytest.raises(DiffError, match="missing '\\+'"):
        apply_patch_text(
            "*** Add File: bad.txt\nhello",
            tmp_path,
        )


# ---------------------------------------------------------------------------
# Delete File
# ---------------------------------------------------------------------------


def test_delete_file(tmp_path: Path) -> None:
    target = tmp_path / "trash.txt"
    target.write_text("garbage", encoding="utf-8")
    touched, _ = apply_patch_text(
        "*** Delete File: trash.txt",
        tmp_path,
    )
    assert not target.exists()
    assert touched == ["trash.txt: [deleted]"]


def test_delete_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DiffError, match="File not found"):
        apply_patch_text("*** Delete File: ghost.txt", tmp_path)


# ---------------------------------------------------------------------------
# Update File
# ---------------------------------------------------------------------------


def test_update_file_inserts_line(tmp_path: Path) -> None:
    """Port of Cline's ``applies the documented freeform patch format`` test."""

    target = tmp_path / "page.tsx"
    target.write_text(
        "\n".join(
            [
                "export default function Page() {",
                "\treturn (",
                "\t\t<div>",
                '\t\t\t<button onClick={() => console.log("clicked")}>Click me</button>',
                "\t\t</div>",
                "\t);",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    patch = "\n".join(
        [
            "*** Update File: page.tsx",
            "@@",
            " export default function Page() {",
            " \treturn (",
            " \t\t<div>",
            ' \t\t\t<button onClick={() => console.log("clicked")}>Click me</button>',
            '+\t\t\t<button onClick={() => console.log("cancel clicked")}>Cancel</button>',
            " \t\t</div>",
            " \t);",
            " }",
        ]
    )
    touched, _ = apply_patch_text(patch, tmp_path)

    body = target.read_text(encoding="utf-8")
    assert 'console.log("cancel clicked")' in body
    assert touched == ["page.tsx"]


def test_update_file_keeps_lines_with_wrapper_tokens(tmp_path: Path) -> None:
    """Lines beginning with ``EOF`` / ``` `` inside the body are not stripped."""

    target = tmp_path / "note.txt"
    target.write_text(
        "\n".join(["alpha", "EOF literal", "``` fence", "omega"]),
        encoding="utf-8",
    )

    patch = "\n".join(
        [
            "*** Update File: note.txt",
            "@@",
            " alpha",
            " EOF literal",
            " ``` fence",
            "+tail",
            " omega",
        ]
    )
    apply_patch_text(patch, tmp_path)
    assert target.read_text(encoding="utf-8") == "\n".join(
        ["alpha", "EOF literal", "``` fence", "tail", "omega"]
    )


def test_update_file_replaces_lines(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")

    patch = "\n".join(
        [
            "*** Update File: config.txt",
            "@@",
            " alpha",
            "-beta",
            "+BETA",
            " gamma",
        ]
    )
    apply_patch_text(patch, tmp_path)
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\ndelta\n"


def test_update_file_with_move(tmp_path: Path) -> None:
    target = tmp_path / "old.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    patch = "\n".join(
        [
            "*** Update File: old.py",
            "*** Move to: new.py",
            "@@",
            " print('hi')",
            "+print('bye')",
        ]
    )
    touched, _ = apply_patch_text(patch, tmp_path)
    assert not target.exists()
    new = tmp_path / "new.py"
    assert new.is_file()
    assert new.read_text(encoding="utf-8") == "print('hi')\nprint('bye')\n"
    assert touched == ["old.py -> new.py"]


def test_update_file_eof_anchor(tmp_path: Path) -> None:
    target = tmp_path / "tail.txt"
    target.write_text("first\nsecond\n", encoding="utf-8")

    # ``*** End of File`` terminates the @@ section, marking it as EOF-
    # anchored. The ``+appended`` line precedes the marker.
    patch = "\n".join(
        [
            "*** Update File: tail.txt",
            "@@",
            " first",
            " second",
            "+appended",
            "*** End of File",
        ]
    )
    apply_patch_text(patch, tmp_path)
    assert target.read_text(encoding="utf-8") == "first\nsecond\nappended\n"


def test_update_file_rejects_unmatched_context(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    original = "alpha\nbeta\ngamma"
    target.write_text(original, encoding="utf-8")

    patch = "\n".join(
        [
            "*** Update File: note.txt",
            "@@",
            " unrelated heading",
            " missing middle",
            "+replacement",
            " absent footer",
        ]
    )
    with pytest.raises(DiffError, match=r"note\.txt: hunk 1: Could not find matching context"):
        apply_patch_text(patch, tmp_path)
    # The file must be untouched after a failure.
    assert target.read_text(encoding="utf-8") == original


def test_update_file_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DiffError, match="File not found"):
        apply_patch_text(
            "*** Update File: phantom.txt\n@@\n+foo",
            tmp_path,
        )


# ---------------------------------------------------------------------------
# Wrapper handling
# ---------------------------------------------------------------------------


def test_legacy_shell_wrapper_is_stripped(tmp_path: Path) -> None:
    patch = "\n".join(
        [
            "%%bash",
            'apply_patch <<"EOF"',
            "*** Begin Patch",
            "*** Add File: note.txt",
            "+hello",
            "*** End Patch",
            "EOF",
        ]
    )
    apply_patch_text(patch, tmp_path)
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"


def test_triple_backtick_fence_is_stripped(tmp_path: Path) -> None:
    patch = "\n".join(
        [
            "```",
            "*** Begin Patch",
            "*** Add File: note.txt",
            "+hello",
            "*** End Patch",
            "```",
        ]
    )
    apply_patch_text(patch, tmp_path)
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"


# ---------------------------------------------------------------------------
# Sentinel validation
# ---------------------------------------------------------------------------


def test_incomplete_sentinels_rejected(tmp_path: Path) -> None:
    with pytest.raises(DiffError, match="incomplete sentinels"):
        apply_patch_text(
            "*** Begin Patch\n*** Add File: note.txt\n+hello",
            tmp_path,
        )


def test_end_sentinel_before_begin_rejected(tmp_path: Path) -> None:
    with pytest.raises(DiffError, match="incomplete sentinels"):
        apply_patch_text(
            "*** End Patch\n*** Begin Patch\n*** Add File: x.txt\n+hi",
            tmp_path,
        )


# ---------------------------------------------------------------------------
# Multi-action
# ---------------------------------------------------------------------------


def test_multiple_actions_in_one_patch(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("one\ntwo\n", encoding="utf-8")
    (tmp_path / "drop.txt").write_text("bye", encoding="utf-8")

    patch = "\n".join(
        [
            "*** Begin Patch",
            "*** Add File: created.txt",
            "+fresh",
            "*** Update File: keep.txt",
            "@@",
            " one",
            "-two",
            "+TWO",
            "*** Delete File: drop.txt",
            "*** End Patch",
        ]
    )
    touched, _ = apply_patch_text(patch, tmp_path)

    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "fresh"
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "one\nTWO\n"
    assert not (tmp_path / "drop.txt").exists()
    assert set(touched) == {"created.txt", "keep.txt", "drop.txt: [deleted]"}


def test_duplicate_action_rejected(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("a\n", encoding="utf-8")
    with pytest.raises(DiffError, match="Duplicate update"):
        apply_patch_text(
            "*** Update File: x.txt\n@@\n+a\n*** Update File: x.txt\n@@\n+b",
            tmp_path,
        )


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def test_restrict_to_cwd_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(DiffError, match="Path must stay within cwd"):
        apply_patch_text(
            "*** Add File: ../escape.txt\n+evil",
            tmp_path,
        )


def test_restrict_to_cwd_disabled_allows_traversal(tmp_path: Path) -> None:
    """When ``restrict_to_cwd=False``, path traversal is permitted (host's
    responsibility to scope)."""

    outside = tmp_path.parent / "myagent_outside_target.txt"
    try:
        apply_patch_text(
            f"*** Add File: {outside}\n+ok",
            tmp_path,
            restrict_to_cwd=False,
        )
        assert outside.read_text(encoding="utf-8") == "ok"
    finally:
        if outside.exists():
            outside.unlink()


# ---------------------------------------------------------------------------
# Pure parse API
# ---------------------------------------------------------------------------


def test_parse_patch_returns_structural_view() -> None:
    patch = parse_patch(
        "\n".join(
            [
                "*** Begin Patch",
                "*** Add File: a.txt",
                "+x",
                "*** Delete File: b.txt",
                "*** End Patch",
            ]
        )
    )
    assert set(patch.actions.keys()) == {"a.txt", "b.txt"}
    assert patch.actions["a.txt"].type.value == "add"
    assert patch.actions["a.txt"].new_file == "x"
    assert patch.actions["b.txt"].type.value == "delete"


def test_compute_patch_changes_does_not_write(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    patch = "\n".join(
        [
            "*** Update File: f.txt",
            "@@",
            " alpha",
            "-beta",
            "+BETA",
        ]
    )
    changes, fuzz = compute_patch_changes(patch, tmp_path)
    assert "f.txt" in changes
    assert changes["f.txt"].old_content == "alpha\nbeta\n"
    assert changes["f.txt"].new_content == "alpha\nBETA\n"
    assert fuzz == 0
    # File on disk untouched.
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


def test_fuzzy_match_tolerates_trailing_whitespace(tmp_path: Path) -> None:
    target = tmp_path / "ws.txt"
    target.write_text("alpha   \nbeta\n", encoding="utf-8")

    # Patch context lacks the trailing spaces; the rstrip pass should match.
    patch = "\n".join(
        [
            "*** Update File: ws.txt",
            "@@",
            " alpha",
            "-beta",
            "+BETA",
        ]
    )
    apply_patch_text(patch, tmp_path)
    assert target.read_text(encoding="utf-8") == "alpha   \nBETA\n"


def test_fuzzy_match_tolerates_smart_quotes(tmp_path: Path) -> None:
    target = tmp_path / "quotes.txt"
    # On-disk file uses ASCII quotes.
    target.write_text('title: "hello"\n', encoding="utf-8")

    # Patch context uses smart quotes; canonicalisation should match.
    patch = "*** Update File: quotes.txt\n@@\n title: \u201chello\u201d\n+second"
    apply_patch_text(patch, tmp_path)
    body = target.read_text(encoding="utf-8")
    assert "second" in body
    assert '"hello"' in body


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


def test_apply_is_atomic_on_parse_failure(tmp_path: Path) -> None:
    """If a later hunk fails, earlier files must not have been written."""

    target = tmp_path / "f.txt"
    target.write_text("alpha\n", encoding="utf-8")

    # First action succeeds, second action references a missing file.
    patch = "\n".join(
        [
            "*** Begin Patch",
            "*** Update File: f.txt",
            "@@",
            " alpha",
            "+beta",
            "*** Update File: missing.txt",
            "@@",
            " nope",
            "*** End Patch",
        ]
    )
    with pytest.raises(DiffError):
        apply_patch_text(patch, tmp_path)

    # ``compute_patch_changes`` parses *all* actions before any writes happen,
    # so ``f.txt`` is untouched.
    assert target.read_text(encoding="utf-8") == "alpha\n"
