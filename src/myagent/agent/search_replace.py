r"""SEARCH/REPLACE edit format parser & applier.

Ports Aider's signature SEARCH/REPLACE block format to Python. The grammar
is intentionally human-friendly and tolerant of common LLM mistakes:

.. code-block:: text

    path/to/file.py
    ```
    <<<<<<< SEARCH
    original code
    =======
    updated code
    >>>>>>> REPLACE
    ```

Supported features (parity with Aider's ``editblock_coder.py``):

* Multiple edits per response (each with its own filename).
* ``<<<<<<<`` / ``=======`` / ``>>>>>>>`` separators (5–9 chars wide).
* Optional filename lookup from the 3 lines preceding the opening
  ``<<<<<<<`` (handles ```` ```python\nfname.py\n``` ```` wrapping).
* Perfect-match replacement first, then leading-whitespace-tolerant match
  (so the LLM can uniformly outdent a block).
* ``...`` elision handling (search and replace sides split on the
  ``r"^\s*\.\.\.\n"`` pattern and paired).
* Fuzzy fallback via :class:`difflib.SequenceMatcher` (similarity ≥ 0.8) so
  near-misses still apply.
* Empty SEARCH block ⇒ "append to file" (or create new file).

References:

* ``competitors/aider/aider/coders/editblock_coder.py``
* ``competitors/aider/aider/coders/editblock_prompts.py``
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from myagent.utils.atomic_write import atomic_write_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FENCE: tuple[str, str] = ("```", "```")
TRIPLE_BACKTICKS = "`" * 3

#: Regex for the SEARCH/REPLACE markers. Aider accepts 5–9 chars wide so
#: typos like ``<<<<<<`` (6 chars) or ``<<<<<<<<<`` (9) still parse.
HEAD_RE = re.compile(r"^<{5,9} SEARCH>?\s*$")
DIVIDER_RE = re.compile(r"^={5,9}\s*$")
UPDATED_RE = re.compile(r"^>{5,9} REPLACE\s*$")

#: Pattern matching the ``...`` elision marker.
DOTS_RE = re.compile(r"(^\s*\.\.\.\n)", re.MULTILINE | re.DOTALL)

#: Minimum :class:`~difflib.SequenceMatcher` ratio for the fuzzy fallback.
SIMILARITY_THRESHOLD = 0.8


class SearchReplaceError(Exception):
    """Raised when a SEARCH/REPLACE block cannot be parsed or applied."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SearchReplaceEdit:
    """A single SEARCH/REPLACE edit.

    ``filename`` is the target path (project-relative). ``search`` is the
    text to find in the file (may be empty to mean "append"). ``replace``
    is the new text to substitute.
    """

    filename: str
    search: str
    replace: str


@dataclass
class SearchReplaceResult:
    """Outcome of applying a batch of edits."""

    touched: list[str] = field(default_factory=list)
    """Files that were written."""

    failed: list[tuple[SearchReplaceEdit, str]] = field(default_factory=list)
    """Edits that did not apply, paired with a human-readable reason."""


# ---------------------------------------------------------------------------
# Filename extraction
# ---------------------------------------------------------------------------


def _strip_filename(raw: str, fence: tuple[str, str]) -> str | None:
    """Extract a filename from a wrapping line.

    Handles ```` ```python\\nfname.py ```` , ``# fname.py``, ``fname.py:``,
    backtick-wrapped names, etc. Returns ``None`` if the line doesn't look
    like a filename marker.
    """

    filename = raw.strip()
    if filename == "...":
        return None

    start_fence = fence[0]
    if filename.startswith(start_fence):
        candidate = filename[len(start_fence) :]
        if candidate and ("." in candidate or "/" in candidate):
            return candidate
        return None

    if filename.startswith(TRIPLE_BACKTICKS):
        candidate = filename[len(TRIPLE_BACKTICKS) :]
        if candidate and ("." in candidate or "/" in candidate):
            return candidate
        return None

    filename = filename.rstrip(":").lstrip("#").strip().strip("`").strip("*")
    # Reject anything that doesn't look like a path — this prevents
    # SEARCH/REPLACE marker lines (e.g. ``>>>>>>> REPLACE``) leaking through
    # as candidate filenames when scanning backwards.
    if not filename or ("." not in filename and "/" not in filename):
        return None
    return filename


def _find_filename(
    preceding_lines: list[str],
    fence: tuple[str, str],
    valid_fnames: list[str] | None,
) -> str | None:
    """Search the 3 lines preceding a ``<<<<<<< SEARCH`` for a filename.

    Mirrors Aider's heuristic: walk backwards through up to 3 preceding
    lines, collecting candidate filenames; then pick the best match
    (exact > basename > fuzzy > first-with-extension).
    """

    import difflib as _difflib

    if valid_fnames is None:
        valid_fnames = []

    # Look at most 3 preceding lines, in reverse order.
    candidates: list[str] = []
    for line in reversed(preceding_lines[-3:]):
        fname = _strip_filename(line, fence)
        if fname:
            candidates.append(fname)
        # Stop scanning if the line isn't a fence line — Aider only walks
        # back through fence/blank markers to avoid false positives.
        if not line.startswith(fence[0]) and not line.startswith(TRIPLE_BACKTICKS):
            break

    if not candidates:
        return None

    # Exact match wins.
    for fname in candidates:
        if fname in valid_fnames:
            return fname

    # Basename match.
    for fname in candidates:
        for vfn in valid_fnames:
            if fname == Path(vfn).name:
                return vfn

    # Fuzzy match against valid filenames.
    for fname in candidates:
        close = _difflib.get_close_matches(fname, valid_fnames, n=1, cutoff=0.8)
        if close:
            return close[0]

    # First candidate with an extension wins.
    for fname in candidates:
        if "." in fname:
            return fname

    return candidates[0]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_search_replace(
    content: str,
    *,
    fence: tuple[str, str] = DEFAULT_FENCE,
    valid_fnames: list[str] | None = None,
) -> list[SearchReplaceEdit]:
    """Parse a model response into a list of SEARCH/REPLACE edits.

    Lines that aren't part of any block are silently skipped. Raises
    :class:`SearchReplaceError` for malformed blocks (missing divider /
    terminator, missing filename when no current filename is in scope).
    """

    lines = content.splitlines(keepends=True)
    edits: list[SearchReplaceEdit] = []
    current_filename: str | None = None
    i = 0

    while i < len(lines):
        line = lines[i]
        if not HEAD_RE.match(line.strip()):
            i += 1
            continue

        # We're at a SEARCH header. Look back up to 3 lines for a filename.
        preceding = lines[max(0, i - 3) : i]
        # If the line immediately after the SEARCH is the DIVIDER, this is a
        # new-file edit (search empty) — disable the "must be in valid_fnames"
        # restriction so a brand-new path can be created.
        if i + 1 < len(lines) and DIVIDER_RE.match(lines[i + 1].strip()):
            filename = _find_filename(preceding, fence, None)
        else:
            filename = _find_filename(preceding, fence, valid_fnames)

        if not filename:
            if current_filename:
                filename = current_filename
            else:
                raise SearchReplaceError(
                    "Bad/missing filename. The filename must be alone on the line "
                    f"before the opening fence {fence[0]}"
                )
        current_filename = filename

        # Collect SEARCH lines until the divider.
        i += 1
        search_lines: list[str] = []
        while i < len(lines) and not DIVIDER_RE.match(lines[i].strip()):
            search_lines.append(lines[i])
            i += 1
        if i >= len(lines) or not DIVIDER_RE.match(lines[i].strip()):
            raise SearchReplaceError("Expected `=======`")

        # Collect REPLACE lines until the terminator (or another divider,
        # which Aider tolerates for chained blocks).
        i += 1
        replace_lines: list[str] = []
        while i < len(lines) and not (
            UPDATED_RE.match(lines[i].strip()) or DIVIDER_RE.match(lines[i].strip())
        ):
            replace_lines.append(lines[i])
            i += 1
        if i >= len(lines) or not (
            UPDATED_RE.match(lines[i].strip()) or DIVIDER_RE.match(lines[i].strip())
        ):
            raise SearchReplaceError("Expected `>>>>>>> REPLACE` or `=======`")

        edits.append(
            SearchReplaceEdit(
                filename=filename,
                search="".join(search_lines),
                replace="".join(replace_lines),
            )
        )
        i += 1

    return edits


# ---------------------------------------------------------------------------
# Applier
# ---------------------------------------------------------------------------


def _prep(content: str) -> tuple[str, list[str]]:
    """Ensure trailing newline and split into lines (keeping endings)."""

    if content and not content.endswith("\n"):
        content += "\n"
    return content, content.splitlines(keepends=True)


def _perfect_replace(
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str]
) -> str | None:
    """Exact line-for-line match."""

    part_tup = tuple(part_lines)
    part_len = len(part_lines)
    for i in range(len(whole_lines) - part_len + 1):
        if part_tup == tuple(whole_lines[i : i + part_len]):
            return "".join(
                whole_lines[:i] + replace_lines + whole_lines[i + part_len :]
            )
    return None


def _match_but_for_leading_whitespace(
    whole_lines: list[str], part_lines: list[str]
) -> str | None:
    """If ``whole_lines`` matches ``part_lines`` modulo a uniform leading
    whitespace offset, return that offset string; else ``None``.
    """

    num = len(whole_lines)
    if not all(
        whole_lines[i].lstrip() == part_lines[i].lstrip() for i in range(num)
    ):
        return None
    add = {
        whole_lines[i][: len(whole_lines[i]) - len(part_lines[i])]
        for i in range(num)
        if whole_lines[i].strip()
    }
    if len(add) != 1:
        return None
    return next(iter(add))


def _replace_part_with_missing_leading_whitespace(
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str]
) -> str | None:
    """Outdent search & replace uniformly, then look for an exact match."""

    leading: list[int] = [
        len(p) - len(p.lstrip()) for p in part_lines if p.strip()
    ]
    leading += [len(p) - len(p.lstrip()) for p in replace_lines if p.strip()]
    num_leading = min(leading, default=0)
    if num_leading > 0:
        part_lines = [p[num_leading:] if p.strip() else p for p in part_lines]
        replace_lines = [
            p[num_leading:] if p.strip() else p for p in replace_lines
        ]

    num_part_lines = len(part_lines)
    for i in range(len(whole_lines) - num_part_lines + 1):
        add_leading = _match_but_for_leading_whitespace(
            whole_lines[i : i + num_part_lines], part_lines
        )
        if add_leading is None:
            continue
        new_replace = [
            add_leading + rline if rline.strip() else rline
            for rline in replace_lines
        ]
        return "".join(
            whole_lines[:i] + new_replace + whole_lines[i + num_part_lines :]
        )
    return None


def _perfect_or_whitespace(
    whole_lines: list[str], part_lines: list[str], replace_lines: list[str]
) -> str | None:
    """Try exact match first, then leading-whitespace-tolerant match."""

    result = _perfect_replace(whole_lines, part_lines, replace_lines)
    if result:
        return result
    return _replace_part_with_missing_leading_whitespace(
        whole_lines, part_lines, replace_lines
    )


def _try_dotdotdots(whole: str, part: str, replace: str) -> str | None:
    """Handle ``...`` elisions in SEARCH/REPLACE blocks.

    The ``...`` markers must appear in matched pairs in both SEARCH and
    REPLACE. Each non-``...`` segment is then substituted in order, with
    ``...`` segments acting as wildcards between them.
    """

    part_pieces = re.split(DOTS_RE, part)
    replace_pieces = re.split(DOTS_RE, replace)

    if len(part_pieces) != len(replace_pieces):
        raise SearchReplaceError("Unpaired ... in SEARCH/REPLACE block")

    if len(part_pieces) == 1:
        # No dots — caller should fall through to other strategies.
        return None

    # Odd-indexed pieces are the ... markers themselves; they must match.
    all_dots_match = all(
        part_pieces[i] == replace_pieces[i] for i in range(1, len(part_pieces), 2)
    )
    if not all_dots_match:
        raise SearchReplaceError("Unmatched ... in SEARCH/REPLACE block")

    part_pieces = [part_pieces[i] for i in range(0, len(part_pieces), 2)]
    replace_pieces = [replace_pieces[i] for i in range(0, len(replace_pieces), 2)]

    for piece_part, piece_replace in zip(part_pieces, replace_pieces, strict=True):
        if not piece_part and not piece_replace:
            continue
        if not piece_part and piece_replace:
            if not whole.endswith("\n"):
                whole += "\n"
            whole += piece_replace
            continue
        if whole.count(piece_part) == 0:
            raise SearchReplaceError("... segment did not match in source file")
        if whole.count(piece_part) > 1:
            raise SearchReplaceError("... segment matched multiple times in source")
        whole = whole.replace(piece_part, piece_replace, 1)
    return whole


def _replace_closest_edit_distance(
    whole_lines: list[str],
    part: str,
    part_lines: list[str],
    replace_lines: list[str],
) -> str | None:
    """Last-resort fuzzy match via :class:`difflib.SequenceMatcher`."""

    max_similarity = 0.0
    best_start = -1
    best_end = -1

    min_len = math.floor(len(part_lines) * 0.9)
    max_len = math.ceil(len(part_lines) * 1.1)

    for length in range(min_len, max_len + 1):
        for i in range(len(whole_lines) - length + 1):
            chunk = "".join(whole_lines[i : i + length])
            similarity = SequenceMatcher(None, chunk, part).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                best_start = i
                best_end = i + length

    if max_similarity < SIMILARITY_THRESHOLD:
        return None

    return "".join(
        whole_lines[:best_start] + replace_lines + whole_lines[best_end:]
    )


def _replace_most_similar_chunk(whole: str, part: str, replace: str) -> str | None:
    """Try every strategy in order. Returns the new content or ``None``."""

    whole, whole_lines = _prep(whole)
    part, part_lines = _prep(part)
    replace, replace_lines = _prep(replace)

    result = _perfect_or_whitespace(whole_lines, part_lines, replace_lines)
    if result:
        return result

    # GPT sometimes spuriously adds a leading blank line to the SEARCH block.
    if len(part_lines) > 2 and not part_lines[0].strip():
        skipped = part_lines[1:]
        result = _perfect_or_whitespace(whole_lines, skipped, replace_lines)
        if result:
            return result

    # Handle ``...`` elisions.
    try:
        result = _try_dotdotdots(whole, part, replace)
        if result:
            return result
    except SearchReplaceError:
        pass

    # Final fallback: fuzzy edit-distance match.
    return _replace_closest_edit_distance(
        whole_lines, part, part_lines, replace_lines
    )


def _strip_quoted_wrapping(
    text: str, fname: str | None, fence: tuple[str, str]
) -> str:
    """Strip ```` ```\n...\n``` ```` wrapping if present."""

    if not text:
        return text
    res = text.splitlines()
    if not res:
        return text
    if fname and res[0].strip().endswith(Path(fname).name):
        res = res[1:]
    if len(res) >= 2 and res[0].startswith(fence[0]) and res[-1].startswith(fence[1]):
        res = res[1:-1]
    out = "\n".join(res)
    if out and not out.endswith("\n"):
        out += "\n"
    return out


def _resolve_path(cwd: Path, input_path: str, restrict_to_cwd: bool) -> Path:
    """Resolve ``input_path`` against ``cwd``, optionally enforcing containment."""

    is_absolute = Path(input_path).is_absolute()
    resolved = (Path(input_path) if is_absolute else (cwd / input_path)).resolve()
    if not restrict_to_cwd or is_absolute:
        return resolved
    try:
        resolved.relative_to(cwd.resolve())
    except ValueError as exc:
        raise SearchReplaceError(
            f"Path must stay within cwd: {input_path}"
        ) from exc
    return resolved


def _do_replace(
    file_path: Path,
    content: str | None,
    search: str,
    replace: str,
    fence: tuple[str, str],
) -> str | None:
    """Apply one SEARCH/REPLACE to a file's content. Returns new content or ``None``."""

    search = _strip_quoted_wrapping(search, str(file_path), fence)
    replace = _strip_quoted_wrapping(replace, str(file_path), fence)

    # New-file case: empty SEARCH + file doesn't exist.
    if not file_path.exists() and not search.strip():
        file_path.touch()
        content = ""

    if content is None:
        return None

    if not search.strip():
        # Append (or seed new file).
        return content + replace

    return _replace_most_similar_chunk(content, search, replace)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_search_replace(
    content: str,
    cwd: Path,
    *,
    restrict_to_cwd: bool = True,
    fence: tuple[str, str] = DEFAULT_FENCE,
    valid_fnames: list[str] | None = None,
    encoding: str = "utf-8",
) -> SearchReplaceResult:
    """Parse ``content`` for SEARCH/REPLACE blocks and apply them under ``cwd``.

    Files are written atomically. Edits that fail to match their SEARCH
    block are collected into :attr:`SearchReplaceResult.failed` rather
    than raising — callers can decide whether to surface them to the LLM
    for repair or abort.
    """

    edits = parse_search_replace(content, fence=fence, valid_fnames=valid_fnames)
    result = SearchReplaceResult()

    for edit in edits:
        file_path = _resolve_path(cwd, edit.filename, restrict_to_cwd)
        existing = (
            file_path.read_text(encoding=encoding) if file_path.exists() else None
        )
        try:
            new_content = _do_replace(
                file_path, existing, edit.search, edit.replace, fence
            )
        except SearchReplaceError as exc:
            result.failed.append((edit, str(exc)))
            continue

        if new_content is None:
            result.failed.append(
                (
                    edit,
                    f"SEARCH block did not match any lines in {edit.filename}",
                )
            )
            continue

        atomic_write_text(file_path, new_content, encoding=encoding)
        result.touched.append(edit.filename)

    return result


__all__ = [
    "DEFAULT_FENCE",
    "SearchReplaceEdit",
    "SearchReplaceError",
    "SearchReplaceResult",
    "apply_search_replace",
    "parse_search_replace",
]
