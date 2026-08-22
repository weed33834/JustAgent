"""``apply_patch`` parser & applier.

Ports the Cline ``apply_patch`` grammar (originally from OpenAI's GPT-5
``apply_patch`` tool) to Python. Supports:

* ``*** Begin Patch`` / ``*** End Patch`` sentinels (optional — raw bodies
  with the file markers are also accepted).
* ``*** Add File: <path>`` followed by ``+line`` content lines.
* ``*** Delete File: <path>``
* ``*** Update File: <path>`` with optional ``*** Move to: <new-path>``
  directive, followed by ``@@``-prefixed sections containing `` `` (keep),
  ``-`` (delete) and ``+`` (insert) lines. ``*** End of File`` anchors a
  section to EOF.
* Legacy shell wrappers — ``%%bash`` / ``apply_patch <<"EOF"`` … ``EOF`` /
  triple-backtick fences — are stripped automatically.
* Fuzzy matching against the on-disk file: NFC normalisation, unicode
  punctuation canonicalisation, trailing-whitespace tolerant, trim-tolerant,
  and finally a Levenshtein similarity pass (threshold 0.66). Mismatches
  produce :class:`PatchWarning` entries rather than silently corrupting the
  file.

References:

* ``competitors/cline/sdk/packages/core/src/extensions/tools/executors/apply-patch-parser.ts``
* ``competitors/cline/sdk/packages/core/src/extensions/tools/executors/apply-patch.ts``
* ``competitors/opencode/packages/opencode/src/patch/index.ts``
"""

from __future__ import annotations

import contextlib
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from justagent.utils.atomic_write import atomic_write_text
from justagent.utils.paths import PathResolutionError, resolve_path

# ---------------------------------------------------------------------------
# Markers & constants
# ---------------------------------------------------------------------------

PATCH_MARKERS = {
    "BEGIN": "*** Begin Patch",
    "END": "*** End Patch",
    "ADD": "*** Add File: ",
    "UPDATE": "*** Update File: ",
    "DELETE": "*** Delete File: ",
    "MOVE": "*** Move to: ",
    "SECTION": "@@",
    "END_FILE": "*** End of File",
}

#: Lines that may legally wrap a patch body when the LLM is asked to invoke
#: ``apply_patch`` via a shell snippet. They are stripped from the head/tail
#: of a patch body that lacks explicit Begin/End sentinels.
BASH_WRAPPERS: tuple[str, ...] = ("%%bash", "apply_patch", "EOF", "```")

#: Minimum Levenshtein-similarity ratio for the fuzzy match fallback.
SIMILARITY_THRESHOLD = 0.66


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class PatchActionType(str, Enum):  # noqa: UP042 - match existing codebase style
    """The kind of file operation a patch action represents."""

    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class PatchChunk:
    """A single contiguous delete/insert hunk within an update action.

    ``orig_index`` is the zero-based line index in the *original* file where
    the chunk starts. ``del_lines`` and ``ins_lines`` are the removed and
    inserted line payloads respectively (without their ``+``/``-``/`` ``
    sigils).
    """

    orig_index: int
    del_lines: list[str]
    ins_lines: list[str]


@dataclass
class PatchAction:
    """A single file-level action in a patch."""

    type: PatchActionType
    new_file: str | None = None
    chunks: list[PatchChunk] = field(default_factory=list)
    move_path: str | None = None


@dataclass
class PatchWarning:
    """A non-fatal problem encountered while parsing/applying a patch.

    Warnings are collected rather than raised so the caller can decide
    whether to abort (default — see :func:`compute_patch_changes`) or
    tolerate partial application.
    """

    path: str
    chunk_index: int | None = None
    message: str = ""
    context: str | None = None


@dataclass
class Patch:
    """Parsed patch — actions keyed by target file path."""

    actions: dict[str, PatchAction] = field(default_factory=dict)
    warnings: list[PatchWarning] = field(default_factory=list)


class DiffError(Exception):
    """Raised when a patch cannot be parsed or applied."""


# ---------------------------------------------------------------------------
# Canonicalisation (fuzzy-match pre-processing)
# ---------------------------------------------------------------------------

#: Map of Unicode punctuation → ASCII equivalent. Mirrors Cline's
#: ``punctuationMap`` so LLM-generated patches that use smart quotes / dashes
#: still match on-disk files written with ASCII punctuation.
_PUNCTUATION_MAP: dict[str, str] = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u201b": "'",
    "\u00a0": " ",
    "\u202f": " ",
}


def _canonicalize(text: str) -> str:
    """NFC-normalise and convert common Unicode punctuation to ASCII.

    Also un-escapes ``\\``` / ``\\'`` / ``\\"`` so LLM output that
    over-escapes quotes still matches.
    """

    normalised = unicodedata.normalize("NFC", text)
    normalised = "".join(_PUNCTUATION_MAP.get(ch, ch) for ch in normalised)
    normalised = normalised.replace("\\`", "`").replace("\\'", "'").replace('\\"', '"')
    return normalised


# ---------------------------------------------------------------------------
# Levenshtein distance (for similarity-scored fuzzy fallback)
# ---------------------------------------------------------------------------


try:  # C++ implementation — 10-100x faster on large files.
    from rapidfuzz import fuzz as _rf_fuzz

    def _similarity(a: str, b: str) -> float:
        """Levenshtein-based similarity in ``[0, 1]`` (RapidFuzz)."""

        if not a and not b:
            return 1.0
        return _rf_fuzz.ratio(a, b) / 100.0

except ImportError:  # pragma: no cover - stdlib fallback

    def _levenshtein(a: str, b: str) -> int:
        """Standard iterative Levenshtein distance."""

        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        previous = list(range(len(b) + 1))
        current = [0] * (len(b) + 1)
        for i, ca in enumerate(a, start=1):
            current[0] = i
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                current[j] = min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            previous, current = current, previous
        return previous[len(b)]

    def _similarity(a: str, b: str) -> float:
        """Levenshtein similarity in ``[0, 1]``."""

        if not a and not b:
            return 1.0
        longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
        if not longer:
            return 1.0
        return (len(longer) - _levenshtein(shorter, longer)) / len(longer)


# ---------------------------------------------------------------------------
# PatchParser
# ---------------------------------------------------------------------------


class _PatchParser:
    """Internal stateful parser. Use :func:`parse_patch` instead.

    ``current_files`` may be ``None`` for parse-only mode (no on-disk
    existence checks); a dict (possibly empty) enables the existence /
    duplicate-via-files checks.
    """

    def __init__(
        self,
        lines: list[str],
        current_files: dict[str, str] | None = None,
    ) -> None:
        self._lines = lines
        self._current_files = current_files
        self._patch = Patch()
        self._index = 0
        self._fuzz = 0
        self._current_path: str | None = None

    # -- public -----------------------------------------------------------

    def parse(self) -> tuple[Patch, int]:
        self._skip_begin_sentinel()
        while self._has_more() and not self._is_end_marker():
            self._parse_next_action()
        return self._patch, self._fuzz

    # -- helpers ----------------------------------------------------------

    def _add_warning(self, warning: PatchWarning) -> None:
        self._patch.warnings.append(warning)

    def _has_more(self) -> bool:
        return self._index < len(self._lines)

    def _current_line(self) -> str | None:
        if 0 <= self._index < len(self._lines):
            return self._lines[self._index]
        return None

    def _skip_begin_sentinel(self) -> None:
        line = self._current_line()
        if line is not None and line.startswith(PATCH_MARKERS["BEGIN"]):
            self._index += 1

    def _is_end_marker(self) -> bool:
        line = self._current_line()
        return line is not None and line.startswith(PATCH_MARKERS["END"])

    def _check_duplicate(self, path: str, operation: str) -> None:
        if path in self._patch.actions:
            raise DiffError(f"Duplicate {operation} for file: {path}")

    @staticmethod
    def _is_stop_marker(line: str, markers: tuple[str, ...]) -> bool:
        return any(line.startswith(m.rstrip()) for m in markers if m)

    # -- action dispatch --------------------------------------------------

    def _parse_next_action(self) -> None:
        line = self._current_line()
        if line is None:
            raise DiffError(f"Unexpected end of input at index {self._index}")
        if line.startswith(PATCH_MARKERS["UPDATE"]):
            self._parse_update(line[len(PATCH_MARKERS["UPDATE"]) :].strip())
            return
        if line.startswith(PATCH_MARKERS["DELETE"]):
            self._parse_delete(line[len(PATCH_MARKERS["DELETE"]) :].strip())
            return
        if line.startswith(PATCH_MARKERS["ADD"]):
            self._parse_add(line[len(PATCH_MARKERS["ADD"]) :].strip())
            return
        raise DiffError(f"Unknown line while parsing: {line}")

    # -- Add / Delete -----------------------------------------------------

    def _parse_delete(self, path: str) -> None:
        self._check_duplicate(path, "delete")
        if self._current_files is not None and path not in self._current_files:
            raise DiffError(f"Delete File Error: Missing File: {path}")
        self._patch.actions[path] = PatchAction(type=PatchActionType.DELETE)
        self._index += 1

    def _parse_add(self, path: str) -> None:
        self._check_duplicate(path, "add")
        if self._current_files is not None and path in self._current_files:
            raise DiffError(f"Add File Error: File already exists: {path}")
        self._index += 1
        stop_markers = (
            PATCH_MARKERS["END"],
            PATCH_MARKERS["UPDATE"],
            PATCH_MARKERS["DELETE"],
            PATCH_MARKERS["ADD"],
        )
        body: list[str] = []
        while self._has_more():
            current = self._current_line()
            assert current is not None  # narrowed by _has_more
            if self._is_stop_marker(current, stop_markers):
                break
            self._index += 1
            if not current.startswith("+"):
                raise DiffError(f"Invalid Add File line (missing '+'): {current}")
            body.append(current[1:])
        self._patch.actions[path] = PatchAction(
            type=PatchActionType.ADD,
            new_file="\n".join(body),
        )

    # -- Update -----------------------------------------------------------

    def _parse_update(self, path: str) -> None:
        self._check_duplicate(path, "update")
        self._current_path = path
        self._index += 1

        move_path: str | None = None
        line = self._current_line()
        if line is not None and line.startswith(PATCH_MARKERS["MOVE"]):
            move_path = line[len(PATCH_MARKERS["MOVE"]) :].strip()
            self._index += 1

        if self._current_files is not None and path not in self._current_files:
            raise DiffError(f"Update File Error: Missing File: {path}")

        text = self._current_files.get(path, "") if self._current_files else ""
        action = self._parse_update_file(text, path)
        action.move_path = move_path
        self._patch.actions[path] = action
        self._current_path = None

    def _parse_update_file(self, text: str, path: str) -> PatchAction:
        action = PatchAction(type=PatchActionType.UPDATE)
        file_lines = text.split("\n")
        index = 0

        stop_markers = (
            PATCH_MARKERS["END"],
            PATCH_MARKERS["UPDATE"],
            PATCH_MARKERS["DELETE"],
            PATCH_MARKERS["ADD"],
            PATCH_MARKERS["END_FILE"],
        )

        while True:
            current = self._current_line()
            if current is None:
                break
            if self._is_stop_marker(current, stop_markers):
                break

            # Optional ``@@ <context>`` header for this section.
            def_str: str | None = None
            if current.startswith("@@ "):
                def_str = current[3:]
            elif current == "@@":
                def_str = ""

            if def_str is not None:
                self._index += 1
            elif index != 0:
                # A non-section line that's not the first section is invalid.
                raise DiffError(f"Invalid Line:\n{current}")

            if def_str is not None and def_str.strip():
                canon_def = _canonicalize(def_str.strip())
                for i in range(index, len(file_lines)):
                    file_line = file_lines[i]
                    if file_line and (
                        _canonicalize(file_line) == canon_def
                        or _canonicalize(file_line.strip()) == canon_def
                    ):
                        index = i + 1
                        if (
                            _canonicalize(file_line.strip()) == canon_def
                            and _canonicalize(file_line) != canon_def
                        ):
                            self._fuzz += 1
                        break

            chunk_context, chunks, end_patch_index, eof = _peek(self._lines, self._index)
            new_index, fuzz, similarity = _find_context(file_lines, chunk_context, index, eof)

            if new_index == -1:
                context_text = "\n".join(chunk_context)
                self._add_warning(
                    PatchWarning(
                        path=self._current_path or path,
                        chunk_index=len(action.chunks),
                        message=(
                            "Could not find matching context "
                            f"(similarity: {similarity:.2f}). Chunk skipped."
                        ),
                        context=(
                            context_text[:200] + "..." if len(context_text) > 200 else context_text
                        ),
                    )
                )
                self._index = end_patch_index
            else:
                self._fuzz += fuzz
                for chunk in chunks:
                    chunk.orig_index += new_index
                    action.chunks.append(chunk)
                index = new_index + len(chunk_context)
                self._index = end_patch_index

        return action


# ---------------------------------------------------------------------------
# Section peek / context search (module-level for testability)
# ---------------------------------------------------------------------------


def _peek(lines: list[str], initial_index: int) -> tuple[list[str], list[PatchChunk], int, bool]:
    """Collect a single ``@@``-delimited section's worth of changes.

    Returns ``(old_lines, chunks, end_index, eof)``:

    * ``old_lines`` — kept + deleted lines, in original order.
    * ``chunks`` — :class:`PatchChunk` list with ``orig_index`` relative to
      ``old_lines`` (caller shifts by the matched file offset).
    * ``end_index`` — index in ``lines`` of the next ``@@`` / ``***`` marker
      (or ``len(lines)``).
    * ``eof`` — True if the section ends with ``*** End of File``.
    """

    index = initial_index
    old: list[str] = []
    del_lines: list[str] = []
    ins_lines: list[str] = []
    chunks: list[PatchChunk] = []
    mode = "keep"

    stop_markers = (
        "@@",
        PATCH_MARKERS["END"],
        PATCH_MARKERS["UPDATE"],
        PATCH_MARKERS["DELETE"],
        PATCH_MARKERS["ADD"],
        PATCH_MARKERS["END_FILE"],
    )

    while index < len(lines):
        source_line = lines[index]
        if not source_line or any(source_line.startswith(m.rstrip()) for m in stop_markers if m):
            break
        if source_line == "***":
            break
        if source_line.startswith("***"):
            raise DiffError(f"Invalid line: {source_line}")

        index += 1
        previous_mode = mode
        line = source_line

        first = line[0] if line else ""
        if first == "+":
            mode = "add"
        elif first == "-":
            mode = "delete"
        elif first == " ":
            mode = "keep"
        else:
            # Lines without a sigil are treated as kept context (matching
            # Cline's tolerant behaviour) — prefix a space so slicing works.
            mode = "keep"
            line = f" {line}"

        line = line[1:]

        if mode == "keep" and previous_mode != mode:
            if ins_lines or del_lines:
                chunks.append(
                    PatchChunk(
                        orig_index=len(old) - len(del_lines),
                        del_lines=del_lines,
                        ins_lines=ins_lines,
                    )
                )
            del_lines = []
            ins_lines = []

        if mode == "delete":
            del_lines.append(line)
            old.append(line)
        elif mode == "add":
            ins_lines.append(line)
        else:  # keep
            old.append(line)

    if ins_lines or del_lines:
        chunks.append(
            PatchChunk(
                orig_index=len(old) - len(del_lines),
                del_lines=del_lines,
                ins_lines=ins_lines,
            )
        )

    if index < len(lines) and lines[index] == PATCH_MARKERS["END_FILE"]:
        index += 1
        return old, chunks, index, True

    return old, chunks, index, False


def _find_context(
    lines: list[str],
    context: list[str],
    start: int,
    eof: bool,
) -> tuple[int, int, float]:
    """Locate ``context`` within ``lines`` starting at ``start``.

    Returns ``(index, fuzz, similarity)``. ``index == -1`` means no match
    found; ``fuzz`` is 0 for an exact match, 1 for trailing-whitespace
    tolerant, 100 for trim-tolerant, 1000 for similarity-threshold fallback,
    and ``+10000`` when the EOF anchor forced a non-anchored match.
    """

    if not context:
        return start, 0, 1.0

    best_similarity = 0.0

    def find_core(start_idx: int) -> tuple[int, int, float]:
        nonlocal best_similarity
        canon_context = _canonicalize("\n".join(context))

        # Pass 1: exact (canonicalised) match.
        for i in range(start_idx, len(lines) - len(context) + 1):
            segment = _canonicalize("\n".join(lines[i : i + len(context)]))
            if segment == canon_context:
                return i, 0, 1.0
            sim = _similarity(segment, canon_context)
            if sim > best_similarity:
                best_similarity = sim

        # Pass 2: rstrip-tolerant.
        canon_rstrip = _canonicalize("\n".join(c.rstrip() for c in context))
        for i in range(start_idx, len(lines) - len(context) + 1):
            segment = _canonicalize(
                "\n".join(line.rstrip() for line in lines[i : i + len(context)])
            )
            if segment == canon_rstrip:
                return i, 1, 1.0

        # Pass 3: trim-tolerant.
        canon_trim = _canonicalize("\n".join(c.strip() for c in context))
        for i in range(start_idx, len(lines) - len(context) + 1):
            segment = _canonicalize("\n".join(line.strip() for line in lines[i : i + len(context)]))
            if segment == canon_trim:
                return i, 100, 1.0

        # Pass 4: similarity-threshold fuzzy match.
        for i in range(start_idx, len(lines) - len(context) + 1):
            segment = _canonicalize("\n".join(lines[i : i + len(context)]))
            sim = _similarity(segment, canon_context)
            if sim >= SIMILARITY_THRESHOLD:
                return i, 1000, sim
            if sim > best_similarity:
                best_similarity = sim

        return -1, 0, best_similarity

    if eof:
        new_index, fuzz, similarity = find_core(len(lines) - len(context))
        if new_index != -1:
            return new_index, fuzz, similarity
        # Fall back to forward scan with a large fuzz penalty so the caller
        # knows the EOF anchor was violated.
        new_index, fuzz, similarity = find_core(start)
        return new_index, fuzz + 10000, similarity

    return find_core(start)


# ---------------------------------------------------------------------------
# Public API: parse, compute, apply
# ---------------------------------------------------------------------------


def _strip_wrappers(lines: list[str]) -> list[str]:
    """Strip leading/trailing ``%%bash`` / ``apply_patch`` / ``EOF`` / ``` `` lines."""

    def is_wrapper(line: str) -> bool:
        if not line.strip():
            return False
        return any(line.startswith(w) for w in BASH_WRAPPERS)

    start = 0
    end = len(lines)
    while start < end and is_wrapper(lines[start]):
        start += 1
    while end > start and is_wrapper(lines[end - 1]):
        end -= 1
    return lines[start:end]


def _normalize_patch_input(patch_text: str) -> list[str]:
    """Normalise ``patch_text`` into a list of lines suitable for the parser.

    Handles CRLF, optional Begin/End sentinels, and shell-wrapper stripping.
    Raises :class:`DiffError` if Begin/End sentinels appear but are
    incomplete — this matches Cline's strict behaviour so partial LLM output
    is rejected loudly rather than silently mis-parsed.
    """

    raw_lines = [line.rstrip("\r") for line in patch_text.split("\n")]

    begin_index = next(
        (i for i, line in enumerate(raw_lines) if line.startswith(PATCH_MARKERS["BEGIN"])),
        -1,
    )
    end_index = -1
    for i in range(len(raw_lines) - 1, -1, -1):
        if raw_lines[i].startswith(PATCH_MARKERS["END"]):
            end_index = i
            break

    if begin_index != -1 or end_index != -1:
        if begin_index == -1 or end_index == -1 or end_index < begin_index:
            raise DiffError(
                "Invalid patch text - incomplete sentinels. Try breaking it into smaller patches."
            )
        return raw_lines[begin_index : end_index + 1]

    stripped = _strip_wrappers(raw_lines)
    while stripped and stripped[0] == "":
        stripped.pop(0)
    while stripped and stripped[-1] == "":
        stripped.pop()
    return [PATCH_MARKERS["BEGIN"], *stripped, PATCH_MARKERS["END"]]


def _resolve_path(cwd: Path, input_path: str, restrict_to_cwd: bool) -> Path:
    """Resolve ``input_path`` against ``cwd``, raising :class:`DiffError`."""

    try:
        return resolve_path(cwd, input_path, restrict_to_cwd=restrict_to_cwd)
    except PathResolutionError as exc:
        raise DiffError(str(exc)) from exc


def _extract_referenced_files(lines: list[str], markers: tuple[str, ...]) -> list[str]:
    """Return file paths referenced by any of ``markers`` in ``lines``."""

    seen: dict[str, None] = {}
    for line in lines:
        for marker in markers:
            if line.startswith(marker):
                path = line[len(marker) :].strip()
                if path:
                    seen.setdefault(path, None)
                break
    return list(seen.keys())


def _load_current_files(lines: list[str], cwd: Path, restrict_to_cwd: bool) -> dict[str, str]:
    """Read the on-disk contents of every file the patch updates/deletes.

    Also verifies that ADD targets do not already exist on disk — preventing
    an LLM from accidentally clobbering a file it should be UPDATE-ing. The
    on-disk existence check is stricter than Cline's parser-side check (which
    only catches intra-patch conflicts) but matches user expectations for an
    agent that edits files in place.
    """

    files: dict[str, str] = {}
    for path in _extract_referenced_files(
        lines, (PATCH_MARKERS["UPDATE"], PATCH_MARKERS["DELETE"])
    ):
        absolute = _resolve_path(cwd, path, restrict_to_cwd)
        if not absolute.is_file():
            raise DiffError(f"File not found: {path}")
        try:
            files[path] = absolute.read_text(encoding="utf-8").replace("\r\n", "\n")
        except OSError as exc:
            raise DiffError(f"File not found: {path}") from exc

    for path in _extract_referenced_files(lines, (PATCH_MARKERS["ADD"],)):
        absolute = _resolve_path(cwd, path, restrict_to_cwd)
        if absolute.exists():
            raise DiffError(f"Add File Error: File already exists: {path}")

    return files


def _apply_chunks(content: str, chunks: list[PatchChunk], file_path: str) -> str:
    """Apply ``chunks`` to ``content`` and return the new text."""

    if not chunks:
        return content

    lines = content.split("\n")
    result: list[str] = []
    current_index = 0

    for chunk in chunks:
        if chunk.orig_index > len(lines):
            raise DiffError(
                f"{file_path}: chunk.orig_index {chunk.orig_index} > lines.length {len(lines)}"
            )
        if current_index > chunk.orig_index:
            raise DiffError(
                f"{file_path}: currentIndex {current_index} > chunk.origIndex {chunk.orig_index}"
            )
        result.extend(lines[current_index : chunk.orig_index])
        result.extend(chunk.ins_lines)
        current_index = chunk.orig_index + len(chunk.del_lines)

    result.extend(lines[current_index:])
    return "\n".join(result)


@dataclass
class PatchFileChange:
    """A computed per-file change. ``old_content`` / ``new_content`` are
    full file contents (not deltas). For ``DELETE``, ``new_content`` is
    ``None``; for ``ADD``, ``old_content`` is ``None``.
    """

    type: PatchActionType
    old_content: str | None = None
    new_content: str | None = None
    move_path: str | None = None


def _patch_to_changes(patch: Patch, original_files: dict[str, str]) -> dict[str, PatchFileChange]:
    changes: dict[str, PatchFileChange] = {}
    for file_path, action in patch.actions.items():
        if action.type is PatchActionType.DELETE:
            changes[file_path] = PatchFileChange(
                type=PatchActionType.DELETE,
                old_content=original_files.get(file_path),
            )
        elif action.type is PatchActionType.ADD:
            if action.new_file is None:
                raise DiffError("ADD action without file content")
            changes[file_path] = PatchFileChange(
                type=PatchActionType.ADD,
                new_content=action.new_file,
            )
        elif action.type is PatchActionType.UPDATE:
            changes[file_path] = PatchFileChange(
                type=PatchActionType.UPDATE,
                old_content=original_files.get(file_path, ""),
                new_content=_apply_chunks(
                    original_files.get(file_path, ""), action.chunks, file_path
                ),
                move_path=action.move_path,
            )
    return changes


def _format_skipped_hunk_failure(warnings: list[PatchWarning]) -> str:
    lines = [
        f"Patch could not be applied because {len(warnings)} hunk"
        f"{'s' if len(warnings) != 1 else ''} did not match the current file content."
    ]
    for warning in warnings:
        hunk = "unknown" if warning.chunk_index is None else str(warning.chunk_index + 1)
        lines.append(f"{warning.path}: hunk {hunk}: {warning.message}")
        if warning.context:
            lines.append(f"Context:\n{warning.context}")
    return "\n".join(lines)


def parse_patch(patch_text: str) -> Patch:
    """Parse ``patch_text`` into a :class:`Patch` (no filesystem access).

    Convenience wrapper for callers that only need the structural view.
    On-disk file-existence checks are skipped — use
    :func:`compute_patch_changes` for full validation.
    """

    lines = _normalize_patch_input(patch_text)
    parser = _PatchParser(lines, current_files=None)
    patch, _ = parser.parse()
    return patch


def compute_patch_changes(
    patch_text: str,
    cwd: Path,
    *,
    restrict_to_cwd: bool = True,
) -> tuple[dict[str, PatchFileChange], int]:
    """Parse ``patch_text`` and compute the per-file changes it implies.

    Reads the on-disk contents of every file the patch references, so the
    returned :class:`PatchFileChange` instances carry full ``old_content``
    and ``new_content``. Raises :class:`DiffError` if any hunk fails to
    match its context.
    """

    lines = _normalize_patch_input(patch_text)
    current_files = _load_current_files(lines, cwd, restrict_to_cwd)
    parser = _PatchParser(lines, current_files=current_files)
    patch, fuzz = parser.parse()
    if patch.warnings:
        raise DiffError(_format_skipped_hunk_failure(patch.warnings))
    return _patch_to_changes(patch, current_files), fuzz


def apply_patch_text(
    patch_text: str,
    cwd: Path,
    *,
    restrict_to_cwd: bool = True,
    encoding: str = "utf-8",
) -> tuple[list[str], int]:
    """Parse & apply ``patch_text`` to files under ``cwd``.

    Returns ``(touched, fuzz)`` where ``touched`` is a list of human-readable
    descriptions (e.g. ``"path/to/file"`` for an in-place update, or
    ``"old -> new"`` for a move) and ``fuzz`` is the cumulative fuzz factor
    reported by the parser. Uses :func:`atomic_write_text` so partial writes
    never corrupt a target file.
    """

    changes, fuzz = compute_patch_changes(patch_text, cwd, restrict_to_cwd=restrict_to_cwd)
    touched: list[str] = []

    for file_path, change in changes.items():
        source_abs = _resolve_path(cwd, file_path, restrict_to_cwd)
        if change.type is PatchActionType.DELETE:
            with contextlib.suppress(FileNotFoundError):
                source_abs.unlink()
            touched.append(f"{file_path}: [deleted]")
        elif change.type is PatchActionType.ADD:
            if change.new_content is None:
                raise DiffError(f"Cannot create {file_path} with no content")
            atomic_write_text(source_abs, change.new_content, encoding=encoding)
            touched.append(file_path)
        elif change.type is PatchActionType.UPDATE:
            if change.new_content is None:
                raise DiffError(f"UPDATE change for {file_path} has no new content")
            if change.move_path:
                move_abs = _resolve_path(cwd, change.move_path, restrict_to_cwd)
                atomic_write_text(move_abs, change.new_content, encoding=encoding)
                with contextlib.suppress(FileNotFoundError):
                    source_abs.unlink()
                touched.append(f"{file_path} -> {change.move_path}")
            else:
                atomic_write_text(source_abs, change.new_content, encoding=encoding)
                touched.append(file_path)

    return touched, fuzz


__all__ = [
    "BASH_WRAPPERS",
    "DiffError",
    "PATCH_MARKERS",
    "Patch",
    "PatchAction",
    "PatchActionType",
    "PatchChunk",
    "PatchFileChange",
    "PatchWarning",
    "apply_patch_text",
    "compute_patch_changes",
    "parse_patch",
]
