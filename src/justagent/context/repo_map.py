"""Repo map generator — scans source files and extracts symbols.

Produces a compact tree representation of a repository's structure,
listing directories, files, and the functions/classes/methods they
contain. Uses regex-based extraction by default (tree-sitter may not be
installed).

Example output::

    src/
      justagent/
        agent/
          runtime.py
            class AgentRuntime
              def run
              def _call_llm
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("justagent.context.repo_map")


class SymbolKind(str, Enum):  # noqa: UP042 - match existing codebase style
    """The kind of a symbol extracted from a source file."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    INTERFACE = "interface"
    TYPE = "type"


@dataclass(frozen=True)
class Symbol:
    """A single symbol (function/class/method/...) found in a source file.

    Attributes:
        name: The symbol's identifier (e.g. ``"AgentRuntime"``).
        kind: What kind of symbol this is.
        line: 1-based line number where the symbol is declared.
        parent: Name of the enclosing symbol (e.g. the class name for a
            method), or an empty string for top-level symbols.
    """

    name: str
    kind: SymbolKind
    line: int
    parent: str = ""


@dataclass(frozen=True)
class FileSymbols:
    """Symbols extracted from a single file.

    Attributes:
        path: File path (relative to the scan root when produced by
            :meth:`RepoMapGenerator.generate`).
        language: Lowercase language name (e.g. ``"python"``) or
            ``"unknown"`` if the extension is not recognised.
        symbols: Symbols found in the file, ordered by line number.
    """

    path: str
    language: str
    symbols: list[Symbol]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _default_ignore_dirs() -> set[str]:
    return {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
        "target",
        ".tox",
    }


def _default_supported_extensions() -> dict[str, str]:
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".rb": "ruby",
    }


@dataclass
class RepoMapConfig:
    """Configuration for :class:`RepoMapGenerator`.

    Attributes:
        max_files: Maximum number of source files to include in the map.
        max_tokens: Soft token budget (used for reporting/filtering; the
            map is hard-truncated to ``max_chars``).
        max_chars: Maximum number of characters in the generated map.
        ignore_dirs: Directory names to skip during the walk.
        supported_extensions: Mapping of file extension (with leading
            dot) to lowercase language name.
        use_tree_sitter: If True and tree-sitter is importable, prefer
            tree-sitter for extraction (falls back to regex otherwise).
    """

    max_files: int = 200
    max_tokens: int = 2048
    max_chars: int = 10000
    ignore_dirs: set[str] = field(default_factory=_default_ignore_dirs)
    supported_extensions: dict[str, str] = field(default_factory=_default_supported_extensions)
    use_tree_sitter: bool = True


# ---------------------------------------------------------------------------
# Symbol-kind → tree prefix
# ---------------------------------------------------------------------------

_KIND_PREFIX: dict[SymbolKind, str] = {
    SymbolKind.FUNCTION: "def",
    SymbolKind.CLASS: "class",
    SymbolKind.METHOD: "def",
    SymbolKind.VARIABLE: "var",
    SymbolKind.IMPORT: "import",
    SymbolKind.INTERFACE: "interface",
    SymbolKind.TYPE: "type",
}


def _line_of(content: str, pos: int) -> int:
    """Return the 1-based line number of ``pos`` in ``content``."""
    return content.count("\n", 0, pos) + 1


# -- tree-sitter (AST-accurate extraction; regex fallback when absent) -------

#: Internal language id → tree-sitter grammar name.
_TS_LANGUAGE_IDS: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "rust": "rust",
    "go": "go",
}

#: Query capture name → :class:`SymbolKind`.
_CAPTURE_KINDS: dict[str, SymbolKind] = {
    "cls": SymbolKind.CLASS,
    "fn": SymbolKind.FUNCTION,
    "meth": SymbolKind.METHOD,
    "var": SymbolKind.VARIABLE,
    "iface": SymbolKind.INTERFACE,
    "typ": SymbolKind.TYPE,
}

#: One query per grammar; capture names must exist in :data:`_CAPTURE_KINDS`.
_TS_QUERIES: dict[str, str] = {
    "python": (
        "(class_definition name: (identifier) @cls)(function_definition name: (identifier) @fn)"
    ),
    "javascript": (
        "(function_declaration name: (identifier) @fn)"
        "(class_declaration name: (identifier) @cls)"
        "(method_definition name: (property_identifier) @meth)"
        "(variable_declarator name: (identifier) @var)"
    ),
    "typescript": (
        "(function_declaration name: (identifier) @fn)"
        "(class_declaration name: (type_identifier) @cls)"
        "(abstract_class_declaration name: (type_identifier) @cls)"
        "(method_definition name: (property_identifier) @meth)"
        "(interface_declaration name: (type_identifier) @iface)"
        "(type_alias_declaration name: (type_identifier) @typ)"
        "(variable_declarator name: (identifier) @var)"
    ),
    "rust": (
        "(function_item name: (identifier) @fn)"
        "(struct_item name: (type_identifier) @cls)"
        "(enum_item name: (type_identifier) @cls)"
        "(trait_item name: (type_identifier) @iface)"
    ),
    "go": (
        "(function_declaration name: (identifier) @fn)"
        "(method_declaration name: (field_identifier) @meth)"
        "(type_spec name: (type_identifier) @cls)"
    ),
}


def _enclosing_class(node: Any) -> str:
    """Return the nearest enclosing class name for a definition node."""
    parent = node.parent
    while parent is not None:
        if parent.type in (
            "class_definition",
            "class_declaration",
            "abstract_class_declaration",
        ):
            name_node = parent.child_by_field_name("name")
            if name_node is not None and name_node.text is not None:
                name_text: str = name_node.text.decode("utf-8", errors="replace")
                return name_text
            return ""
        parent = parent.parent
    return ""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class RepoMapGenerator:
    """Scans a source tree and produces a compact repo map.

    The map is a tree showing directories, files, and the symbols
    (functions/classes/methods) declared in each file. Extraction is
    regex-based; tree-sitter is used opportunistically when available
    and enabled in the config.

    Example::

        >>> gen = RepoMapGenerator()
        >>> print(gen.generate("/path/to/project"))
    """

    def __init__(self, config: RepoMapConfig | None = None) -> None:
        self._config = config or RepoMapConfig()

    # -- public API ---------------------------------------------------------

    def generate(self, root: str | Path) -> str:
        """Scan ``root``, extract symbols, and return a formatted tree.

        The result is truncated to :attr:`RepoMapConfig.max_chars`.
        Returns an empty string if ``root`` is not a directory or
        contains no supported source files.
        """
        root_path = Path(root)
        files = self.scan_files(root_path)
        if not files:
            return ""
        file_symbols: list[FileSymbols] = []
        for f in files:
            fs = self.extract_symbols(f)
            try:
                rel_path = f.relative_to(root_path).as_posix()
            except ValueError:
                rel_path = f.as_posix()
            file_symbols.append(
                FileSymbols(path=rel_path, language=fs.language, symbols=fs.symbols)
            )
        result = self._format_tree(file_symbols)
        if len(result) > self._config.max_chars:
            result = result[: self._config.max_chars]
        return result

    def scan_files(self, root: str | Path) -> list[Path]:
        """Walk ``root`` and return supported source files, sorted.

        Directories listed in :attr:`RepoMapConfig.ignore_dirs` are
        pruned. At most :attr:`RepoMapConfig.max_files` files are
        returned. Returns an empty list if ``root`` is not a directory.
        """
        root_path = Path(root)
        if not root_path.is_dir():
            return []
        ignore = self._config.ignore_dirs
        supported = self._config.supported_extensions
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Prune ignored directories in-place so os.walk does not
            # descend into them.
            dirnames[:] = [d for d in dirnames if d not in ignore]
            for filename in filenames:
                if Path(filename).suffix in supported:
                    files.append(Path(dirpath) / filename)
        files.sort()
        if len(files) > self._config.max_files:
            files = files[: self._config.max_files]
        return files

    def extract_symbols(self, file_path: Path) -> FileSymbols:
        """Detect the language of ``file_path`` and extract its symbols.

        Returns a :class:`FileSymbols` with ``language="unknown"`` and
        no symbols if the extension is not supported. Read errors are
        swallowed and an empty symbol list is returned.
        """
        ext = file_path.suffix
        language = self._config.supported_extensions.get(ext, "")
        if not language:
            return FileSymbols(path=str(file_path), language="unknown", symbols=[])
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return FileSymbols(path=str(file_path), language=language, symbols=[])

        # Preferred: tree-sitter (accurate, C-speed). Fallback: regex.
        ts_symbols = self._extract_tree_sitter(content, language)
        if ts_symbols is not None:
            return FileSymbols(path=str(file_path), language=language, symbols=ts_symbols)

        if language == "python":
            symbols = self._extract_python_regex(content)
        elif language == "javascript":
            symbols = self._extract_javascript_regex(content)
        elif language == "typescript":
            symbols = self._extract_typescript_regex(content)
        elif language == "rust":
            symbols = self._extract_rust_regex(content)
        elif language == "go":
            symbols = self._extract_go_regex(content)
        else:
            symbols = self._extract_generic_regex(content)
        return FileSymbols(path=str(file_path), language=language, symbols=symbols)

    # -- regex extractors ---------------------------------------------------

    def _extract_python_regex(self, content: str) -> list[Symbol]:
        """Extract Python classes, functions, and methods via regex.

        A ``def`` at column 0 is a FUNCTION; an indented ``def`` is a
        METHOD whose ``parent`` is the most recent top-level ``class``.
        """
        # Collect (line, name, kind) tuples, then post-process for parent.
        raw: list[tuple[int, str, SymbolKind]] = []
        for m in re.finditer(r"^class\s+(\w+)", content, re.MULTILINE):
            raw.append((_line_of(content, m.start()), m.group(1), SymbolKind.CLASS))
        for m in re.finditer(r"^def\s+(\w+)", content, re.MULTILINE):
            raw.append((_line_of(content, m.start()), m.group(1), SymbolKind.FUNCTION))
        for m in re.finditer(r"^[ \t]+def\s+(\w+)", content, re.MULTILINE):
            raw.append((_line_of(content, m.start()), m.group(1), SymbolKind.METHOD))
        raw.sort(key=lambda t: t[0])

        symbols: list[Symbol] = []
        current_class = ""
        for line, name, kind in raw:
            if kind is SymbolKind.CLASS:
                current_class = name
                parent = ""
            elif kind is SymbolKind.METHOD:
                parent = current_class
            else:  # FUNCTION — we have left the class scope.
                current_class = ""
                parent = ""
            symbols.append(Symbol(name=name, kind=kind, line=line, parent=parent))
        return symbols

    def _extract_javascript_regex(self, content: str) -> list[Symbol]:
        """Extract JavaScript functions, classes, and const variables."""
        symbols: list[Symbol] = []
        for m in re.finditer(r"function\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.FUNCTION, line=_line_of(content, m.start()))
            )
        for m in re.finditer(r"class\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.CLASS, line=_line_of(content, m.start()))
            )
        for m in re.finditer(r"const\s+(\w+)\s*=", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.VARIABLE, line=_line_of(content, m.start()))
            )
        symbols.sort(key=lambda s: s.line)
        return symbols

    def _extract_typescript_regex(self, content: str) -> list[Symbol]:
        """Extract TypeScript symbols (JavaScript + interfaces and types)."""
        symbols = self._extract_javascript_regex(content)
        for m in re.finditer(r"interface\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                Symbol(
                    name=m.group(1), kind=SymbolKind.INTERFACE, line=_line_of(content, m.start())
                )
            )
        for m in re.finditer(r"type\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.TYPE, line=_line_of(content, m.start()))
            )
        symbols.sort(key=lambda s: s.line)
        return symbols

    def _extract_rust_regex(self, content: str) -> list[Symbol]:
        """Extract Rust functions, structs, enums, and traits."""
        symbols: list[Symbol] = []
        for m in re.finditer(r"fn\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.FUNCTION, line=_line_of(content, m.start()))
            )
        for m in re.finditer(r"struct\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.CLASS, line=_line_of(content, m.start()))
            )
        for m in re.finditer(r"enum\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.CLASS, line=_line_of(content, m.start()))
            )
        for m in re.finditer(r"trait\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                Symbol(
                    name=m.group(1), kind=SymbolKind.INTERFACE, line=_line_of(content, m.start())
                )
            )
        symbols.sort(key=lambda s: s.line)
        return symbols

    def _extract_go_regex(self, content: str) -> list[Symbol]:
        """Extract Go functions and struct types."""
        symbols: list[Symbol] = []
        for m in re.finditer(r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.FUNCTION, line=_line_of(content, m.start()))
            )
        for m in re.finditer(r"type\s+(\w+)\s+struct", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.CLASS, line=_line_of(content, m.start()))
            )
        symbols.sort(key=lambda s: s.line)
        return symbols

    def _extract_generic_regex(self, content: str) -> list[Symbol]:
        """Fallback extractor for languages without a dedicated regex.

        Matches function-like and class-like declarations heuristically.
        """
        symbols: list[Symbol] = []
        for m in re.finditer(r"function\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.FUNCTION, line=_line_of(content, m.start()))
            )
        for m in re.finditer(r"class\s+(\w+)", content, re.MULTILINE):
            symbols.append(
                Symbol(name=m.group(1), kind=SymbolKind.CLASS, line=_line_of(content, m.start()))
            )
        symbols.sort(key=lambda s: s.line)
        return symbols

    # -- formatting ---------------------------------------------------------

    def _format_tree(self, files: list[FileSymbols]) -> str:
        """Format extracted file symbols as an indented tree.

        Directories end with ``/``; files are followed by their symbols
        (methods nested under their parent class). Uses 2-space
        indentation per level.
        """
        if not files:
            return ""
        # Build a nested dict: directory name → dict, file name → FileSymbols.
        tree: dict[str, object] = {}
        for fs in files:
            parts = fs.path.split("/")
            current: dict[str, object] = tree
            for part in parts[:-1]:
                child = current.get(part)
                if not isinstance(child, dict):
                    child = {}
                    current[part] = child
                current = child
            current[parts[-1]] = fs

        lines: list[str] = []
        self._render_tree_node(tree, 0, lines)
        return "\n".join(lines)

    def _render_tree_node(
        self,
        node: dict[str, object],
        depth: int,
        lines: list[str],
    ) -> None:
        """Render one level of the directory tree."""
        indent = "  " * depth
        for name in sorted(node.keys()):
            child = node[name]
            if isinstance(child, FileSymbols):
                lines.append(f"{indent}{name}")
                self._render_symbols(child.symbols, depth + 1, lines)
            elif isinstance(child, dict):
                lines.append(f"{indent}{name}/")
                self._render_tree_node(child, depth + 1, lines)

    def _render_symbols(self, symbols: list[Symbol], depth: int, lines: list[str]) -> None:
        """Render symbols, nesting methods under their parent class."""
        if not symbols:
            return
        by_parent: dict[str, list[Symbol]] = {}
        known_names = {s.name for s in symbols}
        for s in symbols:
            # If the parent is set but the parent symbol is not present in
            # this file, treat the symbol as top-level (orphan-safe).
            parent = s.parent if s.parent in known_names else ""
            by_parent.setdefault(parent, []).append(s)
        self._render_symbol_subtree("", by_parent, depth, lines)

    def _render_symbol_subtree(
        self,
        parent: str,
        by_parent: dict[str, list[Symbol]],
        depth: int,
        lines: list[str],
    ) -> None:
        """Recursively render symbols whose ``parent`` matches."""
        indent = "  " * depth
        for s in sorted(by_parent.get(parent, []), key=lambda s: s.line):
            prefix = _KIND_PREFIX.get(s.kind, "symbol")
            lines.append(f"{indent}{prefix} {s.name}")
            self._render_symbol_subtree(s.name, by_parent, depth + 1, lines)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 characters per token)."""
        return len(text) // 4

    # -- tree-sitter extractor ------------------------------------------------

    def _extract_tree_sitter(self, content: str, language: str) -> list[Symbol] | None:
        """Extract symbols via tree-sitter grammars.

        Returns ``None`` when the grammar pack is unavailable or parsing
        fails, so the caller falls back to the regex extractors. Capture
        names in the queries map 1:1 to :class:`SymbolKind` via
        :data:`_CAPTURE_KINDS`.
        """
        lang_id = _TS_LANGUAGE_IDS.get(language)
        if lang_id is None or not self._has_tree_sitter():
            return None
        try:
            from tree_sitter import Query, QueryCursor
            from tree_sitter_language_pack import get_parser

            parser = get_parser(lang_id)
            ts_language = parser.language
            assert ts_language is not None  # pack guarantees a language
            tree = parser.parse(content.encode("utf-8"))
            cursor = QueryCursor(Query(ts_language, _TS_QUERIES[lang_id]))
            captures: dict[str, list[Any]] = cursor.captures(tree.root_node)
        except Exception:  # noqa: BLE001 - any grammar issue falls back to regex
            logger.debug("tree-sitter extraction failed for %s; using regex", language)
            return None

        symbols: list[Symbol] = []
        for cap_name, nodes in captures.items():
            base_kind = _CAPTURE_KINDS.get(cap_name)
            if base_kind is None:
                continue
            for node in nodes:
                name = node.text.decode("utf-8", errors="replace")
                kind = base_kind
                parent = ""
                if kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                    # A function nested in a class body is a method.
                    cls = _enclosing_class(node)
                    if cls:
                        if kind is SymbolKind.FUNCTION:
                            kind = SymbolKind.METHOD
                        parent = cls
                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        line=node.start_point[0] + 1,
                        parent=parent,
                    )
                )
        symbols.sort(key=lambda s: s.line)
        return symbols

    @staticmethod
    def _has_tree_sitter() -> bool:
        """Return True if the ``tree_sitter`` package is importable."""
        try:
            import tree_sitter  # noqa: F401
        except ImportError:
            return False
        return True


__all__ = [
    "FileSymbols",
    "RepoMapConfig",
    "RepoMapGenerator",
    "Symbol",
    "SymbolKind",
]
