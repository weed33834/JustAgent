"""Tests for the repo map generator."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from autoship.context.repo_map import (
    FileSymbols,
    RepoMapConfig,
    RepoMapGenerator,
    Symbol,
    SymbolKind,
)

# ---------------------------------------------------------------------------
# TestSymbolKind
# ---------------------------------------------------------------------------


class TestSymbolKind:
    def test_is_str_enum(self) -> None:
        assert isinstance(SymbolKind.FUNCTION, str)

    def test_values(self) -> None:
        assert SymbolKind.FUNCTION.value == "function"
        assert SymbolKind.CLASS.value == "class"
        assert SymbolKind.METHOD.value == "method"
        assert SymbolKind.VARIABLE.value == "variable"
        assert SymbolKind.IMPORT.value == "import"
        assert SymbolKind.INTERFACE.value == "interface"
        assert SymbolKind.TYPE.value == "type"

    def test_from_value(self) -> None:
        assert SymbolKind("class") is SymbolKind.CLASS

    def test_all_kinds_present(self) -> None:
        kinds = set(SymbolKind)
        assert kinds == {
            SymbolKind.FUNCTION,
            SymbolKind.CLASS,
            SymbolKind.METHOD,
            SymbolKind.VARIABLE,
            SymbolKind.IMPORT,
            SymbolKind.INTERFACE,
            SymbolKind.TYPE,
        }


# ---------------------------------------------------------------------------
# TestSymbolAndFileSymbols
# ---------------------------------------------------------------------------


class TestSymbolAndFileSymbols:
    def test_symbol_construction(self) -> None:
        s = Symbol(name="foo", kind=SymbolKind.FUNCTION, line=10)
        assert s.name == "foo"
        assert s.kind is SymbolKind.FUNCTION
        assert s.line == 10
        assert s.parent == ""

    def test_symbol_with_parent(self) -> None:
        s = Symbol(name="bar", kind=SymbolKind.METHOD, line=20, parent="Foo")
        assert s.parent == "Foo"

    def test_symbol_is_frozen(self) -> None:
        s = Symbol(name="foo", kind=SymbolKind.FUNCTION, line=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.name = "bar"  # type: ignore[misc]

    def test_symbol_equality(self) -> None:
        a = Symbol(name="foo", kind=SymbolKind.FUNCTION, line=1)
        b = Symbol(name="foo", kind=SymbolKind.FUNCTION, line=1)
        assert a == b

    def test_file_symbols_construction(self) -> None:
        syms = [Symbol(name="a", kind=SymbolKind.CLASS, line=1)]
        fs = FileSymbols(path="foo.py", language="python", symbols=syms)
        assert fs.path == "foo.py"
        assert fs.language == "python"
        assert fs.symbols is syms

    def test_file_symbols_is_frozen(self) -> None:
        fs = FileSymbols(path="foo.py", language="python", symbols=[])
        with pytest.raises(dataclasses.FrozenInstanceError):
            fs.path = "bar.py"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestRepoMapConfig
# ---------------------------------------------------------------------------


class TestRepoMapConfig:
    def test_defaults(self) -> None:
        cfg = RepoMapConfig()
        assert cfg.max_files == 200
        assert cfg.max_tokens == 2048
        assert cfg.max_chars == 10000
        assert cfg.use_tree_sitter is True

    def test_default_ignore_dirs(self) -> None:
        cfg = RepoMapConfig()
        assert ".git" in cfg.ignore_dirs
        assert "__pycache__" in cfg.ignore_dirs
        assert "node_modules" in cfg.ignore_dirs
        assert ".venv" in cfg.ignore_dirs
        assert "venv" in cfg.ignore_dirs
        assert "dist" in cfg.ignore_dirs
        assert "build" in cfg.ignore_dirs
        assert ".mypy_cache" in cfg.ignore_dirs
        assert ".ruff_cache" in cfg.ignore_dirs
        assert "target" in cfg.ignore_dirs
        assert ".tox" in cfg.ignore_dirs

    def test_default_supported_extensions(self) -> None:
        cfg = RepoMapConfig()
        exts = cfg.supported_extensions
        assert exts[".py"] == "python"
        assert exts[".js"] == "javascript"
        assert exts[".ts"] == "typescript"
        assert exts[".rs"] == "rust"
        assert exts[".go"] == "go"
        assert exts[".java"] == "java"
        assert exts[".c"] == "c"
        assert exts[".h"] == "c"
        assert exts[".cpp"] == "cpp"
        assert exts[".hpp"] == "cpp"
        assert exts[".rb"] == "ruby"

    def test_defaults_are_independent_per_instance(self) -> None:
        a = RepoMapConfig()
        b = RepoMapConfig()
        a.ignore_dirs.add("custom")
        assert "custom" not in b.ignore_dirs
        a.supported_extensions[".xyz"] = "lang"
        assert ".xyz" not in b.supported_extensions


# ---------------------------------------------------------------------------
# TestScanFiles
# ---------------------------------------------------------------------------


class TestScanFiles:
    def test_respects_ignore_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "b.py").write_text("y = 2\n", encoding="utf-8")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "c.py").write_text("z = 3\n", encoding="utf-8")

        gen = RepoMapGenerator()
        files = gen.scan_files(tmp_path)
        names = [f.name for f in files]
        assert "a.py" in names
        assert "b.py" not in names
        assert "c.py" not in names

    def test_only_supported_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "b.js").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "c.txt").write_text("not code\n", encoding="utf-8")
        (tmp_path / "d.md").write_text("# doc\n", encoding="utf-8")
        (tmp_path / "e.rs").write_text("fn main() {}\n", encoding="utf-8")

        gen = RepoMapGenerator()
        files = gen.scan_files(tmp_path)
        names = sorted(f.name for f in files)
        assert names == ["a.py", "b.js", "e.rs"]

    def test_sorted_alphabetically(self, tmp_path: Path) -> None:
        for name in ["zeta.py", "alpha.py", "mike.py"]:
            (tmp_path / name).write_text("x = 1\n", encoding="utf-8")

        gen = RepoMapGenerator()
        files = gen.scan_files(tmp_path)
        assert [f.name for f in files] == ["alpha.py", "mike.py", "zeta.py"]

    def test_nested_directories(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b").mkdir()
        (tmp_path / "a" / "b" / "deep.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "a" / "top.py").write_text("y = 2\n", encoding="utf-8")

        gen = RepoMapGenerator()
        files = gen.scan_files(tmp_path)
        names = [f.name for f in files]
        assert "deep.py" in names
        assert "top.py" in names

    def test_max_files_limit(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")

        cfg = RepoMapConfig(max_files=3)
        gen = RepoMapGenerator(cfg)
        files = gen.scan_files(tmp_path)
        assert len(files) == 3

    def test_nonexistent_root_returns_empty(self, tmp_path: Path) -> None:
        gen = RepoMapGenerator()
        assert gen.scan_files(tmp_path / "nope") == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        gen = RepoMapGenerator()
        assert gen.scan_files(tmp_path) == []


# ---------------------------------------------------------------------------
# TestExtractPython
# ---------------------------------------------------------------------------


class TestExtractPython:
    def test_extracts_class(self) -> None:
        gen = RepoMapGenerator()
        content = "class Foo:\n    pass\n"
        symbols = gen._extract_python_regex(content)
        classes = [s for s in symbols if s.kind is SymbolKind.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "Foo"
        assert classes[0].line == 1

    def test_extracts_function(self) -> None:
        gen = RepoMapGenerator()
        content = "def foo():\n    return 1\n"
        symbols = gen._extract_python_regex(content)
        funcs = [s for s in symbols if s.kind is SymbolKind.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "foo"

    def test_method_has_parent(self) -> None:
        gen = RepoMapGenerator()
        content = (
            "class Foo:\n"
            "    def method(self):\n"
            "        return 1\n"
        )
        symbols = gen._extract_python_regex(content)
        methods = [s for s in symbols if s.kind is SymbolKind.METHOD]
        assert len(methods) == 1
        assert methods[0].name == "method"
        assert methods[0].parent == "Foo"

    def test_top_level_function_resets_class_context(self) -> None:
        gen = RepoMapGenerator()
        content = (
            "class Foo:\n"
            "    def m(self):\n"
            "        pass\n"
            "\n"
            "def standalone():\n"
            "    pass\n"
        )
        symbols = gen._extract_python_regex(content)
        standalone = [s for s in symbols if s.name == "standalone"]
        assert len(standalone) == 1
        assert standalone[0].kind is SymbolKind.FUNCTION
        assert standalone[0].parent == ""

    def test_line_numbers(self) -> None:
        gen = RepoMapGenerator()
        content = "\n\nclass A:\n    def b(self):\n        pass\n"
        symbols = gen._extract_python_regex(content)
        cls = next(s for s in symbols if s.name == "A")
        assert cls.line == 3
        mth = next(s for s in symbols if s.name == "b")
        assert mth.line == 4


# ---------------------------------------------------------------------------
# TestExtractJavaScript
# ---------------------------------------------------------------------------


class TestExtractJavaScript:
    def test_extracts_function_and_class_and_const(self) -> None:
        gen = RepoMapGenerator()
        content = (
            "function foo() {}\n"
            "class Bar {}\n"
            "const baz = 42;\n"
        )
        symbols = gen._extract_javascript_regex(content)
        names_kinds = {(s.name, s.kind) for s in symbols}
        assert ("foo", SymbolKind.FUNCTION) in names_kinds
        assert ("Bar", SymbolKind.CLASS) in names_kinds
        assert ("baz", SymbolKind.VARIABLE) in names_kinds

    def test_line_numbers(self) -> None:
        gen = RepoMapGenerator()
        content = "\nfunction foo() {}\n"
        symbols = gen._extract_javascript_regex(content)
        assert symbols[0].line == 2


# ---------------------------------------------------------------------------
# TestExtractTypeScript
# ---------------------------------------------------------------------------


class TestExtractTypeScript:
    def test_extracts_interface_and_type(self) -> None:
        gen = RepoMapGenerator()
        content = (
            "function foo() {}\n"
            "interface Bar {}\n"
            "type Baz = string;\n"
        )
        symbols = gen._extract_typescript_regex(content)
        names_kinds = {(s.name, s.kind) for s in symbols}
        assert ("foo", SymbolKind.FUNCTION) in names_kinds
        assert ("Bar", SymbolKind.INTERFACE) in names_kinds
        assert ("Baz", SymbolKind.TYPE) in names_kinds


# ---------------------------------------------------------------------------
# TestExtractRust
# ---------------------------------------------------------------------------


class TestExtractRust:
    def test_extracts_fn_struct_enum_trait(self) -> None:
        gen = RepoMapGenerator()
        content = (
            "fn main() {}\n"
            "struct Foo {}\n"
            "enum Color {}\n"
            "trait Bar {}\n"
        )
        symbols = gen._extract_rust_regex(content)
        names_kinds = {(s.name, s.kind) for s in symbols}
        assert ("main", SymbolKind.FUNCTION) in names_kinds
        assert ("Foo", SymbolKind.CLASS) in names_kinds
        assert ("Color", SymbolKind.CLASS) in names_kinds
        assert ("Bar", SymbolKind.INTERFACE) in names_kinds


# ---------------------------------------------------------------------------
# TestExtractGo
# ---------------------------------------------------------------------------


class TestExtractGo:
    def test_extracts_func_and_struct(self) -> None:
        gen = RepoMapGenerator()
        content = (
            "func foo() {}\n"
            "func (s *Server) Run() {}\n"
            "type Server struct {}\n"
        )
        symbols = gen._extract_go_regex(content)
        names_kinds = {(s.name, s.kind) for s in symbols}
        assert ("foo", SymbolKind.FUNCTION) in names_kinds
        assert ("Run", SymbolKind.FUNCTION) in names_kinds
        assert ("Server", SymbolKind.CLASS) in names_kinds


# ---------------------------------------------------------------------------
# TestExtractSymbolsDispatch
# ---------------------------------------------------------------------------


class TestExtractSymbolsDispatch:
    def test_python_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("class Foo:\n    def bar(self):\n        pass\n", encoding="utf-8")
        gen = RepoMapGenerator()
        fs = gen.extract_symbols(f)
        assert fs.language == "python"
        assert len(fs.symbols) == 2

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hello\n", encoding="utf-8")
        gen = RepoMapGenerator()
        fs = gen.extract_symbols(f)
        assert fs.language == "unknown"
        assert fs.symbols == []

    def test_read_error_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        # Don't create the file — extract_symbols should handle the OSError.
        gen = RepoMapGenerator()
        fs = gen.extract_symbols(f)
        assert fs.language == "python"
        assert fs.symbols == []


# ---------------------------------------------------------------------------
# TestFormatTree
# ---------------------------------------------------------------------------


class TestFormatTree:
    def test_empty_files(self) -> None:
        gen = RepoMapGenerator()
        assert gen._format_tree([]) == ""

    def test_basic_tree_structure(self) -> None:
        gen = RepoMapGenerator()
        files = [
            FileSymbols(
                path="src/agent/runtime.py",
                language="python",
                symbols=[
                    Symbol(name="AgentRuntime", kind=SymbolKind.CLASS, line=1, parent=""),
                    Symbol(name="run", kind=SymbolKind.METHOD, line=2, parent="AgentRuntime"),
                    Symbol(name="_call_llm", kind=SymbolKind.METHOD, line=3, parent="AgentRuntime"),
                ],
            ),
        ]
        result = gen._format_tree(files)
        lines = result.split("\n")
        assert "src/" in lines
        assert "  autoship/" not in lines  # only src/ in this case
        assert "  agent/" in lines
        assert "    runtime.py" in lines
        assert "      class AgentRuntime" in lines
        assert "        def run" in lines
        assert "        def _call_llm" in lines

    def test_nested_directory_tree(self) -> None:
        gen = RepoMapGenerator()
        files = [
            FileSymbols(path="a/b.py", language="python", symbols=[]),
            FileSymbols(path="a/c.py", language="python", symbols=[]),
        ]
        result = gen._format_tree(files)
        lines = result.split("\n")
        assert lines[0] == "a/"
        assert "  b.py" in lines
        assert "  c.py" in lines

    def test_symbols_indentation(self) -> None:
        gen = RepoMapGenerator()
        files = [
            FileSymbols(
                path="m.py",
                language="python",
                symbols=[
                    Symbol(name="Cls", kind=SymbolKind.CLASS, line=1, parent=""),
                    Symbol(name="meth", kind=SymbolKind.METHOD, line=2, parent="Cls"),
                ],
            ),
        ]
        result = gen._format_tree(files)
        lines = result.split("\n")
        assert lines[0] == "m.py"
        assert lines[1] == "  class Cls"
        assert lines[2] == "    def meth"

    def test_directories_sorted(self) -> None:
        gen = RepoMapGenerator()
        files = [
            FileSymbols(path="z/file.py", language="python", symbols=[]),
            FileSymbols(path="a/file.py", language="python", symbols=[]),
        ]
        result = gen._format_tree(files)
        lines = result.split("\n")
        assert lines[0] == "a/"
        assert "z/" in lines[2]


# ---------------------------------------------------------------------------
# TestGenerate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_end_to_end(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "runtime.py").write_text(
            "class AgentRuntime:\n"
            "    def run(self):\n"
            "        pass\n"
            "    def _call_llm(self):\n"
            "        pass\n",
            encoding="utf-8",
        )
        gen = RepoMapGenerator()
        result = gen.generate(tmp_path)
        assert "src/" in result
        assert "runtime.py" in result
        assert "class AgentRuntime" in result
        assert "def run" in result
        assert "def _call_llm" in result

    def test_empty_dir(self, tmp_path: Path) -> None:
        gen = RepoMapGenerator()
        assert gen.generate(tmp_path) == ""

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        gen = RepoMapGenerator()
        assert gen.generate(tmp_path / "nope") == ""

    def test_truncation(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("class A:\n    pass\n", encoding="utf-8")
        cfg = RepoMapConfig(max_chars=5)
        gen = RepoMapGenerator(cfg)
        result = gen.generate(tmp_path)
        assert len(result) <= 5

    def test_respects_ignore_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "b.py").write_text("y = 2\n", encoding="utf-8")
        gen = RepoMapGenerator()
        result = gen.generate(tmp_path)
        assert "a.py" in result
        assert "b.py" not in result
        assert "node_modules" not in result


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_estimate_tokens(self) -> None:
        assert RepoMapGenerator._estimate_tokens("") == 0
        assert RepoMapGenerator._estimate_tokens("abcd") == 1
        assert RepoMapGenerator._estimate_tokens("abcdefgh") == 2

    def test_has_tree_sitter_returns_bool(self) -> None:
        result = RepoMapGenerator._has_tree_sitter()
        assert isinstance(result, bool)

    def test_line_of(self) -> None:
        from autoship.context.repo_map import _line_of

        content = "a\nb\nc"
        assert _line_of(content, 0) == 1
        assert _line_of(content, 2) == 2
        assert _line_of(content, 4) == 3
