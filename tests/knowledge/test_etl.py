"""Tests for the knowledge ETL pipeline (sources + orchestration)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from justagent.knowledge.etl import (
    APISource,
    DatabaseSource,
    ETLPipeline,
    FilesystemSource,
)


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# Hello\n\nWorld content.", encoding="utf-8")
    (d / "b.txt").write_text("Plain text notes.", encoding="utf-8")
    sub = d / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("Nested doc.", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# FilesystemSource
# ---------------------------------------------------------------------------


class TestFilesystemSource:
    def test_extracts_supported_files(self, docs_dir: Path) -> None:
        src = FilesystemSource("fs1", docs_dir)
        items = list(src.extract())
        # id is the resolved absolute path; match on file_name metadata instead
        names = {i.metadata["file_name"] for i in items}
        assert names == {"a.md", "b.txt", "c.md"}
        assert all(i.source_id == "fs1" for i in items)
        assert all(Path(i.id).is_absolute() for i in items)

    def test_pattern_filter(self, docs_dir: Path) -> None:
        src = FilesystemSource("fs2", docs_dir, pattern="*.md")
        items = list(src.extract())
        assert {i.metadata["file_name"] for i in items} == {"a.md"}

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            FilesystemSource("x", tmp_path / "nope")

    def test_transform_produces_document(self, docs_dir: Path) -> None:
        src = FilesystemSource("fs3", docs_dir)
        item = next(iter(src.extract()))
        doc = src.transform(item)
        assert doc is not None
        assert doc.content  # text parsed into a Document

    def test_incremental_since_skips_old_files(self, docs_dir: Path) -> None:
        src = FilesystemSource("fs4", docs_dir)
        future = 4102444800.0  # 2100-01-01
        assert list(src.extract(since=future)) == []


# ---------------------------------------------------------------------------
# DatabaseSource
# ---------------------------------------------------------------------------


class TestDatabaseSource:
    def _seed_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "data.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT, updated_at REAL)"
        )
        conn.execute(
            "INSERT INTO notes (id, body, updated_at) VALUES (1, 'first row', 100.0)"
        )
        conn.commit()
        conn.close()
        return db

    def test_extract_rows_to_raw_items(self, tmp_path: Path) -> None:
        db = self._seed_db(tmp_path)
        src = DatabaseSource(
            "db1", str(db), "SELECT id, body FROM notes",
            content_columns=["body"],
        )
        items = list(src.extract())
        assert len(items) == 1
        assert items[0].id == "1"
        assert "first row" in (items[0].content or "")

    def test_custom_driver_is_used(self, tmp_path: Path) -> None:
        db = self._seed_db(tmp_path)
        calls: list[str] = []

        def driver(conn_str: str) -> sqlite3.Connection:
            calls.append(conn_str)
            return sqlite3.connect(conn_str)

        src = DatabaseSource(
            "db2", str(db), "SELECT id, body FROM notes",
            content_columns=["body"], driver=driver,
        )
        assert len(list(src.extract())) == 1
        assert calls == [str(db)]


# ---------------------------------------------------------------------------
# APISource
# ---------------------------------------------------------------------------


def _mock_urlopen(payload):
    """Return a context-manager factory mimicking urlopen's response."""
    import io

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return lambda request, timeout=None: _Resp(
        __import__("json").dumps(payload).encode("utf-8")
    )


class TestAPISource:
    def test_extracts_json_items(self, monkeypatch) -> None:
        from justagent.knowledge import etl as etl_mod

        payload = {"items": [{"id": "a", "title": "Alpha"}, {"id": "b", "title": "Beta"}]}
        monkeypatch.setattr(etl_mod, "urlopen", _mock_urlopen(payload))
        src = APISource(
            "api1", "https://api.example.com/items",
            items_path="items", content_fields=["title"],
        )
        items = list(src.extract())
        assert [i.id for i in items] == ["a", "b"]
        docs = [src.transform(i) for i in items]
        assert all(d is not None for d in docs)

    def test_bad_item_is_skipped_not_fatal(self, monkeypatch) -> None:
        from justagent.knowledge import etl as etl_mod

        payload = {"items": [{"id": "", "title": ""}, {"id": "ok", "title": "Fine"}]}
        # empty id falls back to a uuid (never skipped); both rows survive
        monkeypatch.setattr(etl_mod, "urlopen", _mock_urlopen(payload))
        src = APISource(
            "api2", "https://api.example.com/items",
            items_path="items", content_fields=["title"],
        )
        ids = [i.id for i in src.extract()]
        assert len(ids) == 2 and any(i == "ok" for i in ids)


# ---------------------------------------------------------------------------
# ETLPipeline
# ---------------------------------------------------------------------------


class _FakeSource(FilesystemSource):
    """Filesystem source with controllable extraction failures."""

    fail = False

    def extract(self, *, since=None):
        if self.fail:
            raise RuntimeError("boom")
        yield from super().extract()


class TestETLPipeline:
    def test_register_contains_len(self, docs_dir: Path) -> None:
        p = ETLPipeline()
        s = FilesystemSource("s1", docs_dir)
        p.register_source(s)
        assert "s1" in p and len(p) == 1
        assert p.get_source("s1") is s
        assert p.list_source_ids() == ["s1"]

    def test_unregister_and_missing_lookup(self, docs_dir: Path) -> None:
        p = ETLPipeline()
        p.register_source(FilesystemSource("s1", docs_dir))
        assert p.unregister_source("s1") is not None
        assert p.get_source("s1") is None
        with pytest.raises(KeyError):
            p.sync("s1")

    def test_sync_success_updates_state(self, docs_dir: Path) -> None:
        p = ETLPipeline()
        p.register_source(FilesystemSource("s1", docs_dir))
        result = p.sync("s1")
        assert result.error == ""
        assert result.item_count >= 1
        assert len(result.documents) >= 1
        state = p.get_sync_state("s1")
        assert state is not None and state.sync_count == 1
        assert state.last_sync is not None

    def test_sync_failure_records_error(self, tmp_path: Path) -> None:
        d = tmp_path / "d"
        d.mkdir()
        (d / "x.md").write_text("hi", encoding="utf-8")
        src = _FakeSource("bad", d)
        p = ETLPipeline()
        p.register_source(src)
        src.fail = True
        result = p.sync("bad")
        assert "boom" in result.error
        assert result.documents == []
        state = p.get_sync_state("bad")
        assert state is not None and "boom" in state.last_error
        # failed sync does not advance last_sync → next run retries everything
        assert state.last_sync is None

    def test_skipped_count_for_untransformable_items(self, docs_dir: Path) -> None:
        p = ETLPipeline()
        src = FilesystemSource("s1", docs_dir)
        p.register_source(src)
        # Force transform to fail → base extract_and_transform yields doc=None.
        src.transform = lambda item: (_ for _ in ()).throw(ValueError("nope"))
        result = p.sync("s1")
        assert result.error == ""
        assert result.skipped == result.item_count >= 1
        assert result.documents == []

    def test_sync_all_covers_every_source(self, tmp_path: Path) -> None:
        d1 = tmp_path / "d1"
        d1.mkdir()
        (d1 / "a.md").write_text("x", encoding="utf-8")
        d2 = tmp_path / "d2"
        d2.mkdir()
        (d2 / "b.md").write_text("y", encoding="utf-8")
        p = ETLPipeline()
        p.register_source(FilesystemSource("one", d1))
        p.register_source(FilesystemSource("two", d2))
        results = p.sync_all()
        assert set(results) == {"one", "two"}
        assert all(r.error == "" for r in results.values())

    def test_state_persistence_roundtrip(self, docs_dir: Path, tmp_path: Path) -> None:
        path = tmp_path / "states.json"
        p1 = ETLPipeline()
        p1.register_source(FilesystemSource("s1", docs_dir))
        p1.sync("s1")
        p1.save_states(path)

        p2 = ETLPipeline()
        p2.register_source(FilesystemSource("s1", docs_dir))
        loaded = p2.load_states(path)
        assert loaded == 1
        state = p2.get_sync_state("s1")
        assert state is not None and state.sync_count == 1

    def test_reset_states(self, docs_dir: Path) -> None:
        p = ETLPipeline()
        p.register_source(FilesystemSource("s1", docs_dir))
        p.sync("s1")
        p.reset_sync_state("s1")
        st = p.get_sync_state("s1")
        assert st is not None and st.last_sync is None and st.sync_count == 0
