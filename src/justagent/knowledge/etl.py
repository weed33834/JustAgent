"""ETL pipeline for multi-source data ingestion.

Provides a pluggable extract-transform-load framework that ingests data
from multiple source types (database, filesystem, HTTP API) into the
knowledge base as :class:`~justagent.knowledge.document.Document` objects.

Key features:

* **Incremental sync** — each source tracks a ``last_sync`` timestamp.
  Subsequent syncs only extract items modified after that timestamp.
  Pass ``force=True`` to re-sync everything.
* **Pluggable sources** — :class:`ETLSource` is the abstract base.
  Built-in implementations cover the three required source types.
* **Standard-library-first** — filesystem and API sources use only the
  standard library. The database source uses ``sqlite3`` (stdlib) by
  default and supports external drivers (psycopg2, pymysql, etc.) via
  lazy import.
* **Resilient** — extraction errors are caught per-item so one bad row
  or file does not abort the entire sync.

Design:

* :class:`SourceType` — enum of supported source types.
* :class:`SyncState` — tracks incremental sync metadata per source.
* :class:`ETLSource` — abstract base: ``extract() -> raw items``,
  ``transform() -> Document``.
* :class:`FilesystemSource` — scans a directory for files matching a
  glob pattern.
* :class:`DatabaseSource` — executes a SQL query and transforms rows.
* :class:`APISource` — fetches JSON from an HTTP endpoint.
* :class:`ETLPipeline` — orchestrates registration, sync, and state
  tracking.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from justagent.knowledge.document import (
    Document,
    DocumentParser,
)

logger = logging.getLogger("justagent.knowledge")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):  # noqa: UP042
    """Supported ETL source types."""

    FILESYSTEM = "filesystem"
    DATABASE = "database"
    API = "api"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SyncState(BaseModel):
    """Incremental sync state for a single ETL source.

    Attributes:
        source_id: ID of the source this state belongs to.
        last_sync: Unix timestamp of the last successful sync, or None
            if the source has never been synced.
        last_item_count: Number of items processed in the last sync.
        total_items_processed: Cumulative count of items processed.
        last_error: Error message from the last sync attempt, or empty
            if the last sync succeeded.
        sync_count: Number of syncs performed.
    """

    source_id: str
    last_sync: float | None = None
    last_item_count: int = 0
    total_items_processed: int = 0
    last_error: str = ""
    sync_count: int = 0


class SyncResult(BaseModel):
    """Result of a single source sync operation.

    Attributes:
        source_id: ID of the source that was synced.
        documents: List of :class:`Document` objects produced.
        item_count: Number of raw items extracted.
        error: Error message if the sync failed, or empty on success.
        duration_ms: Sync duration in milliseconds.
        skipped: Number of items skipped (e.g. unchanged in incremental mode).
    """

    source_id: str
    documents: list[Document] = Field(default_factory=list)
    item_count: int = 0
    error: str = ""
    duration_ms: float = 0.0
    skipped: int = 0


# ---------------------------------------------------------------------------
# Raw item (intermediate representation)
# ---------------------------------------------------------------------------


class RawItem(BaseModel):
    """A raw item extracted from a source before transformation.

    Attributes:
        source_id: ID of the source this item came from.
        id: Unique identifier for the item within the source (e.g. file
            path, row ID, API object ID).
        content: Text content (for text sources) or None (for binary
            sources that need parsing).
        raw_bytes: Raw bytes (for binary sources like PDF files).
        metadata: Source-specific metadata (e.g. modification time, row
            data, HTTP headers).
        modified_at: Modification timestamp of the item (for incremental
            sync). If None, the item is always considered new.
    """

    source_id: str = ""
    id: str = ""
    content: str | None = None
    raw_bytes: bytes | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    modified_at: float | None = None


# ---------------------------------------------------------------------------
# Abstract ETL source
# ---------------------------------------------------------------------------


class ETLSource(ABC):
    """Abstract base class for an ETL data source.

    Subclasses implement :meth:`extract` (pull raw items from the
    source) and optionally override :meth:`transform` (convert a raw
    item into a :class:`Document`).

    The default :meth:`transform` uses :class:`DocumentParser` to parse
    text or bytes into a document. Override it for custom transformations.
    """

    def __init__(
        self,
        source_id: str,
        source_type: SourceType,
        *,
        parser: DocumentParser | None = None,
    ) -> None:
        self._source_id = source_id
        self._source_type = source_type
        self._parser = parser or DocumentParser()

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def parser(self) -> DocumentParser:
        return self._parser

    @abstractmethod
    def extract(
        self, *, since: float | None = None
    ) -> Iterator[RawItem]:
        """Extract raw items from the source.

        Args:
            since: If provided, only extract items modified after this
                Unix timestamp (incremental sync). If None, extract all
                items.

        Yields:
            :class:`RawItem` objects.
        """

    def transform(self, item: RawItem) -> Document | None:
        """Transform a raw item into a :class:`Document`.

        Returns None if the item should be skipped (e.g. empty content).

        The default implementation:

        * If ``item.raw_bytes`` is set, uses
          :meth:`DocumentParser.parse_bytes`.
        * If ``item.content`` is set, uses
          :meth:`DocumentParser.parse_text`.
        * Otherwise, returns None.
        """
        if item.raw_bytes is not None:
            filename = item.metadata.get("file_name", item.id)
            return self._parser.parse_bytes(
                item.raw_bytes,
                filename=filename,
                title=item.metadata.get("title"),
                metadata={
                    k: v
                    for k, v in item.metadata.items()
                    if k not in ("file_name", "title")
                },
            )
        if item.content is not None and item.content.strip():
            return self._parser.parse_text(
                item.content,
                title=item.metadata.get("title", item.id),
                source=item.id,
                metadata=item.metadata,
            )
        return None

    def extract_and_transform(
        self, *, since: float | None = None
    ) -> Iterator[tuple[RawItem, Document | None]]:
        """Extract and transform in one pass.

        Yields ``(raw_item, document_or_none)`` tuples. Items that fail
        transformation yield ``(raw_item, None)`` and are logged.
        """
        for item in self.extract(since=since):
            try:
                doc = self.transform(item)
            except Exception as exc:
                logger.warning(
                    "Transform failed for item %s from source %s: %s",
                    item.id,
                    self._source_id,
                    exc,
                )
                doc = None
            yield item, doc


# ---------------------------------------------------------------------------
# Filesystem source
# ---------------------------------------------------------------------------


class FilesystemSource(ETLSource):
    """ETL source that scans a filesystem directory for files.

    Recursively scans ``directory`` for files matching ``pattern``
    (a glob, default ``**/*`` for all files). Each file is extracted as
    a :class:`RawItem` with its content read as bytes. Incremental sync
    uses the file's modification time (``st_mtime``).

    Args:
        source_id: Unique source identifier.
        directory: Root directory to scan.
        pattern: Glob pattern for file matching (e.g. ``**/*.md``).
        parser: Optional document parser.
        exclude_dirs: Directory names to exclude (e.g. ``{".git", "__pycache__"}``).
        max_file_size: Maximum file size in bytes to include (default 50 MB).
    """

    def __init__(
        self,
        source_id: str,
        directory: Path | str,
        *,
        pattern: str = "**/*",
        parser: DocumentParser | None = None,
        exclude_dirs: set[str] | None = None,
        max_file_size: int = 50 * 1024 * 1024,
    ) -> None:
        super().__init__(source_id, SourceType.FILESYSTEM, parser=parser)
        self._directory = Path(directory)
        if not self._directory.exists():
            raise FileNotFoundError(f"Directory not found: {self._directory}")
        if not self._directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {self._directory}")
        self._pattern = pattern
        self._exclude_dirs = exclude_dirs or {".git", "__pycache__", ".venv", "node_modules"}
        self._max_file_size = max_file_size

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def pattern(self) -> str:
        return self._pattern

    def extract(
        self, *, since: float | None = None
    ) -> Iterator[RawItem]:
        """Extract files from the directory.

        Yields one :class:`RawItem` per file. If ``since`` is provided,
        only files modified after that timestamp are yielded.
        """
        for file_path in sorted(self._directory.glob(self._pattern)):
            if not file_path.is_file():
                continue
            # Skip excluded directories.
            if any(
                part in self._exclude_dirs
                for part in file_path.relative_to(self._directory).parts
            ):
                continue
            try:
                stat = file_path.stat()
            except OSError as exc:
                logger.warning("Cannot stat %s: %s", file_path, exc)
                continue

            # Size check.
            if stat.st_size > self._max_file_size:
                logger.debug(
                    "Skipping large file %s (%d bytes)", file_path, stat.st_size
                )
                continue

            # Incremental: skip files not modified since last sync.
            if since is not None and stat.st_mtime <= since:
                continue

            try:
                raw_bytes = file_path.read_bytes()
            except OSError as exc:
                logger.warning("Cannot read %s: %s", file_path, exc)
                continue

            yield RawItem(
                source_id=self._source_id,
                id=str(file_path.resolve()),
                raw_bytes=raw_bytes,
                modified_at=stat.st_mtime,
                metadata={
                    "file_name": file_path.name,
                    "file_path": str(file_path.resolve()),
                    "file_size": stat.st_size,
                    "title": file_path.stem,
                    "source_type": "filesystem",
                },
            )


# ---------------------------------------------------------------------------
# Database source
# ---------------------------------------------------------------------------


class DatabaseSource(ETLSource):
    """ETL source that extracts data from a SQL database.

    Uses ``sqlite3`` (standard library) by default. For other databases
    (PostgreSQL, MySQL, etc.), pass ``driver`` with a DB-API 2.0
    compatible connection factory (e.g. ``psycopg2.connect``). The
    driver is imported lazily to avoid hard dependencies.

    Each row is transformed into a :class:`Document` where the content
    is built by joining the text columns. The ``content_columns``
    parameter specifies which columns to include in the content; if
    None, all columns are used.

    Args:
        source_id: Unique source identifier.
        connection_string: Database connection string (e.g. path to
            SQLite file, or DSN for other databases).
        query: SQL query to execute for extraction.
        driver: Optional connection factory (e.g.
            ``psycopg2.connect``). If None, ``sqlite3.connect`` is used.
        content_columns: Column names to include in document content.
            If None, all columns are used.
        id_column: Column name to use as the item ID (default ``"id"``).
        modified_column: Column name containing the modification
            timestamp for incremental sync (default ``"updated_at"``).
        parser: Optional document parser.
    """

    def __init__(
        self,
        source_id: str,
        connection_string: str,
        query: str,
        *,
        driver: Any = None,
        content_columns: list[str] | None = None,
        id_column: str = "id",
        modified_column: str | None = "updated_at",
        parser: DocumentParser | None = None,
    ) -> None:
        super().__init__(source_id, SourceType.DATABASE, parser=parser)
        self._connection_string = connection_string
        self._query = query
        self._driver = driver
        self._content_columns = content_columns
        self._id_column = id_column
        self._modified_column = modified_column

    def _get_connection(self) -> Any:
        """Open a database connection.

        Uses the provided driver, or falls back to sqlite3.
        """
        if self._driver is not None:
            return self._driver(self._connection_string)
        import sqlite3

        return sqlite3.connect(self._connection_string)

    def extract(
        self, *, since: float | None = None
    ) -> Iterator[RawItem]:
        """Execute the SQL query and yield rows as :class:`RawItem` objects.

        If ``since`` is provided and ``modified_column`` is set, a
        WHERE clause is appended to the query to filter by modification
        time. The column value is expected to be a Unix timestamp
        (float or integer).
        """
        query = self._query
        params: list[Any] = []

        if since is not None and self._modified_column:
            # Append a WHERE clause for incremental sync.
            # Naive: assumes the query doesn't already have a WHERE clause
            # or that the caller handles filtering. For safety, we check
            # if the query already contains WHERE (case-insensitive).
            if "where" not in query.lower():
                query += f" WHERE {self._modified_column} > ?"
                params.append(since)
            else:
                query += f" AND {self._modified_column} > ?"
                params.append(since)

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = (
                [desc[0] for desc in cursor.description]
                if cursor.description
                else []
            )
            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row, strict=False))
                item_id = str(row_dict.get(self._id_column, uuid.uuid4().hex))

                # Build content from specified or all columns.
                cols = self._content_columns or columns
                content_parts: list[str] = []
                for col in cols:
                    val = row_dict.get(col)
                    if val is not None:
                        content_parts.append(f"{col}: {val}")
                content = "\n".join(content_parts)

                # Extract modification time.
                modified_at = None
                if self._modified_column and self._modified_column in row_dict:
                    raw_ts = row_dict[self._modified_column]
                    if isinstance(raw_ts, (int, float)):
                        modified_at = float(raw_ts)
                    elif isinstance(raw_ts, str):
                        with contextlib.suppress(ValueError):
                            modified_at = float(raw_ts)

                yield RawItem(
                    source_id=self._source_id,
                    id=item_id,
                    content=content,
                    modified_at=modified_at,
                    metadata={
                        **row_dict,
                        "source_type": "database",
                        "query": self._query,
                    },
                )
        except Exception as exc:
            logger.error(
                "Database extraction failed for source %s: %s",
                self._source_id,
                exc,
            )
            raise
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    logger.debug("Failed to close connection", exc_info=True)
                    pass


# ---------------------------------------------------------------------------
# API source
# ---------------------------------------------------------------------------


class APISource(ETLSource):
    """ETL source that fetches data from an HTTP API.

    Uses ``urllib`` (standard library) for HTTP requests. The response
    is expected to be JSON. Each item in the response (identified by
    ``items_path``) is transformed into a :class:`Document`.

    For APIs requiring authentication, set ``headers`` (e.g.
    ``{"Authorization": "Bearer ..."}``). For paginated APIs,
    ``page_param`` and ``page_size_param`` can be set; the source will
    follow ``next_page`` URLs or increment the page parameter until no
    more items are returned.

    Args:
        source_id: Unique source identifier.
        url: API endpoint URL.
        headers: HTTP headers to send with each request.
        params: Query parameters for the initial request.
        items_path: Dot-separated path to the items array in the JSON
            response (e.g. ``"data.results"``). If None, the response
            itself is treated as the items array.
        id_field: Field name in each item to use as the item ID
            (default ``"id"``).
        content_fields: Fields to include in the document content. If
            None, all fields are used.
        modified_field: Field name containing the modification timestamp
            for incremental sync.
        page_param: Query parameter name for pagination (e.g. ``"page"``).
        max_pages: Maximum number of pages to fetch (default 100).
        timeout: Request timeout in seconds.
        parser: Optional document parser.
    """

    def __init__(
        self,
        source_id: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        items_path: str | None = None,
        id_field: str = "id",
        content_fields: list[str] | None = None,
        modified_field: str | None = "updated_at",
        page_param: str | None = None,
        max_pages: int = 100,
        timeout: float = 30.0,
        parser: DocumentParser | None = None,
    ) -> None:
        super().__init__(source_id, SourceType.API, parser=parser)
        self._url = url
        self._headers = headers or {}
        self._params = dict(params) if params else {}
        self._items_path = items_path
        self._id_field = id_field
        self._content_fields = content_fields
        self._modified_field = modified_field
        self._page_param = page_param
        self._max_pages = max_pages
        self._timeout = timeout

    @property
    def url(self) -> str:
        return self._url

    def extract(
        self, *, since: float | None = None
    ) -> Iterator[RawItem]:
        """Fetch items from the API endpoint.

        If ``since`` is provided and ``modified_field`` is set, a query
        parameter ``modified_after`` is added with the timestamp value.
        """
        page = 1
        total_yielded = 0

        while page <= self._max_pages:
            params = dict(self._params)
            if self._page_param is not None:
                params[self._page_param] = str(page)
            if since is not None and self._modified_field:
                params["modified_after"] = str(since)

            try:
                data = self._fetch(params)
            except Exception as exc:
                logger.error(
                    "API fetch failed for source %s (page %d): %s",
                    self._source_id,
                    page,
                    exc,
                )
                if page == 1:
                    raise
                break

            items = self._extract_items(data)
            if not items:
                break

            for item in items:
                raw_item = self._to_raw_item(item)
                if raw_item is not None:
                    yield raw_item
                    total_yielded += 1

            # Check for next page URL in the response.
            next_url = self._get_next_url(data)
            if next_url:
                self._url = next_url  # Follow pagination URL.
            elif self._page_param is not None:
                page += 1
            else:
                break

        logger.info(
            "API source %s: extracted %d items", self._source_id, total_yielded
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch(self, params: dict[str, str]) -> Any:
        """Make an HTTP GET request and return the parsed JSON response."""
        url = self._url
        if params:
            url = f"{url}?{urlencode(params)}"

        request = Request(url, headers=self._headers, method="GET")
        with urlopen(request, timeout=self._timeout) as response:
            body = response.read()
        return json.loads(body.decode("utf-8"))

    def _extract_items(self, data: Any) -> list[dict[str, Any]]:
        """Extract the items array from the JSON response."""
        if self._items_path is None:
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            return []
        # Navigate the dot-separated path.
        current: Any = data
        for key in self._items_path.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return []
        if isinstance(current, list):
            return [item for item in current if isinstance(item, dict)]
        return []

    def _to_raw_item(self, item: dict[str, Any]) -> RawItem | None:
        """Convert an API item dict to a :class:`RawItem`."""
        item_id = str(item.get(self._id_field, uuid.uuid4().hex))

        # Build content from specified or all fields.
        fields = self._content_fields or list(item.keys())
        content_parts: list[str] = []
        for field_name in fields:
            val = item.get(field_name)
            if val is not None:
                if isinstance(val, (dict, list)):
                    content_parts.append(f"{field_name}: {json.dumps(val)}")
                else:
                    content_parts.append(f"{field_name}: {val}")
        content = "\n".join(content_parts)

        # Extract modification time.
        modified_at = None
        if self._modified_field and self._modified_field in item:
            raw_ts = item[self._modified_field]
            if isinstance(raw_ts, (int, float)):
                modified_at = float(raw_ts)
            elif isinstance(raw_ts, str):
                with contextlib.suppress(ValueError):
                    modified_at = float(raw_ts)

        return RawItem(
            source_id=self._source_id,
            id=item_id,
            content=content,
            modified_at=modified_at,
            metadata={
                **item,
                "source_type": "api",
                "url": self._url,
            },
        )

    @staticmethod
    def _get_next_url(data: Any) -> str | None:
        """Check if the response contains a ``next`` pagination URL."""
        if isinstance(data, dict):
            return data.get("next") or data.get("next_url")
        return None


# ---------------------------------------------------------------------------
# ETL pipeline
# ---------------------------------------------------------------------------


class ETLPipeline:
    """Orchestrates multi-source ETL ingestion with incremental sync.

    Manages a collection of :class:`ETLSource` instances, tracks sync
    state per source, and provides methods for syncing individual
    sources or all sources at once.

    Example::

        >>> pipeline = ETLPipeline()
        >>> pipeline.register_source(FilesystemSource("docs", "./docs"))
        >>> result = pipeline.sync("docs")
        >>> result.item_count > 0
        True
        >>> # Subsequent sync only picks up new/modified files
        >>> result2 = pipeline.sync("docs")
        >>> result2.item_count <= result.item_count
        True
    """

    def __init__(self) -> None:
        self._sources: dict[str, ETLSource] = {}
        self._states: dict[str, SyncState] = {}

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def register_source(self, source: ETLSource) -> None:
        """Register an ETL source.

        Raises:
            ValueError: If a source with the same ID is already registered.
        """
        if source.source_id in self._sources:
            raise ValueError(f"Source already registered: {source.source_id}")
        self._sources[source.source_id] = source
        self._states[source.source_id] = SyncState(source_id=source.source_id)
        logger.info(
            "Registered ETL source %s (%s)",
            source.source_id,
            source.source_type.value,
        )

    def unregister_source(self, source_id: str) -> ETLSource | None:
        """Unregister a source. Returns the removed source, or None."""
        source = self._sources.pop(source_id, None)
        self._states.pop(source_id, None)
        return source

    def get_source(self, source_id: str) -> ETLSource | None:
        """Return a source by ID, or None."""
        return self._sources.get(source_id)

    def list_sources(self) -> list[ETLSource]:
        """Return all registered sources."""
        return list(self._sources.values())

    def list_source_ids(self) -> list[str]:
        """Return all registered source IDs."""
        return list(self._sources.keys())

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._sources

    # ------------------------------------------------------------------
    # Sync state
    # ------------------------------------------------------------------

    def get_sync_state(self, source_id: str) -> SyncState | None:
        """Return the sync state for a source, or None if not registered."""
        return self._states.get(source_id)

    def get_all_sync_states(self) -> dict[str, SyncState]:
        """Return sync states for all sources."""
        return dict(self._states)

    def reset_sync_state(self, source_id: str) -> None:
        """Reset the sync state for a source (forces full re-sync)."""
        if source_id in self._states:
            self._states[source_id] = SyncState(source_id=source_id)
            logger.info("Reset sync state for source %s", source_id)

    def reset_all_sync_states(self) -> None:
        """Reset sync states for all sources."""
        for source_id in self._sources:
            self._states[source_id] = SyncState(source_id=source_id)
        logger.info("Reset sync states for all sources")

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(
        self,
        source_id: str,
        *,
        force: bool = False,
    ) -> SyncResult:
        """Sync a single source.

        Performs an incremental sync (only extracting items modified
        since the last sync) unless ``force=True``.

        Args:
            source_id: ID of the source to sync.
            force: If True, perform a full sync (ignore last_sync
                timestamp).

        Returns:
            :class:`SyncResult` with the extracted documents and stats.

        Raises:
            KeyError: If the source is not registered.
        """
        source = self._sources.get(source_id)
        if source is None:
            raise KeyError(f"Source not found: {source_id}")

        state = self._states[source_id]
        since = None if force else state.last_sync
        start = time.time()

        documents: list[Document] = []
        item_count = 0
        skipped = 0
        error_msg = ""

        try:
            for _item, doc in source.extract_and_transform(since=since):
                item_count += 1
                if doc is not None:
                    documents.append(doc)
                else:
                    skipped += 1
        except Exception as exc:
            error_msg = str(exc)
            logger.error(
                "Sync failed for source %s: %s", source_id, exc, exc_info=True
            )

        duration_ms = (time.time() - start) * 1000

        # Update sync state. We use the sync *start* timestamp (not the
        # end) so that items modified during extraction are not missed
        # in the next incremental sync.
        if not error_msg:
            state.last_sync = start
            state.last_item_count = item_count
            state.total_items_processed += item_count
            state.last_error = ""
            state.sync_count += 1
        else:
            state.last_error = error_msg

        result = SyncResult(
            source_id=source_id,
            documents=documents,
            item_count=item_count,
            error=error_msg,
            duration_ms=duration_ms,
            skipped=skipped,
        )
        logger.info(
            "Sync %s: %d items, %d docs, %d skipped, %.1fms%s",
            source_id,
            item_count,
            len(documents),
            skipped,
            duration_ms,
            f" [ERROR: {error_msg}]" if error_msg else "",
        )
        return result

    def sync_all(
        self,
        *,
        force: bool = False,
    ) -> dict[str, SyncResult]:
        """Sync all registered sources.

        Returns a dict mapping source IDs to :class:`SyncResult` objects.
        """
        results: dict[str, SyncResult] = {}
        for source_id in list(self._sources.keys()):
            results[source_id] = self.sync(source_id, force=force)
        return results

    # ------------------------------------------------------------------
    # Serialization (for persistence across sessions)
    # ------------------------------------------------------------------

    def save_states(self, path: Path | str) -> None:
        """Save sync states to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            source_id: state.model_dump()
            for source_id, state in self._states.items()
        }
        tmp = file_path.with_suffix(file_path.suffix + ".tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(file_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        logger.debug("Saved %d sync states to %s", len(data), file_path)

    def load_states(self, path: Path | str) -> int:
        """Load sync states from a JSON file.

        Returns the number of states loaded. Missing or invalid files
        are silently ignored (returns 0).
        """
        file_path = Path(path)
        if not file_path.exists():
            return 0
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load sync states from %s: %s", file_path, exc)
            return 0
        count = 0
        for source_id, state_data in data.items():
            if source_id in self._states:
                self._states[source_id] = SyncState.model_validate(state_data)
                count += 1
        logger.debug("Loaded %d sync states from %s", count, file_path)
        return count


__all__ = [
    "APISource",
    "DatabaseSource",
    "ETLPipeline",
    "ETLSource",
    "FilesystemSource",
    "RawItem",
    "SourceType",
    "SyncResult",
    "SyncState",
]
