"""Database gateway — unified query interface, pooling and read/write splitting.

A single façade over relational and NoSQL databases so application code
never imports a specific driver. SQL backends (SQLite, MySQL, PostgreSQL)
share a common :class:`DatabaseBackend` contract with connection pooling,
transactions and a normalised placeholder syntax; Redis and MongoDB are
exposed through dedicated NoSQL backends with idiomatic operations.

Design:

* :class:`DatabaseDriver` — enum of supported drivers.
* :class:`ConnectionConfig` — Pydantic model describing how to reach a DB.
* :class:`QueryResult` — normalised result of a SQL statement.
* :class:`ConnectionPool` — bounded, thread-safe connection pool.
* :class:`DatabaseBackend` (ABC) → :class:`SqliteBackend` (stdlib
  ``sqlite3``), :class:`MySQLBackend` / :class:`PostgresBackend` (lazy
  ``pymysql`` / ``psycopg2``).
* :class:`NoSQLBackend` (ABC) → :class:`RedisBackend` (lazy ``redis``),
  :class:`MongoDBBackend` (lazy ``pymongo``).
* :class:`DatabaseGateway` — registers named backends with a role
  (master/replica), routes reads to replicas and writes to the master,
  caches SELECT results and exposes ``transaction`` / ``explain`` helpers.

Only :mod:`sqlite3` (stdlib) is required at import time; every other
driver is imported lazily inside the backend that needs it, so the module
loads cleanly even when optional dependencies are absent.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("justagent.resources")

#: Maximum seconds to wait for a pooled connection before giving up.
DEFAULT_ACQUIRE_TIMEOUT = 30.0


class DatabaseDriver(str, Enum):  # noqa: UP042
    """Supported database drivers."""

    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"


class ConnectionRole(str, Enum):  # noqa: UP042
    """Replication role of a registered backend.

    ``master`` accepts both reads and writes; ``replica`` is read-only and
    is used by the gateway for read splitting.
    """

    MASTER = "master"
    REPLICA = "replica"


class DatabaseError(Exception):
    """Raised for database gateway / backend failures."""


class PoolExhaustedError(DatabaseError):
    """Raised when a connection cannot be acquired within the timeout."""


class QueryResult(BaseModel):
    """Normalised outcome of a SQL statement.

    Attributes:
        sql: The (placeholder-normalised) SQL that ran.
        params: Bound parameters, JSON-safe.
        rows: A list of row dicts (empty for non-SELECT statements).
        columns: Column names in order (empty for non-SELECT statements).
        rowcount: Rows affected by DML, or -1 when unknown.
        last_inserted_id: Auto-increment id of the last INSERT, if any.
        duration: Execution time in seconds.
    """

    sql: str = ""
    params: list[Any] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    rowcount: int = -1
    last_inserted_id: int | None = None
    duration: float = 0.0

    @property
    def row(self) -> dict[str, Any] | None:
        """The first row, or None."""

        return self.rows[0] if self.rows else None


class ConnectionConfig(BaseModel):
    """Connection parameters for a backend.

    Either ``dsn`` (a full URL/DSN) or the individual fields may be set.
    For SQLite, ``database`` is the file path (``":memory:"`` allowed) and
    the network fields are ignored.
    """

    driver: DatabaseDriver
    database: str = ""
    host: str = "localhost"
    port: int = 0
    username: str = ""
    password: str = Field(default="", repr=False)
    dsn: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)

    def port_or_default(self) -> int:
        """Return the configured port or the driver's default."""

        if self.port:
            return self.port
        return _DEFAULT_PORTS.get(self.driver, 0)

    def to_dsn(self) -> str:
        """Build a connection DSN from the fields (when ``dsn`` is unset)."""

        if self.dsn:
            return self.dsn
        if self.driver is DatabaseDriver.SQLITE:
            return self.database or ":memory:"
        port = self.port_or_default()
        auth = ""
        if self.username:
            auth = self.username
            if self.password:
                auth += f":{self.password}"
            auth += "@"
        return f"{self.driver.value}://{auth}{self.host}:{port}/{self.database}"


_DEFAULT_PORTS: dict[DatabaseDriver, int] = {
    DatabaseDriver.MYSQL: 3306,
    DatabaseDriver.POSTGRESQL: 5432,
    DatabaseDriver.MONGODB: 27017,
    DatabaseDriver.REDIS: 6379,
}


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------


class ConnectionPool:
    """Bounded, thread-safe pool of DBAPI connections.

    Connections are created lazily up to ``max_size`` and recycled on
    :meth:`release`. A best-effort ``ping`` is run on checkout; unhealthy
    connections are discarded and replaced.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        min_size: int = 1,
        max_size: int = 10,
        ping: Callable[[Any], bool] | None = None,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> None:
        if min_size < 0:
            raise DatabaseError("min_size must be >= 0")
        if max_size < 1:
            raise DatabaseError("max_size must be >= 1")
        if min_size > max_size:
            raise DatabaseError("min_size must be <= max_size")
        self._factory = factory
        self._min = min_size
        self._max = max_size
        self._ping = ping
        self._acquire_timeout = acquire_timeout
        self._idle: deque[Any] = deque()
        self._in_use: set[int] = set()
        self._size = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._closed = False
        # Prewarm the minimum.
        for _ in range(self._min):
            self._idle.append(self._safe_create())

    @property
    def size(self) -> int:
        """Total connections currently managed (idle + in use)."""

        with self._lock:
            return self._size

    @property
    def in_use(self) -> int:
        """Connections currently checked out."""

        with self._lock:
            return len(self._in_use)

    @property
    def idle(self) -> int:
        """Idle connections ready for reuse."""

        with self._lock:
            return len(self._idle)

    def acquire(self) -> Any:
        """Check out a connection, creating one if room remains.

        Blocks up to ``acquire_timeout`` seconds for a free connection;
        raises :class:`PoolExhaustedError` if none becomes available.
        """

        deadline = time.monotonic() + self._acquire_timeout
        with self._cond:
            while True:
                if self._closed:
                    raise DatabaseError("pool is closed")
                while self._idle:
                    conn = self._idle.popleft()
                    if self._ping is not None and not self._safe_ping(conn):
                        self._safe_close(conn)
                        self._size -= 1
                        continue
                    self._in_use.add(id(conn))
                    return conn
                if self._size < self._max:
                    conn = self._safe_create()
                    self._in_use.add(id(conn))
                    return conn
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PoolExhaustedError(
                        f"could not acquire connection within {self._acquire_timeout}s"
                    )
                self._cond.wait(timeout=remaining)

    def release(self, conn: Any, *, broken: bool = False) -> None:
        """Return a connection to the pool.

        ``broken=True`` discards the connection (e.g. after an error)
        instead of recycling it.
        """

        with self._cond:
            self._in_use.discard(id(conn))
            if self._closed or broken:
                self._safe_close(conn)
                self._size -= 1
            else:
                self._idle.append(conn)
            self._cond.notify()

    def close_all(self) -> None:
        """Close every connection and mark the pool closed."""

        with self._cond:
            self._closed = True
            while self._idle:
                self._safe_close(self._idle.popleft())
            self._size = 0
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_create(self) -> Any:
        try:
            conn = self._factory()
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"failed to create connection: {exc}") from exc
        self._size += 1
        return conn

    def _safe_ping(self, conn: Any) -> bool:
        try:
            return bool(self._ping(conn))  # type: ignore[misc]
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _safe_close(conn: Any) -> None:
        try:
            conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("connection close error: %s", exc)


# ---------------------------------------------------------------------------
# SQL backend abstraction
# ---------------------------------------------------------------------------


class DatabaseBackend(ABC):
    """Abstract relational backend with pooling and a unified SQL interface.

    Subclasses implement :meth:`_create_connection` (return a DBAPI
    connection) and :meth:`_translate_sql` (normalise placeholders).
    Public methods acquire a pooled connection, run the statement and
    release it, so callers never manage connections directly.
    """

    driver: DatabaseDriver

    def __init__(
        self,
        config: ConnectionConfig,
        *,
        min_size: int = 1,
        max_size: int = 10,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> None:
        self.config = config
        self._pool = ConnectionPool(
            self._create_connection,
            min_size=min_size,
            max_size=max_size,
            ping=self._ping,
            acquire_timeout=acquire_timeout,
        )

    # ------------------------------------------------------------------
    # Abstract / overridable
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_connection(self) -> Any:
        """Return a new raw DBAPI connection."""

    def _translate_sql(self, sql: str, params: Any) -> tuple[str, Any]:
        """Normalise placeholders from the canonical ``?``/``:name`` form.

        Default implementation returns the SQL unchanged (suitable for
        SQLite). MySQL/PostgreSQL override to rewrite ``?`` -> ``%s`` and
        ``:name`` -> ``%(name)s``.
        """

        return sql, params

    def _ping(self, conn: Any) -> bool:
        """Return True if ``conn`` is healthy. Default: ``SELECT 1``."""

        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self, sql: str, params: Any = None
    ) -> QueryResult:
        """Run a single statement and return a :class:`QueryResult`.

        DML statements are committed immediately (autocommit-per-statement
        semantics). Use :meth:`transaction` for atomic multi-statement work.

        ``params`` may be a sequence, a dict (named placeholders), a single
        scalar (auto-wrapped into a one-element tuple) or ``None``.
        """

        normalized = _normalize_params(params)
        translated_sql, translated_params = self._translate_sql(sql, normalized)
        conn = self._pool.acquire()
        started = time.perf_counter()
        broken = False
        try:
            cur = conn.cursor()
            cur.execute(translated_sql, translated_params)
            result = self._build_result(translated_sql, params, cur)
            if self._is_write(translated_sql):
                conn.commit()
            cur.close()
            result.duration = time.perf_counter() - started
            return result
        except Exception:
            broken = True
            try:
                conn.rollback()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Rollback failed: %s", exc)
            raise
        finally:
            # NOTE: must release in ``finally`` — a ``return`` inside the
            # ``try`` would otherwise skip an ``else`` clause and leak the
            # connection back to the pool.
            self._pool.release(conn, broken=broken)

    def executemany(
        self, sql: str, params_seq: Sequence[Any]
    ) -> QueryResult:
        """Run ``sql`` once per parameter set in ``params_seq`` (batch)."""

        translated_sql, _ = self._translate_sql(sql, None)
        conn = self._pool.acquire()
        started = time.perf_counter()
        broken = False
        try:
            cur = conn.cursor()
            translated_seq = [
                self._translate_sql(sql, _normalize_params(p))[1]
                for p in params_seq
            ]
            cur.executemany(translated_sql, translated_seq)
            rowcount = cur.rowcount
            last_id = getattr(cur, "lastrowid", None)
            conn.commit()
            cur.close()
            return QueryResult(
                sql=translated_sql,
                params=list(params_seq),
                rowcount=rowcount if rowcount is not None else -1,
                last_inserted_id=int(last_id) if last_id else None,
                duration=time.perf_counter() - started,
            )
        except Exception:
            broken = True
            try:
                conn.rollback()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Rollback failed: %s", exc)
            raise
        finally:
            self._pool.release(conn, broken=broken)

    def fetchone(self, sql: str, params: Any = None) -> dict[str, Any] | None:
        """Run a SELECT and return the first row (or None)."""

        return self.execute(sql, params).row

    def fetchall(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        """Run a SELECT and return every row."""

        return self.execute(sql, params).rows

    @contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        """Yield a raw connection for an atomic multi-statement block.

        Commits on clean exit, rolls back on any exception. The connection
        is held exclusively for the duration of the block.
        """

        conn = self._pool.acquire()
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Transaction rollback failed: %s", exc)
            self._pool.release(conn, broken=True)
            raise
        else:
            self._pool.release(conn)

    def ping(self) -> bool:
        """Return True if the backend can serve at least one query."""

        try:
            conn = self._pool.acquire()
        except DatabaseError:
            return False
        try:
            return self._ping(conn)
        finally:
            self._pool.release(conn)

    def close(self) -> None:
        """Close the underlying pool."""

        self._pool.close_all()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result(self, sql: str, params: Any, cur: Any) -> QueryResult:
        columns: list[str] = []
        rows: list[dict[str, Any]] = []
        if cur.description:
            columns = [d[0] for d in cur.description]
            fetched = cur.fetchall()
            rows = [dict(zip(columns, row, strict=False)) for row in fetched]
        rowcount = cur.rowcount if cur.rowcount is not None else -1
        last_id = getattr(cur, "lastrowid", None)
        return QueryResult(
            sql=sql,
            params=_jsonable_params(params),
            rows=rows,
            columns=columns,
            rowcount=rowcount,
            last_inserted_id=int(last_id) if last_id else None,
        )

    @staticmethod
    def _is_write(sql: str) -> bool:
        """Heuristic: True for DML/DDL that should be committed."""

        stripped = sql.lstrip().upper()
        # Strip leading WITH (...) for CTE-based statements.
        return not _first_keyword_is_read(stripped)


def _first_keyword_is_read(stripped_upper_sql: str) -> bool:
    """Return True if the statement's first SQL keyword is a read keyword."""

    if not stripped_upper_sql:
        return False
    first = stripped_upper_sql.split(None, 1)[0]
    # Strip a leading '(' (e.g. "(SELECT ...)").
    first = first.lstrip("(")
    return first in _READ_KEYWORDS


_READ_KEYWORDS: frozenset[str] = frozenset(
    {"SELECT", "SHOW", "EXPLAIN", "WITH", "DESCRIBE", "PRAGMA"}
)


def _normalize_params(params: Any) -> Any:
    """Coerce ``params`` into a DBAPI-friendly form.

    * ``None`` -> empty tuple.
    * ``dict`` -> returned unchanged (named placeholders).
    * ``list`` / ``tuple`` -> returned unchanged.
    * any other scalar (int, str, bytes, ...) -> wrapped in a one-element
      tuple so callers can pass ``fetchone(sql, 42)`` instead of
      ``fetchone(sql, (42,))``.
    """

    if params is None:
        return ()
    if isinstance(params, dict):
        return params
    if isinstance(params, (list, tuple)):
        return params
    return (params,)


def _jsonable_params(params: Any) -> list[Any]:
    """Coerce params into a JSON-safe list for :class:`QueryResult`."""

    if params is None:
        return []
    if isinstance(params, dict):
        return [{k: _jsonable(v) for k, v in params.items()}]
    if isinstance(params, (list, tuple)):
        return [_jsonable(v) for v in params]
    return [_jsonable(params)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# SQL drivers
# ---------------------------------------------------------------------------


class SqliteBackend(DatabaseBackend):
    """SQLite backend built on the stdlib :mod:`sqlite3` module.

    ``database`` is the file path (``":memory:"`` for an in-memory DB).
    Connections are opened with ``check_same_thread=False`` and WAL mode so
    they can be pooled and shared across threads. The default pool size is
    small (5) since SQLite serialises writes at the file level.
    """

    driver = DatabaseDriver.SQLITE

    def __init__(
        self,
        config: ConnectionConfig,
        *,
        min_size: int = 1,
        max_size: int = 5,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> None:
        if config.driver is not DatabaseDriver.SQLITE:
            raise DatabaseError(
                f"SqliteBackend requires driver=sqlite, got {config.driver.value}"
            )
        super().__init__(
            config,
            min_size=min_size,
            max_size=max_size,
            acquire_timeout=acquire_timeout,
        )

    def _create_connection(self) -> Any:
        path = self.config.database or ":memory:"
        if path == ":memory:":
            # Pooled in-memory connections must share a single database;
            # the default ":memory:" gives each connection its own private
            # DB, which would make CREATE/INSERT land on different backends.
            conn = sqlite3.connect(
                "file::memory:?cache=shared",
                uri=True,
                check_same_thread=False,
            )
        else:
            conn = sqlite3.connect(path, check_same_thread=False)
        # WAL allows concurrent readers alongside a single writer.
        if path != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
            except sqlite3.DatabaseError as exc:
                logger.debug("Failed to set WAL mode: %s", exc)
        conn.row_factory = None
        return conn

    def _translate_sql(self, sql: str, params: Any) -> tuple[str, Any]:
        # sqlite3 accepts ? and :name natively — no translation needed.
        return sql, params

    def _ping(self, conn: Any) -> bool:
        try:
            conn.execute("SELECT 1").close()
            return True
        except Exception:  # noqa: BLE001
            return False


_PARAM_RE = re.compile(r":(\w+)")


class MySQLBackend(DatabaseBackend):
    """MySQL backend using ``pymysql`` (imported lazily on first connect).

    Canonical ``?`` placeholders are rewritten to ``%s`` and ``:name`` to
    ``%(name)s`` to match the MySQLdb paramstyle.
    """

    driver = DatabaseDriver.MYSQL

    def _create_connection(self) -> Any:
        try:
            import pymysql  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise DatabaseError(
                "pymysql is required for the MySQL backend; install with "
                "`pip install pymysql`"
            ) from exc
        try:
            return pymysql.connect(
                host=self.config.host,
                port=self.config.port_or_default(),
                user=self.config.username,
                password=self.config.password,
                database=self.config.database or None,
                autocommit=False,
                cursorclass=pymysql.cursors.DictCursor,
                **self.config.extra,
            )
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"MySQL connect failed: {exc}") from exc

    def _translate_sql(self, sql: str, params: Any) -> tuple[str, Any]:
        if isinstance(params, dict):
            translated = _PARAM_RE.sub(r"%(\1)s", sql)
            return translated, params
        return sql.replace("?", "%s"), params

    def _ping(self, conn: Any) -> bool:
        try:
            conn.ping(reconnect=True)
            return True
        except Exception:  # noqa: BLE001
            return False


class PostgresBackend(DatabaseBackend):
    """PostgreSQL backend using ``psycopg2`` (imported lazily).

    Canonical ``?`` placeholders are rewritten to ``%s`` and ``:name`` to
    ``%(name)s`` to match the psycopg2 paramstyle. Dict cursors are used so
    rows are returned as dicts.
    """

    driver = DatabaseDriver.POSTGRESQL

    def _create_connection(self) -> Any:
        try:
            import psycopg2  # type: ignore[import-not-found]
            from psycopg2.extras import DictCursor  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise DatabaseError(
                "psycopg2 is required for the PostgreSQL backend; install "
                "with `pip install psycopg2-binary`"
            ) from exc
        dsn = self.config.to_dsn()
        try:
            return psycopg2.connect(dsn, cursor_factory=DictCursor, **self.config.extra)
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"PostgreSQL connect failed: {exc}") from exc

    def _translate_sql(self, sql: str, params: Any) -> tuple[str, Any]:
        if isinstance(params, dict):
            translated = _PARAM_RE.sub(r"%(\1)s", sql)
            return translated, params
        return sql.replace("?", "%s"), params

    def _ping(self, conn: Any) -> bool:
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            return True
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
# NoSQL backends
# ---------------------------------------------------------------------------


class NoSQLBackend(ABC):
    """Abstract non-relational backend (Redis / MongoDB).

    NoSQL backends do not implement the SQL :meth:`DatabaseBackend.execute`
    contract; they expose idiomatic operations and a :meth:`raw_client`
    accessor for advanced use.
    """

    driver: DatabaseDriver

    def __init__(self, config: ConnectionConfig) -> None:
        self.config = config
        self._client: Any = None
        self._lock = threading.Lock()

    @abstractmethod
    def _create_client(self) -> Any:
        """Create and return the native client object."""

    def raw_client(self) -> Any:
        """Return the (lazily created) native client."""

        with self._lock:
            if self._client is None:
                self._client = self._create_client()
            return self._client

    def close(self) -> None:
        """Close the native client if it exposes ``close``."""

        with self._lock:
            if self._client is not None:
                closer = getattr(self._client, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("NoSQL client close error: %s", exc)
                self._client = None

    def ping(self) -> bool:
        """Best-effort connectivity check; default returns True after connect."""

        try:
            self.raw_client()
            return True
        except DatabaseError:
            return False


class RedisBackend(NoSQLBackend):
    """Redis backend using ``redis-py`` (imported lazily).

    Exposes common key/value operations. Values are encoded as UTF-8
    strings; pass ``bytes`` to store raw bytes.
    """

    driver = DatabaseDriver.REDIS

    def _create_client(self) -> Any:
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise DatabaseError(
                "redis is required for the Redis backend; install with "
                "`pip install redis`"
            ) from exc
        try:
            return redis.Redis(
                host=self.config.host,
                port=self.config.port_or_default(),
                username=self.config.username or None,
                password=self.config.password or None,
                db=int(self.config.database) if self.config.database.isdigit() else 0,
                decode_responses=True,
                **self.config.extra,
            )
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"Redis connect failed: {exc}") from exc

    def ping(self) -> bool:
        try:
            return bool(self.raw_client().ping())
        except Exception:  # noqa: BLE001
            return False

    def get(self, key: str) -> str | None:
        value = self.raw_client().get(key)
        return value if value is None else str(value)

    def set(self, key: str, value: str | bytes, *, ttl: int | None = None) -> bool:
        client = self.raw_client()
        if ttl is not None:
            return bool(client.set(key, value, ex=ttl))
        return bool(client.set(key, value))

    def delete(self, *keys: str) -> int:
        return int(self.raw_client().delete(*keys))

    def exists(self, key: str) -> bool:
        return bool(self.raw_client().exists(key))

    def expire(self, key: str, ttl: int) -> bool:
        return bool(self.raw_client().expire(key, ttl))

    def incr(self, key: str, amount: int = 1) -> int:
        return int(self.raw_client().incrby(key, amount))

    def keys(self, pattern: str = "*") -> list[str]:
        return [str(k) for k in self.raw_client().keys(pattern)]


class MongoDBBackend(NoSQLBackend):
    """MongoDB backend using ``pymongo`` (imported lazily).

    Operations target a single database (``config.database``); collections
    are passed by name to each method.
    """

    driver = DatabaseDriver.MONGODB

    def _create_client(self) -> Any:
        try:
            from pymongo import MongoClient  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise DatabaseError(
                "pymongo is required for the MongoDB backend; install with "
                "`pip install pymongo`"
            ) from exc
        try:
            client = MongoClient(self.config.to_dsn(), **self.config.extra)
            # Touch the DB to force connection / surface auth errors early.
            _ = client[self.config.database or "admin"].command("ping")
            return client
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"MongoDB connect failed: {exc}") from exc

    def ping(self) -> bool:
        try:
            self.raw_client()[self.config.database or "admin"].command("ping")
            return True
        except Exception:  # noqa: BLE001
            return False

    def _collection(self, name: str) -> Any:
        return self.raw_client()[self.config.database][name]

    def find_one(
        self, collection: str, filter: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        doc = self._collection(collection).find_one(filter or {})
        if doc is None:
            return None
        doc = dict(doc)
        doc["id"] = str(doc.pop("_id", ""))
        return doc

    def find(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,
        *,
        limit: int = 0,
        skip: int = 0,
        sort: list[tuple[str, int]] | None = None,
    ) -> list[dict[str, Any]]:
        cursor = self._collection(collection).find(filter or {})
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        results: list[dict[str, Any]] = []
        for doc in cursor:
            doc = dict(doc)
            doc["id"] = str(doc.pop("_id", ""))
            results.append(doc)
        return results

    def insert_one(self, collection: str, document: dict[str, Any]) -> str:
        result = self._collection(collection).insert_one(document)
        return str(result.inserted_id)

    def insert_many(self, collection: str, documents: list[dict[str, Any]]) -> list[str]:
        result = self._collection(collection).insert_many(documents)
        return [str(_id) for _id in result.inserted_ids]

    def update_one(
        self,
        collection: str,
        filter: dict[str, Any],
        update: dict[str, Any],
        *,
        upsert: bool = False,
    ) -> int:
        result = self._collection(collection).update_one(
            filter, update, upsert=upsert
        )
        return int(result.modified_count)

    def delete_many(self, collection: str, filter: dict[str, Any]) -> int:
        result = self._collection(collection).delete_many(filter)
        return int(result.deleted_count)

    def count(
        self, collection: str, filter: dict[str, Any] | None = None
    ) -> int:
        return int(self._collection(collection).count_documents(filter or {}))


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class _RegisteredBackend:
    """Internal record of a registered backend and its role."""

    def __init__(self, name: str, backend: Any, role: ConnectionRole) -> None:
        self.name = name
        self.backend = backend
        self.role = role


class _QueryCache:
    """Tiny TTL cache for SELECT results keyed by (sql, params)."""

    def __init__(self, ttl: float, max_entries: int) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self._store: dict[tuple[str, str], tuple[float, QueryResult]] = {}
        self._lock = threading.Lock()

    def get(self, sql: str, params: Any) -> QueryResult | None:
        key = (sql, repr(params))
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, result = entry
            if time.monotonic() - ts > self.ttl:
                self._store.pop(key, None)
                return None
            return result

    def put(self, sql: str, params: Any, result: QueryResult) -> None:
        if len(self._store) >= self.max_entries:
            # Drop the oldest entry to bound memory.
            oldest = min(self._store.items(), key=lambda kv: kv[1][0])
            self._store.pop(oldest[0], None)
        self._store[(sql, repr(params))] = (time.monotonic(), result)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class DatabaseGateway:
    """Unified database façade with read/write splitting and caching.

    Register one master (required) and any number of replicas. Read
    statements (SELECT/SHOW/EXPLAIN/WITH/...) are routed round-robin to
    replicas when available, else to the master; everything else goes to
    the master. An optional query cache memoises SELECT results.

    Example:

        >>> gw = DatabaseGateway()
        >>> gw.register("primary", SqliteBackend(ConnectionConfig(
        ...     driver=DatabaseDriver.SQLITE, database=":memory:")),
        ...     role=ConnectionRole.MASTER)
        >>> gw.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        >>> gw.execute("INSERT INTO t (id, name) VALUES (?, ?)", (1, "ada"))
        >>> gw.fetchone("SELECT name FROM t WHERE id = ?", (1))
        {'name': 'ada'}
    """

    def __init__(self, *, query_cache_ttl: float = 0.0) -> None:
        self._backends: dict[str, _RegisteredBackend] = {}
        self._replica_rr = 0
        self._cache: _QueryCache | None = (
            _QueryCache(ttl=query_cache_ttl, max_entries=512)
            if query_cache_ttl > 0
            else None
        )
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        backend: DatabaseBackend | NoSQLBackend,
        *,
        role: ConnectionRole = ConnectionRole.MASTER,
    ) -> None:
        """Register a backend under ``name`` with the given ``role``."""

        if not name:
            raise DatabaseError("backend name must not be empty")
        with self._lock:
            if name in self._backends:
                raise DatabaseError(f"backend {name!r} is already registered")
            self._backends[name] = _RegisteredBackend(name, backend, role)
        logger.info(
            "Registered backend %s (driver=%s, role=%s)",
            name,
            backend.driver.value,
            role.value,
        )

    def unregister(self, name: str) -> Any | None:
        """Remove and return a backend (closing it), or None if absent."""

        with self._lock:
            entry = self._backends.pop(name, None)
        if entry is None:
            return None
        try:
            entry.backend.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("backend close error: %s", exc)
        return entry.backend

    def backend(self, name: str) -> Any | None:
        """Return the registered backend object, or None."""

        with self._lock:
            entry = self._backends.get(name)
            return entry.backend if entry is not None else None

    def sql(self, name: str) -> DatabaseBackend:
        """Return a registered SQL backend; raises if it is not SQL."""

        backend = self.backend(name)
        if backend is None:
            raise DatabaseError(f"no backend named {name!r}")
        if not isinstance(backend, DatabaseBackend):
            raise DatabaseError(f"backend {name!r} is not a SQL backend")
        return backend

    def kv(self, name: str) -> RedisBackend:
        """Return a registered Redis backend."""

        backend = self.backend(name)
        if not isinstance(backend, RedisBackend):
            raise DatabaseError(f"backend {name!r} is not a Redis backend")
        return backend

    def doc(self, name: str) -> MongoDBBackend:
        """Return a registered MongoDB backend."""

        backend = self.backend(name)
        if not isinstance(backend, MongoDBBackend):
            raise DatabaseError(f"backend {name!r} is not a MongoDB backend")
        return backend

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._backends.keys())

    # ------------------------------------------------------------------
    # Query interface (SQL)
    # ------------------------------------------------------------------

    def execute(
        self,
        sql: str,
        params: Any = None,
        *,
        target: str = "auto",
    ) -> QueryResult:
        """Run ``sql`` with auto routing and optional caching.

        ``target`` overrides routing: ``"master"``, ``"replica"`` or
        ``"auto"`` (default). Cached for read statements when a query cache
        is configured.
        """

        is_read = _first_keyword_is_read(sql.lstrip().upper())
        if is_read and self._cache is not None:
            cached = self._cache.get(sql, params)
            if cached is not None:
                return cached
        backend = self._select_backend(target, is_read)
        result = backend.execute(sql, params)
        if is_read and self._cache is not None:
            self._cache.put(sql, params, result)
        return result

    def executemany(
        self, sql: str, params_seq: Sequence[Any], *, target: str = "master"
    ) -> QueryResult:
        """Batch execute; always routes to the master."""

        return self._select_backend(target, is_read=False).executemany(
            sql, params_seq
        )

    def fetchone(self, sql: str, params: Any = None) -> dict[str, Any] | None:
        return self.execute(sql, params).row

    def fetchall(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        return self.execute(sql, params).rows

    @contextmanager
    def transaction(self, name: str | None = None) -> Generator[Any, None, None]:
        """Open a transaction on the master (or a named SQL backend)."""

        backend = self.sql(name) if name else self._select_backend("master", False)
        with backend.transaction() as conn:
            yield conn

    def explain(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        """Run ``EXPLAIN <sql>`` on the master and return the plan rows."""

        backend = self._select_backend("master", True)
        return backend.fetchall(f"EXPLAIN {sql}", params)

    # ------------------------------------------------------------------
    # Cache control
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Invalidate the query result cache."""

        if self._cache is not None:
            self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of entries currently in the query cache."""

        return len(self._cache) if self._cache is not None else 0

    # ------------------------------------------------------------------
    # Stats / lifecycle
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return per-backend pool stats for dashboards."""

        with self._lock:
            entries = list(self._backends.values())
        out: dict[str, Any] = {}
        for entry in entries:
            pool = getattr(entry.backend, "_pool", None)
            if pool is not None:
                out[entry.name] = {
                    "driver": entry.backend.driver.value,
                    "role": entry.role.value,
                    "size": pool.size,
                    "in_use": pool.in_use,
                    "idle": pool.idle,
                }
            else:
                out[entry.name] = {
                    "driver": entry.backend.driver.value,
                    "role": entry.role.value,
                    "pooled": False,
                }
        out["cache_entries"] = self.cache_size
        return out

    def close_all(self) -> None:
        """Close every registered backend."""

        with self._lock:
            entries = list(self._backends.values())
            self._backends.clear()
        for entry in entries:
            try:
                entry.backend.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("backend close error: %s", exc)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _select_backend(self, target: str, is_read: bool) -> DatabaseBackend:
        """Pick the SQL backend for the next statement."""

        with self._lock:
            entries = list(self._backends.values())
        sql_entries = [
            e for e in entries if isinstance(e.backend, DatabaseBackend)
        ]
        if not sql_entries:
            raise DatabaseError("no SQL backend registered")

        if target == "replica":
            replicas = [e for e in sql_entries if e.role is ConnectionRole.REPLICA]
            if not replicas:
                raise DatabaseError("no replica SQL backend registered")
            return self._round_robin(replicas)
        if target == "master":
            masters = [e for e in sql_entries if e.role is ConnectionRole.MASTER]
            if not masters:
                raise DatabaseError("no master SQL backend registered")
            return masters[0].backend  # type: ignore[return-value]
        # auto
        if is_read:
            replicas = [e for e in sql_entries if e.role is ConnectionRole.REPLICA]
            if replicas:
                return self._round_robin(replicas)
        masters = [e for e in sql_entries if e.role is ConnectionRole.MASTER]
        if not masters:
            raise DatabaseError("no master SQL backend registered")
        return masters[0].backend  # type: ignore[return-value]

    def _round_robin(self, replicas: list[_RegisteredBackend]) -> DatabaseBackend:
        idx = self._replica_rr % len(replicas)
        self._replica_rr += 1
        return replicas[idx].backend  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_backend(
    config: ConnectionConfig,
    *,
    min_size: int = 1,
    max_size: int | None = None,
) -> DatabaseBackend | NoSQLBackend:
    """Create the appropriate backend for ``config.driver``.

    ``max_size`` defaults are driver-specific (5 for SQLite, 10 otherwise).
    """

    driver = config.driver
    if driver is DatabaseDriver.SQLITE:
        return SqliteBackend(config, min_size=min_size, max_size=max_size or 5)
    if driver is DatabaseDriver.MYSQL:
        return MySQLBackend(config, min_size=min_size, max_size=max_size or 10)
    if driver is DatabaseDriver.POSTGRESQL:
        return PostgresBackend(config, min_size=min_size, max_size=max_size or 10)
    if driver is DatabaseDriver.REDIS:
        return RedisBackend(config)
    if driver is DatabaseDriver.MONGODB:
        return MongoDBBackend(config)
    raise DatabaseError(f"unsupported driver: {driver!r}")


__all__ = [
    "ConnectionConfig",
    "ConnectionPool",
    "ConnectionRole",
    "DatabaseBackend",
    "DatabaseDriver",
    "DatabaseError",
    "DatabaseGateway",
    "MongoDBBackend",
    "MySQLBackend",
    "NoSQLBackend",
    "PoolExhaustedError",
    "PostgresBackend",
    "QueryResult",
    "RedisBackend",
    "SqliteBackend",
    "create_backend",
]
