"""Unified storage manager — file CRUD, quotas and path abstraction.

Provides a single interface over heterogeneous storage backends so the rest
of the platform never has to know whether a file lives on the local disk,
a NAS mount, an S3 bucket or a MinIO instance.

Design:

* :class:`BackendType` — enum of supported backends (LOCAL, NAS, S3, MINIO).
* :class:`StorageBackend` — abstract base class defining the file CRUD
  contract every backend implements.
* :class:`LocalStorageBackend` / :class:`NASStorageBackend` — concrete
  backends built on :mod:`pathlib` / :mod:`os` (NAS is mounted locally).
* :class:`S3StorageBackend` / :class:`MinioStorageBackend` — concrete
  backends that lazily import ``boto3`` / ``minio`` so the module loads
  without those optional dependencies installed.
* :class:`FileInfo` / :class:`QuotaSpec` / :class:`QuotaUsage` — Pydantic
  data models.
* :class:`QuotaManager` — per-namespace byte/file quotas with live usage
  tracking.
* :class:`StorageManager` — the façade: a mount table maps logical path
  prefixes to backends, enforces quotas, and exposes a uniform file API.

Logical paths use the ``<scheme>://<root>/<key>`` form, e.g.
``local:///data/reports/q1.pdf`` or ``s3://backups/db/dump.sql``.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from justagent.utils import utcnow

logger = logging.getLogger("justagent.resources")


class BackendType(str, Enum):  # noqa: UP042
    """Supported storage backend types."""

    LOCAL = "local"
    NAS = "nas"
    S3 = "s3"
    MINIO = "minio"


class FileInfo(BaseModel):
    """Metadata describing a single stored object.

    Attributes:
        path: Logical path of the object.
        key: Backend-relative key.
        size: Size in bytes (0 for directories).
        modified_at: Last-modified Unix timestamp.
        is_dir: True for directory-like entries.
        content_type: Best-effort MIME type (empty when unknown).
        etag: Backend-specific content hash (e.g. S3 ETag); empty when N/A.
    """

    path: str
    key: str
    size: int = 0
    modified_at: float = 0.0
    is_dir: bool = False
    content_type: str = ""
    etag: str = ""


class QuotaSpec(BaseModel):
    """A namespace quota: hard ceilings on bytes and file count.

    A zero value means "no limit on this axis".
    """

    namespace: str
    max_bytes: int = 0
    max_files: int = 0


class QuotaUsage(BaseModel):
    """Live usage against a :class:`QuotaSpec`."""

    namespace: str
    used_bytes: int = 0
    used_files: int = 0

    def exceeds(self, spec: QuotaSpec) -> bool:
        """True if current usage already violates ``spec``."""

        if spec.max_bytes and self.used_bytes > spec.max_bytes:
            return True
        if spec.max_files and self.used_files > spec.max_files:
            return True
        return False

    def would_exceed(self, spec: QuotaSpec, *, extra_bytes: int = 0, extra_files: int = 0) -> bool:
        """True if adding ``extra_bytes``/``extra_files`` would breach ``spec``."""

        if spec.max_bytes and self.used_bytes + extra_bytes > spec.max_bytes:
            return True
        if spec.max_files and self.used_files + extra_files > spec.max_files:
            return True
        return False


class StorageError(Exception):
    """Raised for invalid storage operations or backend failures."""


class QuotaExceededError(StorageError):
    """Raised when a write would breach the namespace quota."""

    def __init__(self, namespace: str, message: str) -> None:
        super().__init__(message)
        self.namespace = namespace


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


def _normalize_key(key: str) -> str:
    """Normalise a backend-relative key: posix, no leading slash, no ``..``."""

    if not key:
        return ""
    cleaned = PurePosixPath(key).as_posix().lstrip("/")
    if ".." in cleaned.split("/"):
        raise StorageError(f"path traversal not allowed: {key!r}")
    return cleaned


class StorageBackend(ABC):
    """Abstract file-storage backend.

    All keys are backend-relative posix strings (forward slashes, no
    leading slash). Implementations must be safe to call from multiple
    threads; the local backend relies on OS-level locking, the object
    backends create one client per operation.
    """

    backend_type: BackendType

    def __init__(self, name: str, *, root: str = "") -> None:
        self.name = name
        self.root = root

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Return the raw bytes stored at ``key``."""

    @abstractmethod
    def write(self, key: str, data: bytes) -> int:
        """Store ``data`` at ``key``; return the number of bytes written."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete ``key``; return True if something was removed."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if ``key`` exists."""

    @abstractmethod
    def stat(self, key: str) -> FileInfo:
        """Return :class:`FileInfo` for ``key``."""

    @abstractmethod
    def list_dir(self, prefix: str = "") -> list[FileInfo]:
        """List immediate children under ``prefix``."""

    @abstractmethod
    def mkdir(self, key: str) -> None:
        """Create ``key`` as a directory (no-op for object stores)."""

    # Optional operations with default implementations.
    def rename(self, src: str, dst: str) -> None:
        """Move ``src`` to ``dst``. Default: copy-then-delete."""

        data = self.read(src)
        self.write(dst, data)
        self.delete(src)

    def copy(self, src: str, dst: str) -> None:
        """Copy ``src`` to ``dst``."""

        self.write(dst, self.read(src))

    def glob(self, pattern: str) -> list[FileInfo]:
        """Return entries whose key matches a glob ``pattern``.

        Default implementation walks all keys via :meth:`list_dir` and
        matches with :func:`fnmatch`. Backends with server-side filtering
        should override this for efficiency.
        """

        import fnmatch

        norm = _normalize_key(pattern)
        results: list[FileInfo] = []
        seen: set[str] = set()
        stack: list[str] = [""]
        while stack:
            prefix = stack.pop()
            for entry in self.list_dir(prefix):
                if entry.key in seen:
                    continue
                seen.add(entry.key)
                if fnmatch.fnmatch(entry.key, norm):
                    results.append(entry)
                if entry.is_dir:
                    stack.append(entry.key)
        return sorted(results, key=lambda f: f.key)

    def close(self) -> None:
        """Release backend resources (default: no-op)."""


# ---------------------------------------------------------------------------
# Local / NAS backends
# ---------------------------------------------------------------------------


class LocalStorageBackend(StorageBackend):
    """Backend backed by the local filesystem (or a locally-mounted NAS).

    ``root`` is the absolute directory that forms the backend's root; keys
    are resolved relative to it. All paths are confined to ``root`` —
    attempts to escape via ``..`` raise :class:`StorageError`.
    """

    backend_type = BackendType.LOCAL

    def __init__(self, name: str, root: str | Path) -> None:
        super().__init__(name=name, root=str(Path(root).resolve()))
        self._root_path = Path(self.root)
        self._root_path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        norm = _normalize_key(key)
        target = (self._root_path / norm).resolve() if norm else self._root_path
        try:
            target.relative_to(self._root_path.resolve())
        except ValueError as exc:
            raise StorageError(f"path escapes backend root: {key!r}") from exc
        return target

    def read(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise StorageError(f"not found: {key!r}")
        return path.read_bytes()

    def write(self, key: str, data: bytes) -> int:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return len(data)

    def delete(self, key: str) -> bool:
        path = self._resolve(key)
        if not path.exists():
            return False
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def stat(self, key: str) -> FileInfo:
        path = self._resolve(key)
        if not path.exists():
            raise StorageError(f"not found: {key!r}")
        st = path.stat()
        return FileInfo(
            path=f"{self.name}://{key}",
            key=_normalize_key(key),
            size=st.st_size if path.is_file() else 0,
            modified_at=st.st_mtime,
            is_dir=path.is_dir(),
        )

    def list_dir(self, prefix: str = "") -> list[FileInfo]:
        base = self._resolve(prefix)
        if not base.exists() or not base.is_dir():
            return []
        results: list[FileInfo] = []
        for child in sorted(base.iterdir()):
            rel = child.relative_to(self._root_path).as_posix()
            results.append(
                FileInfo(
                    path=f"{self.name}://{rel}",
                    key=rel,
                    size=child.stat().st_size if child.is_file() else 0,
                    modified_at=child.stat().st_mtime,
                    is_dir=child.is_dir(),
                )
            )
        return results

    def mkdir(self, key: str) -> None:
        self._resolve(key).mkdir(parents=True, exist_ok=True)

    def rename(self, src: str, dst: str) -> None:
        src_path = self._resolve(src)
        dst_path = self._resolve(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src_path, dst_path)

    def copy(self, src: str, dst: str) -> None:
        src_path = self._resolve(src)
        dst_path = self._resolve(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


class NASStorageBackend(LocalStorageBackend):
    """Backend for a NAS share mounted into the local filesystem.

    Functionally identical to :class:`LocalStorageBackend` (the OS already
    presents the share as a directory); the distinct type lets the
    platform distinguish NAS capacity from local scratch space and apply
    different monitoring/quotas.
    """

    backend_type = BackendType.NAS

    def __init__(self, name: str, root: str | Path, *, mount_host: str = "") -> None:
        super().__init__(name=name, root=root)
        self.mount_host = mount_host


# ---------------------------------------------------------------------------
# Object-store backends (lazy imports)
# ---------------------------------------------------------------------------


class _ObjectStorageBackend(StorageBackend):
    """Shared logic for S3-compatible object stores.

    Subclasses set ``backend_type`` and implement :meth:`_client_get` /
    :meth:`_client_put` / :meth:`_client_delete` / :meth:`_client_head` /
    :meth:`_client_list` against a concrete SDK. Keys map directly to
    object keys within a single bucket.
    """

    def __init__(
        self,
        name: str,
        *,
        endpoint: str,
        bucket: str,
        access_key: str = "",
        secret_key: str = "",
        region: str = "",
        secure: bool = True,
        root: str = "",
    ) -> None:
        super().__init__(name=name, root=root)
        self.endpoint = endpoint
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.secure = secure

    # -- SDK accessors (implemented by subclasses) ---------------------
    @abstractmethod
    def _ensure_client(self) -> Any:
        """Lazily create and return the SDK client object."""

    def read(self, key: str) -> bytes:
        return self._client_get(_normalize_key(key))

    def write(self, key: str, data: bytes) -> int:
        self._client_put(_normalize_key(key), data)
        return len(data)

    def delete(self, key: str) -> bool:
        return self._client_delete(_normalize_key(key))

    def exists(self, key: str) -> bool:
        return self._client_head(_normalize_key(key))

    def stat(self, key: str) -> FileInfo:
        return self._client_stat(_normalize_key(key))

    def list_dir(self, prefix: str = "") -> list[FileInfo]:
        return self._client_list(_normalize_key(prefix))

    def mkdir(self, key: str) -> None:  # noqa: D102 - no-op for object stores
        return None

    # -- SDK-bound primitives (default impls raise informative errors) -
    def _client_get(self, key: str) -> bytes:
        raise NotImplementedError

    def _client_put(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def _client_delete(self, key: str) -> bool:
        raise NotImplementedError

    def _client_head(self, key: str) -> bool:
        raise NotImplementedError

    def _client_stat(self, key: str) -> FileInfo:
        raise NotImplementedError

    def _client_list(self, prefix: str) -> list[FileInfo]:
        raise NotImplementedError


class S3StorageBackend(_ObjectStorageBackend):
    """S3 backend powered by ``boto3`` (imported lazily on first use)."""

    backend_type = BackendType.S3

    def _ensure_client(self) -> Any:
        if getattr(self, "_client", None) is None:
            try:
                import boto3  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional dep
                raise StorageError(
                    "boto3 is required for the S3 backend; install with "
                    "`pip install boto3`"
                ) from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint or None,
                aws_access_key_id=self.access_key or None,
                aws_secret_access_key=self.secret_key or None,
                region_name=self.region or None,
            )
        return self._client

    def _client_get(self, key: str) -> bytes:
        client = self._ensure_client()
        try:
            resp = client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"S3 get failed for {key!r}: {exc}") from exc
        body = resp["Body"].read()
        return body

    def _client_put(self, key: str, data: bytes) -> None:
        client = self._ensure_client()
        try:
            client.put_object(Bucket=self.bucket, Key=key, Body=data)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"S3 put failed for {key!r}: {exc}") from exc

    def _client_delete(self, key: str) -> bool:
        client = self._ensure_client()
        try:
            client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"S3 delete failed for {key!r}: {exc}") from exc

    def _client_head(self, key: str) -> bool:
        client = self._ensure_client()
        from botocore.exceptions import ClientError  # type: ignore[import-not-found]

        try:
            client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def _client_stat(self, key: str) -> FileInfo:
        client = self._ensure_client()
        try:
            resp = client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"S3 stat failed for {key!r}: {exc}") from exc
        mtime = resp.get("LastModified")
        ts = mtime.timestamp() if mtime else 0.0
        return FileInfo(
            path=f"{self.name}://{key}",
            key=key,
            size=int(resp.get("ContentLength", 0)),
            modified_at=ts,
            content_type=str(resp.get("ContentType", "")),
            etag=str(resp.get("ETag", "")).strip('"'),
        )

    def _client_list(self, prefix: str) -> list[FileInfo]:
        client = self._ensure_client()
        results: list[FileInfo] = []
        token: str | None = None
        seen: set[str] = set()
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            try:
                resp = client.list_objects_v2(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise StorageError(f"S3 list failed for {prefix!r}: {exc}") from exc
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key in seen:
                    continue
                seen.add(key)
                mtime = obj.get("LastModified")
                ts = mtime.timestamp() if mtime else 0.0
                results.append(
                    FileInfo(
                        path=f"{self.name}://{key}",
                        key=key,
                        size=int(obj.get("Size", 0)),
                        modified_at=ts,
                        etag=str(obj.get("ETag", "")).strip('"'),
                    )
                )
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return results


class MinioStorageBackend(_ObjectStorageBackend):
    """MinIO backend powered by the ``minio`` SDK (imported lazily)."""

    backend_type = BackendType.MINIO

    def _ensure_client(self) -> Any:
        if getattr(self, "_client", None) is None:
            try:
                from minio import Minio  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional dep
                raise StorageError(
                    "minio is required for the MinIO backend; install with "
                    "`pip install minio`"
                ) from exc
            from urllib.parse import urlparse as _urlparse

            host = _urlparse(self.endpoint).netloc or self.endpoint
            self._client = Minio(
                host,
                access_key=self.access_key or None,
                secret_key=self.secret_key or None,
                secure=self.secure,
                region=self.region or None,
            )
            if not self._client.bucket_exists(self.bucket):
                self._client.make_bucket(self.bucket)
        return self._client

    def _client_get(self, key: str) -> bytes:
        client = self._ensure_client()
        try:
            resp = client.get_object(self.bucket, key)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"MinIO get failed for {key!r}: {exc}") from exc

    def _client_put(self, key: str, data: bytes) -> None:
        import io

        client = self._ensure_client()
        try:
            client.put_object(
                self.bucket, key, io.BytesIO(data), length=len(data)
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"MinIO put failed for {key!r}: {exc}") from exc

    def _client_delete(self, key: str) -> bool:
        client = self._ensure_client()
        try:
            client.remove_object(self.bucket, key)
            return True
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"MinIO delete failed for {key!r}: {exc}") from exc

    def _client_head(self, key: str) -> bool:
        client = self._ensure_client()
        try:
            client.stat_object(self.bucket, key)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _client_stat(self, key: str) -> FileInfo:
        client = self._ensure_client()
        try:
            obj = client.stat_object(self.bucket, key)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"MinIO stat failed for {key!r}: {exc}") from exc
        ts = obj.last_modified.timestamp() if obj.last_modified else 0.0
        return FileInfo(
            path=f"{self.name}://{key}",
            key=key,
            size=int(obj.size or 0),
            modified_at=ts,
            etag=str(obj.etag or ""),
            content_type=str(obj.content_type or ""),
        )

    def _client_list(self, prefix: str) -> list[FileInfo]:
        client = self._ensure_client()
        results: list[FileInfo] = []
        try:
            for obj in client.list_objects(self.bucket, prefix=prefix, recursive=False):
                ts = obj.last_modified.timestamp() if obj.last_modified else 0.0
                results.append(
                    FileInfo(
                        path=f"{self.name}://{obj.object_name}",
                        key=obj.object_name,
                        size=int(obj.size or 0),
                        modified_at=ts,
                        is_dir=obj.is_dir,
                        etag=str(obj.etag or ""),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"MinIO list failed for {prefix!r}: {exc}") from exc
        return results


# ---------------------------------------------------------------------------
# Quota manager
# ---------------------------------------------------------------------------


class QuotaManager:
    """Per-namespace storage quota tracking.

    Usage is maintained as an in-memory counter updated on every write /
    delete through the :class:`StorageManager`. Call :meth:`recompute` to
    resync the counter from a backend when external processes may have
    modified data.
    """

    def __init__(self) -> None:
        self._specs: dict[str, QuotaSpec] = {}
        self._usage: dict[str, QuotaUsage] = {}
        self._lock = threading.RLock()

    def set_quota(self, namespace: str, *, max_bytes: int = 0, max_files: int = 0) -> QuotaSpec:
        """Set (or replace) the quota for ``namespace``."""

        spec = QuotaSpec(
            namespace=namespace, max_bytes=max_bytes, max_files=max_files
        )
        with self._lock:
            self._specs[namespace] = spec
            self._usage.setdefault(namespace, QuotaUsage(namespace=namespace))
        logger.info(
            "Quota set for %s: max_bytes=%d max_files=%d",
            namespace,
            max_bytes,
            max_files,
        )
        return spec

    def get_quota(self, namespace: str) -> QuotaSpec | None:
        with self._lock:
            return self._specs.get(namespace)

    def remove_quota(self, namespace: str) -> bool:
        with self._lock:
            existed = self._specs.pop(namespace, None) is not None
            self._usage.pop(namespace, None)
            return existed

    def usage(self, namespace: str) -> QuotaUsage:
        with self._lock:
            return self._usage.get(
                namespace, QuotaUsage(namespace=namespace)
            ).model_copy()

    def check(
        self, namespace: str, *, extra_bytes: int = 0, extra_files: int = 0
    ) -> bool:
        """Return True if a write of the given size/count is allowed."""

        with self._lock:
            spec = self._specs.get(namespace)
            if spec is None:
                return True
            usage = self._usage.get(namespace, QuotaUsage(namespace=namespace))
            return not usage.would_exceed(
                spec, extra_bytes=extra_bytes, extra_files=extra_files
            )

    def record(
        self, namespace: str, *, delta_bytes: int, delta_files: int
    ) -> QuotaUsage:
        """Apply a usage delta to ``namespace`` and return the new usage.

        Clamps at zero so deletes of already-removed files don't drift
        negative.
        """

        with self._lock:
            usage = self._usage.setdefault(
                namespace, QuotaUsage(namespace=namespace)
            )
            usage.used_bytes = max(0, usage.used_bytes + delta_bytes)
            usage.used_files = max(0, usage.used_files + delta_files)
            return usage.model_copy()

    def recompute(self, namespace: str, backend: StorageBackend) -> QuotaUsage:
        """Recompute usage for ``namespace`` by scanning ``backend``.

        Walks every object under the backend and sums sizes / counts.
        Expensive for large backends; call sparingly.
        """

        total_bytes = 0
        total_files = 0
        stack: list[str] = [""]
        while stack:
            prefix = stack.pop()
            for entry in backend.list_dir(prefix):
                if entry.is_dir:
                    stack.append(entry.key)
                else:
                    total_bytes += entry.size
                    total_files += 1
        with self._lock:
            usage = self._usage.setdefault(
                namespace, QuotaUsage(namespace=namespace)
            )
            usage.used_bytes = total_bytes
            usage.used_files = total_files
            return usage.model_copy()

    def all_quotas(self) -> list[QuotaSpec]:
        with self._lock:
            return list(self._specs.values())


# ---------------------------------------------------------------------------
# Storage manager (façade)
# ---------------------------------------------------------------------------


class StorageManager:
    """Unified file API over mounted backends with quota enforcement.

    A *mount table* maps a logical scheme (``local`` / ``nas`` / ``s3`` /
    ``minio``) to a :class:`StorageBackend`. Logical paths take the form
    ``<scheme>://<key>``; the scheme selects the backend and ``<key>`` is
    passed to it. An optional *namespace* (defaulting to the scheme) is
    used for quota tracking.

    Example:

        >>> mgr = StorageManager()
        >>> mgr.mount("local", LocalStorageBackend("local", "/tmp/store"))
        >>> mgr.write("local://reports/q1.txt", b"hello")  # doctest: +SKIP
        5
    """

    def __init__(self, quota_manager: QuotaManager | None = None) -> None:
        self._backends: dict[str, StorageBackend] = {}
        self._quota = quota_manager or QuotaManager()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Mount table
    # ------------------------------------------------------------------

    def mount(self, scheme: str, backend: StorageBackend) -> None:
        """Register ``backend`` under ``scheme`` (e.g. ``"local"``)."""

        if not scheme:
            raise StorageError("scheme must not be empty")
        with self._lock:
            self._backends[scheme] = backend
        logger.info("Mounted %s backend as %s://", backend.backend_type.value, scheme)

    def unmount(self, scheme: str) -> StorageBackend | None:
        """Remove a mounted backend; returns the removed backend or None."""

        with self._lock:
            return self._backends.pop(scheme, None)

    def backends(self) -> dict[str, StorageBackend]:
        """Return a copy of the mount table."""

        with self._lock:
            return dict(self._backends)

    @property
    def quotas(self) -> QuotaManager:
        """The quota manager backing this storage manager."""

        return self._quota

    def _resolve(self, logical_path: str) -> tuple[StorageBackend, str]:
        """Split ``logical_path`` into ``(backend, key)``.

        Accepts ``scheme://key`` or a bare ``key`` (resolved against the
        default ``local`` backend when present).

        For object-store schemes (``s3`` / ``minio``) the ``netloc`` is the
        bucket name (already captured in the backend config) and is dropped;
        only the path portion becomes the key. For filesystem schemes
        (``local`` / ``nas``) the ``netloc`` is the first path segment, so
        it is re-joined with the path to form the key.
        """

        if "://" in logical_path:
            parsed = urlparse(logical_path)
            scheme = parsed.scheme
            if scheme in ("s3", "minio"):
                # Bucket lives in netloc and is already part of backend config.
                key = parsed.path.lstrip("/")
            else:
                # local / nas: ``reports`` in ``local://reports/q1.txt`` is
                # the first path segment, not a host.
                if parsed.netloc:
                    key = (parsed.netloc + parsed.path).lstrip("/")
                else:
                    key = parsed.path.lstrip("/")
        else:
            scheme = "local"
            key = logical_path.lstrip("/")
        with self._lock:
            backend = self._backends.get(scheme)
        if backend is None:
            raise StorageError(f"no backend mounted for scheme {scheme!r}")
        return backend, key

    def _namespace_for(self, scheme: str, key: str) -> str:
        """Derive a quota namespace from scheme + top-level path segment."""

        top = key.split("/", 1)[0] if key else ""
        return f"{scheme}:{top}" if top else scheme

    # ------------------------------------------------------------------
    # File CRUD
    # ------------------------------------------------------------------

    def read(self, path: str) -> bytes:
        backend, key = self._resolve(path)
        return backend.read(key)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self.read(path).decode(encoding)

    def write(self, path: str, data: bytes) -> int:
        backend, key = self._resolve(path)
        scheme = self._scheme_of(path)
        namespace = self._namespace_for(scheme, key)
        if not self._quota.check(namespace, extra_bytes=len(data), extra_files=1):
            raise QuotaExceededError(
                namespace,
                f"quota exceeded for {namespace}: writing {len(data)} bytes",
            )
        written = backend.write(key, data)
        self._quota.record(
            namespace, delta_bytes=written, delta_files=1
        )
        return written

    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> int:
        return self.write(path, text.encode(encoding))

    def delete(self, path: str) -> bool:
        backend, key = self._resolve(path)
        scheme = self._scheme_of(path)
        namespace = self._namespace_for(scheme, key)
        try:
            info = backend.stat(key)
            size = info.size
            files = 0 if info.is_dir else 1
        except StorageError as exc:
            size = 0
            files = 0
            logger.debug("stat failed: %s", exc)
        removed = backend.delete(key)
        if removed:
            self._quota.record(
                namespace, delta_bytes=-size, delta_files=-files
            )
        return removed

    def exists(self, path: str) -> bool:
        backend, key = self._resolve(path)
        return backend.exists(key)

    def stat(self, path: str) -> FileInfo:
        backend, key = self._resolve(path)
        info = backend.stat(key)
        return info.model_copy(update={"path": path})

    def list_dir(self, path: str = "") -> list[FileInfo]:
        backend, key = self._resolve(path or f"{self._default_scheme()}://")
        return backend.list_dir(key)

    def mkdir(self, path: str) -> None:
        backend, key = self._resolve(path)
        backend.mkdir(key)

    def rename(self, src: str, dst: str) -> None:
        src_backend, src_key = self._resolve(src)
        dst_backend, dst_key = self._resolve(dst)
        if src_backend is not dst_backend:
            raise StorageError("cross-backend rename is not supported; use copy+delete")
        size = 0
        files = 0
        try:
            info = src_backend.stat(src_key)
            size = info.size
            files = 0 if info.is_dir else 1
        except StorageError as exc:
            logger.debug("stat failed during rename: %s", exc)
        src_backend.rename(src_key, dst_key)
        src_ns = self._namespace_for(self._scheme_of(src), src_key)
        dst_ns = self._namespace_for(self._scheme_of(dst), dst_key)
        if src_ns != dst_ns:
            self._quota.record(src_ns, delta_bytes=-size, delta_files=-files)
            self._quota.record(dst_ns, delta_bytes=size, delta_files=files)

    def copy(self, src: str, dst: str) -> None:
        src_backend, src_key = self._resolve(src)
        dst_backend, dst_key = self._resolve(dst)
        if src_backend is not dst_backend:
            data = src_backend.read(src_key)
            self.write(dst, data)
            return
        scheme = self._scheme_of(dst)
        namespace = self._namespace_for(scheme, dst_key)
        try:
            size = src_backend.stat(src_key).size
        except StorageError as exc:
            size = 0
            logger.debug("stat failed during copy: %s", exc)
        if not self._quota.check(namespace, extra_bytes=size, extra_files=1):
            raise QuotaExceededError(
                namespace,
                f"quota exceeded for {namespace}: copying {size} bytes",
            )
        src_backend.copy(src_key, dst_key)
        self._quota.record(namespace, delta_bytes=size, delta_files=1)

    def glob(self, pattern: str) -> list[FileInfo]:
        backend, key = self._resolve(pattern)
        return backend.glob(key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scheme_of(self, path: str) -> str:
        if "://" in path:
            return urlparse(path).scheme
        return "local"

    def _default_scheme(self) -> str:
        with self._lock:
            if "local" in self._backends:
                return "local"
            if self._backends:
                return next(iter(self._backends))
        return "local"

    def close(self) -> None:
        """Close every mounted backend."""

        with self._lock:
            backends = list(self._backends.values())
        for backend in backends:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("backend close error: %s", exc)


__all__ = [
    "BackendType",
    "FileInfo",
    "LocalStorageBackend",
    "MinioStorageBackend",
    "NASStorageBackend",
    "QuotaExceededError",
    "QuotaManager",
    "QuotaSpec",
    "QuotaUsage",
    "S3StorageBackend",
    "StorageBackend",
    "StorageError",
    "StorageManager",
    "utcnow",
]
