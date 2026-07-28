"""Hardware resource orchestration for the JustAgent platform.

This package unifies five concerns that the platform needs to run work
across enterprise infrastructure:

* :mod:`justagent.resources.registry` — catalog and discover servers,
  storage, GPU clusters, network devices and software licenses.
* :mod:`justagent.resources.scheduler` — priority-queue, preemptive task
  scheduler that matches tasks to registry resources and tracks execution
  (timeout, retry, cancellation).
* :mod:`justagent.resources.storage` — unified file API over local / NAS /
  S3 / MinIO backends with quota management and path abstraction.
* :mod:`justagent.resources.database` — database gateway with connection
  pooling, read/write splitting and query caching (SQLite default, with
  lazy MySQL / PostgreSQL / MongoDB / Redis drivers).
* :mod:`justagent.resources.monitor` — psutil-backed metrics collection,
  configurable threshold alerts, health checks and utilisation reporting.

All public types are re-exported here so callers can do::

    from justagent.resources import (
        ResourceRegistry, TaskScheduler, StorageManager,
        DatabaseGateway, ResourceMonitor,
    )
"""

from __future__ import annotations

from justagent.resources.database import (
    ConnectionConfig,
    ConnectionPool,
    ConnectionRole,
    DatabaseBackend,
    DatabaseDriver,
    DatabaseError,
    DatabaseGateway,
    MongoDBBackend,
    MySQLBackend,
    NoSQLBackend,
    PoolExhaustedError,
    PostgresBackend,
    QueryResult,
    RedisBackend,
    SqliteBackend,
    create_backend,
)
from justagent.resources.monitor import (
    DEFAULT_HISTORY,
    DEFAULT_INTERVAL,
    Alert,
    AlertCallback,
    AlertLevel,
    AlertThreshold,
    DiskIO,
    DiskUsage,
    GpuStats,
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    MetricSnapshot,
    MetricType,
    MonitorError,
    NetworkIO,
    ResourceMonitor,
    default_thresholds,
)
from justagent.resources.registry import (
    ResourceLoad,
    ResourceRecord,
    ResourceRegistry,
    ResourceRegistryError,
    ResourceSpec,
    ResourceStatus,
    ResourceType,
)
from justagent.resources.scheduler import (
    DEFAULT_MAX_CONCURRENT_PER_RESOURCE,
    NO_TIMEOUT,
    ResourceRequirements,
    RetryPolicy,
    SchedulerError,
    Task,
    TaskPriority,
    TaskResult,
    TaskRunner,
    TaskScheduler,
    TaskStatus,
    default_runner,
)
from justagent.resources.storage import (
    BackendType,
    FileInfo,
    LocalStorageBackend,
    MinioStorageBackend,
    NASStorageBackend,
    QuotaExceededError,
    QuotaManager,
    QuotaSpec,
    QuotaUsage,
    S3StorageBackend,
    StorageBackend,
    StorageError,
    StorageManager,
)

__all__ = [
    # Registry
    "ResourceLoad",
    "ResourceRecord",
    "ResourceRegistry",
    "ResourceRegistryError",
    "ResourceSpec",
    "ResourceStatus",
    "ResourceType",
    # Scheduler
    "DEFAULT_MAX_CONCURRENT_PER_RESOURCE",
    "NO_TIMEOUT",
    "ResourceRequirements",
    "RetryPolicy",
    "SchedulerError",
    "Task",
    "TaskPriority",
    "TaskResult",
    "TaskRunner",
    "TaskScheduler",
    "TaskStatus",
    "default_runner",
    # Storage
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
    # Database
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
    # Monitor
    "DEFAULT_HISTORY",
    "DEFAULT_INTERVAL",
    "Alert",
    "AlertCallback",
    "AlertLevel",
    "AlertThreshold",
    "DiskIO",
    "DiskUsage",
    "GpuStats",
    "HealthCheck",
    "HealthCheckResult",
    "HealthStatus",
    "MetricSnapshot",
    "MetricType",
    "MonitorError",
    "NetworkIO",
    "ResourceMonitor",
    "default_thresholds",
]
