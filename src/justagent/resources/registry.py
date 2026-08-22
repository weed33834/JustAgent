"""Resource registry — catalog and discover enterprise hardware/software resources.

The registry is the single source of truth for every physical or logical
resource the JustAgent platform can schedule work onto: bare-metal and
virtual servers, network-attached storage, GPU clusters, network devices
and software licenses.

Design:

* :class:`ResourceType` / :class:`ResourceStatus` — typed enumerations.
* :class:`ResourceSpec` — hardware capability declaration (CPU, memory,
  disk, GPU, network). Used both to advertise a resource and to express
  what a task needs.
* :class:`ResourceLoad` — live utilisation snapshot used by the scheduler
  for load balancing.
* :class:`ResourceRecord` — a registered resource: identity, status,
  endpoint, capabilities, load and free-form metadata.
* :class:`ResourceRegistry` — thread-safe in-memory catalog supporting
  registration, tag/capability discovery, heartbeats and
  load-aware best-match selection.

The registry keeps no hard dependency on the scheduler or the monitor:
they read from it (and write load/heartbeat back) through its public API.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("justagent.resources")


class ResourceType(str, Enum):  # noqa: UP042 - match existing codebase style
    """The kind of enterprise resource a record describes."""

    SERVER = "server"
    STORAGE = "storage"
    GPU_CLUSTER = "gpu_cluster"
    NETWORK_DEVICE = "network_device"
    SOFTWARE_LICENSE = "software_license"


class ResourceStatus(str, Enum):  # noqa: UP042
    """Operational state of a registered resource."""

    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


#: Resources in one of these states are considered schedulable.
_SCHEDULABLE_STATES: frozenset[ResourceStatus] = frozenset(
    {ResourceStatus.ONLINE, ResourceStatus.DEGRADED}
)


class ResourceSpec(BaseModel):
    """Hardware capability declaration for a resource.

    ``cpu_cores`` / ``memory_gb`` / ``disk_gb`` describe the total capacity
    the resource advertises. A task expresses *minimum* requirements which
    are checked against these values via :meth:`satisfies`.
    """

    cpu_cores: int = 0
    memory_gb: float = 0.0
    disk_gb: float = 0.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    network_bandwidth_mbps: int = 0
    architecture: str = ""
    tags: list[str] = Field(default_factory=list)

    def satisfies(
        self,
        *,
        min_cpu: int = 0,
        min_memory: float = 0.0,
        min_disk: float = 0.0,
        min_gpu: int = 0,
        min_gpu_memory: float = 0.0,
        min_bandwidth: int = 0,
        required_tags: list[str] | None = None,
        architecture: str = "",
    ) -> bool:
        """Return True if this spec meets every minimum requirement.

        All comparisons are ``>=`` so a resource with exactly the requested
        capacity still qualifies. ``required_tags`` must be a subset of the
        spec's tags. An empty ``architecture`` requirement matches anything.
        """

        if self.cpu_cores < min_cpu:
            return False
        if self.memory_gb + 1e-9 < min_memory:
            return False
        if self.disk_gb + 1e-9 < min_disk:
            return False
        if self.gpu_count < min_gpu:
            return False
        if self.gpu_memory_gb + 1e-9 < min_gpu_memory:
            return False
        if self.network_bandwidth_mbps < min_bandwidth:
            return False
        if architecture and self.architecture and self.architecture != architecture:
            return False
        if required_tags:
            have = set(self.tags)
            if not set(required_tags).issubset(have):
                return False
        return True


class ResourceLoad(BaseModel):
    """Live utilisation snapshot used for load balancing.

    The scheduler combines these into a single 0–100 :meth:`score` and
    picks the lowest-scoring eligible resource. ``running_tasks`` is an
    integer weight so a busy-by-count resource is deprioritised even when
    its percentage gauges are low.
    """

    running_tasks: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    updated_at: float = Field(default_factory=time.time)

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp a percentage into the 0–100 range."""

        if value < 0.0:
            return 0.0
        if value > 100.0:
            return 100.0
        return value

    def score(self) -> float:
        """Composite load score in 0–100; higher means more loaded.

        CPU/memory each contribute 35 %, GPU 20 % (resources without GPUs
        report 0 GPU usage and so are unaffected), and each running task
        adds a fixed 1 % so headroom is visible on otherwise-idle hosts.
        """

        cpu = self._clamp(self.cpu_usage_percent)
        mem = self._clamp(self.memory_usage_percent)
        gpu = self._clamp(self.gpu_usage_percent)
        return cpu * 0.35 + mem * 0.35 + gpu * 0.20 + float(self.running_tasks)


class ResourceRecord(BaseModel):
    """A registered enterprise resource.

    Attributes:
        id: Stable unique identifier (auto-generated UUID4 hex when omitted).
        name: Human-readable name; must be unique within the registry.
        type: The :class:`ResourceType` of this resource.
        status: Current :class:`ResourceStatus`.
        endpoint: Connection string or URL (e.g. ``ssh://host``, ``s3://bucket``).
        address: Low-level network address (IP/FQDN) for monitoring.
        capabilities: Advertised :class:`ResourceSpec`.
        tags: Free-form labels for discovery filters.
        metadata: Arbitrary structured metadata (vendor, model, location...).
        owner: Team or user accountable for the resource.
        registered_at: Unix timestamp of registration.
        last_heartbeat: Unix timestamp of the most recent heartbeat (0 = never).
        load: Most recent :class:`ResourceLoad` snapshot.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    type: ResourceType
    status: ResourceStatus = ResourceStatus.UNKNOWN
    endpoint: str = ""
    address: str = ""
    capabilities: ResourceSpec = Field(default_factory=ResourceSpec)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    owner: str = ""
    registered_at: float = Field(default_factory=time.time)
    last_heartbeat: float = 0.0
    load: ResourceLoad = Field(default_factory=ResourceLoad)

    def is_schedulable(self) -> bool:
        """True if the resource is in a state that accepts new work."""

        return self.status in _SCHEDULABLE_STATES

    def matches(
        self,
        *,
        type: ResourceType | None = None,
        status: ResourceStatus | None = None,
        tags: list[str] | None = None,
        name_contains: str = "",
    ) -> bool:
        """Return True if this record matches all provided filters.

        A ``None``/empty filter is treated as "do not filter on this field".
        ``tags`` uses subset semantics: the record must carry every requested
        tag (additional tags are allowed).
        """

        if type is not None and self.type is not type:
            return False
        if status is not None and self.status is not status:
            return False
        if tags and not set(tags).issubset(set(self.tags)):
            return False
        return not (name_contains and name_contains.lower() not in self.name.lower())


class ResourceRegistryError(Exception):
    """Raised for invalid registry operations (duplicate name, missing id...)."""


class ResourceRegistry:
    """Thread-safe in-memory catalog of enterprise resources.

    The registry stores records by id and enforces name uniqueness. All
    mutating and querying operations take a lock so the registry can be
    shared safely across the scheduler, monitor and CLI threads.

    Example:

        >>> registry = ResourceRegistry()
        >>> record = registry.register(ResourceRecord(
        ...     name="gpu-node-1",
        ...     type=ResourceType.GPU_CLUSTER,
        ...     status=ResourceStatus.ONLINE,
        ...     capabilities=ResourceSpec(cpu_cores=32, gpu_count=8),
        ...     tags=["a100", "training"],
        ... ))
        >>> best = registry.best_match(min_gpu=4, required_tags=["training"])
        >>> best is not None and best.name
        'gpu-node-1'
    """

    def __init__(self) -> None:
        self._by_id: dict[str, ResourceRecord] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, record: ResourceRecord) -> ResourceRecord:
        """Add a new resource; raises if the name is already taken.

        Returns the record as stored (with a generated id filled in when the
        caller omitted one).
        """

        with self._lock:
            if self._name_taken(record.name, exclude_id=record.id):
                raise ResourceRegistryError(
                    f"a resource named {record.name!r} is already registered"
                )
            self._by_id[record.id] = record
            logger.info(
                "Registered resource %s (%s, id=%s, status=%s)",
                record.name,
                record.type.value,
                record.id,
                record.status.value,
            )
            return record

    def unregister(self, resource_id: str) -> ResourceRecord | None:
        """Remove a resource by id; return the removed record or None."""

        with self._lock:
            record = self._by_id.pop(resource_id, None)
            if record is not None:
                logger.info("Unregistered resource %s (id=%s)", record.name, record.id)
            return record

    def unregister_by_name(self, name: str) -> ResourceRecord | None:
        """Remove a resource by name; return the removed record or None."""

        with self._lock:
            for rid, record in self._by_id.items():
                if record.name == name:
                    return self.unregister(rid)
            return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, resource_id: str) -> ResourceRecord | None:
        """Return a record by id (a copy), or None if absent."""

        with self._lock:
            return self._by_id.get(resource_id)

    def get_by_name(self, name: str) -> ResourceRecord | None:
        """Return a record by name (a copy), or None if absent."""

        with self._lock:
            for record in self._by_id.values():
                if record.name == name:
                    return record
            return None

    def list_all(self) -> list[ResourceRecord]:
        """Return all records (copies), ordered by name."""

        with self._lock:
            return sorted(self._by_id.values(), key=lambda r: r.name)

    def discover(
        self,
        *,
        type: ResourceType | None = None,
        status: ResourceStatus | None = None,
        tags: list[str] | None = None,
        name_contains: str = "",
        schedulable_only: bool = False,
    ) -> list[ResourceRecord]:
        """Discover resources matching every provided filter.

        When ``schedulable_only`` is True, only resources whose
        :meth:`ResourceRecord.is_schedulable` returns True are returned.
        """

        with self._lock:
            results = [
                record
                for record in self._by_id.values()
                if record.matches(
                    type=type,
                    status=status,
                    tags=tags,
                    name_contains=name_contains,
                )
            ]
        if schedulable_only:
            results = [r for r in results if r.is_schedulable()]
        return sorted(results, key=lambda r: r.name)

    def best_match(
        self,
        *,
        min_cpu: int = 0,
        min_memory: float = 0.0,
        min_disk: float = 0.0,
        min_gpu: int = 0,
        min_gpu_memory: float = 0.0,
        min_bandwidth: int = 0,
        required_tags: list[str] | None = None,
        architecture: str = "",
        type: ResourceType | None = None,
    ) -> ResourceRecord | None:
        """Pick the least-loaded schedulable resource that satisfies the requirements.

        Filters out non-schedulable resources, then resources whose
        :class:`ResourceSpec` does not meet the minimums, then returns the
        one with the lowest :meth:`ResourceLoad.score` (ties broken by name
        for deterministic output). Returns None when nothing qualifies.
        """

        with self._lock:
            candidates = [
                record
                for record in self._by_id.values()
                if record.is_schedulable()
                and (type is None or record.type is type)
                and record.capabilities.satisfies(
                    min_cpu=min_cpu,
                    min_memory=min_memory,
                    min_disk=min_disk,
                    min_gpu=min_gpu,
                    min_gpu_memory=min_gpu_memory,
                    min_bandwidth=min_bandwidth,
                    required_tags=required_tags,
                    architecture=architecture,
                )
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda r: (r.load.score(), r.name))
        return candidates[0]

    # ------------------------------------------------------------------
    # State mutation
    # ------------------------------------------------------------------

    def update_status(self, resource_id: str, status: ResourceStatus) -> ResourceRecord | None:
        """Set the status of a resource; return the updated record or None."""

        with self._lock:
            record = self._by_id.get(resource_id)
            if record is None:
                return None
            updated = record.model_copy(update={"status": status})
            self._by_id[resource_id] = updated
            logger.info("Resource %s status -> %s", record.name, status.value)
            return updated

    def heartbeat(self, resource_id: str, at: float | None = None) -> ResourceRecord | None:
        """Refresh the last-heartbeat timestamp for a resource.

        A heartbeat implicitly marks the resource ONLINE if it was UNKNOWN.
        Returns the updated record or None when the id is unknown.
        """

        ts = time.time() if at is None else at
        with self._lock:
            record = self._by_id.get(resource_id)
            if record is None:
                return None
            new_status = record.status
            if new_status is ResourceStatus.UNKNOWN:
                new_status = ResourceStatus.ONLINE
            updated = record.model_copy(update={"last_heartbeat": ts, "status": new_status})
            self._by_id[resource_id] = updated
            return updated

    def update_load(self, resource_id: str, load: ResourceLoad) -> ResourceRecord | None:
        """Replace the load snapshot for a resource (called by the monitor)."""

        with self._lock:
            record = self._by_id.get(resource_id)
            if record is None:
                return None
            updated = record.model_copy(update={"load": load})
            self._by_id[resource_id] = updated
            return updated

    def mark_stale(self, max_age_seconds: float, at: float | None = None) -> list[str]:
        """Mark resources whose heartbeat is older than ``max_age_seconds`` as OFFLINE.

        Resources that have never heartbeated (``last_heartbeat == 0``) are
        left untouched. Returns the ids of the resources that were flipped.
        """

        now = time.time() if at is None else at
        flipped: list[str] = []
        with self._lock:
            for rid, record in list(self._by_id.items()):
                if record.last_heartbeat <= 0.0:
                    continue
                if (
                    now - record.last_heartbeat > max_age_seconds
                    and record.status is not ResourceStatus.OFFLINE
                ):
                    self._by_id[rid] = record.model_copy(update={"status": ResourceStatus.OFFLINE})
                    flipped.append(rid)
                    logger.warning(
                        "Resource %s marked OFFLINE (stale heartbeat, age=%.0fs)",
                        record.name,
                        now - record.last_heartbeat,
                    )
        return flipped

    # ------------------------------------------------------------------
    # Aggregate reporting
    # ------------------------------------------------------------------

    def count(self, type: ResourceType | None = None) -> int:
        """Return the number of registered resources, optionally by type."""

        with self._lock:
            if type is None:
                return len(self._by_id)
            return sum(1 for r in self._by_id.values() if r.type is type)

    def count_by_status(self) -> dict[ResourceStatus, int]:
        """Return a mapping of status -> count across all resources."""

        counts: dict[ResourceStatus, int] = dict.fromkeys(ResourceStatus, 0)
        with self._lock:
            for record in self._by_id.values():
                counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for dashboards / CLI output."""

        with self._lock:
            by_type: dict[str, int] = {}
            for record in self._by_id.values():
                key = record.type.value
                by_type[key] = by_type.get(key, 0) + 1
            status_counts = self.count_by_status()
            return {
                "total": len(self._by_id),
                "by_type": by_type,
                "by_status": {s.value: c for s, c in status_counts.items()},
                "schedulable": sum(1 for r in self._by_id.values() if r.is_schedulable()),
            }

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialise all records to a list of JSON-safe dicts (for persistence)."""

        with self._lock:
            return [record.model_dump(mode="json") for record in self._by_id.values()]

    def load_from_dict(self, records: list[dict[str, Any]]) -> int:
        """Bulk-load records from a serialised list; returns the count loaded.

        Existing records with matching ids are replaced. Invalid entries are
        skipped (and logged) rather than aborting the whole import.
        """

        loaded = 0
        with self._lock:
            for item in records:
                try:
                    record = ResourceRecord.model_validate(item)
                except Exception as exc:  # noqa: BLE001 - best-effort import
                    logger.warning("Skipping invalid resource record: %s", exc)
                    continue
                self._by_id[record.id] = record
                loaded += 1
        logger.info("Loaded %d resource record(s) from dict", loaded)
        return loaded

    def clear(self) -> None:
        """Remove every registered resource."""

        with self._lock:
            self._by_id.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _name_taken(self, name: str, exclude_id: str = "") -> bool:
        """Return True if ``name`` is in use by a record other than ``exclude_id``."""

        for rid, record in self._by_id.items():
            if rid == exclude_id:
                continue
            if record.name == name:
                return True
        return False


__all__ = [
    "ResourceLoad",
    "ResourceRecord",
    "ResourceRegistry",
    "ResourceRegistryError",
    "ResourceSpec",
    "ResourceStatus",
    "ResourceType",
]
