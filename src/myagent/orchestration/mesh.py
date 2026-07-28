"""Agent Mesh networking — distributed multi-agent discovery, routing and messaging.

Provides an in-process mesh fabric over which cooperating agents register
themselves, advertise :class:`AgentCapability` sets, exchange
:class:`MeshMessage` packets and keep each other alive via heartbeats. Routing
supports direct (point-to-point), broadcast and capability-based delivery,
governed by a pluggable :class:`MeshTopology`.

Design:

* :class:`MeshNodeStatus` — operational state of a node.
* :class:`AgentCapability` — the skills a node advertises (reasoning, code
  generation, data analysis, ...).
* :class:`MeshTopology` — STAR / MESH / TREE / RING routing shapes.
* :class:`MeshNode` — a registered agent: identity, endpoint, capabilities,
  status, heartbeat and metadata.
* :class:`MeshMessage` — a routed packet with optional acknowledgement.
* :class:`MeshRouter` — computes the recipient set for a message and finds
  the best node for a given capability, maintaining a routing table.
* :class:`AgentMesh` — async, thread-safe registry + delivery engine.

Delivery is pluggable: each node may register an async message handler; nodes
without a handler buffer messages in an in-memory inbox that can be drained
later. ``requires_ack`` messages await the handler and report acknowledgement.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("myagent.orchestration.mesh")


class MeshError(Exception):
    """Raised for invalid mesh operations (unknown node, routing failure, ...)."""


class MeshNodeStatus(str, Enum):  # noqa: UP042 - match existing codebase style
    """Operational state of a :class:`MeshNode`."""

    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    DEGRADED = "degraded"


class AgentCapability(str, Enum):  # noqa: UP042
    """The skills a mesh node can advertise.

    Capabilities drive capability-based routing
    (:meth:`MeshRouter.find_best_node`) and task delegation in
    :mod:`myagent.orchestration.coordinator`.
    """

    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    COMMUNICATION = "communication"
    MONITORING = "monitoring"
    PLANNING = "planning"
    EXECUTION = "execution"


class MeshTopology(str, Enum):  # noqa: UP042
    """Logical routing shape of the mesh."""

    STAR = "star"
    MESH = "mesh"
    TREE = "tree"
    RING = "ring"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class MeshNode(BaseModel):
    """A registered agent in the mesh.

    Attributes:
        id: Unique node identifier (auto-generated UUID4 hex when omitted).
        name: Human-readable name.
        address: Network address (IP / FQDN / unix socket path).
        port: TCP/UDP port (0 when not applicable).
        capabilities: The :class:`AgentCapability` set this node advertises.
        status: Current :class:`MeshNodeStatus`.
        last_heartbeat: Unix timestamp of the most recent heartbeat (0 = never).
        metadata: Arbitrary structured metadata (parent id, region, load, ...).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    address: str = ""
    port: int = 0
    capabilities: set[AgentCapability] = Field(default_factory=set)
    status: MeshNodeStatus = MeshNodeStatus.ONLINE
    last_heartbeat: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def has_capability(self, capability: AgentCapability) -> bool:
        """True when this node advertises *capability*."""

        return capability in self.capabilities

    def is_reachable(self) -> bool:
        """True when the node can receive messages (ONLINE, BUSY or DEGRADED)."""

        return self.status in (
            MeshNodeStatus.ONLINE,
            MeshNodeStatus.BUSY,
            MeshNodeStatus.DEGRADED,
        )


class MeshMessage(BaseModel):
    """A routed packet exchanged between mesh nodes.

    Attributes:
        id: Unique message identifier.
        source: ID of the sending node.
        target: ID of the recipient node, ``None`` for broadcast, or a
            capability specifier of the form ``"cap:<capability>"`` for
            capability-based routing.
        message_type: Application-level message type (e.g. ``"task"``,
            ``"query"``, ``"result"``).
        payload: Arbitrary structured payload.
        timestamp: UTC creation timestamp.
        requires_ack: When True the sender expects an acknowledgement; the
            mesh awaits the recipient handler and reports ack status.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source: str
    target: str | None = None
    message_type: str = "message"
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    requires_ack: bool = False


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

#: Signature of an async per-node message handler.
MessageHandler = Callable[[MeshMessage], Awaitable[dict[str, Any] | None]]

#: Prefix marking a capability-based target specifier.
_CAPABILITY_PREFIX = "cap:"


class MeshRouter:
    """Computes recipients for a message and resolves capability requests.

    The router maintains a routing table mapping node ids to nodes and
    capability values to the set of node ids that advertise them. It is kept
    in sync by :meth:`maintain_routing_table`, which the mesh calls whenever
    nodes are registered or deregistered.

    Attributes:
        topology: The :class:`MeshTopology` governing broadcast/ring routing.
        hub_id: The central hub node id for STAR topology (optional).
    """

    def __init__(
        self,
        *,
        topology: MeshTopology = MeshTopology.MESH,
        hub_id: str = "",
    ) -> None:
        self.topology = topology
        self.hub_id = hub_id
        self._nodes: dict[str, MeshNode] = {}
        self._capability_index: dict[AgentCapability, list[str]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Table maintenance
    # ------------------------------------------------------------------

    def set_nodes(self, nodes: dict[str, MeshNode]) -> None:
        """Replace the router's node map and rebuild the routing table."""

        with self._lock:
            self._nodes = dict(nodes)
        self.maintain_routing_table()

    def maintain_routing_table(self) -> None:
        """Rebuild the capability -> [node_id] index from the current nodes."""

        with self._lock:
            index: dict[AgentCapability, list[str]] = {}
            for node in self._nodes.values():
                for capability in node.capabilities:
                    index.setdefault(capability, []).append(node.id)
            # Sort each capability list deterministically.
            for cap in index:
                index[cap].sort()
            self._capability_index = index

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        message: MeshMessage,
        *,
        exclude_source: bool = True,
        load: dict[str, int] | None = None,
    ) -> list[str]:
        """Return the recipient node ids for *message*.

        * Direct: ``message.target`` is a node id -> ``[target]`` (if reachable).
        * Broadcast: ``message.target`` is ``None`` -> all reachable nodes,
          shaped by :attr:`topology` (with the source excluded unless
          *exclude_source* is False).
        * Capability: ``message.target`` is ``"cap:<capability>"`` -> the best
          node advertising that capability (via :meth:`find_best_node`).

        Args:
            message: The message to route.
            exclude_source: When ``True`` (default), exclude the sending node
                from broadcast recipients.
            load: Optional in-flight message counts per node, used to pick the
                least-loaded node for capability-based routing.
        """

        with self._lock:
            nodes = dict(self._nodes)
        target = message.target
        if target is None:
            return self._broadcast_recipients(nodes, message.source, exclude_source)
        if target.startswith(_CAPABILITY_PREFIX):
            cap_name = target[len(_CAPABILITY_PREFIX) :].strip()
            try:
                capability = AgentCapability(cap_name)
            except ValueError:
                logger.warning("Unknown capability in target %r", target)
                return []
            best = self.find_best_node(capability, nodes=nodes, load=load)
            return [best] if best else []
        # Direct delivery.
        node = nodes.get(target)
        if node is None:
            logger.warning("Direct route to unknown node %r", target)
            return []
        if not node.is_reachable():
            logger.debug("Direct route target %r is not reachable (%s)", target, node.status.value)
            return []
        return [target]

    def _broadcast_recipients(
        self,
        nodes: dict[str, MeshNode],
        source: str,
        exclude_source: bool,
    ) -> list[str]:
        """Compute the broadcast recipient set for the active topology."""

        reachable = [nid for nid, node in nodes.items() if node.is_reachable()]
        if self.topology is MeshTopology.STAR and self.hub_id:
            # STAR broadcasts are mediated by the hub; the hub must be present.
            if self.hub_id in nodes and nodes[self.hub_id].is_reachable():
                # All reachable leaves plus the hub.
                recipients = [self.hub_id]
                recipients.extend(nid for nid in reachable if nid != self.hub_id)
            else:
                recipients = reachable
        elif self.topology is MeshTopology.RING:
            # Order the ring by name for determinism.
            recipients = sorted(reachable)
        elif self.topology is MeshTopology.TREE:
            # Propagate from the root downward; root first (sorted).
            recipients = sorted(reachable)
        else:  # MESH
            recipients = sorted(reachable)
        if exclude_source and source in recipients:
            recipients.remove(source)
        return recipients

    def find_best_node(
        self,
        capability: AgentCapability,
        *,
        nodes: dict[str, MeshNode] | None = None,
        load: dict[str, int] | None = None,
    ) -> str | None:
        """Return the id of the best reachable node advertising *capability*.

        "Best" is the least-loaded reachable node (fewest in-flight messages)
        with the requested capability; ties are broken by name for
        determinism. Returns ``None`` when no qualifying node exists.
        """

        with self._lock:
            node_map = dict(nodes) if nodes is not None else dict(self._nodes)
            candidates = list(self._capability_index.get(capability, []))
        load_map = load or {}
        # Prefer STAR hub when it has the capability.
        if self.topology is MeshTopology.STAR and self.hub_id:
            hub = node_map.get(self.hub_id)
            if hub is not None and hub.is_reachable() and capability in hub.capabilities:
                return self.hub_id
        eligible = [
            nid
            for nid in candidates
            if nid in node_map
            and node_map[nid].is_reachable()
            and capability in node_map[nid].capabilities
        ]
        if not eligible:
            # Fallback: scan all nodes (index may be stale before rebuild).
            eligible = [
                nid
                for nid, node in node_map.items()
                if node.is_reachable() and capability in node.capabilities
            ]
        if not eligible:
            return None
        eligible.sort(key=lambda nid: (load_map.get(nid, 0), node_map[nid].name))
        return eligible[0]

    def get_topology(self) -> MeshTopology:
        """Return the configured :class:`MeshTopology`."""

        return self.topology


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


class AgentMesh:
    """Async, thread-safe agent mesh registry and message-delivery engine.

    Nodes register with an optional async message handler. Messages are routed
    by the internal :class:`MeshRouter` (direct, broadcast or capability-based)
    and delivered concurrently. Nodes without a handler buffer messages in an
    in-memory inbox (:meth:`drain_inbox`). Stale heartbeats can be reaped with
    :meth:`mark_stale`.

    Example::

        mesh = AgentMesh(topology=MeshTopology.MESH)
        await mesh.register_node(
            MeshNode(name="reasoner", capabilities={AgentCapability.REASONING}),
            handler=my_async_handler,
        )
        msg = MeshMessage(source="coordinator", target="cap:reasoning",
                          message_type="task", payload={"question": "why?"})
        results = await mesh.send_message("cap:reasoning", msg)
    """

    def __init__(
        self,
        *,
        topology: MeshTopology = MeshTopology.MESH,
        hub_id: str = "",
        heartbeat_timeout: float = 60.0,
    ) -> None:
        self._nodes: dict[str, MeshNode] = {}
        self._handlers: dict[str, MessageHandler] = {}
        self._inboxes: dict[str, asyncio.Queue[MeshMessage]] = {}
        self._load: dict[str, int] = {}
        self._router = MeshRouter(topology=topology, hub_id=hub_id)
        self._heartbeat_timeout = heartbeat_timeout
        self._lock = threading.RLock()
        self._delivery_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Node registration
    # ------------------------------------------------------------------

    async def register_node(
        self,
        node: MeshNode,
        *,
        handler: MessageHandler | None = None,
    ) -> MeshNode:
        """Register a node and optional message handler. Returns the node."""

        with self._lock:
            if node.id in self._nodes:
                raise MeshError(f"node {node.id!r} is already registered")
            if node.status is MeshNodeStatus.ONLINE:
                node.last_heartbeat = time.time()
            self._nodes[node.id] = node
            self._inboxes[node.id] = asyncio.Queue()
            self._load[node.id] = 0
            if handler is not None:
                self._handlers[node.id] = handler
            self._router.set_nodes(self._nodes)
        logger.info(
            "Registered mesh node %s (%s, id=%s, capabilities=%s)",
            node.name,
            node.status.value,
            node.id,
            sorted(c.value for c in node.capabilities),
        )
        return node

    async def deregister_node(self, node_id: str) -> MeshNode | None:
        """Remove a node by id; return the removed node or ``None``."""

        with self._lock:
            node = self._nodes.pop(node_id, None)
            self._handlers.pop(node_id, None)
            self._inboxes.pop(node_id, None)
            self._load.pop(node_id, None)
            self._router.set_nodes(self._nodes)
        if node is not None:
            logger.info("Deregistered mesh node %s (id=%s)", node.name, node.id)
        return node

    async def get_node(self, node_id: str) -> MeshNode | None:
        """Return the node by id, or ``None``."""

        with self._lock:
            return self._nodes.get(node_id)

    async def list_nodes(self, *, status: MeshNodeStatus | None = None) -> list[MeshNode]:
        """List nodes, optionally filtered by status."""

        with self._lock:
            nodes = list(self._nodes.values())
        if status is not None:
            nodes = [n for n in nodes if n.status is status]
        return sorted(nodes, key=lambda n: n.name)

    async def discover_nodes(self, capability: AgentCapability) -> list[MeshNode]:
        """Return reachable nodes advertising *capability*."""

        with self._lock:
            nodes = [
                node
                for node in self._nodes.values()
                if node.is_reachable() and capability in node.capabilities
            ]
        return sorted(nodes, key=lambda n: (self._load.get(n.id, 0), n.name))

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def heartbeat(self, node_id: str) -> MeshNode | None:
        """Refresh a node's heartbeat and mark it ONLINE if reachable.

        Returns the updated node or ``None`` when the id is unknown.
        """

        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            node.last_heartbeat = time.time()
            if node.status in (MeshNodeStatus.OFFLINE, MeshNodeStatus.DEGRADED):
                node.status = MeshNodeStatus.ONLINE
            self._router.set_nodes(self._nodes)
            return node

    async def mark_stale(
        self, *, timeout: float | None = None, at: float | None = None
    ) -> list[str]:
        """Mark nodes with stale heartbeats as OFFLINE.

        Returns the ids of the nodes that were flipped. Nodes that have never
        heartbeated (``last_heartbeat == 0``) are left untouched.
        """

        max_age = timeout if timeout is not None else self._heartbeat_timeout
        now = time.time() if at is None else at
        flipped: list[str] = []
        with self._lock:
            for nid, node in self._nodes.items():
                if node.last_heartbeat <= 0.0:
                    continue
                if (
                    now - node.last_heartbeat > max_age
                    and node.status is not MeshNodeStatus.OFFLINE
                ):
                    node.status = MeshNodeStatus.OFFLINE
                    flipped.append(nid)
                    logger.warning(
                        "Mesh node %s marked OFFLINE (stale heartbeat, age=%.0fs)",
                        node.name,
                        now - node.last_heartbeat,
                    )
            if flipped:
                self._router.set_nodes(self._nodes)
        return flipped

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send_message(self, target: str, message: MeshMessage) -> list[dict[str, Any]]:
        """Route and deliver *message* to *target*.

        *target* is a node id (direct), ``None``/``"broadcast"`` (broadcast)
        or ``"cap:<capability>"`` (capability-based). Returns one delivery
        record per recipient.
        """

        if target in (None, "broadcast", "*"):
            return await self.broadcast(message)
        # Normalise capability specifiers onto the message target.
        if target.startswith(_CAPABILITY_PREFIX):
            message.target = target
        else:
            message.target = target
        return await self._deliver(message)

    async def broadcast(self, message: MeshMessage) -> list[dict[str, Any]]:
        """Deliver *message* to all reachable nodes (per topology)."""

        message.target = None
        return await self._deliver(message)

    async def _deliver(self, message: MeshMessage) -> list[dict[str, Any]]:
        """Route *message* and deliver to each recipient concurrently."""

        with self._lock:
            recipients = self._router.route(message, load=self._load)
            nodes = dict(self._nodes)
        if not recipients:
            logger.debug(
                "Message %s from %s had no recipients (target=%r)",
                message.id,
                message.source,
                message.target,
            )
            return []
        logger.debug(
            "Delivering message %s to %d recipient(s): %s",
            message.id,
            len(recipients),
            recipients,
        )
        # Bump load counts.
        with self._lock:
            for nid in recipients:
                self._load[nid] = self._load.get(nid, 0) + 1
        results = await asyncio.gather(
            *[self._deliver_to(nid, message) for nid in recipients],
            return_exceptions=True,
        )
        with self._lock:
            for nid in recipients:
                self._load[nid] = max(0, self._load.get(nid, 0) - 1)
        records: list[dict[str, Any]] = []
        for nid, outcome in zip(recipients, results, strict=True):
            node = nodes.get(nid)
            if isinstance(outcome, BaseException):
                records.append(
                    {
                        "node_id": nid,
                        "node_name": node.name if node else "",
                        "delivered": False,
                        "acked": False,
                        "error": str(outcome),
                    }
                )
            else:
                records.append(outcome)
        return records

    async def _deliver_to(self, node_id: str, message: MeshMessage) -> dict[str, Any]:
        """Deliver *message* to a single node via handler or inbox."""

        with self._lock:
            node = self._nodes.get(node_id)
            handler = self._handlers.get(node_id)
            inbox = self._inboxes.get(node_id)
        if node is None:
            return {"node_id": node_id, "delivered": False, "acked": False, "error": "unknown node"}
        if not node.is_reachable():
            return {
                "node_id": node_id,
                "node_name": node.name,
                "delivered": False,
                "acked": False,
                "error": f"node not reachable ({node.status.value})",
            }
        # Handler path.
        if handler is not None:
            try:
                response = await handler(message)
            except Exception as exc:  # noqa: BLE001
                return {
                    "node_id": node_id,
                    "node_name": node.name,
                    "delivered": False,
                    "acked": False,
                    "error": str(exc),
                }
            return {
                "node_id": node_id,
                "node_name": node.name,
                "delivered": True,
                "acked": message.requires_ack,
                "response": response,
            }
        # Inbox path (no handler): buffer the message.
        if inbox is not None:
            await inbox.put(message)
            return {
                "node_id": node_id,
                "node_name": node.name,
                "delivered": True,
                "acked": False,
                "buffered": True,
            }
        return {
            "node_id": node_id,
            "node_name": node.name,
            "delivered": False,
            "acked": False,
            "error": "no inbox available",
        }

    async def drain_inbox(self, node_id: str, *, timeout: float = 0.0) -> list[MeshMessage]:
        """Remove and return all buffered messages for *node_id*."""

        with self._lock:
            inbox = self._inboxes.get(node_id)
        if inbox is None:
            return []
        result: list[MeshMessage] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    result.append(inbox.get_nowait())
                except asyncio.QueueEmpty:
                    break
            else:
                try:
                    result.append(await asyncio.wait_for(inbox.get(), timeout=remaining))
                except TimeoutError:
                    break
        return result

    # ------------------------------------------------------------------
    # Routing access
    # ------------------------------------------------------------------

    def _route_message(self, message: MeshMessage) -> list[str]:
        """Return the recipient node ids for *message* (no delivery)."""

        with self._lock:
            return self._router.route(message, load=self._load)

    async def get_topology(self) -> MeshTopology:
        """Return the mesh's :class:`MeshTopology`."""

        return self._router.get_topology()

    def get_router(self) -> MeshRouter:
        """Return the underlying :class:`MeshRouter`."""

        return self._router

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of mesh state for dashboards."""

        with self._lock:
            by_status: dict[str, int] = {}
            for node in self._nodes.values():
                by_status[node.status.value] = by_status.get(node.status.value, 0) + 1
            return {
                "total_nodes": len(self._nodes),
                "by_status": by_status,
                "topology": self._router.topology.value,
                "in_flight": sum(self._load.values()),
            }


__all__ = [
    "AgentCapability",
    "AgentMesh",
    "MessageHandler",
    "MeshError",
    "MeshMessage",
    "MeshNode",
    "MeshNodeStatus",
    "MeshRouter",
    "MeshTopology",
]
