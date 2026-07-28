"""Tests for :mod:`myagent.orchestration.mesh` (agent mesh networking)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from myagent.orchestration.mesh import (
    AgentCapability,
    AgentMesh,
    MeshError,
    MeshMessage,
    MeshNode,
    MeshNodeStatus,
    MeshRouter,
    MeshTopology,
)

# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


class TestMeshEnums:
    def test_mesh_node_status_values(self) -> None:
        assert MeshNodeStatus.ONLINE.value == "online"
        assert MeshNodeStatus.OFFLINE.value == "offline"
        assert MeshNodeStatus.BUSY.value == "busy"
        assert MeshNodeStatus.DEGRADED.value == "degraded"

    def test_agent_capability_values(self) -> None:
        assert AgentCapability.REASONING.value == "reasoning"
        assert AgentCapability.CODE_GENERATION.value == "code_generation"
        assert AgentCapability.DATA_ANALYSIS.value == "data_analysis"
        assert AgentCapability.COMMUNICATION.value == "communication"
        assert AgentCapability.EXECUTION.value == "execution"

    def test_mesh_topology_values(self) -> None:
        assert MeshTopology.STAR.value == "star"
        assert MeshTopology.MESH.value == "mesh"
        assert MeshTopology.TREE.value == "tree"
        assert MeshTopology.RING.value == "ring"

    def test_enums_are_str(self) -> None:
        assert isinstance(MeshNodeStatus.ONLINE, str)
        assert isinstance(AgentCapability.REASONING, str)
        assert isinstance(MeshTopology.MESH, str)


# ---------------------------------------------------------------------------
# MeshNode / MeshMessage models
# ---------------------------------------------------------------------------


class TestMeshNode:
    def test_construction_defaults(self) -> None:
        node = MeshNode(name="reasoner")
        assert node.name == "reasoner"
        assert node.address == ""
        assert node.port == 0
        assert node.capabilities == set()
        assert node.status is MeshNodeStatus.ONLINE
        assert node.metadata == {}
        assert node.id
        assert node.last_heartbeat > 0

    def test_construction_with_capabilities(self) -> None:
        node = MeshNode(
            name="coder",
            address="10.0.0.1",
            port=8080,
            capabilities={AgentCapability.CODE_GENERATION, AgentCapability.REASONING},
            metadata={"region": "us-east"},
        )
        assert node.address == "10.0.0.1"
        assert node.port == 8080
        assert AgentCapability.CODE_GENERATION in node.capabilities
        assert node.metadata["region"] == "us-east"

    def test_has_capability(self) -> None:
        node = MeshNode(name="n", capabilities={AgentCapability.PLANNING})
        assert node.has_capability(AgentCapability.PLANNING) is True
        assert node.has_capability(AgentCapability.REASONING) is False

    def test_is_reachable_online(self) -> None:
        assert MeshNode(name="n", status=MeshNodeStatus.ONLINE).is_reachable() is True

    def test_is_reachable_busy(self) -> None:
        assert MeshNode(name="n", status=MeshNodeStatus.BUSY).is_reachable() is True

    def test_is_reachable_degraded(self) -> None:
        assert MeshNode(name="n", status=MeshNodeStatus.DEGRADED).is_reachable() is True

    def test_is_reachable_offline(self) -> None:
        assert MeshNode(name="n", status=MeshNodeStatus.OFFLINE).is_reachable() is False


class TestMeshMessage:
    def test_construction_defaults(self) -> None:
        msg = MeshMessage(source="coordinator")
        assert msg.source == "coordinator"
        assert msg.target is None
        assert msg.message_type == "message"
        assert msg.payload == {}
        assert msg.requires_ack is False
        assert msg.id
        assert msg.timestamp is not None

    def test_construction_full(self) -> None:
        msg = MeshMessage(
            source="a",
            target="b",
            message_type="task",
            payload={"question": "why?"},
            requires_ack=True,
        )
        assert msg.target == "b"
        assert msg.message_type == "task"
        assert msg.payload["question"] == "why?"
        assert msg.requires_ack is True


# ---------------------------------------------------------------------------
# MeshRouter — routing
# ---------------------------------------------------------------------------


def _make_nodes() -> dict[str, MeshNode]:
    return {
        "alpha": MeshNode(
            id="alpha",
            name="alpha",
            capabilities={AgentCapability.REASONING},
        ),
        "beta": MeshNode(
            id="beta",
            name="beta",
            capabilities={AgentCapability.REASONING, AgentCapability.CODE_GENERATION},
        ),
        "gamma": MeshNode(
            id="gamma",
            name="gamma",
            capabilities={AgentCapability.DATA_ANALYSIS},
        ),
    }


class TestMeshRouterDirect:
    def test_route_direct_to_reachable_node(self) -> None:
        router = MeshRouter()
        router.set_nodes(_make_nodes())
        msg = MeshMessage(source="coordinator", target="alpha")
        assert router.route(msg) == ["alpha"]

    def test_route_direct_to_offline_node_returns_empty(self) -> None:
        router = MeshRouter()
        nodes = _make_nodes()
        nodes["alpha"].status = MeshNodeStatus.OFFLINE
        router.set_nodes(nodes)
        msg = MeshMessage(source="coordinator", target="alpha")
        assert router.route(msg) == []

    def test_route_direct_to_unknown_node_returns_empty(self) -> None:
        router = MeshRouter()
        router.set_nodes(_make_nodes())
        msg = MeshMessage(source="coordinator", target="ghost")
        assert router.route(msg) == []


class TestMeshRouterBroadcast:
    def test_route_broadcast_all_reachable_sorted(self) -> None:
        router = MeshRouter()
        router.set_nodes(_make_nodes())
        msg = MeshMessage(source="coordinator", target=None)
        # Source excluded; recipients sorted by id.
        assert router.route(msg) == ["alpha", "beta", "gamma"]

    def test_route_broadcast_excludes_source(self) -> None:
        router = MeshRouter()
        router.set_nodes(_make_nodes())
        msg = MeshMessage(source="beta", target=None)
        assert router.route(msg) == ["alpha", "gamma"]

    def test_route_broadcast_include_source(self) -> None:
        router = MeshRouter()
        router.set_nodes(_make_nodes())
        msg = MeshMessage(source="beta", target=None)
        assert router.route(msg, exclude_source=False) == ["alpha", "beta", "gamma"]

    def test_route_broadcast_skips_offline(self) -> None:
        router = MeshRouter()
        nodes = _make_nodes()
        nodes["beta"].status = MeshNodeStatus.OFFLINE
        router.set_nodes(nodes)
        msg = MeshMessage(source="coordinator", target=None)
        assert router.route(msg) == ["alpha", "gamma"]


class TestMeshRouterCapability:
    def test_route_capability_returns_best_node(self) -> None:
        router = MeshRouter()
        router.set_nodes(_make_nodes())
        msg = MeshMessage(source="coordinator", target="cap:reasoning")
        recipients = router.route(msg)
        assert len(recipients) == 1
        # Both alpha and beta advertise reasoning; ties broken by name -> alpha.
        assert recipients[0] == "alpha"

    def test_route_capability_respects_load(self) -> None:
        router = MeshRouter()
        router.set_nodes(_make_nodes())
        msg = MeshMessage(source="coordinator", target="cap:reasoning")
        # beta is less loaded -> preferred.
        recipients = router.route(msg, load={"alpha": 5, "beta": 0})
        assert recipients == ["beta"]

    def test_route_capability_unknown_returns_empty(self) -> None:
        router = MeshRouter()
        router.set_nodes(_make_nodes())
        msg = MeshMessage(source="coordinator", target="cap:monitoring")
        assert router.route(msg) == []

    def test_route_capability_invalid_returns_empty(self) -> None:
        router = MeshRouter()
        router.set_nodes(_make_nodes())
        msg = MeshMessage(source="coordinator", target="cap:not-a-capability")
        assert router.route(msg) == []


class TestMeshRouterFindBest:
    def test_find_best_node_least_loaded(self) -> None:
        router = MeshRouter()
        nodes = _make_nodes()
        router.set_nodes(nodes)
        assert router.find_best_node(AgentCapability.REASONING) == "alpha"
        assert (
            router.find_best_node(AgentCapability.REASONING, load={"alpha": 3}) == "beta"
        )

    def test_find_best_node_none_when_no_candidate(self) -> None:
        router = MeshRouter()
        router.set_nodes(_make_nodes())
        assert router.find_best_node(AgentCapability.MONITORING) is None

    def test_find_best_node_skips_offline(self) -> None:
        router = MeshRouter()
        nodes = _make_nodes()
        nodes["alpha"].status = MeshNodeStatus.OFFLINE
        router.set_nodes(nodes)
        assert router.find_best_node(AgentCapability.REASONING) == "beta"

    def test_find_best_node_star_prefers_hub(self) -> None:
        router = MeshRouter(topology=MeshTopology.STAR, hub_id="hub")
        nodes = {
            "hub": MeshNode(id="hub", name="hub", capabilities={AgentCapability.REASONING}),
            "leaf": MeshNode(id="leaf", name="leaf", capabilities={AgentCapability.REASONING}),
        }
        router.set_nodes(nodes)
        assert router.find_best_node(AgentCapability.REASONING) == "hub"

    def test_get_topology(self) -> None:
        router = MeshRouter(topology=MeshTopology.RING)
        assert router.get_topology() is MeshTopology.RING


# ---------------------------------------------------------------------------
# AgentMesh — registration
# ---------------------------------------------------------------------------


class TestMeshRegistration:
    @pytest.mark.asyncio
    async def test_register_and_get_node(self) -> None:
        mesh = AgentMesh()
        node = MeshNode(id="a", name="alpha", capabilities={AgentCapability.REASONING})
        registered = await mesh.register_node(node)
        assert registered is node
        assert await mesh.get_node("a") is node

    @pytest.mark.asyncio
    async def test_register_node_sets_heartbeat_when_online(self) -> None:
        mesh = AgentMesh()
        node = MeshNode(
            id="a", name="alpha", last_heartbeat=0.0, status=MeshNodeStatus.ONLINE
        )
        await mesh.register_node(node)
        assert node.last_heartbeat > 0.0

    @pytest.mark.asyncio
    async def test_register_duplicate_raises(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(MeshNode(id="a", name="alpha"))
        with pytest.raises(MeshError, match="already registered"):
            await mesh.register_node(MeshNode(id="a", name="alpha-2"))

    @pytest.mark.asyncio
    async def test_deregister_node_returns_removed(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(MeshNode(id="a", name="alpha"))
        removed = await mesh.deregister_node("a")
        assert removed is not None
        assert removed.name == "alpha"
        assert await mesh.get_node("a") is None

    @pytest.mark.asyncio
    async def test_deregister_unknown_returns_none(self) -> None:
        mesh = AgentMesh()
        assert await mesh.deregister_node("ghost") is None

    @pytest.mark.asyncio
    async def test_list_nodes_sorted_by_name(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(MeshNode(id="z", name="zeta"))
        await mesh.register_node(MeshNode(id="a", name="alpha"))
        names = [n.name for n in await mesh.list_nodes()]
        assert names == ["alpha", "zeta"]

    @pytest.mark.asyncio
    async def test_list_nodes_filtered_by_status(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(
            MeshNode(id="a", name="alpha", status=MeshNodeStatus.ONLINE)
        )
        await mesh.register_node(
            MeshNode(id="b", name="beta", status=MeshNodeStatus.OFFLINE)
        )
        online = await mesh.list_nodes(status=MeshNodeStatus.ONLINE)
        assert [n.id for n in online] == ["a"]


# ---------------------------------------------------------------------------
# AgentMesh — discover_nodes by capability
# ---------------------------------------------------------------------------


class TestMeshDiscovery:
    @pytest.mark.asyncio
    async def test_discover_nodes_by_capability(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(
            MeshNode(id="a", name="alpha", capabilities={AgentCapability.REASONING})
        )
        await mesh.register_node(
            MeshNode(
                id="b",
                name="beta",
                capabilities={AgentCapability.REASONING, AgentCapability.CODE_GENERATION},
            )
        )
        await mesh.register_node(
            MeshNode(id="g", name="gamma", capabilities={AgentCapability.DATA_ANALYSIS})
        )
        discovered = await mesh.discover_nodes(AgentCapability.REASONING)
        assert {n.id for n in discovered} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_discover_nodes_excludes_offline(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(
            MeshNode(
                id="a",
                name="alpha",
                capabilities={AgentCapability.REASONING},
                status=MeshNodeStatus.OFFLINE,
            )
        )
        assert await mesh.discover_nodes(AgentCapability.REASONING) == []

    @pytest.mark.asyncio
    async def test_discover_nodes_empty_when_no_match(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(
            MeshNode(id="a", name="alpha", capabilities={AgentCapability.REASONING})
        )
        assert await mesh.discover_nodes(AgentCapability.MONITORING) == []


# ---------------------------------------------------------------------------
# AgentMesh — heartbeat & stale detection
# ---------------------------------------------------------------------------


class TestMeshHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_refreshes_timestamp(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(
            MeshNode(id="a", name="alpha", last_heartbeat=100.0)
        )
        before = 100.0
        node = await mesh.heartbeat("a")
        assert node is not None
        assert node.last_heartbeat > before

    @pytest.mark.asyncio
    async def test_heartbeat_revives_offline_node(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(
            MeshNode(id="a", name="alpha", status=MeshNodeStatus.OFFLINE)
        )
        node = await mesh.heartbeat("a")
        assert node is not None
        assert node.status is MeshNodeStatus.ONLINE

    @pytest.mark.asyncio
    async def test_heartbeat_unknown_returns_none(self) -> None:
        mesh = AgentMesh()
        assert await mesh.heartbeat("ghost") is None

    @pytest.mark.asyncio
    async def test_mark_stale_flips_old_nodes_to_offline(self) -> None:
        mesh = AgentMesh(heartbeat_timeout=60.0)
        # register_node resets last_heartbeat to time.time() for ONLINE nodes,
        # so we register first and then backdate the heartbeat manually.
        node = await mesh.register_node(MeshNode(id="a", name="alpha"))
        node.last_heartbeat = 100.0
        # now well beyond the 60s timeout.
        flipped = await mesh.mark_stale(at=1000.0)
        assert flipped == ["a"]
        stale = await mesh.get_node("a")
        assert stale is not None
        assert stale.status is MeshNodeStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_mark_stale_skips_fresh_nodes(self) -> None:
        mesh = AgentMesh(heartbeat_timeout=60.0)
        await mesh.register_node(MeshNode(id="a", name="alpha"))
        flipped = await mesh.mark_stale(at=time.time())
        assert flipped == []

    @pytest.mark.asyncio
    async def test_mark_stale_skips_never_heartbeated(self) -> None:
        mesh = AgentMesh(heartbeat_timeout=60.0)
        await mesh.register_node(
            MeshNode(
                id="a",
                name="alpha",
                last_heartbeat=0.0,
                status=MeshNodeStatus.DEGRADED,
            )
        )
        flipped = await mesh.mark_stale(at=1000.0)
        assert flipped == []


# ---------------------------------------------------------------------------
# AgentMesh — direct messaging
# ---------------------------------------------------------------------------


class TestMeshDirectMessaging:
    @pytest.mark.asyncio
    async def test_send_message_direct_with_handler(self) -> None:
        mesh = AgentMesh()
        received: list[MeshMessage] = []

        async def handler(message: MeshMessage) -> dict[str, Any] | None:
            received.append(message)
            return {"echo": message.payload}

        await mesh.register_node(
            MeshNode(id="a", name="alpha", capabilities={AgentCapability.REASONING}),
            handler=handler,
        )
        msg = MeshMessage(
            source="coordinator",
            target="a",
            message_type="task",
            payload={"q": "why?"},
            requires_ack=True,
        )
        records = await mesh.send_message("a", msg)
        assert len(records) == 1
        record = records[0]
        assert record["node_id"] == "a"
        assert record["delivered"] is True
        assert record["acked"] is True
        assert record["response"] == {"echo": {"q": "why?"}}
        assert received[0].payload == {"q": "why?"}

    @pytest.mark.asyncio
    async def test_send_message_direct_without_handler_buffers(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(MeshNode(id="a", name="alpha"))
        msg = MeshMessage(source="coordinator", target="a", payload={"x": 1})
        records = await mesh.send_message("a", msg)
        assert records[0]["delivered"] is True
        assert records[0]["buffered"] is True
        assert records[0]["acked"] is False
        buffered = await mesh.drain_inbox("a")
        assert len(buffered) == 1
        assert buffered[0].payload == {"x": 1}

    @pytest.mark.asyncio
    async def test_send_message_to_offline_returns_empty(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(
            MeshNode(id="a", name="alpha", status=MeshNodeStatus.OFFLINE)
        )
        msg = MeshMessage(source="coordinator", target="a")
        assert await mesh.send_message("a", msg) == []

    @pytest.mark.asyncio
    async def test_send_message_to_unknown_returns_empty(self) -> None:
        mesh = AgentMesh()
        msg = MeshMessage(source="coordinator", target="ghost")
        assert await mesh.send_message("ghost", msg) == []

    @pytest.mark.asyncio
    async def test_handler_exception_recorded_as_error(self) -> None:
        mesh = AgentMesh()

        async def boom(message: MeshMessage) -> dict[str, Any] | None:
            raise RuntimeError("handler down")

        await mesh.register_node(
            MeshNode(id="a", name="alpha"), handler=boom
        )
        msg = MeshMessage(source="coordinator", target="a")
        records = await mesh.send_message("a", msg)
        assert records[0]["delivered"] is False
        assert "handler down" in records[0]["error"]


# ---------------------------------------------------------------------------
# AgentMesh — broadcast
# ---------------------------------------------------------------------------


class TestMeshBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_delivers_to_all_reachable(self) -> None:
        mesh = AgentMesh()
        received: list[str] = []

        async def handler(message: MeshMessage) -> dict[str, Any] | None:
            received.append(message.target or "broadcast")
            return None

        await mesh.register_node(
            MeshNode(id="a", name="alpha"), handler=handler
        )
        await mesh.register_node(
            MeshNode(id="b", name="beta"), handler=handler
        )
        # Offline node should be skipped.
        await mesh.register_node(
            MeshNode(id="c", name="gamma", status=MeshNodeStatus.OFFLINE)
        )
        msg = MeshMessage(source="coordinator", payload={"ping": 1})
        records = await mesh.broadcast(msg)
        assert {r["node_id"] for r in records} == {"a", "b"}
        assert all(r["delivered"] for r in records)
        assert msg.target is None  # broadcast normalises target

    @pytest.mark.asyncio
    async def test_send_message_with_broadcast_target(self) -> None:
        mesh = AgentMesh()
        received: list[str] = []

        async def handler(message: MeshMessage) -> dict[str, Any] | None:
            received.append(message.id)
            return None

        await mesh.register_node(MeshNode(id="a", name="alpha"), handler=handler)
        await mesh.register_node(MeshNode(id="b", name="beta"), handler=handler)
        msg = MeshMessage(source="coordinator")
        records = await mesh.send_message("broadcast", msg)
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_broadcast_excludes_source(self) -> None:
        mesh = AgentMesh()
        targets: list[str] = []

        async def handler(message: MeshMessage) -> dict[str, Any] | None:
            targets.append(message.target or "")
            return None

        await mesh.register_node(MeshNode(id="a", name="alpha"), handler=handler)
        await mesh.register_node(MeshNode(id="b", name="beta"), handler=handler)
        msg = MeshMessage(source="a")
        records = await mesh.broadcast(msg)
        assert {r["node_id"] for r in records} == {"b"}


# ---------------------------------------------------------------------------
# AgentMesh — capability-based messaging
# ---------------------------------------------------------------------------


class TestMeshCapabilityMessaging:
    @pytest.mark.asyncio
    async def test_send_message_capability_routes_to_best(self) -> None:
        mesh = AgentMesh()
        received: list[str] = []

        async def handler(message: MeshMessage) -> dict[str, Any] | None:
            received.append(message.id)
            return None

        await mesh.register_node(
            MeshNode(
                id="b",
                name="beta",
                capabilities={AgentCapability.REASONING, AgentCapability.CODE_GENERATION},
            ),
            handler=handler,
        )
        await mesh.register_node(
            MeshNode(id="a", name="alpha", capabilities={AgentCapability.REASONING}),
            handler=handler,
        )
        msg = MeshMessage(
            source="coordinator",
            target="cap:reasoning",
            payload={"question": "why?"},
        )
        records = await mesh.send_message("cap:reasoning", msg)
        # Tie broken by name -> node "a" (name "alpha") wins over "b" (name "beta").
        assert len(records) == 1
        assert records[0]["node_id"] == "a"
        assert records[0]["node_name"] == "alpha"
        assert records[0]["delivered"] is True

    @pytest.mark.asyncio
    async def test_send_message_capability_no_match_returns_empty(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(
            MeshNode(id="a", name="alpha", capabilities={AgentCapability.REASONING})
        )
        msg = MeshMessage(source="coordinator", target="cap:monitoring")
        assert await mesh.send_message("cap:monitoring", msg) == []


# ---------------------------------------------------------------------------
# AgentMesh — drain_inbox & reporting
# ---------------------------------------------------------------------------


class TestMeshInboxAndReporting:
    @pytest.mark.asyncio
    async def test_drain_inbox_unknown_returns_empty(self) -> None:
        mesh = AgentMesh()
        assert await mesh.drain_inbox("ghost") == []

    @pytest.mark.asyncio
    async def test_drain_inbox_returns_buffered_messages(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(MeshNode(id="a", name="alpha"))
        await mesh.send_message("a", MeshMessage(source="c", target="a", payload={"i": 1}))
        await mesh.send_message("a", MeshMessage(source="c", target="a", payload={"i": 2}))
        buffered = await mesh.drain_inbox("a")
        assert len(buffered) == 2
        # Inbox is drained after reading.
        assert await mesh.drain_inbox("a") == []

    @pytest.mark.asyncio
    async def test_get_topology(self) -> None:
        mesh = AgentMesh(topology=MeshTopology.STAR)
        assert await mesh.get_topology() is MeshTopology.STAR

    @pytest.mark.asyncio
    async def test_get_router(self) -> None:
        mesh = AgentMesh()
        assert isinstance(mesh.get_router(), MeshRouter)

    @pytest.mark.asyncio
    async def test_stats_reflects_state(self) -> None:
        mesh = AgentMesh()
        await mesh.register_node(MeshNode(id="a", name="alpha"))
        await mesh.register_node(
            MeshNode(id="b", name="beta", status=MeshNodeStatus.OFFLINE)
        )
        stats = mesh.stats()
        assert stats["total_nodes"] == 2
        assert stats["by_status"]["online"] == 1
        assert stats["by_status"]["offline"] == 1
        assert stats["topology"] == "mesh"
        assert stats["in_flight"] == 0
