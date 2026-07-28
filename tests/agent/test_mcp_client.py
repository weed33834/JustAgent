"""Tests for ``justagent.agent.mcp_client`` (MCP client + JSON-RPC layer)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from justagent.agent.mcp_client import (
    JSONRPCError,
    MCPClient,
    MCPConnectionState,
    MCPError,
    MCPManager,
    MCPOAuthToken,
    MCPResource,
    MCPServerConfig,
    MCPTool,
    MCPTransportType,
    OAuthTokenProvider,
    StaticTokenProvider,
    _jsonrpc_error,
    _jsonrpc_notification,
    _jsonrpc_request,
    _jsonrpc_response,
)
from justagent.exceptions import MyAgentError

# ---------------------------------------------------------------------------
# Mock transport — records sent messages, returns canned responses
# ---------------------------------------------------------------------------


class _MockTransport:
    """Scripted async transport for testing :class:`MCPClient`.

    Each call to :meth:`receive` pops the next response from the
    ``responses`` list. When empty, a default empty-result response is
    returned. All sent messages are recorded in :attr:`sent`.
    """

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._receive_delay: float = 0.0

    def set_receive_delay(self, delay: float) -> None:
        self._receive_delay = delay

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def receive(self) -> dict[str, Any]:
        if self._receive_delay > 0:
            await asyncio.sleep(self._receive_delay)
        if not self._responses:
            return {"jsonrpc": "2.0", "id": 0, "result": {}}
        return self._responses.pop(0)

    async def close(self) -> None:
        self.closed = True


def _init_response(req_id: int = 1) -> dict[str, Any]:
    """Build a canned ``initialize`` success response."""

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "test-server", "version": "1.0.0"},
            "capabilities": {},
        },
    }


def _result_response(req_id: int, result: dict[str, Any]) -> dict[str, Any]:
    """Build a canned success response with an arbitrary result."""

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error_response(req_id: int, code: int, message: str) -> dict[str, Any]:
    """Build a canned error response."""

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _make_stdio_config(name: str = "test") -> MCPServerConfig:
    """Build a minimal stdio server config."""

    return MCPServerConfig(
        name=name,
        transport=MCPTransportType.STDIO,
        command="echo",
        args=["hello"],
    )


def _make_http_config(name: str = "http-test") -> MCPServerConfig:
    """Build a minimal HTTP server config."""

    return MCPServerConfig(
        name=name,
        transport=MCPTransportType.HTTP,
        url="http://localhost:8080/mcp",
        timeout_seconds=5.0,
    )


# ---------------------------------------------------------------------------
# MCPTransportType
# ---------------------------------------------------------------------------


class TestMCPTransportType:
    def test_stdio_value(self) -> None:
        assert MCPTransportType.STDIO.value == "stdio"

    def test_sse_value(self) -> None:
        assert MCPTransportType.SSE.value == "sse"

    def test_http_value(self) -> None:
        assert MCPTransportType.HTTP.value == "http"

    def test_is_str_subclass(self) -> None:
        assert isinstance(MCPTransportType.STDIO, str)
        assert MCPTransportType.STDIO == "stdio"

    def test_all_transports(self) -> None:
        expected = {"stdio", "sse", "http"}
        actual = {t.value for t in MCPTransportType}
        assert actual == expected


# ---------------------------------------------------------------------------
# MCPConnectionState
# ---------------------------------------------------------------------------


class TestMCPConnectionState:
    def test_disconnected_value(self) -> None:
        assert MCPConnectionState.DISCONNECTED.value == "disconnected"

    def test_connecting_value(self) -> None:
        assert MCPConnectionState.CONNECTING.value == "connecting"

    def test_connected_value(self) -> None:
        assert MCPConnectionState.CONNECTED.value == "connected"

    def test_error_value(self) -> None:
        assert MCPConnectionState.ERROR.value == "error"

    def test_is_str_subclass(self) -> None:
        assert isinstance(MCPConnectionState.CONNECTED, str)
        assert MCPConnectionState.DISCONNECTED == "disconnected"


# ---------------------------------------------------------------------------
# MCPTool
# ---------------------------------------------------------------------------


class TestMCPTool:
    def test_construction(self) -> None:
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object"},
            server_name="fs",
        )
        assert tool.name == "read_file"
        assert tool.description == "Read a file"
        assert tool.input_schema == {"type": "object"}
        assert tool.server_name == "fs"

    def test_frozen(self) -> None:
        tool = MCPTool(
            name="x", description="", input_schema={}, server_name="s"
        )
        with pytest.raises((AttributeError, Exception)):
            tool.name = "changed"  # type: ignore[misc]

    def test_input_schema_defaults_independent(self) -> None:
        a = MCPTool(name="a", description="", input_schema={}, server_name="s")
        b = MCPTool(name="b", description="", input_schema={}, server_name="s")
        a.input_schema["key"] = "val"
        assert "key" not in b.input_schema


# ---------------------------------------------------------------------------
# MCPResource
# ---------------------------------------------------------------------------


class TestMCPResource:
    def test_construction(self) -> None:
        resource = MCPResource(
            uri="file:///tmp/x",
            name="x",
            description="a file",
            mimeType="text/plain",
        )
        assert resource.uri == "file:///tmp/x"
        assert resource.name == "x"
        assert resource.description == "a file"
        assert resource.mimeType == "text/plain"

    def test_default_mimetype(self) -> None:
        resource = MCPResource(uri="u", name="n", description="d")
        assert resource.mimeType == ""

    def test_frozen(self) -> None:
        resource = MCPResource(uri="u", name="n", description="d")
        with pytest.raises((AttributeError, Exception)):
            resource.uri = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MCPServerConfig
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    def test_defaults(self) -> None:
        config = MCPServerConfig(
            name="x", transport=MCPTransportType.STDIO
        )
        assert config.command == ""
        assert config.args == []
        assert config.url == ""
        assert config.env == {}
        assert config.headers == {}
        assert config.enabled is True
        assert config.timeout_seconds == 30.0

    def test_stdio_config(self) -> None:
        config = MCPServerConfig(
            name="fs",
            transport=MCPTransportType.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            env={"NODE_PATH": "/usr/lib"},
        )
        assert config.command == "npx"
        assert config.args == ["-y", "@modelcontextprotocol/server-filesystem"]
        assert config.env == {"NODE_PATH": "/usr/lib"}

    def test_http_config(self) -> None:
        config = MCPServerConfig(
            name="api",
            transport=MCPTransportType.HTTP,
            url="https://example.com/mcp",
            headers={"X-Custom": "yes"},
            timeout_seconds=10.0,
        )
        assert config.url == "https://example.com/mcp"
        assert config.headers == {"X-Custom": "yes"}
        assert config.timeout_seconds == 10.0

    def test_frozen(self) -> None:
        config = MCPServerConfig(name="x", transport=MCPTransportType.STDIO)
        with pytest.raises((AttributeError, Exception)):
            config.name = "changed"  # type: ignore[misc]

    def test_disabled(self) -> None:
        config = MCPServerConfig(
            name="x", transport=MCPTransportType.STDIO, enabled=False
        )
        assert config.enabled is False

    def test_defaults_independent(self) -> None:
        a = MCPServerConfig(name="a", transport=MCPTransportType.STDIO)
        b = MCPServerConfig(name="b", transport=MCPTransportType.STDIO)
        a.args.append("shared")
        assert b.args == []


# ---------------------------------------------------------------------------
# MCPOAuthToken
# ---------------------------------------------------------------------------


class TestMCPOAuthToken:
    def test_construction(self) -> None:
        token = MCPOAuthToken(
            access_token="abc123",
            token_type="Bearer",
            expires_at=1000.0,
            refresh_token="rfresh",
        )
        assert token.access_token == "abc123"
        assert token.token_type == "Bearer"
        assert token.expires_at == 1000.0
        assert token.refresh_token == "rfresh"

    def test_defaults(self) -> None:
        token = MCPOAuthToken(access_token="x")
        assert token.token_type == "Bearer"
        assert token.expires_at == 0.0
        assert token.refresh_token == ""

    def test_frozen(self) -> None:
        token = MCPOAuthToken(access_token="x")
        with pytest.raises((AttributeError, Exception)):
            token.access_token = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StaticTokenProvider
# ---------------------------------------------------------------------------


class TestStaticTokenProvider:
    @pytest.mark.asyncio
    async def test_get_token_returns_stored(self) -> None:
        token = MCPOAuthToken(access_token="my-token")
        provider = StaticTokenProvider(token)
        result = await provider.get_token()
        assert result is token

    @pytest.mark.asyncio
    async def test_refresh_returns_same(self) -> None:
        token = MCPOAuthToken(access_token="my-token")
        provider = StaticTokenProvider(token)
        refreshed = await provider.refresh()
        assert refreshed is token

    @pytest.mark.asyncio
    async def test_get_token_none_not_returned(self) -> None:
        token = MCPOAuthToken(access_token="abc")
        provider = StaticTokenProvider(token)
        assert await provider.get_token() is not None


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


class TestJSONRPC:
    def test_request_format(self) -> None:
        msg = _jsonrpc_request(1, "tools/list")
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 1
        assert msg["method"] == "tools/list"
        assert "params" not in msg

    def test_request_with_params(self) -> None:
        params = {"name": "tool1", "arguments": {"x": 1}}
        msg = _jsonrpc_request(2, "tools/call", params)
        assert msg["params"] == params

    def test_notification_has_no_id(self) -> None:
        msg = _jsonrpc_notification("notifications/initialized")
        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "notifications/initialized"
        assert "id" not in msg

    def test_notification_with_params(self) -> None:
        msg = _jsonrpc_notification("progress", {"pct": 50})
        assert msg["params"] == {"pct": 50}
        assert "id" not in msg

    def test_response_format(self) -> None:
        msg = _jsonrpc_response(1, {"tools": []})
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 1
        assert msg["result"] == {"tools": []}

    def test_error_format(self) -> None:
        msg = _jsonrpc_error(1, -32601, "Method not found")
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 1
        assert msg["error"] == {"code": -32601, "message": "Method not found"}

    def test_jsonrpc_error_construction(self) -> None:
        err = JSONRPCError(-32601, "Method not found", "extra")
        assert err.rpc_code == -32601
        assert err.data == "extra"
        assert "Method not found" in str(err)

    def test_jsonrpc_error_is_mcp_error(self) -> None:
        err = JSONRPCError(-32601, "Method not found")
        assert isinstance(err, MCPError)
        assert isinstance(err, MyAgentError)

    def test_jsonrpc_error_default_data(self) -> None:
        err = JSONRPCError(-32700, "Parse error")
        assert err.data is None

    def test_jsonrpc_error_in_details(self) -> None:
        err = JSONRPCError(-32603, "boom", {"key": "val"})
        assert err.details["rpc_code"] == -32603
        assert err.details["data"] == {"key": "val"}


# ---------------------------------------------------------------------------
# MCPClient (stdio, using mock transport)
# ---------------------------------------------------------------------------


class TestMCPClientStdio:
    @pytest.mark.asyncio
    async def test_connect_handshake(self) -> None:
        transport = _MockTransport([_init_response(1)])
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()

        assert client.state is MCPConnectionState.CONNECTED
        # Two messages sent: initialize request + initialized notification.
        assert len(transport.sent) == 2
        assert transport.sent[0]["method"] == "initialize"
        assert transport.sent[0]["id"] == 1
        assert transport.sent[1]["method"] == "notifications/initialized"
        assert "id" not in transport.sent[1]

    @pytest.mark.asyncio
    async def test_connect_sends_correct_protocol_version(self) -> None:
        transport = _MockTransport([_init_response(1)])
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()

        init_msg = transport.sent[0]
        params = init_msg["params"]
        assert params["protocolVersion"] == "2024-11-05"
        assert params["clientInfo"]["name"] == "justagent"
        assert params["clientInfo"]["version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_list_tools(self) -> None:
        responses = [
            _init_response(1),
            _result_response(
                2,
                {
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "Read a file",
                            "inputSchema": {"type": "object"},
                        },
                        {
                            "name": "write_file",
                            "description": "Write a file",
                            "inputSchema": {"type": "object"},
                        },
                    ]
                },
            ),
        ]
        transport = _MockTransport(responses)
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        tools = await client.list_tools()

        assert len(tools) == 2
        assert tools[0].name == "read_file"
        assert tools[0].description == "Read a file"
        assert tools[0].input_schema == {"type": "object"}
        assert tools[0].server_name == "test"
        assert tools[1].name == "write_file"

    @pytest.mark.asyncio
    async def test_list_tools_empty(self) -> None:
        transport = _MockTransport([_init_response(1), _result_response(2, {"tools": []})])
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        tools = await client.list_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_call_tool_text_content(self) -> None:
        responses = [
            _init_response(1),
            _result_response(
                2,
                {
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ],
                    "isError": False,
                },
            ),
        ]
        transport = _MockTransport(responses)
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        result = await client.call_tool("greet", {"name": "x"})

        assert result == "hello\nworld"
        # Verify the tools/call request was sent correctly.
        call_msg = transport.sent[-1]
        assert call_msg["method"] == "tools/call"
        assert call_msg["params"] == {"name": "greet", "arguments": {"name": "x"}}

    @pytest.mark.asyncio
    async def test_call_tool_mixed_content_returns_raw(self) -> None:
        responses = [
            _init_response(1),
            _result_response(
                2,
                {
                    "content": [
                        {"type": "text", "text": "desc"},
                        {"type": "image", "data": "base64..."},
                    ],
                    "isError": False,
                },
            ),
        ]
        transport = _MockTransport(responses)
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        result = await client.call_tool("render", {})

        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self) -> None:
        responses = [
            _init_response(1),
            _result_response(
                2,
                {
                    "content": [{"type": "text", "text": "file not found"}],
                    "isError": True,
                },
            ),
        ]
        transport = _MockTransport(responses)
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        with pytest.raises(MCPError, match="returned an error"):
            await client.call_tool("read_file", {"path": "/missing"})

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        transport = _MockTransport([_init_response(1)])
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        assert client.state is MCPConnectionState.CONNECTED

        await client.disconnect()
        assert client.state is MCPConnectionState.DISCONNECTED
        assert transport.closed is True

    @pytest.mark.asyncio
    async def test_state_transitions(self) -> None:
        transport = _MockTransport([_init_response(1)])
        client = MCPClient(_make_stdio_config(), _transport=transport)
        assert client.state is MCPConnectionState.DISCONNECTED

        await client.connect()
        assert client.state is MCPConnectionState.CONNECTED

        await client.disconnect()
        assert client.state is MCPConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_failure_sets_error_state(self) -> None:
        # Server returns an error for initialize → connect raises, state ERROR.
        transport = _MockTransport([_error_response(1, -32603, "boom")])
        client = MCPClient(_make_stdio_config(), _transport=transport)

        with pytest.raises(JSONRPCError):
            await client.connect()
        assert client.state is MCPConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_server_name_property(self) -> None:
        client = MCPClient(_make_stdio_config("my-server"), _transport=_MockTransport())
        assert client.server_name == "my-server"

    @pytest.mark.asyncio
    async def test_list_resources(self) -> None:
        responses = [
            _init_response(1),
            _result_response(
                2,
                {
                    "resources": [
                        {
                            "uri": "file:///a",
                            "name": "a",
                            "description": "file a",
                            "mimeType": "text/plain",
                        }
                    ]
                },
            ),
        ]
        transport = _MockTransport(responses)
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        resources = await client.list_resources()

        assert len(resources) == 1
        assert resources[0].uri == "file:///a"
        assert resources[0].mimeType == "text/plain"

    @pytest.mark.asyncio
    async def test_read_resource(self) -> None:
        responses = [
            _init_response(1),
            _result_response(
                2,
                {"contents": [{"uri": "file:///a", "text": "hello"}]},
            ),
        ]
        transport = _MockTransport(responses)
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        result = await client.read_resource("file:///a")

        assert result["contents"][0]["text"] == "hello"
        assert transport.sent[-1]["params"] == {"uri": "file:///a"}


# ---------------------------------------------------------------------------
# MCPClient (HTTP config, using mock transport)
# ---------------------------------------------------------------------------


class TestMCPClientHTTP:
    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        transport = _MockTransport([_init_response(1)])
        client = MCPClient(_make_http_config(), _transport=transport)
        await client.connect()

        assert client.state is MCPConnectionState.CONNECTED
        assert len(transport.sent) == 2

    @pytest.mark.asyncio
    async def test_list_tools(self) -> None:
        responses = [
            _init_response(1),
            _result_response(
                2,
                {"tools": [{"name": "search", "description": "search web", "inputSchema": {}}]},
            ),
        ]
        transport = _MockTransport(responses)
        client = MCPClient(_make_http_config(), _transport=transport)
        await client.connect()
        tools = await client.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "search"
        assert tools[0].server_name == "http-test"

    @pytest.mark.asyncio
    async def test_call_tool(self) -> None:
        responses = [
            _init_response(1),
            _result_response(
                2,
                {"content": [{"type": "text", "text": "result"}], "isError": False},
            ),
        ]
        transport = _MockTransport(responses)
        client = MCPClient(_make_http_config(), _transport=transport)
        await client.connect()
        result = await client.call_tool("search", {"q": "test"})

        assert result == "result"

    @pytest.mark.asyncio
    async def test_disconnect_closes_transport(self) -> None:
        transport = _MockTransport([_init_response(1)])
        client = MCPClient(_make_http_config(), _transport=transport)
        await client.connect()
        await client.disconnect()

        assert transport.closed is True
        assert client.state is MCPConnectionState.DISCONNECTED


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------


class TestMCPManager:
    def test_add_server(self) -> None:
        manager = MCPManager()
        config = _make_stdio_config("s1")
        manager.add_server(config)
        assert manager.list_servers() == [config]

    def test_add_multiple_servers(self) -> None:
        manager = MCPManager()
        c1 = _make_stdio_config("s1")
        c2 = _make_http_config("s2")
        manager.add_server(c1)
        manager.add_server(c2)
        assert len(manager.list_servers()) == 2

    def test_remove_server(self) -> None:
        manager = MCPManager()
        manager.add_server(_make_stdio_config("s1"))
        assert manager.remove_server("s1") is True
        assert manager.list_servers() == []

    def test_remove_server_not_found(self) -> None:
        manager = MCPManager()
        assert manager.remove_server("missing") is False

    def test_list_servers_empty(self) -> None:
        manager = MCPManager()
        assert manager.list_servers() == []

    @pytest.mark.asyncio
    async def test_connect_all_one_fails_one_succeeds(self) -> None:
        manager = MCPManager()
        # Good server — inject a mock transport via a subclass override.
        good_config = _make_stdio_config("good")
        manager.add_server(good_config)
        bad_config = _make_http_config("bad")
        manager.add_server(bad_config)

        # Patch connect by injecting transports through a custom connect.
        good_transport = _MockTransport([_init_response(1)])
        bad_transport = _MockTransport([_error_response(1, -32603, "nope")])

        original_init = MCPClient.__init__

        def patched_init(self: MCPClient, config: MCPServerConfig, **kwargs: Any) -> None:
            if config.name == "good":
                kwargs["_transport"] = good_transport
            elif config.name == "bad":
                kwargs["_transport"] = bad_transport
            original_init(self, config, **kwargs)

        MCPClient.__init__ = patched_init  # type: ignore[method-assign]
        try:
            results = await manager.connect_all()
        finally:
            MCPClient.__init__ = original_init  # type: ignore[method-assign]

        assert results["good"] is None
        assert results["bad"] is not None
        assert isinstance(results["bad"], JSONRPCError)
        assert manager.get_server("good") is not None
        assert manager.get_server("bad") is None

    @pytest.mark.asyncio
    async def test_connect_all_skips_disabled(self) -> None:
        manager = MCPManager()
        manager.add_server(
            MCPServerConfig(
                name="off",
                transport=MCPTransportType.STDIO,
                command="echo",
                enabled=False,
            )
        )
        results = await manager.connect_all()
        assert "off" not in results

    @pytest.mark.asyncio
    async def test_list_all_tools_aggregates(self) -> None:
        manager = MCPManager()
        manager.add_server(_make_stdio_config("s1"))
        manager.add_server(_make_http_config("s2"))

        t1 = _MockTransport([_init_response(1), _result_response(2, {"tools": [
            {"name": "a", "description": "", "inputSchema": {}}]})])
        t2 = _MockTransport([_init_response(1), _result_response(2, {"tools": [
            {"name": "b", "description": "", "inputSchema": {}}]})])

        original_init = MCPClient.__init__

        def patched_init(self: MCPClient, config: MCPServerConfig, **kwargs: Any) -> None:
            if config.name == "s1":
                kwargs["_transport"] = t1
            elif config.name == "s2":
                kwargs["_transport"] = t2
            original_init(self, config, **kwargs)

        MCPClient.__init__ = patched_init  # type: ignore[method-assign]
        try:
            await manager.connect_all()
            tools = await manager.list_all_tools()
        finally:
            MCPClient.__init__ = original_init  # type: ignore[method-assign]

        names = {t.name for t in tools}
        assert names == {"a", "b"}

    @pytest.mark.asyncio
    async def test_call_tool_routes_to_correct_server(self) -> None:
        manager = MCPManager()
        manager.add_server(_make_stdio_config("s1"))

        transport = _MockTransport([
            _init_response(1),
            _result_response(2, {"content": [{"type": "text", "text": "ok"}], "isError": False}),
        ])
        original_init = MCPClient.__init__

        def patched_init(self: MCPClient, config: MCPServerConfig, **kwargs: Any) -> None:
            kwargs["_transport"] = transport
            original_init(self, config, **kwargs)

        MCPClient.__init__ = patched_init  # type: ignore[method-assign]
        try:
            await manager.connect_all()
            result = await manager.call_tool("s1", "do_thing", {})
        finally:
            MCPClient.__init__ = original_init  # type: ignore[method-assign]

        assert result == "ok"

    @pytest.mark.asyncio
    async def test_call_tool_unknown_server_raises(self) -> None:
        manager = MCPManager()
        with pytest.raises(MCPError, match="not connected"):
            await manager.call_tool("missing", "tool", {})

    @pytest.mark.asyncio
    async def test_disconnect_all(self) -> None:
        manager = MCPManager()
        manager.add_server(_make_stdio_config("s1"))

        transport = _MockTransport([_init_response(1)])
        original_init = MCPClient.__init__

        def patched_init(self: MCPClient, config: MCPServerConfig, **kwargs: Any) -> None:
            kwargs["_transport"] = transport
            original_init(self, config, **kwargs)

        MCPClient.__init__ = patched_init  # type: ignore[method-assign]
        try:
            await manager.connect_all()
            assert manager.get_server("s1") is not None
            await manager.disconnect_all()
        finally:
            MCPClient.__init__ = original_init  # type: ignore[method-assign]

        assert manager.get_server("s1") is None
        assert transport.closed is True

    @pytest.mark.asyncio
    async def test_get_server_returns_none_when_not_connected(self) -> None:
        manager = MCPManager()
        manager.add_server(_make_stdio_config("s1"))
        assert manager.get_server("s1") is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_connect_when_already_connected(self) -> None:
        transport = _MockTransport([_init_response(1)])
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()

        with pytest.raises(MCPError, match="already connected"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        client = MCPClient(_make_stdio_config())
        # Should be a no-op, not raise.
        await client.disconnect()
        assert client.state is MCPConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_call_tool_on_disconnected(self) -> None:
        client = MCPClient(_make_stdio_config())
        with pytest.raises(MCPError, match="not connected"):
            await client.call_tool("x", {})

    @pytest.mark.asyncio
    async def test_list_tools_on_disconnected(self) -> None:
        client = MCPClient(_make_stdio_config())
        with pytest.raises(MCPError, match="not connected"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_list_resources_on_disconnected(self) -> None:
        client = MCPClient(_make_stdio_config())
        with pytest.raises(MCPError, match="not connected"):
            await client.list_resources()

    @pytest.mark.asyncio
    async def test_read_resource_on_disconnected(self) -> None:
        client = MCPClient(_make_stdio_config())
        with pytest.raises(MCPError, match="not connected"):
            await client.read_resource("file:///x")

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        transport = _MockTransport([_init_response(1)])
        transport.set_receive_delay(1.0)
        config = MCPServerConfig(
            name="slow",
            transport=MCPTransportType.STDIO,
            command="echo",
            timeout_seconds=0.05,
        )
        client = MCPClient(config, _transport=transport)
        with pytest.raises(MCPError, match="Timed out"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_jsonrpc_error_from_server(self) -> None:
        transport = _MockTransport([
            _init_response(1),
            _error_response(2, -32601, "Method not found"),
        ])
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        with pytest.raises(JSONRPCError) as exc_info:
            await client.list_tools()
        assert exc_info.value.rpc_code == -32601

    @pytest.mark.asyncio
    async def test_disconnect_after_error_state(self) -> None:
        transport = _MockTransport([_error_response(1, -32603, "boom")])
        client = MCPClient(_make_stdio_config(), _transport=transport)
        with pytest.raises(JSONRPCError):
            await client.connect()
        assert client.state is MCPConnectionState.ERROR
        # Disconnect should still work and reset state.
        await client.disconnect()
        assert client.state is MCPConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_call_tool_single_text_block(self) -> None:
        responses = [
            _init_response(1),
            _result_response(
                2,
                {"content": [{"type": "text", "text": "only"}], "isError": False},
            ),
        ]
        transport = _MockTransport(responses)
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        result = await client.call_tool("x", {})
        assert result == "only"

    @pytest.mark.asyncio
    async def test_call_tool_empty_content(self) -> None:
        responses = [
            _init_response(1),
            _result_response(2, {"content": [], "isError": False}),
        ]
        transport = _MockTransport(responses)
        client = MCPClient(_make_stdio_config(), _transport=transport)
        await client.connect()
        result = await client.call_tool("x", {})
        assert result == []


# ---------------------------------------------------------------------------
# OAuthTokenProvider protocol conformance
# ---------------------------------------------------------------------------


class TestOAuthProtocol:
    def test_static_provider_satisfies_protocol(self) -> None:
        provider: OAuthTokenProvider = StaticTokenProvider(
            MCPOAuthToken(access_token="x")
        )
        assert provider is not None

    @pytest.mark.asyncio
    async def test_http_transport_injects_auth_header(self) -> None:
        """The HTTP transport adds an Authorization header from the token."""
        from justagent.agent.mcp_client import _HTTPTransport

        token = MCPOAuthToken(access_token="secret-token", token_type="Bearer")
        provider = StaticTokenProvider(token)
        config = MCPServerConfig(
            name="auth",
            transport=MCPTransportType.HTTP,
            url="http://localhost:9999/mcp",
        )
        transport = _HTTPTransport(config, provider)
        await transport.start()

        # The httpx client should have the Authorization header set.
        assert transport._client is not None
        auth_header = transport._client.headers.get("authorization")
        assert auth_header == "Bearer secret-token"
        await transport.close()


# ---------------------------------------------------------------------------
# _HTTPTransport with mocked httpx
# ---------------------------------------------------------------------------


class TestHTTPTransportMocked:
    @pytest.mark.asyncio
    async def test_send_receive_roundtrip(self) -> None:
        """Test _HTTPTransport send/receive using a mocked httpx client."""
        from unittest.mock import AsyncMock, MagicMock

        from justagent.agent.mcp_client import _HTTPTransport

        config = MCPServerConfig(
            name="t",
            transport=MCPTransportType.HTTP,
            url="http://localhost:1234/mcp",
        )
        transport = _HTTPTransport(config)
        # Inject a mock httpx client.
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        transport._client = mock_client

        await transport.send({"jsonrpc": "2.0", "id": 1, "method": "test"})
        result = await transport.receive()

        assert result == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
        mock_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_http_error_raises_mcp_error(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from justagent.agent.mcp_client import _HTTPTransport

        config = MCPServerConfig(
            name="t",
            transport=MCPTransportType.HTTP,
            url="http://localhost:1234/mcp",
        )
        transport = _HTTPTransport(config)
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        transport._client = mock_client

        with pytest.raises(MCPError, match="HTTP request failed"):
            await transport.send({"jsonrpc": "2.0", "id": 1, "method": "test"})

    @pytest.mark.asyncio
    async def test_parse_sse(self) -> None:
        from justagent.agent.mcp_client import _HTTPTransport

        sse_text = (
            "event: message\n"
            'data: {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}\n'
            "\n"
        )
        parsed = _HTTPTransport._parse_sse(sse_text)
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["result"] == {"tools": []}

    @pytest.mark.asyncio
    async def test_parse_sse_no_data_raises(self) -> None:
        from justagent.agent.mcp_client import _HTTPTransport

        with pytest.raises(MCPError, match="No data"):
            _HTTPTransport._parse_sse("event: ping\n\n")

    @pytest.mark.asyncio
    async def test_start_requires_url(self) -> None:
        from justagent.agent.mcp_client import _HTTPTransport

        config = MCPServerConfig(
            name="t",
            transport=MCPTransportType.HTTP,
            url="",
        )
        transport = _HTTPTransport(config)
        with pytest.raises(MCPError, match="requires a url"):
            await transport.start()


# ---------------------------------------------------------------------------
# _StdioTransport validation
# ---------------------------------------------------------------------------


class TestStdioTransportValidation:
    @pytest.mark.asyncio
    async def test_start_requires_command(self) -> None:
        from justagent.agent.mcp_client import _StdioTransport

        config = MCPServerConfig(
            name="t",
            transport=MCPTransportType.STDIO,
            command="",
        )
        transport = _StdioTransport(config)
        with pytest.raises(MCPError, match="requires a command"):
            await transport.start()

    @pytest.mark.asyncio
    async def test_send_before_start_raises(self) -> None:
        from justagent.agent.mcp_client import _StdioTransport

        config = MCPServerConfig(
            name="t",
            transport=MCPTransportType.STDIO,
            command="echo",
        )
        transport = _StdioTransport(config)
        with pytest.raises(MCPError, match="not started"):
            await transport.send({"jsonrpc": "2.0", "method": "test"})

    @pytest.mark.asyncio
    async def test_close_when_not_started_is_noop(self) -> None:
        from justagent.agent.mcp_client import _StdioTransport

        config = MCPServerConfig(
            name="t",
            transport=MCPTransportType.STDIO,
            command="echo",
        )
        transport = _StdioTransport(config)
        await transport.close()  # Should not raise.
