"""MCP (Model Context Protocol) client — connect to external tool servers.

MCP is an open protocol (JSON-RPC 2.0) that lets the agent call tools
provided by external servers. Servers can run as:

* **stdio** — a subprocess launched by the client, communicating via
  stdin/stdout JSON-RPC lines.
* **SSE** — a Server-Sent Events endpoint for server→client messages
  plus an HTTP POST endpoint for client→server.
* **HTTP** (streamable) — a single HTTP endpoint supporting both
  directions via streaming.

This module implements a lightweight client using ``httpx`` (already a
dependency) and ``subprocess``/``asyncio``. If the official ``mcp``
Python SDK is installed, it is preferred as the backend.

Reference: https://modelcontextprotocol.io spec.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, cast

import httpx

from justagent.exceptions import MyAgentError

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

_MCP_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_NAME = "justagent"
_CLIENT_VERSION = "2.0.0"


def _is_mcp_sdk_available() -> bool:
    """Return True if the official ``mcp`` SDK is installed.

    The SDK is an optional backend (declared via the ``mcp`` extra in
    ``pyproject.toml``). When unavailable, the lightweight client
    implemented in this module is used instead. The import is performed
    inside the function so the module loads cleanly without the SDK.
    """

    try:
        import mcp  # noqa: F401  — presence check only
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MCPError(MyAgentError):
    """Base error for all MCP client operations."""


class JSONRPCError(MCPError):
    """Raised when a JSON-RPC response contains an ``error`` object.

    Attributes:
        rpc_code: The JSON-RPC error code from the server.
        data: Optional error data payload from the server.
    """

    def __init__(self, rpc_code: int, message: str, data: Any = None) -> None:
        super().__init__(message, details={"rpc_code": rpc_code, "data": data})
        self.rpc_code = rpc_code
        self.data = data


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MCPTransportType(str, Enum):  # noqa: UP042
    """Transport mechanism used to talk to an MCP server."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class MCPConnectionState(str, Enum):  # noqa: UP042
    """Lifecycle state of an :class:`MCPClient` connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPTool:
    """A tool exposed by an MCP server.

    Attributes:
        name: Tool name (used in ``tools/call``).
        description: Human-readable description.
        input_schema: JSON Schema describing accepted arguments.
        server_name: Name of the server that owns this tool.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass(frozen=True)
class MCPResource:
    """A resource exposed by an MCP server.

    Attributes:
        uri: Resource URI (e.g. ``file:///path``).
        name: Human-readable name.
        description: Human-readable description.
        mimeType: MIME type of the resource content.
    """

    uri: str
    name: str
    description: str
    mimeType: str = ""  # noqa: N815  — MCP protocol field is mixedCase


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for a single MCP server connection.

    Attributes:
        name: Unique server name (used for routing tool calls).
        transport: Transport type (stdio / sse / http).
        command: Executable command for stdio transport.
        args: Command-line arguments for stdio transport.
        url: Endpoint URL for SSE / HTTP transport.
        env: Extra environment variables for stdio transport.
        headers: Extra HTTP headers for SSE / HTTP transport.
        enabled: If False, the server is skipped by :class:`MCPManager`.
        timeout_seconds: Per-request timeout.
    """

    name: str
    transport: MCPTransportType
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout_seconds: float = 30.0


# ---------------------------------------------------------------------------
# OAuth support
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPOAuthToken:
    """An OAuth bearer token for authenticating to an MCP server.

    Attributes:
        access_token: The token string sent in the ``Authorization``
            header.
        token_type: Token type (default ``Bearer``).
        expires_at: Unix epoch seconds when the token expires (0 = never
            / unknown).
        refresh_token: Optional refresh token.
    """

    access_token: str
    token_type: str = "Bearer"
    expires_at: float = 0.0
    refresh_token: str = ""


class OAuthTokenProvider(Protocol):
    """Protocol for objects that can supply OAuth tokens.

    Implementations may fetch tokens from disk, a keyring, or an
    authorization-code flow. The HTTP transport calls
    :meth:`get_token` when establishing a connection.
    """

    async def get_token(self) -> MCPOAuthToken | None:
        """Return the current token, or ``None`` if none is available."""
        ...

    async def refresh(self) -> MCPOAuthToken | None:
        """Refresh and return a new token, or ``None`` on failure."""
        ...


class StaticTokenProvider:
    """A token provider that always returns the same fixed token.

    Useful for testing or when a long-lived API key is used in place of
    a real OAuth flow.
    """

    def __init__(self, token: MCPOAuthToken) -> None:
        self._token = token

    async def get_token(self) -> MCPOAuthToken | None:
        return self._token

    async def refresh(self) -> MCPOAuthToken | None:
        return self._token


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 helpers
# ---------------------------------------------------------------------------


def _jsonrpc_request(
    id: int | str, method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request object (with ``id``)."""

    message: dict[str, Any] = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def _jsonrpc_notification(
    method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 notification (no ``id``, no response)."""

    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    return message


def _jsonrpc_response(id: int | str, result: Any) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 success response."""

    return {"jsonrpc": "2.0", "id": id, "result": result}


def _jsonrpc_error(id: int | str, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response."""

    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message},
    }


# ---------------------------------------------------------------------------
# Transport abstraction
# ---------------------------------------------------------------------------


class _Transport(Protocol):
    """Async transport for sending/receiving JSON-RPC messages."""

    async def send(self, message: dict[str, Any]) -> None:
        """Send a single JSON-RPC message to the server."""
        ...

    async def receive(self) -> dict[str, Any]:
        """Receive a single JSON-RPC message from the server."""
        ...

    async def close(self) -> None:
        """Close the transport and release resources."""
        ...


class _StdioTransport:
    """JSON-RPC over a subprocess's stdin/stdout (newline-delimited).

    The subprocess is launched via :func:`asyncio.create_subprocess_exec`
    using ``config.command`` and ``config.args``. Messages are written as
    one JSON object per line on stdin and read line-by-line from stdout.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Launch the subprocess."""

        if not self._config.command:
            raise MCPError(
                "stdio transport requires a command",
                details={"server": self._config.name},
            )
        env = {**os.environ, **self._config.env}
        self._process = await asyncio.create_subprocess_exec(
            self._config.command,
            *self._config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def send(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise MCPError("stdio transport not started")
        line = json.dumps(message) + "\n"
        data = line.encode("utf-8")
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise MCPError("stdio transport not started")
        line = await self._process.stdout.readline()
        if not line:
            raise MCPError("Connection closed by server")
        try:
            return cast(dict[str, Any], json.loads(line.decode("utf-8")))
        except json.JSONDecodeError as exc:
            raise MCPError(
                f"Invalid JSON-RPC line: {exc}",
                details={"line": line.decode("utf-8", errors="replace")},
            ) from exc

    async def close(self) -> None:
        if self._process is None:
            return
        proc = self._process
        if proc.stdin is not None:
            proc.stdin.close()
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        self._process = None


class _HTTPTransport:
    """JSON-RPC over HTTP (single POST per request) or SSE.

    Each :meth:`send` POSTs the JSON-RPC message to ``config.url`` and
    buffers the response body. :meth:`receive` returns the buffered
    response. When the server responds with ``text/event-stream``, the
    first ``data:`` line is parsed as the JSON-RPC message.
    """

    def __init__(
        self,
        config: MCPServerConfig,
        token_provider: OAuthTokenProvider | None = None,
    ) -> None:
        self._config = config
        self._token_provider = token_provider
        self._client: httpx.AsyncClient | None = None
        self._pending: dict[str, Any] | None = None

    async def start(self) -> None:
        """Create the underlying ``httpx.AsyncClient``."""

        if not self._config.url:
            raise MCPError(
                "HTTP/SSE transport requires a url",
                details={"server": self._config.name},
            )
        headers = dict(self._config.headers)
        if self._token_provider is not None:
            token = await self._token_provider.get_token()
            if token is not None:
                headers["Authorization"] = f"{token.token_type} {token.access_token}"
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=self._config.timeout_seconds,
        )

    async def send(self, message: dict[str, Any]) -> None:
        if self._client is None:
            raise MCPError("HTTP transport not started")
        try:
            response = await self._client.post(self._config.url, json=message)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MCPError(
                f"HTTP request failed: {exc}",
                details={"server": self._config.name},
            ) from exc
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            self._pending = self._parse_sse(response.text)
        else:
            try:
                self._pending = response.json()
            except json.JSONDecodeError as exc:
                raise MCPError(
                    f"Invalid JSON response: {exc}",
                    details={"body": response.text},
                ) from exc

    async def receive(self) -> dict[str, Any]:
        if self._pending is None:
            raise MCPError("No pending response")
        result = self._pending
        self._pending = None
        return result

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _parse_sse(text: str) -> dict[str, Any]:
        """Parse the first ``data:`` payload from an SSE stream."""

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:"):
                payload = stripped[5:].strip()
                if payload:
                    try:
                        return cast(dict[str, Any], json.loads(payload))
                    except json.JSONDecodeError as exc:
                        raise MCPError(
                            f"Invalid SSE data: {exc}",
                            details={"data": payload},
                        ) from exc
        raise MCPError("No data in SSE response")


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------


class MCPClient:
    """A client for a single MCP server.

    Handles the MCP handshake (``initialize`` + ``notifications/initialized``)
    and exposes ``tools/list``, ``tools/call``, ``resources/list``, and
    ``resources/read``.

    Example::

        config = MCPServerConfig(
            name="fs",
            transport=MCPTransportType.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        client = MCPClient(config)
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "/tmp/x"})
        await client.disconnect()
    """

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        token_provider: OAuthTokenProvider | None = None,
        _transport: _Transport | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            config: Server configuration.
            token_provider: Optional OAuth token provider for HTTP/SSE.
            _transport: Test hook — inject a pre-built transport instead
                of creating one from ``config``. When provided, the
                client skips transport creation and uses this directly.
        """

        self._config = config
        self._token_provider = token_provider
        self._transport: _Transport | None = _transport
        self._state = MCPConnectionState.DISCONNECTED
        self._id_counter = 0
        self._transport_owned = _transport is None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> MCPConnectionState:
        """Current connection state."""

        return self._state

    @property
    def server_name(self) -> str:
        """Name of the server this client is connected to."""

        return self._config.name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish the connection and perform the MCP handshake.

        Sends ``initialize`` then ``notifications/initialized``. On
        failure, the state is set to :attr:`MCPConnectionState.ERROR`.

        Raises:
            MCPError: If already connected or the handshake fails.
        """

        if self._state is MCPConnectionState.CONNECTED:
            raise MCPError(
                f"Server {self._config.name!r} is already connected"
            )
        self._state = MCPConnectionState.CONNECTING
        try:
            if self._transport is None:
                self._transport = await self._create_and_start_transport()
            await self._do_handshake()
            self._state = MCPConnectionState.CONNECTED
        except Exception:
            self._state = MCPConnectionState.ERROR
            # Close any half-open transport we created.
            if self._transport_owned and self._transport is not None:
                with contextlib.suppress(Exception):
                    await self._transport.close()
                self._transport = None
            raise

    async def disconnect(self) -> None:
        """Close the connection gracefully.

        Safe to call when already disconnected (no-op).
        """

        if self._transport is not None:
            with contextlib.suppress(Exception):
                await self._transport.close()
            if self._transport_owned:
                self._transport = None
        self._state = MCPConnectionState.DISCONNECTED

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[MCPTool]:
        """Call ``tools/list`` and return the advertised tools."""

        self._require_connected()
        result = await self._request("tools/list")
        tools_data = (
            result.get("tools", []) if isinstance(result, dict) else []
        )
        return [
            MCPTool(
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
                server_name=self._config.name,
            )
            for tool in tools_data
            if isinstance(tool, dict)
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call ``tools/call`` and return the result content.

        If all content blocks are text, their text is joined with
        newlines and returned as a string. Otherwise the raw content
        list is returned.

        Raises:
            MCPError: If the server reports an error (``isError``).
        """

        self._require_connected()
        result = await self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        if isinstance(result, dict) and result.get("isError"):
            content = result.get("content", [])
            text = self._join_text(content)
            raise MCPError(
                f"Tool {name!r} returned an error: {text}",
                details={"tool": name, "content": content},
            )
        if isinstance(result, dict):
            content = result.get("content", [])
            return self._format_content(content)
        return result

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    async def list_resources(self) -> list[MCPResource]:
        """Call ``resources/list`` and return the advertised resources."""

        self._require_connected()
        result = await self._request("resources/list")
        resources_data = (
            result.get("resources", []) if isinstance(result, dict) else []
        )
        return [
            MCPResource(
                uri=resource.get("uri", ""),
                name=resource.get("name", ""),
                description=resource.get("description", ""),
                mimeType=resource.get("mimeType", ""),
            )
            for resource in resources_data
            if isinstance(resource, dict)
        ]

    async def read_resource(self, uri: str) -> Any:
        """Call ``resources/read`` and return the raw result."""

        self._require_connected()
        return await self._request("resources/read", {"uri": uri})

    # ------------------------------------------------------------------
    # Internal: handshake
    # ------------------------------------------------------------------

    async def _do_handshake(self) -> None:
        """Send ``initialize`` + ``notifications/initialized``."""

        params = {
            "protocolVersion": _MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": _CLIENT_NAME, "version": _CLIENT_VERSION},
        }
        await self._request("initialize", params)
        await self._notification("notifications/initialized")

    # ------------------------------------------------------------------
    # Internal: JSON-RPC request / notification
    # ------------------------------------------------------------------

    async def _request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send a JSON-RPC request and wait for the matching response."""

        if self._transport is None:
            raise MCPError("Transport is not available")
        self._id_counter += 1
        req_id = self._id_counter
        message = _jsonrpc_request(req_id, method, params)
        await self._transport.send(message)
        try:
            response = await asyncio.wait_for(
                self._transport.receive(),
                timeout=self._config.timeout_seconds,
            )
        except TimeoutError as exc:
            raise MCPError(
                f"Timed out waiting for response to {method!r}",
                details={"method": method, "timeout": self._config.timeout_seconds},
            ) from exc
        return self._check_response(response, req_id)

    async def _notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""

        if self._transport is None:
            raise MCPError("Transport is not available")
        message = _jsonrpc_notification(method, params)
        await self._transport.send(message)

    @staticmethod
    def _check_response(response: dict[str, Any], expected_id: int | str) -> Any:
        """Validate a JSON-RPC response and return its ``result``."""

        if "error" in response:
            err = response["error"]
            if not isinstance(err, dict):
                raise JSONRPCError(-32603, "Malformed error object", err)
            raise JSONRPCError(
                err.get("code", -32603),
                err.get("message", "Unknown error"),
                err.get("data"),
            )
        if "result" not in response:
            raise JSONRPCError(
                -32603,
                f"Response missing 'result' for id {expected_id!r}",
                response,
            )
        return response["result"]

    # ------------------------------------------------------------------
    # Internal: content formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_content(content: Any) -> Any:
        """Join text blocks or return the raw content list."""

        if not isinstance(content, list):
            return content
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            else:
                # Non-text block present — return the raw list.
                return content
        if texts:
            return "\n".join(texts)
        return content

    @staticmethod
    def _join_text(content: Any) -> str:
        """Join all text blocks in ``content`` into a single string."""

        if not isinstance(content, list):
            return str(content)
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal: transport creation
    # ------------------------------------------------------------------

    async def _create_and_start_transport(self) -> _Transport:
        """Create and start a real transport based on the config."""

        transport: _Transport
        if self._config.transport is MCPTransportType.STDIO:
            stdio = _StdioTransport(self._config)
            await stdio.start()
            transport = stdio
        elif self._config.transport in (MCPTransportType.SSE, MCPTransportType.HTTP):
            http = _HTTPTransport(self._config, self._token_provider)
            await http.start()
            transport = http
        else:
            raise MCPError(
                f"Unsupported transport: {self._config.transport}",
                details={"server": self._config.name},
            )
        return transport

    # ------------------------------------------------------------------
    # Internal: guards
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        """Raise :class:`MCPError` if the client is not connected."""

        if self._state is not MCPConnectionState.CONNECTED:
            raise MCPError(
                f"Server {self._config.name!r} is not connected "
                f"(state={self._state.value})",
                details={"state": self._state.value},
            )


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------


class MCPManager:
    """Manages multiple :class:`MCPClient` instances.

    The manager owns server configs and their connected clients. It
    provides aggregate operations like connecting all servers at once,
    listing tools across all servers, and routing a tool call to the
    correct server.

    Example::

        manager = MCPManager()
        manager.add_server(stdio_config)
        manager.add_server(http_config)
        results = await manager.connect_all()
        all_tools = await manager.list_all_tools()
        result = await manager.call_tool("fs", "read_file", {"path": "/x"})
        await manager.disconnect_all()
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------

    def add_server(self, config: MCPServerConfig) -> None:
        """Register a server configuration (does not connect)."""

        self._servers[config.name] = config

    def remove_server(self, name: str) -> bool:
        """Remove a server and disconnect its client if connected.

        Returns ``True`` if the server was registered.
        """

        existed = name in self._servers
        self._servers.pop(name, None)
        client = self._clients.pop(name, None)
        if client is not None and client.state is MCPConnectionState.CONNECTED:
            # Best-effort disconnect; the caller can await
            # :meth:`disconnect_all` for a clean shutdown.
            asyncio.ensure_future(client.disconnect())
        return existed

    def list_servers(self) -> list[MCPServerConfig]:
        """Return all registered server configs (insertion order)."""

        return list(self._servers.values())

    def get_server(self, name: str) -> MCPClient | None:
        """Return the connected client for ``name``, or ``None``."""

        return self._clients.get(name)

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------

    async def connect_all(self) -> dict[str, Exception | None]:
        """Connect all enabled servers concurrently.

        Returns a map of server name → exception (``None`` on success).
        Disabled servers are skipped (not present in the result).
        """

        enabled = [
            (name, config)
            for name, config in self._servers.items()
            if config.enabled
        ]

        async def _connect_one(
            name: str, config: MCPServerConfig
        ) -> tuple[str, Exception | None]:
            client = MCPClient(config)
            try:
                await client.connect()
                self._clients[name] = client
                return name, None
            except Exception as exc:  # noqa: BLE001 — report all failures
                return name, exc

        results_raw = await asyncio.gather(
            *(_connect_one(name, cfg) for name, cfg in enabled),
            return_exceptions=False,
        )
        return dict(results_raw)

    async def disconnect_all(self) -> None:
        """Disconnect all connected clients."""

        clients = list(self._clients.values())
        self._clients.clear()
        await asyncio.gather(
            *(client.disconnect() for client in clients),
            return_exceptions=True,
        )

    # ------------------------------------------------------------------
    # Aggregate operations
    # ------------------------------------------------------------------

    async def list_all_tools(self) -> list[MCPTool]:
        """Aggregate tools from all connected servers."""

        connected = [
            client
            for client in self._clients.values()
            if client.state is MCPConnectionState.CONNECTED
        ]
        batches = await asyncio.gather(
            *(client.list_tools() for client in connected),
            return_exceptions=True,
        )
        tools: list[MCPTool] = []
        for batch in batches:
            if isinstance(batch, list):
                tools.extend(batch)
        return tools

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """Route a tool call to the named server.

        Raises:
            MCPError: If the server is not connected.
        """

        client = self._clients.get(server_name)
        if client is None:
            raise MCPError(
                f"Server {server_name!r} is not connected",
                details={"server": server_name},
            )
        return await client.call_tool(tool_name, arguments)


__all__ = [
    "JSONRPCError",
    "MCPClient",
    "MCPConnectionState",
    "MCPError",
    "MCPManager",
    "MCPOAuthToken",
    "MCPResource",
    "MCPServerConfig",
    "MCPTool",
    "MCPTransportType",
    "OAuthTokenProvider",
    "StaticTokenProvider",
]
