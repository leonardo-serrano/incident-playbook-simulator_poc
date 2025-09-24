# orchestrator/mcp_tools.py
"""
MCP toolbox with selectable transport:
- stdio (legacy, kept for backward compatibility)
- http  (recommended): calls minimal HTTP wrapper around MCP tools

Direct-mode (import module and call functions) has been removed to simplify the design
and align with the recommendation to avoid custom fallbacks.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from utils.logging_utils import async_tool_span
from orchestrator.config import MCP_TRANSPORT, MCP_SERVER_URL
import httpx


class MCPToolbox:
    """Facade for calling our tools via either MCP stdio or HTTP."""

    def __init__(self, server_path: Path, handshake_timeout: float = 10.0):
        self._log = logging.getLogger("tools")
        self.server_path = Path(server_path)
        self.handshake_timeout = handshake_timeout

        # transport selection
        self._transport = (MCP_TRANSPORT or "stdio").strip().lower()

        # stdio mode internals
        self._ctx = None
        self._streams: Tuple[Any, Any] | None = None
        self.session: ClientSession | None = None
        
        # http mode internals
        self._http: Optional[httpx.AsyncClient] = None

    # --------------------------- lifecycle -------------------------------- #
    async def __aenter__(self) -> "MCPToolbox":
        """
        Initialize the selected transport.
        """
        if self._transport == "http":
            base_url = MCP_SERVER_URL.strip()
            if not base_url:
                raise RuntimeError("MCP_TRANSPORT=http requires MCP_SERVER_URL to be set (e.g., http://127.0.0.1:8765)")
            # Optional shared token for hardened HTTP server
            token = os.getenv("MCP_HTTP_TOKEN", "").strip()
            headers = {"X-Auth-Token": token} if token else None
            self._http = httpx.AsyncClient(base_url=base_url, timeout=self.handshake_timeout, headers=headers)
            # Optional: ping a non-existent endpoint to warm DNS/connection
            return self

        # default: stdio
        if not self.server_path.exists():
            raise FileNotFoundError(f"MCP server not found at {self.server_path}")

        params = StdioServerParameters(
            command=sys.executable,
            args=["-u", str(self.server_path)],  # unbuffered Python to avoid stalled pipes
        )

        # Open stdio transport itself with a timeout
        self._ctx = stdio_client(params)
        read, write = await asyncio.wait_for(self._ctx.__aenter__(), timeout=self.handshake_timeout)
        self._streams = (read, write)
        self.session = ClientSession(read, write)

        # Perform the JSON-RPC handshake with a timeout
        await asyncio.wait_for(self.session.initialize(), timeout=self.handshake_timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._http is not None:
            with contextlib.suppress(Exception):
                await self._http.aclose()
            self._http = None
            return
        with contextlib.suppress(Exception):
            if self.session is not None:
                await self.session.close()
        with contextlib.suppress(Exception):
            if self._ctx is not None:
                await self._ctx.__aexit__(exc_type, exc, tb)
        self._ctx = None
        self._streams = None
        self.session = None

    # --------------------------- call helpers ------------------------------ #
    async def _call_stdio(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        assert self.session is not None, "MCP stdio session not initialized"
        result = await self.session.call_tool(name, args)
        if not result.content:
            return {}
        part = result.content[0]
        data = getattr(part, "text", None) or getattr(part, "data", None) or part
        return json.loads(data) if isinstance(data, str) else data

    async def _call_http(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        assert self._http is not None, "MCP http client not initialized"
        resp = await self._http.post(f"/tool/{name}", json={"args": args})
        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP tool error: {data}")
        return data.get("result") or {}
        
    async def _call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._transport == "http":
            return await self._call_http(name, args)
        return await self._call_stdio(name, args)

    # ------------------------------ tools --------------------------------- #
    async def search_playbooks(self, query: str, k: int = 3) -> Dict[str, Any]:
        params = {"query": query, "k": k}
        async with async_tool_span(self._log, "search_playbooks", params):
            # call underlying implementation (direct or stdio)
            return await self._call("search_playbooks", params)

    async def get_metrics(self, service: str) -> Dict[str, Any]:
        """Adapter: map 'service' -> 'service_name' for the direct-call server function."""
        params = {"service_name": service}  # <-- was {"service": service}
        async with async_tool_span(self._log, "get_metrics", params):
            return await self._call("get_metrics", params)

    async def query_logs(
        self,
        service: str,
        *,
        pattern: str | None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Adapter: map 'service' -> 'service_name' for the direct-call server function."""
        params = {"service_name": service, "pattern": pattern, "limit": limit}  # <-- was {"service": service, ...}
        async with async_tool_span(self._log, "query_logs", params):
            return await self._call("query_logs", params)

    async def scale_instance(self, service: str, target: int) -> Dict[str, Any]:
        """Adapter: map 'service' -> 'service_name' for direct-call server function."""
        params = {"service_name": service, "target": target}  # <-- key fix
        async with async_tool_span(self._log, "scale_instance", params):
            return await self._call("scale_instance", params)

    async def rollback_deployment(self, service: str) -> Dict[str, Any]:
        """Adapter: map 'service' -> 'service_name' for direct-call server function."""
        params = {"service_name": service}  # <-- key fix
        async with async_tool_span(self._log, "rollback_deployment", params):
            return await self._call("rollback_deployment", params)

    async def clear_cache(self, scope: str = "global") -> Dict[str, Any]:
        params = {"scope": scope}
        async with async_tool_span(self._log, "clear_cache", params):
            return await self._call("clear_cache", params)

    async def notify_team(self, message: str, channel: str = "incident-bridge") -> Dict[str, Any]:
        params = {"message": message, "channel": channel}
        async with async_tool_span(self._log, "notify_team", params):
            return await self._call("notify_team", params)

    async def ask_human(self, prompt: str) -> Dict[str, Any]:
        params = {"prompt": prompt}
        async with async_tool_span(self._log, "ask_human", params):
            return await self._call("ask_human", params)
