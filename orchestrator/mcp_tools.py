# orchestrator/mcp_tools.py
"""
MCP toolbox with a resilient startup:
- Tries MCP stdio first (unbuffered -u on Windows).
- Applies a handshake timeout.
- If anything goes wrong, falls back to "direct mode" (importing mcp_server.server).
- Cleanup during failure is fully suppressed so no exception leaks out.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from utils.logging_utils import async_tool_span


class MCPToolbox:
    """Facade for calling our tools via MCP stdio, with a transparent direct fallback."""

    def __init__(self, server_path: Path, handshake_timeout: float = 3.0):
        self._log = logging.getLogger("tools")
        self.server_path = Path(server_path)
        self.handshake_timeout = handshake_timeout

        # stdio mode internals
        self._ctx = None
        self._streams: Tuple[Any, Any] | None = None
        self.session: ClientSession | None = None

        # direct mode internals
        self._direct = False
        self._srv_mod = None  # module reference (mcp_server.server)

    # --------------------------- lifecycle -------------------------------- #
    async def __aenter__(self) -> "MCPToolbox":
        """
        Try stdio MCP first; if it doesn't initialize within timeout,
        switch to direct-import mode.
        """
        if not self.server_path.exists():
            raise FileNotFoundError(f"MCP server not found at {self.server_path}")

        params = StdioServerParameters(
            command=sys.executable,
            args=["-u", str(self.server_path)],  # unbuffered Python to avoid stalled pipes
        )

        try:
            # Open stdio transport itself with a timeout
            self._ctx = stdio_client(params)
            read, write = await asyncio.wait_for(self._ctx.__aenter__(), timeout=self.handshake_timeout)
            self._streams = (read, write)
            self.session = ClientSession(read, write)

            # Perform the JSON-RPC handshake with a timeout
            await asyncio.wait_for(self.session.initialize(), timeout=self.handshake_timeout)
            return self  # stdio mode is ready

        except Exception as e:
            # Best-effort cleanup; never allow exceptions to escape from here
            with contextlib.suppress(Exception):
                if self.session is not None:
                    await self.session.close()
            with contextlib.suppress(Exception):
                if self._ctx is not None:
                    await self._ctx.__aexit__(type(e), e, e.__traceback__)

            # Reset stdio internals
            self.session = None
            self._ctx = None
            self._streams = None

            # Enable direct mode so the graph can still run
            self._enable_direct_mode()
            return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._direct:
            return  # nothing to close in direct mode
        with contextlib.suppress(Exception):
            if self.session is not None:
                await self.session.close()
        with contextlib.suppress(Exception):
            if self._ctx is not None:
                await self._ctx.__aexit__(exc_type, exc, tb)
        self._ctx = None
        self._streams = None
        self.session = None

    def _enable_direct_mode(self):
        """Import the server module and mark the toolbox as direct-mode."""
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import importlib
        self._srv_mod = importlib.import_module("mcp_server.server")
        self._direct = True

    # --------------------------- call helpers ------------------------------ #
    async def _call_stdio(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        assert self.session is not None, "MCP stdio session not initialized"
        result = await self.session.call_tool(name, args)
        if not result.content:
            return {}
        part = result.content[0]
        data = getattr(part, "text", None) or getattr(part, "data", None) or part
        return json.loads(data) if isinstance(data, str) else data

    async def _call_direct(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the underlying Python function when running in 'direct mode'.
        Tools decorated with @mcp.tool() are wrappers (e.g., FunctionTool).
        We try to recover the real callable from:
          1) The attribute itself (if it's a plain function).
          2) Unwrapping common attributes on the wrapper (func/fn/_func/__wrapped__/_callable/handler/_handler).
          3) The MCP registry (mcp.tools / mcp._tools / mcp.registry / mcp._registry) as dict or list.
          4) Last resort: scan any callable attribute on the wrapper object.
        """
        def _unwrap_callable(obj):
            # Try common attributes that usually hold the original function
            for attr in ("func", "fn", "_func", "__wrapped__", "_callable", "callable", "handler", "_handler"):
                fn = getattr(obj, attr, None)
                if callable(fn):
                    return fn
            # Fallback: scan object __dict__ looking for any callable
            try:
                for _, val in vars(obj).items():
                    if callable(val):
                        return val
            except Exception:
                pass
            return None
        # (1) Attribute directly callable?
        obj = getattr(self._srv_mod, name, None)
        if callable(obj):
            return obj(**args)
        # (2) Try to unwrap the attribute itself (FunctionTool wrapper, etc.)
        if obj is not None:
            fn = _unwrap_callable(obj)
            if callable(fn):
                return fn(**args)
        # (3) Look inside the MCP registry for a tool by name and unwrap it
        mcp_inst = getattr(self._srv_mod, "mcp", None)
        if mcp_inst is not None:
            for reg_name in ("tools", "_tools", "registry", "_registry"):
                reg = getattr(mcp_inst, reg_name, None)
                if reg is None:
                    continue
                tool_obj = None
                # dict-like
                if hasattr(reg, "get"):
                    tool_obj = reg.get(name)
                # list/tuple of tool objects with .name
                if tool_obj is None and isinstance(reg, (list, tuple)):
                    tool_obj = next((t for t in reg if getattr(t, "name", None) == name), None)

                if tool_obj is not None:
                    fn = _unwrap_callable(tool_obj)
                    if callable(fn):
                        return fn(**args)
        # (4) Give up with a helpful error
        tname = type(obj).__name__ if obj is not None else "None"
        raise TypeError(f"Cannot call tool '{name}' in direct mode; got object type={tname}")
        
    async def _call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return await (self._call_direct if self._direct else self._call_stdio)(name, args)

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
