# mcp_server/http_server.py
"""
Minimal HTTP server exposing MCP tools over HTTP.

Endpoints:
- POST /tool/<name>  with JSON body {"args": {...}}
  Returns JSON result of the tool call or error payload.

Usage:
  python3 -m mcp_server.http_server --host 127.0.0.1 --port 8765

This wraps the existing functions defined in mcp_server/server.py
without requiring stdio. Useful while migrating away from stdio MCP.

Security hardening (added):
- Optional token auth via header "X-Auth-Token". Token read from env MCP_HTTP_TOKEN.
- If no token is set, only loopback clients are allowed.
- Allowlist of permitted tools enforced server-side.
- Simple request body size limit for POSTs.
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# Import the MCP server module to reuse the tool functions
from mcp_server import server as mcp_srv  # type: ignore

# --------------------- Security configuration --------------------- #
# Allowlist of tool names exposed over HTTP. Keep in sync with server.py tools.
ALLOWED_TOOLS = {
    "get_metrics",
    "query_logs",
    "search_playbooks",
    "scale_instance",
    "rollback_deployment",
    "clear_cache",
    "notify_team",
    "ask_human",
    "ping",
}

# Optional shared token for HTTP auth. When set, clients must send X-Auth-Token.
MCP_HTTP_TOKEN = os.getenv("MCP_HTTP_TOKEN", "").strip()

# Max request body size (bytes) for POST /tool/<name>
MAX_BODY_BYTES = int(os.getenv("MCP_HTTP_MAX_BODY", "262144"))  # 256 KiB default


def _resolve_tool(name: str):
    """Return a callable for tool 'name' or None if not found."""
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

    # Direct attribute on module
    fn = getattr(mcp_srv, name, None)
    call = fn if callable(fn) else (_unwrap_callable(fn) if fn is not None else None)
    if callable(call):
        return call
    # Registry lookup
    mcp_inst = getattr(mcp_srv, "mcp", None)
    if mcp_inst is not None:
        for reg_name in ("tools", "_tools", "registry", "_registry"):
            reg = getattr(mcp_inst, reg_name, None)
            if reg is None:
                continue
            tool_obj = None
            if hasattr(reg, "get"):
                tool_obj = reg.get(name)
            if tool_obj is None and isinstance(reg, (list, tuple)):
                tool_obj = next((t for t in reg if getattr(t, "name", None) == name), None)
            if tool_obj is not None:
                call = _unwrap_callable(tool_obj) or (tool_obj if callable(tool_obj) else None)
                if callable(call):
                    return call
    return None


class ToolHTTPHandler(BaseHTTPRequestHandler):
    server_version = "MCPHTTP/0.1"

    def _send(self, code: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # --------------------------- Auth helpers --------------------------- #
    def _is_loopback(self) -> bool:
        host = (self.client_address[0] if self.client_address else "") or ""
        return host in {"127.0.0.1", "::1", "localhost"}

    def _check_auth(self) -> bool:
        """Return True if the request is authorized, else send error and return False.

        Policy:
        - If MCP_HTTP_TOKEN is set, require header X-Auth-Token to match exactly.
        - If MCP_HTTP_TOKEN is not set, only allow loopback clients.
        """
        if MCP_HTTP_TOKEN:
            token = self.headers.get("X-Auth-Token", "")
            if token != MCP_HTTP_TOKEN:
                self._send(401, {"error": "unauthorized", "detail": "Missing or invalid X-Auth-Token"})
                return False
            return True
        # No token configured → restrict to local loopback only
        if not self._is_loopback():
            self._send(403, {"error": "forbidden", "detail": "Remote access requires MCP_HTTP_TOKEN"})
            return False
        return True

    def do_GET(self):  # noqa: N802
        if not self._check_auth():
            return
        parsed = urlparse(self.path or "/")
        parts = [p for p in (parsed.path or "/").split("/") if p]
        if len(parts) == 1 and parts[0] == "tools":
            # list visible tool names by checking module attrs and registry, then filter by allowlist
            names = set()
            # module attributes
            for k, v in vars(mcp_srv).items():
                if callable(v):
                    names.add(k)
                else:
                    uv = getattr(v, "name", None)
                    if isinstance(uv, str):
                        names.add(uv)
            # registry
            mcp_inst = getattr(mcp_srv, "mcp", None)
            if mcp_inst is not None:
                for reg_name in ("tools", "_tools", "registry", "_registry"):
                    reg = getattr(mcp_inst, reg_name, None)
                    if hasattr(reg, "keys"):
                        names.update(list(reg.keys()))
                    if isinstance(reg, (list, tuple)):
                        for t in reg:
                            n = getattr(t, "name", None)
                            if isinstance(n, str):
                                names.add(n)
            allowed = sorted(n for n in names if isinstance(n, str) and n in ALLOWED_TOOLS)
            self._send(200, {"tools": allowed})
            return
        self._send(404, {"error": "not_found", "detail": "Use GET /tools or POST /tool/<name>"})

    def do_POST(self):  # noqa: N802
        if not self._check_auth():
            return
        try:
            parsed = urlparse(self.path or "/")
            parts = [p for p in (parsed.path or "/").split("/") if p]
            # Expect path like /tool/<name>
            if len(parts) != 2 or parts[0] != "tool":
                self._send(404, {"error": "not_found", "detail": "Use POST /tool/<name>"})
                return
            name = parts[1]

            if name not in ALLOWED_TOOLS:
                self._send(403, {"error": "forbidden", "detail": f"Tool not allowed: {name}"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_BODY_BYTES:
                self._send(413, {"error": "payload_too_large", "detail": f"Body exceeds {MAX_BODY_BYTES} bytes"})
                return
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception as e:
                self._send(400, {"error": "invalid_json", "detail": str(e)})
                return
            args = body.get("args") or {}
            if not isinstance(args, dict):
                self._send(400, {"error": "invalid_args", "detail": "'args' must be an object"})
                return

            call = _resolve_tool(name)

            if call is None or not callable(call):
                self._send(404, {"error": "unknown_tool", "detail": f"No such tool: {name}"})
                return

            try:
                result = call(**args)
                self._send(200, {"ok": True, "result": result})
            except Exception as e:
                self._send(500, {"error": "tool_error", "detail": str(e)})
        except Exception as e:
            self._send(500, {"error": "internal_error", "detail": str(e)})


def main():
    ap = argparse.ArgumentParser(description="MCP HTTP tool server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    httpd = HTTPServer((args.host, args.port), ToolHTTPHandler)
    print(f"MCP HTTP server listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
