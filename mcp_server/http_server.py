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
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# Import the MCP server module to reuse the tool functions
from mcp_server import server as mcp_srv  # type: ignore


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

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path or "/")
        parts = [p for p in (parsed.path or "/").split("/") if p]
        if len(parts) == 1 and parts[0] == "tools":
            # list visible tool names by checking module attrs and registry
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
            self._send(200, {"tools": sorted(n for n in names if isinstance(n, str))})
            return
        self._send(404, {"error": "not_found", "detail": "Use GET /tools or POST /tool/<name>"})

    def do_POST(self):  # noqa: N802
        try:
            parsed = urlparse(self.path or "/")
            parts = [p for p in (parsed.path or "/").split("/") if p]
            # Expect path like /tool/<name>
            if len(parts) != 2 or parts[0] != "tool":
                self._send(404, {"error": "not_found", "detail": "Use POST /tool/<name>"})
                return
            name = parts[1]

            length = int(self.headers.get("Content-Length", "0"))
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
