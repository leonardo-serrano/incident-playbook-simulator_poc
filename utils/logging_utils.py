# utils/logging_utils.py
"""
Centralized logging for the PoC:
- Human-readable console logs + rotating file logs under reports/logs/app.log
- Correlation ID (per incident) via ContextVar injected as [cid] into every record
- Optional JSON file logs (LOG_JSON=true)
- Helpers to trace MCP tool calls with duration and outcome

Env vars (optional):
  LOG_LEVEL=INFO|DEBUG|WARNING   -> console level (default INFO)
  LOG_FILE_LEVEL=DEBUG|INFO      -> file level (default DEBUG)
  LOG_JSON=true|false            -> JSON logs in file handler (default false)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager, asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional
import contextvars

# Correlation ID (per run / incident)
_CORR_ID: contextvars.ContextVar[str] = contextvars.ContextVar("corr_id", default="-")

def set_correlation_id(cid: str) -> None:
    """Set correlation id for the current context (propagates to all log records)."""
    _CORR_ID.set(cid or "-")

def clear_correlation_id() -> None:
    """Clear correlation id (useful for tests or separate runs)."""
    _CORR_ID.set("-")


class _CorrelationFilter(logging.Filter):
    """Inject correlation id into every record as 'cid'."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.cid = _CORR_ID.get()
        return True


class _JsonFormatter(logging.Formatter):
    """Compact JSON formatter for file logs."""
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "cid": getattr(record, "cid", "-"),
            "msg": record.getMessage(),
        }
        # Include a few useful extras if present
        for key in ("tool", "status", "ms", "route", "action", "alert_id"):
            val = getattr(record, key, None)
            if val is not None:
                base[key] = val
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def setup_logging(log_dir: str | Path = "reports/logs") -> None:
    """Configure root logger with console + rotating file handler."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    console_level = os.getenv("LOG_LEVEL", "INFO").upper()
    file_level = os.getenv("LOG_FILE_LEVEL", "DEBUG").upper()
    json_file = os.getenv("LOG_JSON", "false").strip().lower() in {"1", "true", "yes", "on"}

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers decide what to show

    # Common filter
    corr_filter = _CorrelationFilter()

    # Console handler (human-readable)
    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(getattr(logging, console_level, logging.INFO))
    ch.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s [%(cid)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    ch.addFilter(corr_filter)
    root.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(str(Path(log_dir) / "app.log"), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(getattr(logging, file_level, logging.DEBUG))
    if json_file:
        fh.setFormatter(_JsonFormatter())
    else:
        fh.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)-5s [%(cid)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    fh.addFilter(corr_filter)
    root.addHandler(fh)


@contextmanager
def tool_span(logger: logging.Logger, tool: str, params: Optional[Dict[str, Any]] = None):
    """
    Sync context manager to trace a tool call. Use for sync tools if any appear.
    """
    t0 = time.perf_counter()
    logger.info(f"tool.start {tool} params={params}")
    try:
        yield
        ms = int((time.perf_counter() - t0) * 1000)
        # attach structured fields when possible
        logger.info(f"tool.end   {tool} ok ({ms} ms)")
    except Exception:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.exception(f"tool.error {tool} failed ({ms} ms)")
        raise


@asynccontextmanager
async def async_tool_span(logger: logging.Logger, tool: str, params: Optional[Dict[str, Any]] = None):
    """
    Async context manager to trace an async tool call (MCP tools).
    Example:
        async with async_tool_span(log, "search_playbooks", {"query": q}):
            res = await self.search_playbooks(...)
    """
    t0 = time.perf_counter()
    logger.info(f"tool.start {tool} params={params}")
    try:
        yield
        ms = int((time.perf_counter() - t0) * 1000)
        logger.info(f"tool.end   {tool} ok ({ms} ms)")
    except Exception:
        ms = int((time.perf_counter() - t0) * 1000)
        logger.exception(f"tool.error {tool} failed ({ms} ms)")
        raise
