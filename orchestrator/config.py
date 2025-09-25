# orchestrator/config.py
"""Simple config loader for the POC (Gemini-only).
- USE_LLM: enable/disable LLM-powered planner/summarizer.
- GOOGLE_API_KEY and GEMINI_MODEL (or MODEL) configure Google Gemini.
Values are read from environment variables so you can tweak them without code changes.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # loads variables from a .env file in the project root

# Project root directory (resolve relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _as_bool(val: str | None, default: bool=False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}

# Public config flags
USE_LLM: bool = _as_bool(os.getenv("SIM_USE_LLM"), default=False)

# Google Gemini (defaults preserved for backward compatibility)
GEMINI_MODEL   = os.getenv("MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001"))
GOOGLE_API_KEY = os.getenv("API_KEY", os.getenv("GOOGLE_API_KEY"))

# MCP transport configuration
# MCP_TRANSPORT: 'stdio' (default) or 'http'
MCP_TRANSPORT = (os.getenv("MCP_TRANSPORT") or "stdio").lower()
# MCP_SERVER_URL: e.g., http://127.0.0.1:8765 (only used when MCP_TRANSPORT=http)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")
