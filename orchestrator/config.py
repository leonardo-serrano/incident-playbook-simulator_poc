# orchestrator/config.py
"""Simple config loader for the POC.
- USE_LLM: enable/disable LLM-powered planner/summarizer.
- OPENAI_MODEL: model name (default 'gpt-5').
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
LLM_PROVIDER   = (os.getenv("LLM_PROVIDER") or "google").lower()
GEMINI_MODEL   = os.getenv("MODEL", "gemini-2.0-flash-001")
GOOGLE_API_KEY = os.getenv("API_KEY")
