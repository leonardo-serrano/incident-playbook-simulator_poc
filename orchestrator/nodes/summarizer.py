# orchestrator/nodes/summarizer.py
"""Summarizer node:
- Produces a concise textual summary of the incident, evidence and action taken.
- If an LLM client is provided it will be used; otherwise a deterministic line is returned.
"""
from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import logging

log = logging.getLogger("nodes.summarizer")

def _read_prompt(name: str) -> str:
    """
    Locate a prompt file inside orchestrator/prompts/.
    Prefer .txt (step 6), fall back to .md for backward compatibility.
    """
    base = Path(__file__).resolve().parents[1] / "prompts" / name
    for ext in (".txt", ".md"):
        p = base.with_suffix(ext)
        if p.exists():
            return p.read_text(encoding="utf-8")
    return "You are a helpful summarizer."

def _fallback_summary(state: Dict[str, Any]) -> str:
    alert = state.get("alert") or {}
    plan = state.get("plan") or []
    metrics = state.get("metrics")
    logs = state.get("logs") or []
    actions = state.get("actions_taken") or []
    proposed = next((s for s in plan if s.get("step") == "proposed_action"), {})
    action = proposed.get("action", "none")
    line1 = f"Incident {alert.get('id', '<unknown>')}: {alert.get('title', '')}"
    ev1 = f"Metrics present: {metrics is not None}; Logs lines: {len(logs)}"
    ev2 = f"Proposed action: {action}"
    done = f"Executed: {actions[-1]['action'] if actions else 'none'}"
    return " | ".join([line1, ev1, ev2, done])

def make_summarizer_node(llm=None):
    async def summarizer(state: Dict[str, Any]) -> Dict[str, Any]:
        summary = None
        if llm is not None:
            try:
                print("LLM summarizer using Gemini!") # TESTING LLM EXEC
                system = _read_prompt("summarizer")
                summary = llm.summarize(state, system_prompt=system) or None
            except Exception:
                summary = None
        if not summary:
            summary = _fallback_summary(state)
        trace_entry = {"node": "summarizer"}
        log.info("summarizer.done")
        return {"summary": summary, "reasoning_trace": (state.get("reasoning_trace") or []) + [trace_entry]}
    return summarizer
