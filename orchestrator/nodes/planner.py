# orchestrator/nodes/planner.py
# -*- coding: utf-8 -*-
"""
Planner node:
- Ask the LLM to produce a structured plan (list of tool-call steps).
- Robustly coerce LLM output into List[Dict].
- If parsing fails, fall back to a deterministic minimal plan.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import logging
import re

log = logging.getLogger("nodes.planner")

def _read_prompt(name: str) -> str:
    """Load planner prompt from orchestrator/prompts or prompts directory."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "orchestrator" / "prompts" / name,
        here.parents[1] / "prompts" / name,
    ]
    for base in candidates:
        for ext in (".txt", ".md"):
            p = base.with_suffix(ext)
            if p.exists():
                return p.read_text(encoding="utf-8")
    log.warning("Planner prompt not found; using minimal inline prompt.")
    return (
        "You are the planner. Return ONLY a JSON array of steps. "
        "Each step is an object with 'step' and optional 'params'. "
        "Allowed steps: search_playbooks, get_metrics, query_logs. No prose."
    )

def _json_extract_array(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Try to parse a JSON array from a string:
    - direct json.loads
    - fenced ```json blocks
    - first [...] bracket pair
    """
    # direct
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data  # type: ignore[return-value]
    except Exception:
        pass

    # fenced ```json ... ```
    fenced = re.findall(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, flags=re.IGNORECASE)
    for block in fenced:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                return data  # type: ignore[return-value]
        except Exception:
            continue

    # first bracket slice
    try:
        i = text.index("[")
        j = text.rindex("]") + 1
        data = json.loads(text[i:j])
        if isinstance(data, list):
            return data  # type: ignore[return-value]
    except Exception:
        pass
    return None

def _default_plan(alert: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Deterministic minimal plan used as fallback:
    - search_playbooks
    - get_metrics
    - query_logs
    """
    title = (alert or {}).get("title", "") or ""
    service = (alert or {}).get("service", "") or ""
    # naive guess for logs pattern
    if "cpu" in title.lower():
        pattern = "slow|threshold|cpu|saturation|load"
    elif "500" in title or "error" in title.lower():
        pattern = "500|error|exception|timeout"
    else:
        pattern = "warn|error|slow|timeout"

    query = f"{title} {service}".strip() or "incident playbook"
    return [
        {"step": "search_playbooks", "params": {"query": query, "k": 3}},
        {"step": "get_metrics", "params": {"service_name": service or "unknown"}},
        {"step": "query_logs", "params": {"service_name": service or "unknown", "pattern": pattern, "limit": 200}},
    ]

def make_planner_node(llm: Optional[Any] = None, prompts_dir: Optional[Path] = None):
    """
    Build the planner node. If llm is provided, we invoke it with the planner prompt;
    otherwise we immediately return the deterministic fallback plan.
    """
    def _read_planner_prompt() -> str:
        if prompts_dir:
            for ext in (".txt", ".md"):
                p = (prompts_dir / "planner_prompt").with_suffix(ext)
                if p.exists():
                    return p.read_text(encoding="utf-8")
        return _read_prompt("planner_prompt")

    async def planner(state: Dict[str, Any]) -> Dict[str, Any]:
        alert: Dict[str, Any] = state.get("alert") or {}
        prompt = _read_planner_prompt()

        plan: List[Dict[str, Any]] = []
        used_llm = False

        if llm is not None:
            # For visibility in demos
            print("LLM planner using Gemini!")
            try:
                payload = {
                    "alert": {
                        "id": alert.get("id"),
                        "title": alert.get("title"),
                        "service": alert.get("service"),
                        "symptoms": alert.get("symptoms"),
                    },
                    "allowed_steps": ["search_playbooks", "get_metrics", "query_logs"],
                    "schema": {"step": "string", "params": "object (optional)"},
                }
                raw = llm.plan(payload, system_prompt=prompt)
                # If adapter already returns a list, accept it
                if isinstance(raw, list):
                    plan = raw  # type: ignore[assignment]
                else:
                    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                    parsed = _json_extract_array(text)
                    plan = parsed if parsed is not None else _default_plan(alert)
                used_llm = True
            except Exception as e:
                log.error("planner: LLM call failed, using fallback. err=%s", e)
                plan = _default_plan(alert)
        else:
            plan = _default_plan(alert)

        # Keep only a reasonable number of steps to avoid flooding
        if len(plan) > 20:
            plan = plan[:20]

        # Trace + return
        log.info("planner.end plan_len=%s", len(json.dumps(plan, ensure_ascii=False)))
        (state.setdefault("reasoning_trace", [])).append({
            "node": "planner",
            "used_llm": used_llm,
            "steps": [s.get("step") for s in plan if isinstance(s, dict)][:6]
        })
        return {"plan": plan}

    return planner
