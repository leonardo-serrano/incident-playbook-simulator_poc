# orchestrator/nodes/hitl.py
"""
HITL node:
- Requests approval/denial using the `ask_human` MCP tool.
- Stores the response and sets `hitl_approved` for routing.
"""

from __future__ import annotations

from typing import Any, Dict, List
from orchestrator.mcp_tools import MCPToolbox
import logging

log = logging.getLogger("nodes.hitl")

def make_hitl_node(tools: MCPToolbox):
    async def hitl(state: Dict[str, Any]) -> Dict[str, Any]:
        plan: List[Dict[str, Any]] = state.get("plan") or []
        proposed = next((s for s in plan if s.get("step") == "proposed_action"), {})
        service = proposed.get("service", "")
        action = proposed.get("action", "unknown")
        msg = f"Approve action '{action}' on service '{service}'?"

        res = await tools.ask_human(msg)  # policy is read from hitl_policy.json if present
        approved = bool(res.get("result", {}).get("approved", True))

        trace_entry = {"node": "hitl", "approved": approved}
        log.info(f"hitl.approved={approved}")
        return {
            "hitl_response": str(res),
            "hitl_needed": False,  # consumed
            "hitl_approved": approved,
            "reasoning_trace": (state.get("reasoning_trace") or []) + [trace_entry],
        }
    return hitl


def hitl_router(state: Dict[str, Any]) -> str:
    """After HITL, route to executor if approved; otherwise summarizer."""
    return "executor" if state.get("hitl_approved") else "summarizer"
