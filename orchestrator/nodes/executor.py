# orchestrator/nodes/executor.py
"""
Executor node:
- Executes the proposed action using MCP tools.
- Appends the result to `actions_taken`.
- Is tolerant to where the proposal lives (inside `plan` or as `state["proposed_action"]`).
"""

from __future__ import annotations

from typing import Any, Dict, List
import logging

from orchestrator.mcp_tools import MCPToolbox

log = logging.getLogger("nodes.executor")


def _pick_proposed_action(state: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return a normalized 'proposed_action' dict, or None if missing.

    Accepted shapes:
      1) Inside the plan as: {"step": "proposed_action", "action": "...", ...}
      2) Top-level state as: state["proposed_action"] (dict)
    """
    plan: List[Dict[str, Any]] = state.get("plan") or []
    for step in plan:
        if isinstance(step, dict) and step.get("step") == "proposed_action":
            return step

    top = state.get("proposed_action")
    if isinstance(top, dict):
        # Make sure it carries a step label for trace consistency
        return {"step": "proposed_action", **top}

    return None


def make_executor_node(tools: MCPToolbox):
    async def executor(state: Dict[str, Any]) -> Dict[str, Any]:
        proposed = _pick_proposed_action(state)

        if not proposed:
            # Nothing to execute → noop with explicit reason
            log.warning("executor: missing proposed_action in plan/state; executing noop")
            result: Dict[str, Any] = {"action": "noop", "payload": {"reason": "missing proposed_action"}}
            actions_taken = (state.get("actions_taken") or []) + [result]
            trace = (state.get("reasoning_trace") or []) + [{"node": "executor", "action": "noop"}]
            return {"actions_taken": actions_taken, "reasoning_trace": trace}

        # Normalize common fields coming from planner/decision
        action = proposed.get("action")
        # Accept both 'service' and 'service_name'
        service = proposed.get("service") or proposed.get("service_name") or ""
        payload = proposed.get("payload") or {}

        log.info(f"executor.action={action} service={service}")

        # Execute using MCP tools based on normalized action
        result: Dict[str, Any]
        if action == "scale_instance":
            # Accept 'target' in root or in payload (string/number)
            raw_target = proposed.get("target", payload.get("target", 2))
            try:
                target = int(raw_target)
            except Exception:
                target = 2
            result = await tools.scale_instance(service, target)

        elif action == "rollback_deployment":
            result = await tools.rollback_deployment(service)

        elif action == "clear_cache":
            # Accept 'scope' at root or payload
            scope = proposed.get("scope") or payload.get("scope") or "global"
            result = await tools.clear_cache(scope)

        else:
            # Unknown or missing action → noop with reason
            result = {"action": "noop", "payload": {"reason": f"unknown action '{action}'"}}

        actions_taken = (state.get("actions_taken") or []) + [result]
        trace_entry = {"node": "executor", "action": action or "noop"}
        trace = (state.get("reasoning_trace") or []) + [trace_entry]
        return {"actions_taken": actions_taken, "reasoning_trace": trace}

    return executor
