# orchestrator/nodes/retriever.py
"""
Retriever node:
- Executes the retrieval steps from the plan using the MCP tools:
  * search_playbooks
  * get_metrics
  * query_logs
- Writes results into state: retrieved_docs, metrics, logs.
- Adds 'doc_titles' to the reasoning trace so scoring can evaluate
  retrieval relevance (see utils/scoring.py).
"""

from __future__ import annotations

from typing import Any, Dict, List
import json
import logging
from orchestrator.mcp_tools import MCPToolbox

log = logging.getLogger("nodes.retriever")


def make_retriever_node(tools: MCPToolbox):
    async def retriever(state: Dict[str, Any]) -> Dict[str, Any]:
        # --- Coerce plan into a List[Dict] defensively ---
        plan = state.get("plan") or []
        if isinstance(plan, str):
            # Plan came as a JSON string → try to parse
            try:
                plan = json.loads(plan)
            except Exception:
                log.warning("retriever: plan is a string but JSON parse failed; using empty plan.")
                plan = []
        if not isinstance(plan, list):
            log.warning("retriever: plan is not a list; using empty plan.")
            plan = []

        # Results we will accumulate
        docs: List[Dict[str, Any]] = []
        metrics: Dict[str, Any] | None = None
        logs: List[Dict[str, Any]] | None = None

        # --- Execute each plan step deterministically over the mock tools ---
        for step in plan:
            if not isinstance(step, dict):
                # Skip malformed items defensively
                continue

            name = step.get("step")
            # Planner emits params under "params". Keep a fallback to tolerate older shapes.
            params = step.get("params", {}) if isinstance(step.get("params"), dict) else {}

            if name == "search_playbooks":
                query = params.get("query") or step.get("query") or ""
                k = int(params.get("k", step.get("k", 3)))
                res = await tools.search_playbooks(query, k=k)
                # Expected shape: {"results": [ {...}, ... ]}
                docs = res.get("results", []) or []

            elif name == "get_metrics":
                service = (
                    params.get("service_name")
                    or params.get("service")
                    or step.get("service_name")
                    or step.get("service")
                    or ""
                )
                res = await tools.get_metrics(service)
                metrics = res.get("metrics")

            elif name == "query_logs":
                service = (
                    params.get("service_name")
                    or params.get("service")
                    or step.get("service_name")
                    or step.get("service")
                    or ""
                )
                pattern = params.get("pattern", step.get("pattern"))
                limit = int(params.get("limit", step.get("limit", 200)))
                res = await tools.query_logs(service, pattern=pattern, limit=limit)
                logs = res.get("logs")

        # --- Build compact trace info for scoring/observability ---
        titles: List[str] = []
        for d in (docs or []):
            if isinstance(d, dict):
                t = d.get("title") or d.get("name") or d.get("id") or ""
                if isinstance(t, str) and t:
                    titles.append(t)

        log.info(
            "retriever.done docs=%s metrics=%s logs=%s",
            len(docs or []),
            metrics is not None,
            logs is not None,
        )

        trace_entry = {
            "node": "retriever",
            "docs": len(docs or []),
            "has_metrics": bool(metrics),
            "has_logs": bool(logs),
            "doc_titles": titles[:5],  # keep it short in traces
        }

        # Return partial state; LangGraph will merge this into the graph state
        return {
            "retrieved_docs": docs,
            "metrics": metrics,
            "logs": logs,
            "reasoning_trace": (state.get("reasoning_trace") or []) + [trace_entry],
        }

    return retriever
