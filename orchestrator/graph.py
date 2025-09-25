# orchestrator/graph.py
"""Builds the LangGraph orchestration and provides a small CLI to run a single alert."""

from __future__ import annotations
import argparse, json, asyncio
from pathlib import Path
from typing import Any, Dict, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.config import (
    USE_LLM,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    PROJECT_ROOT,
)
from orchestrator.llm import build_llm_or_none, ensure_text_iface
from orchestrator.mcp_tools import MCPToolbox
from orchestrator.nodes.planner import make_planner_node
from orchestrator.nodes.retriever import make_retriever_node
from orchestrator.nodes.decision import make_decision_node
from orchestrator.nodes.hitl import make_hitl_node, hitl_router
from orchestrator.nodes.executor import make_executor_node
from orchestrator.nodes.summarizer import make_summarizer_node
from orchestrator.state import IncidentState


class GraphState(TypedDict, total=False):
    alert: Dict[str, Any]
    plan: list
    retrieved_docs: list
    metrics: Dict[str, Any]
    logs: list
    actions_taken: list
    hitl_needed: bool
    hitl_response: Dict[str, Any]
    hitl_approved: bool
    summary: str
    reasoning_trace: list
    next: str


# Read the next hop decided by the Decision node (LLM or fallback).
def _next_route(state: dict) -> str:
    # Normalize and default to summarizer if missing/invalid
    nxt = (state.get("next") or "").strip().lower()
    return nxt if nxt in {"executor", "hitl", "summarizer"} else "summarizer"


def build_graph(server_path: Path):
    """Create and compile the graph; return (app, tools)."""
    # MCP toolbox with fast fallback to direct mode (Windows-friendly)
    tools = MCPToolbox(server_path, handshake_timeout=3.0)

    # LLM is optional; when disabled or misconfigured, nodes use deterministic fallbacks
    llm = None
    if USE_LLM:
        llm = build_llm_or_none(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)

    graph = StateGraph(GraphState)

    # Nodes (planner/summarizer receive the LLM; others use MCP tools)
    graph.add_node("planner", make_planner_node(llm=llm))
    graph.add_node("retriever", make_retriever_node(tools))
    graph.add_node("decision", make_decision_node(llm=llm))
    graph.add_node("hitl", make_hitl_node(tools))
    graph.add_node("executor", make_executor_node(tools))
    graph.add_node("summarizer", make_summarizer_node(llm=llm))

    # Topology
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "decision")
    graph.add_conditional_edges(
        "decision",
        _next_route,
        {
            "executor": "executor",
            "hitl": "hitl",
            "summarizer": "summarizer",
        },
    )
    graph.add_conditional_edges(
        "hitl", hitl_router,
        {"executor": "executor", "summarizer": "summarizer"},
    )
    graph.add_edge("executor", "summarizer")
    graph.add_edge("summarizer", END)

    app = graph.compile(checkpointer=MemorySaver())
    return app, tools


def _load_alert(fixtures_dir: Path, alert_id: str) -> Dict[str, Any]:
    alerts_path = fixtures_dir / "alerts.json"
    data = json.loads(alerts_path.read_text(encoding="utf-8"))
    for a in data:
        if a.get("id") == alert_id:
            return a
    raise SystemExit(f"Alert id '{alert_id}' not found in {alerts_path}")


async def _run_once(alert_id: str) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    fixtures = repo_root / "mcp_server" / "sample_data"
    if not fixtures.exists():
        raise SystemExit(
            "Required sample_data folder is missing.\n"
            f"Expected at: {fixtures}\n"
            "Please create it and include at least:\n"
            "  - alerts.json\n  - playbooks.md\n  - metrics.json\n  - logs.json"
        )

    server_path = repo_root / "mcp_server" / "server.py"
    app, tools = build_graph(server_path)

    state = IncidentState(
        alert=_load_alert(fixtures, alert_id)
    )

    initial_state: GraphState = state.model_dump(exclude_none=True) # excluding keys with None to avoid type inconsistencies

    # Required when using a checkpointer
    run_config = {"configurable": {"thread_id": f"incident-{alert_id}"}}

    async with tools:
        return await app.ainvoke(initial_state, config=run_config)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alert", default="ALERT-1001",
                   help="Alert id to simulate (e.g., ALERT-1001 or ALERT-1002)")
    args = p.parse_args()
    final_state = asyncio.run(_run_once(args.alert))

    print("\n=== SUMMARY ===")
    print(final_state.get("summary", ""))

    print("\n=== ACTIONS ===")
    for a in (final_state.get("actions_taken", []) or []):
        print(a)

    print("\n=== TRACE (nodes) ===")
    for t in (final_state.get("reasoning_trace", []) or []):
        print(t)
        
    print("\n")


if __name__ == "__main__":
    main()
