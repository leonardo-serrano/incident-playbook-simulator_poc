# orchestrator/state.py
"""
IncidentState: shared state model for the orchestration graph.

This Pydantic model mirrors the state fields described in the PoC spec:
- alert, plan, retrieved_docs, metrics, logs, actions_taken,
  hitl_needed, hitl_response, summary, reasoning_trace.  :contentReference[oaicite:0]{index=0}

Notes:
- Lists use default_factory to avoid shared mutable defaults.
- `reasoning_trace` stores rich entries (dicts) so we can log node names,
  decisions, etc. If you prefer plain strings, change the type to List[str].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IncidentState(BaseModel):
    """Typed, serializable container for the incident run state."""

    # Core alert being processed (parsed from sample_data/alerts.json)
    alert: Dict[str, Any]

    # Planner output: an ordered list of steps (each step is a dict)
    plan: List[Dict[str, Any]] = Field(default_factory=list)

    # Retriever output: relevant snippets/documents (e.g., playbooks)
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)

    # Observability data (None if not fetched in this run)
    metrics: Optional[Dict[str, Any]] = None
    logs: List[Dict[str, Any]] = Field(default_factory=list)

    # Side effects executed by the Executor node (append-only log)
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)

    # Human-in-the-loop flags/results
    hitl_needed: bool = False
    hitl_response: Optional[str] = None

    # Final output
    summary: Optional[str] = None

    # Trace of node-level reasoning/decisions (dict entries for richer context)
    reasoning_trace: List[Dict[str, Any]] = Field(default_factory=list)

    # ----------------------- Convenience helpers ----------------------- #
    def add_trace(self, **entry: Any) -> None:
        """Append a trace entry (e.g., {'node': 'planner', 'plan_len': 3})."""
        self.reasoning_trace.append(entry)

    def record_action(self, action: str, **payload: Any) -> None:
        """Append an action taken by the Executor with its payload."""
        self.actions_taken.append({"action": action, "payload": payload})

    def as_graph_input(self) -> Dict[str, Any]:
        """
        Dump into a plain dict for LangGraph.
        By default we include None fields; set exclude_none=True if desired.
        """
        return self.model_dump()

    @classmethod
    def from_graph(cls, d: Dict[str, Any]) -> "IncidentState":
        """Build an IncidentState from an existing dict-like graph state."""
        return cls.model_validate(d)
