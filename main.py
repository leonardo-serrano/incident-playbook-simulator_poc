# main.py
"""
Simple CLI for the PoC:
- Run a single alert ( --alert ALERT-XXXX )
- Or run a batch evaluation from eval/scenarios.json ( --batch )

Outputs:
- Per-run traces as JSONL in reports/traces/
- Aggregated metrics in reports/metrics.json

Design notes:
- We reuse the orchestrator.graph _run_once coroutine to execute the graph end-to-end.
- The aggregated metrics are lightweight, deterministic, and file-based (no DB).
- The tool is Windows-friendly (no POSIX-only features) and uses only the stdlib.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from utils.logging_utils import setup_logging, set_correlation_id, clear_correlation_id
import logging

log = logging.getLogger("cli")

# Read build/runtime flags (optional; used only for metadata)
try:
    from orchestrator.config import USE_LLM, GEMINI_MODEL  # type: ignore
except Exception:
    USE_LLM, GEMINI_MODEL = False, ""

# Reuse the existing graph runner (async)
from orchestrator.graph import _run_once as run_graph_once  # type: ignore


# ------------------------------ filesystem helpers ------------------------------ #
def _ensure_dir(p: Path) -> None:
    """Create directory p if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _safe_name(s: str) -> str:
    """Return a filename-safe string for Windows (very conservative)."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


# ------------------------------ trace & metrics -------------------------------- #
@dataclass
class RunSummary:
    """Minimal summary per run used for aggregation."""
    alert_id: str
    duration_ms: int
    route: str                   # 'executor' | 'hitl' | 'summarizer' | 'unknown'
    actions: List[str]           # e.g., ["rollback_deployment"] or ["scale_instance"] or []
    hitl_approved: Optional[bool]
    logs_count: int
    metrics_present: bool
    proposed_action: Optional[str]


def _extract_route(final_state: Dict[str, Any]) -> Tuple[str, Optional[bool]]:
    """
    Inspect reasoning_trace to extract decision route and HITL approval.
    Returns (route, hitl_approved) with route ∈ {'executor','hitl','summarizer','unknown'}.
    """
    route, approved = "unknown", None
    for entry in final_state.get("reasoning_trace") or []:
        if entry.get("node") == "decision":
            raw = str(entry.get("next", "unknown"))
            route = raw.strip().lower()
        elif entry.get("node") == "hitl":
            # keep as None if not present
            if "approved" in entry:
                approved = bool(entry.get("approved"))
    if route not in {"executor", "hitl", "summarizer"}:
        route = "unknown"
    return route, approved


# --- Routing extraction for metrics (ADDED) ---------------------------------
def _extract_route_from_trace_for_metrics(state: dict) -> str:
    """Look into reasoning_trace for the decision route as a last resort."""
    trace = state.get("reasoning_trace") or []
    for item in reversed(trace):
        if isinstance(item, dict) and item.get("node") == "decision":
            r = item.get("route") or item.get("next")
            if isinstance(r, str) and r in {"executor", "hitl", "summarizer"}:
                return r
    return "unknown"

def extract_final_route(state: dict) -> str:
    """
    Robustly extract the final route to be counted in metrics.
    Priority:
      1) state['route'] (set by decision.py)
      2) state['next']  (some graphs keep this)
      3) reasoning_trace last 'decision.route' or 'decision.next'
    """
    r = state.get("route") or state.get("next")
    if isinstance(r, str) and r in {"executor", "hitl", "summarizer"}:
        return r
    return _extract_route_from_trace_for_metrics(state)
# ---------------------------------------------------------------------------


def _write_trace_file(
    out_dir: Path,
    alert_id: str,
    started_at: str,
    finished_at: str,
    duration_ms: int,
    final_state: Dict[str, Any],
    error: Optional[str] = None,
) -> Path:
    """
    Write a JSONL trace for a single run. One file per run with a unique timestamp.
    Layout:
      - line 1: meta header (timestamps, duration, llm flags)
      - lines 2..N: reasoning_trace entries (if any)
      - last line: final summary (actions + human summary)
      - optional: error record
    """
    _ensure_dir(out_dir)
    fname = f"{_safe_name(alert_id)}__{_safe_name(started_at)}.jsonl"
    fpath = out_dir / fname

    header = {
        "type": "header",
        "alert_id": alert_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "llm_enabled": bool(USE_LLM),
        "llm_model": GEMINI_MODEL if USE_LLM else None,
    }

    with fpath.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")

        # Stream reasoning trace
        for entry in final_state.get("reasoning_trace") or []:
            f.write(json.dumps({"type": "trace", **entry}, ensure_ascii=False) + "\n")

        # Final summary record
        f.write(
            json.dumps(
                {
                    "type": "final",
                    "actions_taken": final_state.get("actions_taken") or [],
                    "summary": final_state.get("summary") or "",
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        if error:
            f.write(json.dumps({"type": "error", "message": error}, ensure_ascii=False) + "\n")

    return fpath


def _summarize_run(alert_id: str, started_at: datetime, final_state: Dict[str, Any]) -> RunSummary:
    """Compute a compact summary used for aggregation."""
    finished_at = datetime.now(timezone.utc)
    duration_ms = int((finished_at - started_at).total_seconds() * 1000)

    route, hitl_approved = _extract_route(final_state)
    actions = [a.get("action") for a in (final_state.get("actions_taken") or []) if isinstance(a, dict)]
    logs_count = len(final_state.get("logs") or [])
    metrics_present = bool(final_state.get("metrics"))
    # Scan plan for a proposed action (optional)
    proposed = None
    for step in final_state.get("plan") or []:
        if step.get("step") == "proposed_action":
            proposed = step.get("action")
            break

    return RunSummary(
        alert_id=alert_id,
        duration_ms=duration_ms,
        route=route,
        actions=[a for a in actions if a],
        hitl_approved=hitl_approved,
        logs_count=logs_count,
        metrics_present=metrics_present,
        proposed_action=proposed,
    )


def _update_aggregates(agg: Dict[str, Any], rs: RunSummary) -> None:
    """Update in-memory aggregated metrics with one run summary."""
    agg["runs"] = agg.get("runs", 0) + 1

    by_alert = agg.setdefault("by_alert", {})
    a = by_alert.setdefault(rs.alert_id, {"runs": 0, "total_ms": 0})
    a["runs"] += 1
    a["total_ms"] += rs.duration_ms

    routes = agg.setdefault("routes", {"executor": 0, "hitl": 0, "summarizer": 0, "unknown": 0})
    route_key = rs.route if rs.route in routes else "unknown"
    routes[route_key] += 1

    actions = agg.setdefault("actions", {})
    if rs.actions:
        for act in rs.actions:
            actions[act] = actions.get(act, 0) + 1
    else:
        actions["none"] = actions.get("none", 0) + 1

    if rs.hitl_approved is not None:
        hitl = agg.setdefault("hitl", {"approved": 0, "rejected": 0})
        hitl["approved" if rs.hitl_approved else "rejected"] += 1

    ev = agg.setdefault("evidence", {"metrics_present": 0, "total_log_lines": 0})
    if rs.metrics_present:
        ev["metrics_present"] += 1
    ev["total_log_lines"] += rs.logs_count



def _finalize_aggregates(agg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute derived fields such as averages."""
    out = dict(agg)
    out.setdefault("runs", 0)
    # Average duration per alert
    for aid, rec in out.get("by_alert", {}).items():
        runs = max(1, rec["runs"])
        rec["avg_ms"] = int(rec["total_ms"] / runs)
    # Average log lines per run
    runs = max(1, out.get("runs", 0))
    ev = out.get("evidence", {})
    if ev:
        ev["avg_log_lines_per_run"] = float(ev.get("total_log_lines", 0)) / float(runs)
    # LLM metadata
    out["llm_enabled"] = bool(USE_LLM)
    out["llm_model"] = GEMINI_MODEL if USE_LLM else None
    out["generated_at"] = _now_iso()
    return out


# ------------------------------ scenarios loader ------------------------------ #
def _load_scenarios(path: Path) -> List[str]:
    """
    Load scenarios from JSON.

    Accepted shapes:
    - {"scenarios": [{"id": "ALERT-1001"}, {"id": "ALERT-1002"}]}
    - [{"id": "ALERT-1001"}, {"id": "ALERT-1002"}]
    - ["ALERT-1001", "ALERT-1002"]

    Returns a flat list of alert ids.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    ids: List[str] = []

    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
        data = data["scenarios"]

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict) and "id" in item:
                ids.append(str(item["id"]))

    if not ids:
        raise SystemExit(f"No scenarios found in {path}")
    return ids


# ------------------------------ runners --------------------------------------- #
async def _run_single(alert_id: str, traces_dir: Path) -> RunSummary:
    """Run the graph for one alert id and persist a JSONL trace."""
    started_iso = _now_iso()
    started_dt = datetime.now(timezone.utc)

    # Set correlation id for this run (appears as [cid] in logs)
    set_correlation_id(f"incident-{alert_id}")
    try:
        error_msg: Optional[str] = None
        try:
            # Run the orchestration graph once
            final_state = await run_graph_once(alert_id)
        except Exception as e:
            # Log the exception so as not to lose the reason for “unknown/none.”
            log.exception("Graph run failed for %s", alert_id)
            final_state = {"reasoning_trace": [], "actions_taken": [], "summary": ""}
            error_msg = f"{type(e).__name__}: {e}"

        rs = _summarize_run(alert_id, started_dt, final_state)

        # --- ADDED: override route in RunSummary using final_state (more robust) ---
        try:
            corrected_route = extract_final_route(final_state)
            if corrected_route and corrected_route in {"executor", "hitl", "summarizer", "unknown"}:
                rs.route = corrected_route
        except Exception:
            # Keep original if anything goes wrong
            pass
        # ---------------------------------------------------------------------------

        _write_trace_file(
            traces_dir,
            alert_id=alert_id,
            started_at=started_iso,
            finished_at=_now_iso(),
            duration_ms=rs.duration_ms,
            final_state=final_state,
            error=error_msg,
        )
        return rs
    finally:
        # Always clear correlation id after the run
        clear_correlation_id()


def main():

    setup_logging()  # console+file; honors LOG_LEVEL/LOG_FILE_LEVEL/LOG_JSON
    
    parser = argparse.ArgumentParser(description="Incident Playbook Simulator CLI")
    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument("--alert", help="Run a single alert id (e.g., ALERT-1001)")
    mx.add_argument("--batch", action="store_true", help="Run a batch from eval/scenarios.json")
    parser.add_argument(
        "--scenarios",
        default=str(Path("eval") / "scenarios.json"),
        help="Path to scenarios file for --batch (default: eval/scenarios.json)",
    )
    parser.add_argument(
        "--traces-dir",
        default=str(Path("reports") / "traces"),
        help="Directory for per-run traces (default: reports/traces)",
    )
    parser.add_argument(
        "--metrics-out",
        default=str(Path("reports") / "metrics.json"),
        help="Aggregated metrics output (default: reports/metrics.json)",
    )
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)
    _ensure_dir(traces_dir)
    _ensure_dir(Path("reports"))  # ensure top-level

    agg: Dict[str, Any] = {}

    if args.alert:
        # Single run
        rs = asyncio.run(_run_single(args.alert, traces_dir))
        _update_aggregates(agg, rs)
    else:
        # Batch run
        scenarios_path = Path(args.scenarios)
        if not scenarios_path.exists():
            raise SystemExit(
                f"Scenarios file not found: {scenarios_path}\n"
                "Create it or pass a custom path with --scenarios."
            )
        ids = _load_scenarios(scenarios_path)
        for alert_id in ids:
            rs = asyncio.run(_run_single(alert_id, traces_dir))
            _update_aggregates(agg, rs)

    # Finalize and write aggregated metrics
    final_metrics = _finalize_aggregates(agg)
    metrics_out = Path(args.metrics_out)
    metrics_out.write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Console summary (short)
    print("\n=== METRICS ===")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
