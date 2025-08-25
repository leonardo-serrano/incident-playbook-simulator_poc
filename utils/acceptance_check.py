# utils/acceptance_check.py
"""
Automated acceptance checks for the PoC (Definition of Done).

It verifies:
- There are JSONL traces for both ALERT-1001 and ALERT-1002.
- Each trace contains: header, at least one 'retriever' trace, and a 'final' record.
- ALERT-1001 route == 'executor' and final action == 'scale_instance'.
- ALERT-1002 route == 'hitl' (HITL required) and final action == 'rollback_deployment'.
- reports/metrics.json exists (aggregated counters).
- reports/score.json exists and has the four success metrics + per-scenario scoring.

Exit code: 0 on PASS; 1 on FAIL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_jsonl(path: Path) -> Dict[str, Any]:
    """Parse minimal info from a run trace JSONL."""
    seen_header = False
    seen_retriever = False
    seen_final = False
    route = "unknown"
    actions: List[str] = []
    alert_id = "UNKNOWN"

    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        typ = rec.get("type")
        if typ == "header":
            seen_header = True
            alert_id = rec.get("alert_id", alert_id)
        elif typ == "trace":
            if rec.get("node") == "retriever":
                seen_retriever = True
            if rec.get("node") == "decision":
                raw = str(rec.get("next", "unknown")).strip().lower()
                route = raw if raw in {"executor", "hitl", "summarizer"} else "unknown"
        elif typ == "final":
            seen_final = True
            for a in rec.get("actions_taken") or []:
                if isinstance(a, dict) and "action" in a:
                    actions.append(str(a["action"]))

    return {
        "alert_id": alert_id,
        "seen_header": seen_header,
        "seen_retriever": seen_retriever,
        "seen_final": seen_final,
        "route": route,
        "final_action": actions[-1] if actions else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Acceptance checks for the PoC.")
    ap.add_argument("--traces", default=str(Path("reports") / "traces"))
    ap.add_argument("--metrics", default=str(Path("reports") / "metrics.json"))
    ap.add_argument("--score", default=str(Path("reports") / "score.json"))
    ap.add_argument("--expected", default=str(Path("eval") / "expected_actions.json"))
    args = ap.parse_args()

    traces_dir = Path(args.traces)
    metrics_path = Path(args.metrics)
    score_path = Path(args.score)
    expected_path = Path(args.expected)

    if not traces_dir.exists():
        _fail(f"Traces directory not found: {traces_dir}")
    if not metrics_path.exists():
        _fail(f"Missing aggregated metrics: {metrics_path}")
    if not score_path.exists():
        _fail(f"Missing scoring output: {score_path}")
    if not expected_path.exists():
        _fail(f"Missing expected actions spec: {expected_path}")

    # Load and index traces
    files = sorted(traces_dir.glob("*.jsonl"))
    if len(files) < 2:
        _fail("Need traces for at least two runs (ALERT-1001 and ALERT-1002).")

    parsed = [_parse_jsonl(p) for p in files]
    by_alert = {p["alert_id"]: p for p in parsed}

    # Required scenarios
    for aid in ("ALERT-1001", "ALERT-1002"):
        if aid not in by_alert:
            _fail(f"Missing required scenario trace: {aid}")
        info = by_alert[aid]
        if not (info["seen_header"] and info["seen_retriever"] and info["seen_final"]):
            _fail(f"Trace for {aid} is incomplete (header/retriever/final).")

    # Scenario-specific checks
    if by_alert["ALERT-1001"]["route"] != "executor":
        _fail("ALERT-1001 expected route 'executor'.")
    if by_alert["ALERT-1001"]["final_action"] != "scale_instance":
        _fail("ALERT-1001 expected final action 'scale_instance'.")

    if by_alert["ALERT-1002"]["route"] != "hitl":
        _fail("ALERT-1002 expected route 'hitl'.")
    if by_alert["ALERT-1002"]["final_action"] != "rollback_deployment":
        _fail("ALERT-1002 expected final action 'rollback_deployment'.")

    _ok("Scenario traces present and valid (1001 executor / 1002 HITL).")

    # Success metrics: validate in score.json (where scoring writes them)
    score = _load_json(score_path)
    for key in ("autonomy_rate", "hitl_rate", "retrieval_relevance_rate", "action_correctness_rate"):
        if key not in score:
            _fail(f"Score missing '{key}' success metric.")
    _ok("Score contains the four success metrics.")

    # Per-scenario scoring present
    by_alert_score = score.get("by_alert") or {}
    for aid in ("ALERT-1001", "ALERT-1002"):
        if aid not in by_alert_score:
            _fail(f"Score missing per-scenario entry: {aid}")
    _ok("Scoring contains per-scenario results for 1001 and 1002.")

    print("\nALL CHECKS PASSED ✅")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
