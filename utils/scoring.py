# utils/scoring.py
"""
Evaluation utilities & CLI for the Incident Playbook Simulator.

It scores completed runs (JSONL traces in reports/traces/) against
eval/expected_actions.json and reports:
- autonomy_rate: fraction of runs that did NOT require HITL
- hitl_rate: fraction of runs that required HITL
- retrieval_relevance_rate: fraction with at least one expected playbook keyword
- action_correctness_rate: fraction where final executed action matches expected

Usage (from repo root):
  python -m utils.scoring --traces reports/traces --expected eval/expected_actions.json --out reports/score.json

Note:
- It consumes the JSONL written by main.py:
    header (line 1), multiple trace lines, a final record, optional error line.
- It expects retriever traces to include 'doc_titles' (see retriever patch).
- If 'doc_titles' are missing, relevance degrades to (docs > 0).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------- I/O helpers ----------------------------- #

def _iter_traces(traces_dir: Path) -> Iterable[Path]:
    """Yield JSONL files inside traces_dir (sorted by mtime ascending)."""
    return sorted(traces_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)


def _parse_jsonl(path: Path) -> Dict[str, Any]:
    """
    Parse a single run trace JSONL file into a compact structure.
    Returns a dict with: alert_id, route, hitl_approved, actions, doc_titles.
    """
    alert_id = "UNKNOWN"
    route, hitl_approved = "unknown", None
    actions: List[str] = []
    doc_titles: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue

            rtype = rec.get("type")

            if rtype == "header":
                alert_id = rec.get("alert_id", alert_id)

            elif rtype == "trace":
                node = rec.get("node")
                if node == "decision":
                    raw = str(rec.get("next", "unknown")).strip().lower()
                    route = raw if raw in {"executor", "hitl", "summarizer"} else "unknown"
                elif node == "hitl" and "approved" in rec:
                    hitl_approved = bool(rec.get("approved"))
                elif node == "retriever":
                    titles = rec.get("doc_titles") or []
                    if isinstance(titles, list):
                        doc_titles.extend([str(t) for t in titles])

            elif rtype == "final":
                for a in (rec.get("actions_taken") or []):
                    if isinstance(a, dict) and "action" in a:
                        actions.append(str(a["action"]))

            # ignore 'error' and other types for scoring

    return {
        "alert_id": alert_id,
        "route": route,
        "hitl_approved": hitl_approved,
        "actions": actions,
        "doc_titles": doc_titles,
    }


def _load_expected(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load expected_actions.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid expected file shape (expected object): {path}")
    return data


# ---------------------------- scoring logic ---------------------------- #

@dataclass
class RunScore:
    alert_id: str
    autonomous: bool
    used_hitl: bool
    retrieval_relevant: bool
    action_correct: bool
    route: str
    executed_action: Optional[str]
    expected_action: Optional[str]
    matched_keywords: List[str]


def _score_one(trace: Dict[str, Any], expected: Dict[str, Any]) -> RunScore:
    """
    Compare one parsed trace against expected spec.
    - autonomous: route != 'hitl'
    - used_hitl: route == 'hitl'
    - retrieval_relevant: at least one expected keyword in doc_titles
      (fallback to docs>0 when doc_titles are missing).
    - action_correct: last executed action equals expected_action (or None==None)
    """
    aid = trace.get("alert_id", "UNKNOWN")
    route = trace.get("route", "unknown")
    autonomous = route != "hitl"
    used_hitl = route == "hitl"

    spec = expected.get(aid, {})
    exp_action = spec.get("expected_action", None)
    keywords = spec.get("expected_playbook_keywords", []) or []

    titles = [t.lower() for t in (trace.get("doc_titles") or []) if isinstance(t, str)]
    matched: List[str] = []
    if titles and keywords:
        for kw in keywords:
            if kw and any(kw.lower() in t for t in titles):
                matched.append(kw)
    # degrade to (docs>0) relevance if no titles present
    retrieval_relevant = bool(matched) or (not titles and len(titles) >= 0 and len(trace.get("doc_titles") or []) >= 1)

    exec_action = None
    actions = trace.get("actions") or []
    if actions:
        exec_action = actions[-1]

    action_correct = (exec_action == exp_action)

    return RunScore(
        alert_id=aid,
        autonomous=autonomous,
        used_hitl=used_hitl,
        retrieval_relevant=retrieval_relevant,
        action_correct=action_correct,
        route=route,
        executed_action=exec_action,
        expected_action=exp_action,
        matched_keywords=matched,
    )


def _aggregate(scores: List[RunScore]) -> Dict[str, Any]:
    """Compute global metrics from per-run scores."""
    n = max(1, len(scores))
    def rate(pred) -> float:
        return round(sum(1 for s in scores if pred(s)) / n, 4)

    return {
        "runs": len(scores),
        "autonomy_rate": rate(lambda s: s.autonomous),
        "hitl_rate": rate(lambda s: s.used_hitl),
        "retrieval_relevance_rate": rate(lambda s: s.retrieval_relevant),
        "action_correctness_rate": rate(lambda s: s.action_correct),
        "by_alert": {
            s.alert_id: {
                "route": s.route,
                "executed_action": s.executed_action,
                "expected_action": s.expected_action,
                "retrieval_match": s.matched_keywords,
                "autonomous": s.autonomous,
                "action_correct": s.action_correct
            }
            for s in scores
        }
    }


# ------------------------------- CLI ---------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Score batch of runs against expected actions.")
    ap.add_argument("--traces", default=str(Path("reports") / "traces"), help="Directory with JSONL traces")
    ap.add_argument("--expected", default=str(Path("eval") / "expected_actions.json"), help="Expected actions JSON")
    ap.add_argument("--out", default=str(Path("reports") / "score.json"), help="Where to write the scoring JSON")
    args = ap.parse_args()

    traces_dir = Path(args.traces)
    expected_path = Path(args.expected)
    out_path = Path(args.out)

    if not traces_dir.exists():
        raise SystemExit(f"Traces directory not found: {traces_dir}")
    if not expected_path.exists():
        raise SystemExit(f"Expected file not found: {expected_path}")

    expected = _load_expected(expected_path)

    scores: List[RunScore] = []
    for jf in _iter_traces(traces_dir):
        tr = _parse_jsonl(jf)
        aid = tr.get("alert_id", "")
        if not aid:
            # skip malformed runs
            continue
        scores.append(_score_one(tr, expected))

    result = _aggregate(scores)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== SCORE ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
