from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_policy_paths() -> list[Path]:
    # Allow override via environment variable
    env = os.getenv("DECISION_POLICY_PATH")
    if env:
        return [Path(env)]
    root = _repo_root()
    return [
        root / "sample_data" / "decision_policy.json",
        root / "mcp_server" / "sample_data" / "decision_policy.json",
    ]


def load_decision_policy() -> Dict[str, Any]:
    """
    Load decision policy JSON from known locations. Returns an empty policy if not found.
    Structure expected:
    {
      "default": {"route": "summarizer", "risk": "none", "justification": "..."},
      "rules": [
        {
          "when": {"alert_id": "ALERT-1002"},
          "enforce": true,
          "decision": {"route": "hitl", "risk": "high", "justification": "..."},
          "proposed_action": {"action": "rollback_deployment", "service": "checkout-api", "payload": {}}
        }
      ]
    }
    """
    for p in _candidate_policy_paths():
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            # Ignore malformed file; treat as no policy
            pass
    return {}


def _norm_route(val: Optional[str]) -> str:
    raw = str(val or "").strip().lower()
    return raw if raw in {"executor", "hitl", "summarizer"} else "unknown"


def evaluate_decision_policy(
    alert: Dict[str, Any],
    base_route: str,
    base_risk: str,
    base_justification: str,
) -> Tuple[str, str, str, Optional[Dict[str, Any]]]:
    """
    Apply decision policy rules to possibly override route/risk/justification and to provide
    an optional proposed_action suggestion.

    Returns: (route, risk, justification, proposed_action_or_none)
    """
    policy = load_decision_policy()
    default = policy.get("default") or {}
    rules = policy.get("rules") or []

    route = _norm_route(base_route)
    risk = str(base_risk or "none").strip().lower()
    justification = base_justification or ""
    suggested_action: Optional[Dict[str, Any]] = None

    aid = (alert or {}).get("id") or (alert or {}).get("alert_id") or ""
    svc = (alert or {}).get("service") or (alert or {}).get("service_name") or ""
    sev = (alert or {}).get("severity") or ""
    # Normalize tags to a set of lowercased strings
    raw_tags = (alert or {}).get("tags") or []
    if isinstance(raw_tags, str):
        # allow comma/space-separated strings
        parts = [t.strip() for t in re.split(r"[,\s]+", raw_tags) if t.strip()]
    else:
        parts = [str(t).strip() for t in (raw_tags if isinstance(raw_tags, (list, tuple)) else [])]
    alert_tags = {t.lower() for t in parts}

    def apply_decision(dec: Dict[str, Any]) -> None:
        nonlocal route, risk, justification
        r = _norm_route(dec.get("route"))
        if r != "unknown":
            route = r
        if "risk" in dec:
            risk = str(dec.get("risk") or risk).strip().lower()
        if dec.get("justification"):
            # Append rather than replace to preserve LLM reasoning when present
            extra = str(dec["justification"]).strip()
            if justification:
                justification = f"{justification} | {extra}"
            else:
                justification = extra

    # First, apply default if base route is invalid
    if route == "unknown" and default:
        apply_decision(default)

    # Then evaluate rules; the first matching rule applies. If `enforce` is true,
    # it overrides regardless of base values.
    for rule in rules:
        when = rule.get("when") or {}
        # AND semantics across provided conditions
        match = True
        # match alert_id
        if "alert_id" in when:
            match = match and (str(when["alert_id"]).strip() == aid)
        # match service (accept service or service_name in policy)
        w_service = when.get("service") or when.get("service_name")
        if w_service is not None:
            match = match and (str(w_service).strip() == svc)
        # match severity
        if "severity" in when:
            match = match and (str(when["severity"]).strip() == sev)
        # match tags: require non-empty intersection
        if "tags" in when:
            wtags = when["tags"]
            if isinstance(wtags, str):
                wtags_list = [t.strip() for t in re.split(r"[,\s]+", wtags) if t.strip()]
            else:
                wtags_list = [str(t).strip() for t in (wtags if isinstance(wtags, (list, tuple)) else [])]
            wset = {t.lower() for t in wtags_list}
            match = match and (len(alert_tags.intersection(wset)) > 0)

        if not match:
            continue

        if rule.get("enforce", False):
            # Force override
            dec = rule.get("decision") or {}
            apply_decision(dec)
        else:
            # Only apply if we don't already have a valid route
            if route == "unknown":
                dec = rule.get("decision") or {}
                apply_decision(dec)

        pa = rule.get("proposed_action")
        if isinstance(pa, dict):
            # Ensure normalized shape
            action = (pa.get("action") or "").strip()
            if action:
                svc = pa.get("service") or pa.get("service_name") or (alert or {}).get("service") or ""
                payload = pa.get("payload") or {}
                # Ensure both service fields exist for compatibility
                payload = {"service": svc, "service_name": svc, **payload}
                suggested_action = {
                    "step": "proposed_action",
                    "action": action,
                    "service": svc,
                    "service_name": svc,
                    "payload": payload,
                }
        break  # first-match wins

    return route, risk, justification, suggested_action
