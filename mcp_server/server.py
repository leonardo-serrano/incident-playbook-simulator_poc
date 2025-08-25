# mcp_server/server.py
from __future__ import annotations

# Logging setup (to stderr so it won't interfere with MCP stdout)
import logging
import sys

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastmcp import FastMCP

# -----------------------------------------------------------------------------
# Data location: try mcp_server/sample_data first, then sample_data.
# You can override with the environment variable SAMPLE_DATA_DIR.
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [
    REPO_ROOT / "mcp_server" / "sample_data",
    REPO_ROOT / "sample_data",
]
SAMPLE_DATA_DIR = (
    Path(os.environ["SAMPLE_DATA_DIR"])
    if "SAMPLE_DATA_DIR" in os.environ
    else next((p for p in DEFAULT_DIRS if p.exists()), DEFAULT_DIRS[-1])
)

REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TRACES_DIR = REPORTS_DIR / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS_LOG = REPORTS_DIR / "actions.jsonl"
NOTIFY_LOG = REPORTS_DIR / "notifications.jsonl"
HITL_LOG = REPORTS_DIR / "hitl_decisions.jsonl"

METRICS_PATH = SAMPLE_DATA_DIR / "metrics.json"
LOGS_PATH = SAMPLE_DATA_DIR / "logs.json"
PLAYBOOKS_PATH = SAMPLE_DATA_DIR / "playbooks.md"
HITL_POLICY_PATH = SAMPLE_DATA_DIR / "hitl_policy.json"  # optional

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")

# -----------------------------------------------------------------------------
# MCP Server initialization
# -----------------------------------------------------------------------------
mcp = FastMCP("Incident Playbook MCP Server (mock)")

# -----------------------------------------------------------------------------
# 1) get_metrics(service_name)
# Reads metrics.json and returns the block for the given service.
# -----------------------------------------------------------------------------
@mcp.tool()
def get_metrics(service_name: str) -> Dict[str, Any]:
    """
    Return metrics for a given service from metrics.json
    """
    _ensure_exists(METRICS_PATH, "metrics")
    data = _read_json(METRICS_PATH)
    if service_name not in data:
        raise ValueError(f"service '{service_name}' not found in metrics.json")
    return {
        "service": service_name,
        "metrics": data[service_name],
        "source": str(METRICS_PATH),
        "timestamp": _now_iso(),
    }

# -----------------------------------------------------------------------------
# 2) query_logs(service_name, pattern?)
# Filters logs.json by service and (optionally) by regex pattern (case-insensitive).
# -----------------------------------------------------------------------------
@mcp.tool()
def query_logs(service_name: str, pattern: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    """
    Return logs for a service, optionally filtered by regex pattern (case-insensitive).
    """
    _ensure_exists(LOGS_PATH, "logs")
    data = _read_json(LOGS_PATH)
    if service_name not in data:
        raise ValueError(f"service '{service_name}' not found in logs.json")

    logs: List[Dict[str, Any]] = data[service_name]
    if pattern:
        try:
            rx = re.compile(pattern, flags=re.IGNORECASE)
            logs = [row for row in logs if rx.search(row.get("message", ""))]
        except re.error as e:
            raise ValueError(f"invalid regex pattern: {e}")

    if limit and limit > 0:
        logs = logs[:limit]

    return {
        "service": service_name,
        "pattern": pattern,
        "logs": logs,
        "source": str(LOGS_PATH),
        "timestamp": _now_iso(),
    }

# -----------------------------------------------------------------------------
# 3) search_playbooks(query, k=3)
# Searches sections in playbooks.md (split by '## '), ranks by token occurrences.
# -----------------------------------------------------------------------------
def _split_playbook_sections(text: str) -> List[Dict[str, Any]]:
    # Split by sections starting with '## '
    sections: List[Dict[str, Any]] = []
    current = {"title": None, "body": []}
    for line in text.splitlines():
        if line.startswith("## "):
            if current["title"]:
                sections.append({"title": current["title"], "body": "\n".join(current["body"]).strip()})
            current = {"title": line[3:].strip(), "body": []}
        else:
            current["body"].append(line)
    if current["title"]:
        sections.append({"title": current["title"], "body": "\n".join(current["body"]).strip()})
    return sections

@mcp.tool()
def search_playbooks(query: str, k: int = 3) -> Dict[str, Any]:
    """
    Return top-k playbook sections that match the query (simple token count scoring).
    """
    _ensure_exists(PLAYBOOKS_PATH, "playbooks")
    text = PLAYBOOKS_PATH.read_text(encoding="utf-8")
    sections = _split_playbook_sections(text)

    q = query.lower().strip()
    tokens = [t for t in re.split(r"\W+", q) if t]
    def score(section: Dict[str, Any]) -> int:
        blob = f"{section['title']}\n{section['body']}".lower()
        return sum(blob.count(tok) for tok in tokens)

    ranked = sorted(
        (s for s in sections),
        key=lambda s: (score(s), s["title"] is not None),
        reverse=True,
    )
    topk = ranked[: max(1, k)]
    results = [
        {
            "title": s["title"],
            "snippet": s["body"][:400],
        }
        for s in topk
    ]
    return {
        "query": query,
        "results": results,
        "source": str(PLAYBOOKS_PATH),
        "timestamp": _now_iso(),
    }

# -----------------------------------------------------------------------------
# 4) scale_instance(service_name, target)
# 5) rollback_deployment(service_name)
# 6) clear_cache(scope)
# Actions are recorded into reports/actions.jsonl for traceability.
# -----------------------------------------------------------------------------
def _record_action(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "action": kind,
        "payload": payload,
        "timestamp": _now_iso(),
    }
    _append_jsonl(ACTIONS_LOG, entry)
    return entry

@mcp.tool()
def scale_instance(service_name: str, target: int) -> Dict[str, Any]:
    """
    Mock scaling action - records the action and returns a deterministic response.
    """
    if target < 1:
        raise ValueError("target must be >= 1")
    payload = {"service": service_name, "target_instances": target}
    return _record_action("scale_instance", payload)

@mcp.tool()
def rollback_deployment(service_name: str) -> Dict[str, Any]:
    """
    Mock rollback - records the action and returns a deterministic response.
    """
    payload = {"service": service_name}
    return _record_action("rollback_deployment", payload)

@mcp.tool()
def clear_cache(scope: str) -> Dict[str, Any]:
    """
    Mock cache clear - records the action and returns a deterministic response.
    """
    payload = {"scope": scope}
    return _record_action("clear_cache", payload)

# -----------------------------------------------------------------------------
# 7) notify_team(message, channel?)
# Append the notification into reports/notifications.jsonl (mock channel).
# -----------------------------------------------------------------------------
@mcp.tool()
def notify_team(message: str, channel: Optional[str] = None) -> Dict[str, Any]:
    """
    Mock notification - appends to notifications log.
    """
    entry = {
        "channel": channel or "incident-bridge",
        "message": message,
        "timestamp": _now_iso(),
    }
    _append_jsonl(NOTIFY_LOG, entry)
    return {"status": "sent", **entry}

# -----------------------------------------------------------------------------
# 8) ask_human(prompt, policy?)
# Reads policy from sample_data/hitl_policy.json (if present), otherwise approves by default.
# Example structure:
# { "default": "approve",
#   "rules": [
#     {"contains": "rollback", "decision": "approve"},
#     {"contains": "delete",   "decision": "deny"}
#   ]
# }
# -----------------------------------------------------------------------------
def _decide_hitl(prompt: str, policy: Optional[str]) -> Dict[str, Any]:
    decision = "approve"
    used_rule = "default"

    # If an explicit policy ("approve"/"deny") is passed, it overrides the file policy
    if policy and policy.lower() in {"approve", "deny"}:
        decision = policy.lower()
        used_rule = "explicit"
    elif HITL_POLICY_PATH.exists():
        cfg = _read_json(HITL_POLICY_PATH)
        decision = str(cfg.get("default", "approve")).lower()
        rules = cfg.get("rules", [])
        for r in rules:
            if str(r.get("contains", "")).lower() in prompt.lower():
                d = str(r.get("decision", decision)).lower()
                if d in {"approve", "deny"}:
                    decision = d
                    used_rule = f"rule:{r.get('contains')}"
                    break

    return {
        "approved": decision == "approve",
        "decision": decision,
        "used_rule": used_rule,
    }

@mcp.tool()
def ask_human(prompt: str, policy: Optional[str] = None) -> Dict[str, Any]:
    """
    HITL mock - returns approval/denial based on hitl_policy.json or explicit policy.
    """
    result = _decide_hitl(prompt, policy)
    entry = {
        "prompt": prompt,
        "result": result,
        "timestamp": _now_iso(),
    }
    _append_jsonl(HITL_LOG, entry)
    return entry

# -----------------------------------------------------------------------------
# Optional health check
# -----------------------------------------------------------------------------
@mcp.tool()
def ping() -> str:
    return "pong"

# -----------------------------------------------------------------------------
# MCP stdio server entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Configure root logging only when running this file as a program.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.info("Using sample data at: %s", SAMPLE_DATA_DIR)
    mcp.run()
