import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aiv_dse.core.validator import ValidationResult

MAX_HISTORY = 3
MAX_LESSONS = 10
METRIC_FIELDS = ["latency_ns", "area_units", "power_mw"]


def load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"history": [], "lessons_learned": []}
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    # Backward compat: older state files lack lessons_learned
    state.setdefault("lessons_learned", [])
    return state


def save_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def append_result(
    state: Dict[str, Any],
    result: ValidationResult,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """Append a validation result to state history, keeping last MAX_HISTORY entries."""
    entry = {
        "run_id": report.get("run_id", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": result.status,
        "metrics": {f: report.get(f) for f in METRIC_FIELDS},
        "violations": result.violations,
    }
    history = state.get("history", [])
    history.append(entry)

    # Trim to last MAX_HISTORY entries
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    return {"history": history}


def compute_deltas(state: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Compute % change between the last two runs. Returns None if <2 runs."""
    history = state.get("history", [])
    if len(history) < 2:
        return None

    prev = history[-2]["metrics"]
    curr = history[-1]["metrics"]
    deltas = {}

    for field in METRIC_FIELDS:
        p = prev.get(field)
        c = curr.get(field)
        if p and c and p != 0:
            deltas[field] = round(((c - p) / p) * 100, 1)
        else:
            deltas[field] = None

    return deltas


def append_lesson(
    state: Dict[str, Any],
    iteration: int,
    proposed_change: str,
    rejection_reason: str,
) -> Dict[str, Any]:
    """Append a Reflexion-style lesson when the judge rejects a proposal.

    Stored in state["lessons_learned"]. Capped at MAX_LESSONS to keep prompts
    bounded. Subsequent advisor calls read these so the advisor learns
    from past judge rejections instead of repeating the same mistake.
    """
    lessons = state.get("lessons_learned", [])
    lessons.append({
        "iteration": iteration,
        "proposed_change": proposed_change,
        "rejection_reason": rejection_reason,
    })
    if len(lessons) > MAX_LESSONS:
        lessons = lessons[-MAX_LESSONS:]
    state["lessons_learned"] = lessons
    return state


def history_summary(state: Dict[str, Any]) -> str:
    """Human-readable summary of recent runs and trends."""
    history = state.get("history", [])
    if not history:
        return "No runs recorded."

    n = len(history)
    statuses = [h["status"] for h in history]
    counts = {}
    for s in statuses:
        counts[s] = counts.get(s, 0) + 1

    parts = [f"{n} run{'s' if n != 1 else ''}:"]
    parts.extend(f"{count} {status}" for status, count in sorted(counts.items()))

    deltas = compute_deltas(state)
    if deltas:
        delta_strs = []
        for field, pct in deltas.items():
            if pct is not None:
                sign = "+" if pct >= 0 else ""
                delta_strs.append(f"{field} {sign}{pct}%")
        if delta_strs:
            parts.append("Latest deltas: " + ", ".join(delta_strs))

    return ". ".join(parts)
