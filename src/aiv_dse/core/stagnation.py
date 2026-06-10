"""Convergence and stagnation detection.

Checks whether recent runs show meaningful improvement or if the
exploration loop has stalled.
"""

from typing import Any, Dict, Optional

from aiv_dse.core.state import METRIC_FIELDS


def detect_stagnation(
    state: Dict[str, Any],
    threshold_pct: float = 2.0,
    window: int = 3,
) -> Optional[str]:
    """Check if the last `window` runs all have deltas below threshold_pct.

    Returns None if not stagnant, or a descriptive message if stagnant.
    """
    history = state.get("history", [])
    if len(history) < window:
        return None

    recent = history[-window:]

    # Compare each consecutive pair in the window
    for i in range(1, len(recent)):
        prev_m = recent[i - 1].get("metrics", {})
        curr_m = recent[i].get("metrics", {})

        for field in METRIC_FIELDS:
            p = prev_m.get(field)
            c = curr_m.get(field)
            if p and c and p != 0:
                pct_change = abs(((c - p) / p) * 100)
                if pct_change >= threshold_pct:
                    return None  # Found meaningful change

    return (
        f"Stagnation detected: last {window} runs show <{threshold_pct}% "
        f"improvement in all metrics. Consider relaxing constraints or "
        f"changing approach."
    )


def compute_deltas_vs_baseline(
    state: Dict[str, Any],
    baseline_run_id: str,
) -> Optional[Dict[str, float]]:
    """Compute % change between the latest run and a specific baseline run.

    Returns None if baseline_run_id is not found or no runs exist.
    """
    history = state.get("history", [])
    if not history:
        return None

    # Find the baseline entry
    baseline = None
    for entry in history:
        if entry.get("run_id") == baseline_run_id:
            baseline = entry
            break

    if baseline is None:
        return None

    latest = history[-1]
    baseline_m = baseline.get("metrics", {})
    latest_m = latest.get("metrics", {})
    deltas = {}

    for field in METRIC_FIELDS:
        b = baseline_m.get(field)
        l = latest_m.get(field)
        if b and l and b != 0:
            deltas[field] = round(((l - b) / b) * 100, 1)
        else:
            deltas[field] = None

    return deltas
