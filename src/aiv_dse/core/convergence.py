"""Enhanced convergence detection with weighted tradeoff scoring.

Stagnation = stuck and failing (detected by stagnation.py).
Convergence = stable and passing (detected here).
Both are reasons to stop the loop.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aiv_dse.core.pareto import ParetoTracker


def compute_weighted_score(
    metrics: Dict[str, Any],
    policy: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted tradeoff score (lower = better).

    Normalizes each metric against its policy max, then weighted sum.
    Default weights: latency=0.4, area=0.3, power=0.3.

    A score of 1.0 means every metric is exactly at its threshold.
    Below 1.0 = all metrics have headroom. Above 1.0 = some violations.
    """
    if weights is None:
        weights = {"latency_ns": 0.4, "area_units": 0.3, "power_mw": 0.3}

    # Build field -> max lookup from policy constraints
    thresholds = {}
    for c in policy.get("constraints", []):
        thresholds[c["field"]] = c["max"]

    score = 0.0
    total_weight = 0.0
    for field, w in weights.items():
        observed = metrics.get(field)
        threshold = thresholds.get(field)
        if observed is not None and threshold and threshold > 0:
            score += w * (observed / threshold)
            total_weight += w

    if total_weight > 0:
        score /= total_weight  # Normalize so weights don't need to sum to 1

    return round(score, 4)


def check_convergence(
    state: Dict[str, Any],
    policy: Dict[str, Any],
    threshold_pct: float = 2.0,
    window: int = 3,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """Check if the loop has converged (stable and passing).

    Convergence = last `window` runs are ALL APPROVED and the weighted
    score delta between consecutive runs is < threshold_pct.

    Returns a message suggesting to stop, or None if not converged.
    """
    history = state.get("history", [])
    if len(history) < window:
        return None

    recent = history[-window:]

    # All must be APPROVED
    if not all(entry.get("status") == "APPROVED" for entry in recent):
        return None

    # Compute scores for each run in window
    scores = []
    for entry in recent:
        m = entry.get("metrics", {})
        scores.append(compute_weighted_score(m, policy, weights))

    # Check if score deltas are small
    for i in range(1, len(scores)):
        if scores[i - 1] > 0:
            pct_change = abs((scores[i] - scores[i - 1]) / scores[i - 1]) * 100
            if pct_change >= threshold_pct:
                return None

    avg_score = sum(scores) / len(scores)
    return (
        f"Converged: last {window} runs all APPROVED with stable "
        f"weighted score (~{avg_score:.3f}). Consider stopping."
    )


def check_pareto_convergence(
    tracker: "ParetoTracker",
    window: int = 3,
) -> Optional[str]:
    """Thin wrapper delegating Pareto frontier convergence to the tracker."""
    return tracker.check_frontier_convergence(window=window)
