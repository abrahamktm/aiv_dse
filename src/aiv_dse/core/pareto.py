"""Multi-objective Pareto front tracking.

Provides dominance checks, Pareto front computation, and a tracker
that accumulates points across iterations and supports frontier-based
convergence detection.
"""

from typing import Any, Dict, List, Optional


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """Return True if point *a* Pareto-dominates point *b*.

    All objectives must be <= (lower is better) and at least one must be
    strictly <.  Both dicts must share the same keys.
    """
    dominated_keys = a.keys() & b.keys()
    if not dominated_keys:
        return False
    all_leq = all(a[k] <= b[k] for k in dominated_keys)
    any_lt = any(a[k] < b[k] for k in dominated_keys)
    return all_leq and any_lt


def compute_pareto_front(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the non-dominated subset of *points*.

    Each element must contain a ``"metrics"`` dict with comparable numeric
    values.  Deduplicates by metrics (keeps the first occurrence).
    O(n^2) — fine for < 1000 points.
    """
    if not points:
        return []

    # Deduplicate by metrics tuple (keep first occurrence)
    seen_metrics: set = set()
    unique_points: List[Dict[str, Any]] = []
    for p in points:
        key = tuple(sorted(p["metrics"].items()))
        if key not in seen_metrics:
            seen_metrics.add(key)
            unique_points.append(p)

    front: List[Dict[str, Any]] = []
    for p in unique_points:
        p_metrics = p["metrics"]
        is_dominated = False
        for q in unique_points:
            if q is p:
                continue
            if dominates(q["metrics"], p_metrics):
                is_dominated = True
                break
        if not is_dominated:
            front.append(p)
    return front


class ParetoTracker:
    """Accumulates synthesis results and maintains a live Pareto front.

    Only points with status ``"APPROVED"`` enter the front computation.
    """

    def __init__(self) -> None:
        self._all_points: List[Dict[str, Any]] = []
        self._approved_points: List[Dict[str, Any]] = []
        self._front: List[Dict[str, Any]] = []
        self._front_size_history: List[int] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_point(
        self,
        run_id: str,
        metrics: Dict[str, float],
        synth_params: Dict[str, Any],
        status: str,
    ) -> None:
        """Record a synthesis result.  Recomputes the front if APPROVED."""
        point = {
            "run_id": run_id,
            "metrics": dict(metrics),
            "synth_params": dict(synth_params),
            "status": status,
        }
        self._all_points.append(point)
        if status == "APPROVED":
            self._approved_points.append(point)
            self._front = compute_pareto_front(self._approved_points)
            self._front_size_history.append(len(self._front))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def all_points(self) -> List[Dict[str, Any]]:
        """All explored points including VETO (for visualization)."""
        return list(self._all_points)

    @property
    def front(self) -> List[Dict[str, Any]]:
        return list(self._front)

    @property
    def front_size(self) -> int:
        return len(self._front)

    def select_by_weights(
        self,
        policy: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Pick the front point with the lowest weighted normalised score.

        Normalises each metric by its policy threshold, applies weights,
        and returns the point with the smallest weighted sum.
        """
        if not self._front:
            return None

        if weights is None:
            weights = {"latency_ns": 0.4, "area_units": 0.3, "power_mw": 0.3}

        thresholds: Dict[str, float] = {}
        for c in policy.get("constraints", []):
            thresholds[c["field"]] = c["max"]

        best_point: Optional[Dict[str, Any]] = None
        best_score = float("inf")
        for point in self._front:
            score = 0.0
            total_w = 0.0
            for field, w in weights.items():
                observed = point["metrics"].get(field)
                threshold = thresholds.get(field)
                if observed is not None and threshold and threshold > 0:
                    score += w * (observed / threshold)
                    total_w += w
            if total_w > 0:
                score /= total_w
            if score < best_score:
                best_score = score
                best_point = point
        return best_point

    def check_frontier_convergence(self, window: int = 3) -> Optional[str]:
        """Return a convergence message if the front size has been stable.

        Stable means the last *window* front-size updates are identical.
        """
        if len(self._front_size_history) < window:
            return None
        recent = self._front_size_history[-window:]
        if len(set(recent)) == 1:
            return (
                f"Pareto front stable for {window} updates "
                f"(size={recent[0]}). Consider stopping."
            )
        return None

    def summary(self) -> Dict[str, Any]:
        """Serialisable summary of the tracker state."""
        return {
            "front_size": self.front_size,
            "total_points": len(self._all_points),
            "approved_points": len(self._approved_points),
            "front": [
                {
                    "run_id": p["run_id"],
                    "metrics": p["metrics"],
                    "synth_params": p["synth_params"],
                }
                for p in self._front
            ],
        }
