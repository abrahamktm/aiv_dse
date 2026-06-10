"""Automatic constraint relaxation.

Detects unreachable constraints from consecutive VETO streaks and
optionally relaxes policy thresholds to allow convergence.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UnreachableConstraint:
    constraint_id: str
    field: str
    current_threshold: float
    closest_observed: float
    consecutive_vetos: int
    gap_pct: float
    suggested_threshold: float


@dataclass
class RelaxationReport:
    unreachable: List[UnreachableConstraint] = field(default_factory=list)
    relaxed_policy: Optional[Dict[str, Any]] = None
    relaxed_constraints: List[str] = field(default_factory=list)


def detect_unreachable_constraints(
    full_history: List[Dict[str, Any]],
    policy: Dict[str, Any],
    consecutive_veto_threshold: int = 5,
) -> List[UnreachableConstraint]:
    """Scan history from tail to detect constraints with consecutive VETOs.

    For each constraint, count how many consecutive most-recent runs
    violated it.  If >= threshold, flag it as unreachable.  Also find
    the closest observed value across all runs for that field.
    """
    if not full_history:
        return []

    constraints = policy.get("constraints", [])
    unreachable: List[UnreachableConstraint] = []

    for c in constraints:
        cid = c["id"]
        c_field = c["field"]
        threshold = c["max"]

        # Count consecutive VETOs from the tail
        consecutive = 0
        for entry in reversed(full_history):
            violations = entry.get("violations", [])
            violated_this = any(
                v.get("constraint_id") == cid for v in violations
            )
            if violated_this:
                consecutive += 1
            else:
                break

        if consecutive < consecutive_veto_threshold:
            continue

        # Find closest observed value across all history
        closest = None
        for entry in full_history:
            metrics = entry.get("metrics", {})
            observed = metrics.get(c_field)
            if observed is not None:
                if closest is None or abs(observed - threshold) < abs(closest - threshold):
                    closest = observed

        if closest is None:
            continue

        gap_pct = ((closest - threshold) / threshold) * 100 if threshold else 0.0
        suggested = threshold * (1 + 10.0 / 100)  # default 10% step

        unreachable.append(UnreachableConstraint(
            constraint_id=cid,
            field=c_field,
            current_threshold=threshold,
            closest_observed=closest,
            consecutive_vetos=consecutive,
            gap_pct=gap_pct,
            suggested_threshold=suggested,
        ))

    return unreachable


def relax_policy(
    policy: Dict[str, Any],
    unreachable: List[UnreachableConstraint],
    step_pct: float = 10.0,
) -> Dict[str, Any]:
    """Deep-copy the policy and increase thresholds for flagged constraints.

    Increases ``max`` by *step_pct* percent for each unreachable constraint.
    """
    relaxed = copy.deepcopy(policy)
    flagged_ids = {u.constraint_id for u in unreachable}

    for c in relaxed.get("constraints", []):
        if c["id"] in flagged_ids:
            c["max"] = c["max"] * (1 + step_pct / 100)

    return relaxed


def analyze_and_relax(
    full_history: List[Dict[str, Any]],
    policy: Dict[str, Any],
    consecutive_veto_threshold: int = 5,
    step_pct: float = 10.0,
    auto_relax: bool = False,
) -> RelaxationReport:
    """Orchestrate detection and optional relaxation.

    Returns a report with unreachable constraints and, if *auto_relax*
    is True, the relaxed policy.
    """
    unreachable = detect_unreachable_constraints(
        full_history, policy, consecutive_veto_threshold,
    )

    report = RelaxationReport(unreachable=unreachable)

    if auto_relax and unreachable:
        report.relaxed_policy = relax_policy(policy, unreachable, step_pct)
        report.relaxed_constraints = [u.constraint_id for u in unreachable]

    return report
