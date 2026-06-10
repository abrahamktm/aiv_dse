import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


@dataclass
class ValidationResult:
    status: str                                  # APPROVED | VETO | ESCALATE | HALT
    violations: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    suggested_relaxations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "violations": self.violations,
            "reasons": self.reasons,
            "suggested_relaxations": self.suggested_relaxations,
        }


def load_policy(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate(report: Dict[str, Any], policy: Dict[str, Any], *, is_poison: bool = False) -> ValidationResult:
    """Compare report metrics against policy constraints.
    Returns a ValidationResult with status based on worst severity."""

    if is_poison:
        return ValidationResult(
            status="HALT",
            reasons=["Poison data detected -- halting."],
        )

    constraints = policy.get("constraints", [])
    violations: List[Dict[str, Any]] = []
    reasons: List[str] = []
    relaxations: List[str] = []

    worst_severity = None

    for c in constraints:
        field_name = c["field"]
        threshold = c["max"]
        observed = report.get(field_name)

        if observed is None:
            continue

        if observed > threshold:
            severity = c.get("severity", "WARNING")
            action = c.get("on_violation", "ESCALATE")

            violations.append({
                "constraint_id": c["id"],
                "field": field_name,
                "observed": observed,
                "threshold": threshold,
                "severity": severity,
                "action": action,
            })

            pct_over = ((observed - threshold) / threshold) * 100
            reasons.append(
                f"{c['id']}: {field_name}={observed} exceeds max {threshold} "
                f"({pct_over:.0f}% over, severity={severity})"
            )

            # Suggest a relaxed threshold: observed + 10% margin, rounded up
            suggested = math.ceil(observed * 1.1)
            relaxations.append(
                f"increase {field_name} max from {threshold} to {suggested}"
            )

            if worst_severity is None or severity == "CRITICAL":
                worst_severity = severity

    if not violations:
        return ValidationResult(status="APPROVED")

    if worst_severity == "CRITICAL":
        status = "VETO"
    else:
        status = "ESCALATE"

    return ValidationResult(
        status=status,
        violations=violations,
        reasons=reasons,
        suggested_relaxations=relaxations,
    )
