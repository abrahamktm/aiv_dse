"""Policy-to-prompt bridge.

Converts policy YAML, state history, and validation results into a
structured text block for the LLM. Contains only synthetic metrics
and policy thresholds -- no private/company data.
"""

from typing import Any, Dict

from aiv_dse.core.state import compute_deltas, METRIC_FIELDS
from aiv_dse.core.validator import ValidationResult


def format_context(
    policy: Dict[str, Any],
    state: Dict[str, Any],
    result: ValidationResult,
) -> str:
    """Build a prompt section from policy, state history, and latest result."""
    sections = []

    # Current constraints
    sections.append("## Current constraints")
    for c in policy.get("constraints", []):
        sections.append(
            f"- {c['id']}: max {c['max']} {c['field']} "
            f"({c.get('severity', 'WARNING')})"
        )

    # Run history
    history = state.get("history", [])
    if history:
        sections.append("")
        sections.append(f"## Run history (last {len(history)})")
        prev_metrics = None
        for entry in history:
            m = entry.get("metrics", {})
            metric_parts = []
            for field in METRIC_FIELDS:
                val = m.get(field)
                if val is not None:
                    metric_parts.append(f"{field}={val}")
            line = f"{entry['run_id']}: {', '.join(metric_parts)} -> {entry['status']}"

            # Add deltas vs previous entry
            if prev_metrics:
                delta_parts = []
                for field in METRIC_FIELDS:
                    p = prev_metrics.get(field)
                    c_val = m.get(field)
                    if p and c_val and p != 0:
                        pct = ((c_val - p) / p) * 100
                        sign = "+" if pct >= 0 else ""
                        delta_parts.append(f"{field} {sign}{pct:.1f}%")
                if delta_parts:
                    line += f" ({', '.join(delta_parts)})"

            sections.append(line)
            prev_metrics = m

    # Latest violations
    if result.violations:
        sections.append("")
        sections.append("## Latest violations")
        for v in result.violations:
            threshold = v["threshold"]
            observed = v["observed"]
            pct_over = ((observed - threshold) / threshold) * 100
            sections.append(
                f"- {v['constraint_id']}: {observed} > {threshold} "
                f"({pct_over:.0f}% over, {v['severity']})"
            )

    # Suggested relaxations from validator
    if result.suggested_relaxations:
        sections.append("")
        sections.append("## Suggested relaxations (from validator)")
        for s in result.suggested_relaxations:
            sections.append(f"- {s}")

    return "\n".join(sections)
