"""Engineering Decision Record (EDR) writer.

Writes a markdown EDR summarizing a design exploration run:
run history table, violations, LLM proposal, and human decision.
Analogous to AIV-DE's ADR but for iterative exploration.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aiv_dse.core.state import METRIC_FIELDS
from aiv_dse.core.validator import ValidationResult
from aiv_dse.llm.models import LLMProposal


def write_edr(
    run_history: List[Dict[str, Any]],
    proposal: Optional[LLMProposal],
    validation: ValidationResult,
    output_path: str,
    human_decision: Optional[str] = None,
) -> str:
    """Write a markdown EDR to disk.

    Args:
        run_history: List of state history entries.
        proposal: LLM proposal (None if no LLM was invoked).
        validation: Latest validation result.
        output_path: Path to write the EDR file.
        human_decision: "Accepted" | "Modified" | "Rejected" | None.

    Returns:
        The output file path.
    """
    latest = run_history[-1] if run_history else {}
    run_id = latest.get("run_id", "unknown")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = []
    lines.append(f"# EDR: Design Space Exploration -- {run_id}")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Status:** {validation.status}")
    lines.append("")

    # Run history table
    lines.append("## Run History")
    lines.append("")
    header_fields = ["Run"] + METRIC_FIELDS + ["Status"]
    lines.append("| " + " | ".join(header_fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_fields)) + " |")
    for entry in run_history:
        m = entry.get("metrics", {})
        row = [entry.get("run_id", "?")]
        for field in METRIC_FIELDS:
            row.append(str(m.get(field, "N/A")))
        row.append(entry.get("status", "?"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Violations
    if validation.violations:
        lines.append("## Violations")
        lines.append("")
        for v in validation.violations:
            lines.append(
                f"- **{v['constraint_id']}**: {v['observed']} > {v['threshold']} "
                f"({v['severity']})"
            )
        lines.append("")

    # Suggested relaxations
    if validation.suggested_relaxations:
        lines.append("## Suggested Relaxations (Deterministic)")
        lines.append("")
        for s in validation.suggested_relaxations:
            lines.append(f"- {s}")
        lines.append("")

    # LLM proposal
    if proposal:
        lines.append(f"## LLM Proposal (confidence: {proposal.confidence:.2f})")
        lines.append("")
        lines.append(f"**Cited runs:** {', '.join(proposal.cited_runs)}")
        lines.append(f"**Overall reasoning:** {proposal.overall_reasoning}")
        lines.append("")
        for adj in proposal.adjustments:
            lines.append(
                f"- Relax **{adj.constraint_id}**: "
                f"{adj.current_max} -> {adj.proposed_max}"
            )
            lines.append(f"  - Reasoning: {adj.reasoning}")
        lines.append("")

    # Human decision
    if human_decision:
        lines.append("## Decision")
        lines.append("")
        lines.append(f"{human_decision} by human operator.")
        lines.append("")

    content = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path
