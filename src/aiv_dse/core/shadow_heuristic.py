"""Shadow heuristic -- deterministic baseline to prove the LLM adds value.

This is NOT a fallback. It runs every iteration alongside the LLM and
Bayesian advisor. Its proposals are logged for comparison but never
applied (unless --strategy shadow is explicitly chosen).

The rule is deliberately simple:
  - Find the most-violated constraint (highest % over threshold)
  - If latency violated  -> increase unroll_factor by 1
  - If area/power violated -> decrease unroll_factor by 1
  - If conflicting directions -> do nothing (stuck)

This proves the LLM is better than a single-dimensional heuristic
that can't balance multi-dimensional tradeoffs.
"""

from typing import Any, Dict

from aiv_dse.core.validator import ValidationResult
from aiv_dse.llm.models import SynthParamAdjustment, SynthParamProposal, SynthesisParams


def shadow_propose(
    result: ValidationResult,
    current_params: SynthesisParams,
    policy: Dict[str, Any],
) -> SynthParamProposal:
    """Propose parameter changes using the dumb heuristic rule."""

    if not result.violations:
        return SynthParamProposal(
            adjustments=[],
            overall_reasoning="Shadow heuristic: no violations, no changes needed.",
            confidence=1.0,
            cited_runs=["N/A"],
        )

    # Find worst violation by % over threshold
    worst = None
    worst_pct = 0.0
    for v in result.violations:
        threshold = v["threshold"]
        observed = v["observed"]
        if threshold > 0:
            pct_over = ((observed - threshold) / threshold) * 100
            if pct_over > worst_pct:
                worst_pct = pct_over
                worst = v

    if worst is None:
        return SynthParamProposal(
            adjustments=[],
            overall_reasoning="Shadow heuristic: could not determine worst violation.",
            confidence=1.0,
            cited_runs=["N/A"],
        )

    cid = worst["constraint_id"]
    adjustments = []

    # Determine direction
    if cid == "latency":
        # Need more parallelism -> increase unroll
        new_val = min(current_params.unroll_factor + 1, 64)
        if new_val != current_params.unroll_factor:
            adjustments.append(SynthParamAdjustment(
                param_name="unroll_factor",
                current_value=float(current_params.unroll_factor),
                proposed_value=float(new_val),
                reasoning=(
                    f"Shadow heuristic: latency violated by {worst_pct:.0f}%, "
                    f"increasing unroll_factor to add parallelism."
                ),
            ))
    elif cid in ("area", "power"):
        # Need less resources -> decrease unroll
        new_val = max(current_params.unroll_factor - 1, 1)
        if new_val != current_params.unroll_factor:
            adjustments.append(SynthParamAdjustment(
                param_name="unroll_factor",
                current_value=float(current_params.unroll_factor),
                proposed_value=float(new_val),
                reasoning=(
                    f"Shadow heuristic: {cid} violated by {worst_pct:.0f}%, "
                    f"decreasing unroll_factor to reduce resource usage."
                ),
            ))
    else:
        # Unknown constraint, do nothing
        pass

    reasoning = (
        f"Shadow heuristic: worst violation is {cid} "
        f"({worst_pct:.0f}% over). "
        + (f"Adjusting unroll_factor." if adjustments else "No actionable change.")
    )

    return SynthParamProposal(
        adjustments=adjustments,
        overall_reasoning=reasoning,
        confidence=1.0,
        cited_runs=["N/A"],
    )
