"""Human-in-the-loop review step.

Presents LLM proposals to the human operator for accept/modify/reject.
Triggered by low confidence, repeated vetoes, stagnation, or CRITICAL
constraint violations in the proposal.
"""

import copy
from typing import Optional

from aiv_dse.llm.models import LLMProposal


def _print_proposal(proposal: LLMProposal) -> None:
    """Display proposal in a human-readable format."""
    print("\n=== LLM Proposal (HITL Review Required) ===")
    print(f"  Confidence: {proposal.confidence:.2f}")
    print(f"  Cited runs: {', '.join(proposal.cited_runs)}")
    print(f"  Overall reasoning: {proposal.overall_reasoning}")
    print()
    for i, adj in enumerate(proposal.adjustments, 1):
        print(f"  [{i}] {adj.constraint_id}: "
              f"{adj.current_max} -> {adj.proposed_max}")
        print(f"      Reasoning: {adj.reasoning}")
    print()


def hitl_review(proposal: LLMProposal, reason: str) -> Optional[LLMProposal]:
    """Present a proposal for human review.

    Args:
        proposal: The LLM-generated proposal to review.
        reason: Why HITL was triggered (e.g. "low confidence", "stagnation").

    Returns:
        The proposal (possibly modified) if accepted, or None if rejected.
    """
    print(f"\n*** HITL triggered: {reason} ***")
    _print_proposal(proposal)

    while True:
        choice = input("  Accept (a), Modify (m), Reject (r)? ").strip().lower()

        if choice == "a":
            print("  -> Proposal ACCEPTED.")
            return proposal

        elif choice == "r":
            print("  -> Proposal REJECTED.")
            return None

        elif choice == "m":
            modified = copy.deepcopy(proposal)
            for i, adj in enumerate(modified.adjustments):
                new_val = input(
                    f"  {adj.constraint_id} proposed_max "
                    f"[{adj.proposed_max}] (Enter to keep): "
                ).strip()
                if new_val:
                    try:
                        modified.adjustments[i].proposed_max = float(new_val)
                    except ValueError:
                        print(f"  Invalid number, keeping {adj.proposed_max}")
            print("  -> Proposal MODIFIED and accepted.")
            return modified

        else:
            print("  Invalid choice. Enter 'a', 'm', or 'r'.")


def should_trigger_hitl(
    proposal: LLMProposal,
    consecutive_vetoes: int,
    stagnation_msg: Optional[str],
    policy: dict,
) -> Optional[str]:
    """Determine if HITL review is needed. Returns the reason or None."""
    if proposal.confidence < 0.7:
        return f"Low confidence ({proposal.confidence:.2f})"

    if consecutive_vetoes >= 2:
        return f"Repeated vetoes ({consecutive_vetoes} consecutive)"

    if stagnation_msg:
        return stagnation_msg

    # Check if any proposed adjustment touches a CRITICAL constraint
    critical_ids = set()
    for c in policy.get("constraints", []):
        if c.get("severity") == "CRITICAL":
            critical_ids.add(c["id"])

    for adj in proposal.adjustments:
        if adj.constraint_id in critical_ids:
            return f"Adjustment to CRITICAL constraint: {adj.constraint_id}"

    return None
