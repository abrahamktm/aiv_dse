"""Stage 2 LLM-powered runner.

Usage:
    python -m aiv_dse.run_stage2 samples/report_fail.json
    python -m aiv_dse.run_stage2 samples/report_fail.json --baseline RUN-001
    python -m aiv_dse.run_stage2 samples/report_fail.json --sdk anthropic

Flow:
    1. Load report -> validate_physics (poison check)
    2. Load policy + state
    3. Validate report against policy -> ValidationResult
    4. Append to state, compute deltas
    5. If APPROVED -> print summary, write EDR, done
    6. If VETO/ESCALATE:
       a. Check stagnation (last 3 runs)
       b. Call constraint_advisor -> LLMProposal
       c. If HITL triggered -> human review
       d. Write EDR with proposal + human decision
    7. Print history_summary

Governance: max_iters from policy YAML is enforced. If exceeded, HALT.
"""

import argparse
import sys
from pathlib import Path

from aiv_dse.adapters.report_parser import PoisonDataError, load_report, validate_physics
from aiv_dse.core.state import (
    append_result,
    compute_deltas,
    history_summary,
    load_state,
    save_state,
)
from aiv_dse.core.stagnation import compute_deltas_vs_baseline, detect_stagnation
from aiv_dse.core.validator import ValidationResult, load_policy, validate
from aiv_dse.llm.config import LLMSettings
from aiv_dse.llm.constraint_advisor import propose_adjustments
from aiv_dse.llm.models import LLMProposal
from aiv_dse.workflow.edr_writer import write_edr
from aiv_dse.workflow.hitl import hitl_review, should_trigger_hitl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_PATH = str(_PROJECT_ROOT / "policy" / "default_policy.yaml")
STATE_PATH = str(_PROJECT_ROOT / "out" / "state.json")
EDR_DIR = str(_PROJECT_ROOT / "out")


def _count_consecutive_vetoes(state: dict) -> int:
    """Count how many consecutive VETO results from the tail of history."""
    history = state.get("history", [])
    count = 0
    for entry in reversed(history):
        if entry.get("status") == "VETO":
            count += 1
        else:
            break
    return count


def main(report_path: str, baseline_run_id: str | None = None, sdk_mode: str = "langchain") -> None:
    # Load report
    print(f"Loading report: {report_path}")
    report = load_report(report_path)
    print(f"  run_id: {report.get('run_id')}")

    # Physics validation (poison detection)
    is_poison = False
    try:
        validate_physics(report)
        print("  physics: OK")
    except PoisonDataError as e:
        print(f"  physics: FAILED -- {e}")
        is_poison = True

    # Policy validation
    policy = load_policy(POLICY_PATH)
    result = validate(report, policy, is_poison=is_poison)

    print(f"\n--- ValidationResult ---")
    print(f"  status: {result.status}")
    if result.violations:
        print("  violations:")
        for v in result.violations:
            print(f"    - {v['constraint_id']}: {v['field']}={v['observed']} "
                  f"(max {v['threshold']}, {v['severity']})")
    if result.reasons:
        print("  reasons:")
        for r in result.reasons:
            print(f"    - {r}")
    if result.suggested_relaxations:
        print("  suggested relaxations:")
        for s in result.suggested_relaxations:
            print(f"    - {s}")

    # Update state (skip if poison)
    if is_poison:
        print("\n--- State ---")
        print("  Poison run not persisted to state.")
        return

    state = load_state(STATE_PATH)

    # Governance: max_iters enforcement
    governance = policy.get("governance", {})
    max_iters = governance.get("max_iters", 20)
    current_iter = len(state.get("history", []))
    if current_iter >= max_iters:
        print(f"\n*** HALT: max_iters ({max_iters}) reached. ***")
        print("  Reset state or increase max_iters in policy to continue.")
        return

    state = append_result(state, result, report)
    save_state(STATE_PATH, state)

    # Deltas
    deltas = compute_deltas(state)
    if deltas:
        print("\n--- Deltas (vs previous run) ---")
        for field, pct in deltas.items():
            if pct is not None:
                sign = "+" if pct >= 0 else ""
                print(f"  {field}: {sign}{pct}%")

    # Baseline-relative deltas
    if baseline_run_id:
        bl_deltas = compute_deltas_vs_baseline(state, baseline_run_id)
        if bl_deltas:
            print(f"\n--- Deltas (vs baseline {baseline_run_id}) ---")
            for field, pct in bl_deltas.items():
                if pct is not None:
                    sign = "+" if pct >= 0 else ""
                    print(f"  {field}: {sign}{pct}%")
        else:
            print(f"\n  Baseline {baseline_run_id} not found in state history.")

    print(f"\n--- State ---")
    print(f"  {history_summary(state)}")

    # If APPROVED, write EDR and done
    proposal: LLMProposal | None = None
    human_decision: str | None = None

    if result.status == "APPROVED":
        print("\n  Result: APPROVED. No constraint changes needed.")
    else:
        # VETO or ESCALATE -> invoke LLM advisor
        print(f"\n--- LLM Constraint Advisor (sdk_mode={sdk_mode}) ---")

        # Check stagnation
        stagnation_msg = detect_stagnation(state)
        if stagnation_msg:
            print(f"  Warning: {stagnation_msg}")

        # Build LLM settings
        settings = LLMSettings.from_env()
        settings.sdk_mode = sdk_mode
        if sdk_mode == "anthropic":
            settings.provider = "anthropic"
            if settings.model_name == "gpt-4o-mini":
                settings.model_name = "claude-sonnet-4-20250514"

        try:
            proposal = propose_adjustments(policy, state, result, settings)

            print(f"  Confidence: {proposal.confidence:.2f}")
            print(f"  Cited runs: {', '.join(proposal.cited_runs)}")
            print(f"  Reasoning: {proposal.overall_reasoning}")
            for adj in proposal.adjustments:
                print(f"  -> {adj.constraint_id}: "
                      f"{adj.current_max} -> {adj.proposed_max}")
                print(f"     {adj.reasoning}")

            # Check HITL triggers
            consecutive_vetoes = _count_consecutive_vetoes(state)
            hitl_reason = should_trigger_hitl(
                proposal, consecutive_vetoes, stagnation_msg, policy
            )

            if hitl_reason:
                reviewed = hitl_review(proposal, hitl_reason)
                if reviewed is None:
                    human_decision = "Rejected"
                    proposal = None
                elif reviewed is proposal:
                    human_decision = "Accepted"
                else:
                    human_decision = "Modified"
                    proposal = reviewed
            else:
                human_decision = "Auto-accepted (high confidence)"

        except RuntimeError as e:
            print(f"  LLM error: {e}")
            print("  Falling back to deterministic suggestions only.")

    # Write EDR
    run_id = report.get("run_id", "unknown")
    edr_path = str(Path(EDR_DIR) / f"edr_{run_id}.md")
    write_edr(
        run_history=state.get("history", []),
        proposal=proposal,
        validation=result,
        output_path=edr_path,
        human_decision=human_decision,
    )
    print(f"\n  EDR written to {edr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIV-DSE Stage 2 runner")
    parser.add_argument("report", help="Path to synthesis report JSON")
    parser.add_argument("--baseline", default=None, help="Baseline run_id for delta comparison")
    parser.add_argument(
        "--sdk", default="langchain", choices=["langchain", "anthropic"],
        help="SDK mode: 'langchain' (default) or 'anthropic' (direct SDK)",
    )
    args = parser.parse_args()
    main(args.report, baseline_run_id=args.baseline, sdk_mode=args.sdk)
