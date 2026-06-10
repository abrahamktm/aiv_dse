"""Stage 1 deterministic runner.

Usage:
    python -m aiv_dse.run_stage1 samples/report_pass.json
    python -m aiv_dse.run_stage1 samples/report_fail.json
    python -m aiv_dse.run_stage1 samples/poison_report.json
"""

import json
import os
import sys
from pathlib import Path

from aiv_dse.adapters.report_parser import load_report, validate_physics, PoisonDataError
from aiv_dse.core.validator import load_policy, validate
from aiv_dse.core.state import load_state, save_state, append_result, history_summary

# Resolve paths relative to the project root (two levels up from this file's src/aiv_dse/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_PATH = str(_PROJECT_ROOT / "policy" / "default_policy.yaml")
STATE_PATH = str(_PROJECT_ROOT / "out" / "state.json")


def main(report_path: str) -> None:
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
        print(f"  violations:")
        for v in result.violations:
            print(f"    - {v['constraint_id']}: {v['field']}={v['observed']} "
                  f"(max {v['threshold']}, {v['severity']})")
    if result.reasons:
        print(f"  reasons:")
        for r in result.reasons:
            print(f"    - {r}")
    if result.suggested_relaxations:
        print(f"  suggested relaxations:")
        for s in result.suggested_relaxations:
            print(f"    - {s}")

    # Update state (skip if poison -- don't persist bad data)
    if not is_poison:
        state = load_state(STATE_PATH)
        state = append_result(state, result, report)
        save_state(STATE_PATH, state)
        print(f"\n--- State ---")
        print(f"  {history_summary(state)}")
        print(f"  saved to {STATE_PATH}")
    else:
        print(f"\n--- State ---")
        print(f"  Poison run not persisted to state.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m aiv_dse.run_stage1 <report.json>")
        sys.exit(1)
    main(sys.argv[1])
