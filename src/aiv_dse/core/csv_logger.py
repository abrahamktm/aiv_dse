"""CSV logger for all synthesis runs.

Append-only log with one row per run, capturing all params and metrics.
Useful for post-hoc analysis, plotting, ML training data, and
cross-run comparisons.
"""

import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict

from aiv_dse.llm.models import SynthesisParams

CSV_HEADERS = [
    "run_id",
    "timestamp",
    "status",
    "latency_ns",
    "area_units",
    "power_mw",
    "unroll_factor",
    "pipeline_depth",
    "clock_period_ns",
    "array_partition_factor",
    "clock_slack_ns",
    "dpo_mode",
    "flatten",
    "inline",
    "loop_merge",
    "bitwidth_reduce",
    "resource_sharing",
]


def init_csv_log(path: str) -> None:
    """Create CSV file with headers if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()


def log_run(
    path: str,
    run_id: str,
    status: str,
    report: Dict[str, Any],
    params: SynthesisParams,
) -> None:
    """Append a run to the CSV log.

    Args:
        path:   Path to runs.csv
        run_id: Run identifier (e.g. "RUN-003")
        status: Validation status (APPROVED, VETO, ESCALATE, HALT)
        report: Report dict from adapter
        params: SynthesisParams used for this run
    """
    init_csv_log(path)

    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "latency_ns": report.get("latency_ns"),
        "area_units": report.get("area_units"),
        "power_mw": report.get("power_mw"),
        "unroll_factor": params.unroll_factor,
        "pipeline_depth": params.pipeline_depth,
        "clock_period_ns": params.clock_period_ns,
        "array_partition_factor": params.array_partition_factor,
        "clock_slack_ns": params.clock_slack_ns,
        "dpo_mode": params.dpo_mode,
        "flatten": params.flatten,
        "inline": params.inline,
        "loop_merge": params.loop_merge,
        "bitwidth_reduce": params.bitwidth_reduce,
        "resource_sharing": params.resource_sharing,
    }

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)
