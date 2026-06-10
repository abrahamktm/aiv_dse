"""Long-term memory -- full history that is never trimmed.

state.py keeps a rolling window (MAX_HISTORY=3) for LLM context size.
This module keeps everything, forever. Used for:
  - Cross-run analysis
  - ML training data (future phases)
  - Constraint combo tracking (which param combos were tried + outcomes)
"""

import json
import os
from typing import Any, Dict, List, Optional

FULL_HISTORY_PATH = "out/full_history.json"


def append_full_history(
    entry: Dict[str, Any],
    path: str = FULL_HISTORY_PATH,
) -> None:
    """Append an entry to the full history file. Never trims."""
    history = load_full_history(path)
    history.append(entry)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_full_history(path: str = FULL_HISTORY_PATH) -> List[Dict[str, Any]]:
    """Load the complete history. Returns empty list if file doesn't exist."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_history(
    history: List[Dict[str, Any]],
    constraint_id: str,
) -> List[Dict[str, Any]]:
    """Find past runs that violated a specific constraint."""
    results = []
    for entry in history:
        for v in entry.get("violations", []):
            if v.get("constraint_id") == constraint_id:
                results.append(entry)
                break
    return results


def get_tried_combos(
    history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract param combos tried and their outcomes.

    Returns list of {params: {...}, status: str, metrics: {...}}.
    Useful for constraint pruning in future phases.
    """
    combos = []
    for entry in history:
        params = entry.get("synth_params")
        if params:
            combos.append({
                "params": params,
                "status": entry.get("status", "unknown"),
                "metrics": entry.get("metrics", {}),
            })
    return combos
