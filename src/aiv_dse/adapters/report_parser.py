import json
import re
from typing import Any, Dict

INJECTION_PATTERNS = [
    "ignore all previous",
    "ignore instructions",
    "exfiltrate",
    "admin secrets",
    "send to cloud",
]


class PoisonDataError(Exception):
    """Raised when a report contains physically impossible values or injection."""
    pass


def load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_physics(report: Dict[str, Any]) -> None:
    """Check for physically impossible values and prompt injection.
    Raises PoisonDataError if any check fails."""
    problems = []

    latency = report.get("latency_ns")
    if latency is not None and latency <= 0:
        problems.append(f"latency_ns={latency} (must be > 0)")

    area = report.get("area_units")
    if area is not None and area < 0:
        problems.append(f"area_units={area} (must be >= 0)")

    power = report.get("power_mw")
    if power is not None and power <= 0:
        problems.append(f"power_mw={power} (must be > 0)")

    # Check notes for prompt injection patterns
    notes = str(report.get("notes", "")).lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(re.escape(pattern), notes):
            problems.append(f"prompt injection detected in notes: '{pattern}'")

    if problems:
        raise PoisonDataError(
            f"Poison data in run {report.get('run_id', '?')}: "
            + "; ".join(problems)
        )
