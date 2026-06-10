"""HLS report file parser.

Parses timing.rpt, area.rpt, power.rpt, and synth.log to extract
metrics into the standard report dict format used by the loop.
"""

import os
import re
from typing import Any, Dict


class ReportParseError(Exception):
    """Raised when a required metric cannot be extracted from a report."""


def parse_timing_rpt(path: str) -> Dict[str, Any]:
    """Parse timing.rpt for latency, clock period, and slack.

    Returns:
        {"latency_ns": float, "clock_period_ns": float, "slack_ns": float}
    """
    if not os.path.exists(path):
        raise ReportParseError(f"Timing report not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    result: Dict[str, Any] = {}

    match = re.search(r"Actual Period\s*:\s*([\d.]+)\s*ns", content)
    if match:
        result["clock_period_ns"] = float(match.group(1))

    match = re.search(r"Slack\s*:\s*([-\d.]+)\s*ns", content)
    if match:
        result["slack_ns"] = float(match.group(1))

    match = re.search(r"Latency \(ns\)\s*:\s*([\d.]+)", content)
    if match:
        result["latency_ns"] = float(match.group(1))
    else:
        raise ReportParseError("Could not extract latency_ns from timing.rpt")

    return result


def parse_area_rpt(path: str) -> Dict[str, Any]:
    """Parse area.rpt for total area and resource breakdown.

    Returns:
        {"area_units": int, "breakdown": {"luts": int, "ffs": int, ...}}
    """
    if not os.path.exists(path):
        raise ReportParseError(f"Area report not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    result: Dict[str, Any] = {"breakdown": {}}

    match = re.search(r"Total Area Units\s+(\d+)", content)
    if match:
        result["area_units"] = int(match.group(1))
    else:
        raise ReportParseError("Could not extract area_units from area.rpt")

    for pattern, key in [
        (r"LUTs\s+(\d+)", "luts"),
        (r"Flip-Flops\s+(\d+)", "ffs"),
        (r"BRAM Blocks\s+(\d+)", "bram"),
        (r"DSP Slices\s+(\d+)", "dsp"),
    ]:
        match = re.search(pattern, content)
        if match:
            result["breakdown"][key] = int(match.group(1))

    return result


def parse_power_rpt(path: str) -> Dict[str, Any]:
    """Parse power.rpt for total power and breakdown.

    Returns:
        {"power_mw": float, "breakdown": {"dynamic_mw": float, "static_mw": float}}
    """
    if not os.path.exists(path):
        raise ReportParseError(f"Power report not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    result: Dict[str, Any] = {"breakdown": {}}

    match = re.search(r"Total Power\s*:\s*([\d.]+)\s*mW", content)
    if match:
        result["power_mw"] = float(match.group(1))
    else:
        raise ReportParseError("Could not extract power_mw from power.rpt")

    match = re.search(r"Dynamic Power\s*:\s*([\d.]+)\s*mW", content)
    if match:
        result["breakdown"]["dynamic_mw"] = float(match.group(1))

    match = re.search(r"Static Power\s*:\s*([\d.]+)\s*mW", content)
    if match:
        result["breakdown"]["static_mw"] = float(match.group(1))

    return result


def parse_synth_log(path: str) -> Dict[str, Any]:
    """Parse synth.log for warnings, suggestions, and errors.

    Returns:
        {"warnings": list, "suggestions": list, "errors": list, "exit_status": str}
    """
    if not os.path.exists(path):
        raise ReportParseError(f"Synthesis log not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    warnings = re.findall(r"WARNING:\s*\[([^\]]+)\]\s*(.+)", content)
    suggestions = re.findall(r"SUGGESTION:\s*\[([^\]]+)\]\s*(.+)", content)
    errors = re.findall(r"ERROR:\s*\[([^\]]+)\]\s*(.+)", content)

    match = re.search(r"Exit status:\s*(\w+)", content)
    exit_status = match.group(1) if match else "UNKNOWN"

    return {
        "warnings": [f"{code}: {msg}" for code, msg in warnings],
        "suggestions": [f"{code}: {msg}" for code, msg in suggestions],
        "errors": [f"{code}: {msg}" for code, msg in errors],
        "exit_status": exit_status,
    }


def parse_all_reports(report_dir: str) -> Dict[str, Any]:
    """Parse all HLS reports from a directory into a standard report dict."""
    timing = parse_timing_rpt(os.path.join(report_dir, "timing.rpt"))
    area = parse_area_rpt(os.path.join(report_dir, "area.rpt"))
    power = parse_power_rpt(os.path.join(report_dir, "power.rpt"))

    log_path = os.path.join(report_dir, "synth.log")
    log = parse_synth_log(log_path) if os.path.exists(log_path) else {
        "warnings": [], "suggestions": [], "errors": [], "exit_status": "UNKNOWN",
    }

    return {
        "latency_ns": timing["latency_ns"],
        "area_units": area["area_units"],
        "power_mw": power["power_mw"],
        "clock_period_ns": timing.get("clock_period_ns"),
        "slack_ns": timing.get("slack_ns"),
        "area_breakdown": area.get("breakdown"),
        "power_breakdown": power.get("breakdown"),
        "warnings": log.get("warnings", []),
        "suggestions": log.get("suggestions", []),
        "errors": log.get("errors", []),
        "synthesis_status": log.get("exit_status", "UNKNOWN"),
    }
