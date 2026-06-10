"""Tests for automatic constraint relaxation (Phase 13)."""

import copy
import pytest

from aiv_dse.core.constraint_relaxer import (
    UnreachableConstraint,
    analyze_and_relax,
    detect_unreachable_constraints,
    relax_policy,
)


POLICY = {
    "constraints": [
        {"id": "latency", "field": "latency_ns", "max": 10000, "severity": "CRITICAL", "on_violation": "VETO"},
        {"id": "area", "field": "area_units", "max": 50000, "severity": "WARNING", "on_violation": "ESCALATE"},
        {"id": "power", "field": "power_mw", "max": 500, "severity": "CRITICAL", "on_violation": "VETO"},
    ],
}


def _make_entry(status, latency=8000, area=40000, power=300, violations=None):
    return {
        "run_id": "R",
        "status": status,
        "metrics": {"latency_ns": latency, "area_units": area, "power_mw": power},
        "violations": violations or [],
    }


def _veto_entry(constraint_id, latency=15000, area=60000, power=600):
    return _make_entry(
        "VETO",
        latency=latency, area=area, power=power,
        violations=[{"constraint_id": constraint_id}],
    )


# ---------------------------------------------------------------------------
# detect_unreachable_constraints
# ---------------------------------------------------------------------------
class TestDetectUnreachable:
    def test_no_history_returns_empty(self):
        result = detect_unreachable_constraints([], POLICY)
        assert result == []

    def test_below_threshold_not_flagged(self):
        """3 VETOs is below default threshold of 5."""
        history = [_veto_entry("latency") for _ in range(3)]
        result = detect_unreachable_constraints(history, POLICY, consecutive_veto_threshold=5)
        assert len(result) == 0

    def test_consecutive_vetos_flagged(self):
        """5 consecutive VETOs should be detected."""
        history = [_veto_entry("latency") for _ in range(5)]
        result = detect_unreachable_constraints(history, POLICY, consecutive_veto_threshold=5)
        assert len(result) == 1
        assert result[0].constraint_id == "latency"
        assert result[0].consecutive_vetos == 5

    def test_broken_streak_not_flagged(self):
        """An APPROVED in the middle resets the consecutive count."""
        history = [
            _veto_entry("latency"),
            _veto_entry("latency"),
            _make_entry("APPROVED"),  # breaks the streak
            _veto_entry("latency"),
            _veto_entry("latency"),
        ]
        result = detect_unreachable_constraints(history, POLICY, consecutive_veto_threshold=5)
        assert len(result) == 0

    def test_gap_pct_computed(self):
        """gap_pct = (closest - threshold) / threshold * 100."""
        # threshold=10000, closest observed=12000 -> gap = 20%
        history = [
            _veto_entry("latency", latency=12000) for _ in range(5)
        ]
        result = detect_unreachable_constraints(history, POLICY, consecutive_veto_threshold=5)
        assert len(result) == 1
        assert abs(result[0].gap_pct - 20.0) < 0.1
        assert result[0].closest_observed == 12000

    def test_multiple_constraints_flagged(self):
        """Two constraints can both be unreachable."""
        history = [
            _make_entry(
                "VETO", latency=15000, power=700,
                violations=[
                    {"constraint_id": "latency"},
                    {"constraint_id": "power"},
                ],
            )
            for _ in range(5)
        ]
        result = detect_unreachable_constraints(history, POLICY, consecutive_veto_threshold=5)
        ids = {u.constraint_id for u in result}
        assert "latency" in ids
        assert "power" in ids


# ---------------------------------------------------------------------------
# relax_policy
# ---------------------------------------------------------------------------
class TestRelaxPolicy:
    def test_relax_increases_threshold(self):
        unreachable = [UnreachableConstraint(
            constraint_id="latency", field="latency_ns",
            current_threshold=10000, closest_observed=12000,
            consecutive_vetos=5, gap_pct=20.0, suggested_threshold=11000,
        )]
        relaxed = relax_policy(POLICY, unreachable, step_pct=10.0)
        lat = next(c for c in relaxed["constraints"] if c["id"] == "latency")
        assert lat["max"] == pytest.approx(11000.0)

    def test_original_policy_not_mutated(self):
        original_max = POLICY["constraints"][0]["max"]
        unreachable = [UnreachableConstraint(
            constraint_id="latency", field="latency_ns",
            current_threshold=10000, closest_observed=12000,
            consecutive_vetos=5, gap_pct=20.0, suggested_threshold=11000,
        )]
        relax_policy(POLICY, unreachable, step_pct=10.0)
        assert POLICY["constraints"][0]["max"] == original_max

    def test_only_flagged_constraints_relaxed(self):
        unreachable = [UnreachableConstraint(
            constraint_id="latency", field="latency_ns",
            current_threshold=10000, closest_observed=12000,
            consecutive_vetos=5, gap_pct=20.0, suggested_threshold=11000,
        )]
        relaxed = relax_policy(POLICY, unreachable, step_pct=10.0)
        area = next(c for c in relaxed["constraints"] if c["id"] == "area")
        power = next(c for c in relaxed["constraints"] if c["id"] == "power")
        assert area["max"] == 50000  # unchanged
        assert power["max"] == 500    # unchanged


# ---------------------------------------------------------------------------
# analyze_and_relax
# ---------------------------------------------------------------------------
class TestAnalyzeAndRelax:
    def test_no_auto_relax_returns_no_policy(self):
        history = [_veto_entry("latency") for _ in range(5)]
        report = analyze_and_relax(history, POLICY, auto_relax=False)
        assert len(report.unreachable) == 1
        assert report.relaxed_policy is None
        assert report.relaxed_constraints == []

    def test_auto_relax_returns_policy(self):
        history = [_veto_entry("latency") for _ in range(5)]
        report = analyze_and_relax(history, POLICY, auto_relax=True)
        assert report.relaxed_policy is not None
        assert "latency" in report.relaxed_constraints
        lat = next(c for c in report.relaxed_policy["constraints"] if c["id"] == "latency")
        assert lat["max"] > 10000
