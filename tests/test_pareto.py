"""Tests for multi-objective Pareto front (Phase 6).

Covers: dominates(), compute_pareto_front(), ParetoTracker,
Bayesian multi-objective mode, and loop integration.
"""

import pytest

from aiv_dse.core.pareto import ParetoTracker, compute_pareto_front, dominates
from aiv_dse.core.bayesian_advisor import BayesianAdvisor
from aiv_dse.llm.models import SynthParamProposal, SynthesisParams


# ---------------------------------------------------------------------------
# dominates() tests
# ---------------------------------------------------------------------------
class TestDominates:
    def test_strict_domination(self):
        """a is strictly better on all objectives."""
        a = {"latency_ns": 5, "area_units": 100, "power_mw": 10}
        b = {"latency_ns": 10, "area_units": 200, "power_mw": 20}
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_tradeoff_no_domination(self):
        """Neither dominates the other — they trade off."""
        a = {"latency_ns": 5, "area_units": 200}
        b = {"latency_ns": 10, "area_units": 100}
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_equal_no_domination(self):
        """Identical points do not dominate each other."""
        a = {"latency_ns": 5, "area_units": 100}
        assert dominates(a, a) is False

    def test_one_strictly_better_rest_equal(self):
        """Better on one metric, equal on others => dominates."""
        a = {"latency_ns": 5, "area_units": 100, "power_mw": 10}
        b = {"latency_ns": 5, "area_units": 100, "power_mw": 20}
        assert dominates(a, b) is True
        assert dominates(b, a) is False


# ---------------------------------------------------------------------------
# compute_pareto_front() tests
# ---------------------------------------------------------------------------
class TestComputeParetoFront:
    def test_single_point(self):
        points = [{"metrics": {"latency_ns": 10, "area_units": 100}}]
        front = compute_pareto_front(points)
        assert len(front) == 1

    def test_two_non_dominated(self):
        """Two tradeoff points => both on front."""
        points = [
            {"metrics": {"latency_ns": 5, "area_units": 200}},
            {"metrics": {"latency_ns": 10, "area_units": 100}},
        ]
        front = compute_pareto_front(points)
        assert len(front) == 2

    def test_one_dominated(self):
        """One point strictly dominated => excluded."""
        points = [
            {"metrics": {"latency_ns": 5, "area_units": 100}},
            {"metrics": {"latency_ns": 10, "area_units": 200}},  # dominated
        ]
        front = compute_pareto_front(points)
        assert len(front) == 1
        assert front[0]["metrics"]["latency_ns"] == 5

    def test_empty(self):
        assert compute_pareto_front([]) == []


# ---------------------------------------------------------------------------
# ParetoTracker tests
# ---------------------------------------------------------------------------
_POLICY = {"constraints": [
    {"id": "latency", "field": "latency_ns", "max": 10000},
    {"id": "area", "field": "area_units", "max": 50000},
    {"id": "power", "field": "power_mw", "max": 500},
]}


class TestParetoTracker:
    def test_excludes_veto(self):
        """VETO points should not appear in the front."""
        tracker = ParetoTracker()
        tracker.add_point("R1", {"latency_ns": 5000, "area_units": 30000, "power_mw": 200},
                          {"unroll_factor": 2}, "VETO")
        assert tracker.front_size == 0

    def test_select_by_weights(self):
        tracker = ParetoTracker()
        # Two APPROVED points on the front (tradeoff)
        tracker.add_point("R1", {"latency_ns": 5000, "area_units": 40000, "power_mw": 200},
                          {"unroll_factor": 2}, "APPROVED")
        tracker.add_point("R2", {"latency_ns": 8000, "area_units": 25000, "power_mw": 150},
                          {"unroll_factor": 4}, "APPROVED")
        assert tracker.front_size == 2

        # Weight latency heavily => should prefer R1 (lower latency)
        sel = tracker.select_by_weights(_POLICY, {"latency_ns": 0.9, "area_units": 0.05, "power_mw": 0.05})
        assert sel is not None
        assert sel["run_id"] == "R1"

    def test_convergence_detection(self):
        tracker = ParetoTracker()
        # First point establishes the front (size=1)
        tracker.add_point(
            "R1",
            {"latency_ns": 5000, "area_units": 30000, "power_mw": 200},
            {"unroll_factor": 2},
            "APPROVED",
        )
        # Next two are dominated by R1, so front stays at size 1
        tracker.add_point(
            "R2",
            {"latency_ns": 6000, "area_units": 31000, "power_mw": 210},
            {"unroll_factor": 3},
            "APPROVED",
        )
        tracker.add_point(
            "R3",
            {"latency_ns": 7000, "area_units": 32000, "power_mw": 220},
            {"unroll_factor": 4},
            "APPROVED",
        )
        # Front size has been [1, 1, 1] => stable
        msg = tracker.check_frontier_convergence(window=3)
        assert msg is not None
        assert "stable" in msg.lower()

    def test_summary(self):
        tracker = ParetoTracker()
        tracker.add_point("R1", {"latency_ns": 5000, "area_units": 30000, "power_mw": 200},
                          {"unroll_factor": 2}, "APPROVED")
        s = tracker.summary()
        assert s["front_size"] == 1
        assert s["total_points"] == 1
        assert s["approved_points"] == 1
        assert len(s["front"]) == 1


# ---------------------------------------------------------------------------
# Bayesian multi-objective tests
# ---------------------------------------------------------------------------
class TestBayesianMultiObjective:
    def test_nsga2_created(self):
        bo = BayesianAdvisor(multi_objective=True, seed=42)
        assert bo._multi_objective is True
        # Study should have 3 directions
        assert len(bo._study.directions) == 3

    def test_observe_and_propose(self):
        bo = BayesianAdvisor(multi_objective=True, seed=42)
        params = SynthesisParams(unroll_factor=4)
        report = {"latency_ns": 12000, "area_units": 60000, "power_mw": 400}
        policy = _POLICY

        for _ in range(3):
            bo.observe(params, report, policy)

        proposal = bo.propose(params)
        assert isinstance(proposal, SynthParamProposal)
        assert proposal.confidence > 0

    def test_backward_compat_single_objective(self):
        """Default (multi_objective=False) still creates single-direction study."""
        bo = BayesianAdvisor(seed=42)
        assert bo._multi_objective is False
        assert bo._study.direction.name == "MINIMIZE"


# ---------------------------------------------------------------------------
# Loop multi-objective integration tests
# ---------------------------------------------------------------------------
class TestLoopMultiObjective:
    def test_multi_objective_runs(self):
        """Multi-objective loop runs and returns pareto_summary."""
        from aiv_dse.adapters.dummy_hls import DummyHLSAdapter
        from aiv_dse.core.validator import load_policy
        from aiv_dse.run_loop import run_loop

        adapter = DummyHLSAdapter(noise_pct=0.0, seed=42)
        policy = load_policy("policy/default_policy.yaml")
        params = SynthesisParams(unroll_factor=2, pipeline_depth=2)

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=5,
            strategy="shadow",
            bayesian_seed=42,
            multi_objective=True,
        )
        assert result["pareto_summary"] is not None
        assert result["pareto_summary"]["total_points"] > 0

    def test_single_objective_unchanged(self):
        """Explicit single-objective returns None for pareto fields."""
        from aiv_dse.adapters.dummy_hls import DummyHLSAdapter
        from aiv_dse.core.validator import load_policy
        from aiv_dse.run_loop import run_loop

        adapter = DummyHLSAdapter(noise_pct=0.0)
        policy = load_policy("policy/default_policy.yaml")
        params = SynthesisParams(unroll_factor=2, pipeline_depth=2)

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=3,
            strategy="shadow",
            multi_objective=False,
        )
        assert result["pareto_summary"] is None
        assert result["pareto_selection"] is None
