"""Tests for Pareto front visualization (Phase 11)."""

import pytest

from aiv_dse.adapters.dummy_hls import DummyHLSAdapter
from aiv_dse.core.pareto import ParetoTracker
from aiv_dse.core.validator import load_policy
from aiv_dse.llm.models import SynthesisParams
from aiv_dse.run_loop import run_loop


# ---------------------------------------------------------------------------
# Plot tests
# ---------------------------------------------------------------------------
class TestPlotParetoFront:
    def test_plot_creates_file(self, tmp_path):
        plt = pytest.importorskip("matplotlib")
        from aiv_dse.core.visualize import plot_pareto_front

        all_pts = [
            {"metrics": {"latency_ns": 8000, "area_units": 40000, "power_mw": 300}},
            {"metrics": {"latency_ns": 12000, "area_units": 30000, "power_mw": 400}},
            {"metrics": {"latency_ns": 15000, "area_units": 60000, "power_mw": 600}},
        ]
        front_pts = [
            {"metrics": {"latency_ns": 8000, "area_units": 40000, "power_mw": 300}},
            {"metrics": {"latency_ns": 12000, "area_units": 30000, "power_mw": 400}},
        ]
        selected = {"metrics": {"latency_ns": 8000, "area_units": 40000, "power_mw": 300}}

        out = str(tmp_path / "pareto.png")
        result = plot_pareto_front(all_pts, front_pts, selected, out)
        assert (tmp_path / "pareto.png").exists()
        assert result.endswith("pareto.png")

    def test_plot_empty_front(self, tmp_path):
        pytest.importorskip("matplotlib")
        from aiv_dse.core.visualize import plot_pareto_front

        all_pts = [
            {"metrics": {"latency_ns": 15000, "area_units": 60000, "power_mw": 600}},
        ]
        out = str(tmp_path / "empty.png")
        # Should not raise
        plot_pareto_front(all_pts, [], None, out)
        assert (tmp_path / "empty.png").exists()

    def test_plot_no_selected(self, tmp_path):
        pytest.importorskip("matplotlib")
        from aiv_dse.core.visualize import plot_pareto_front

        all_pts = [
            {"metrics": {"latency_ns": 8000, "area_units": 40000, "power_mw": 300}},
        ]
        front_pts = [
            {"metrics": {"latency_ns": 8000, "area_units": 40000, "power_mw": 300}},
        ]
        out = str(tmp_path / "nosel.png")
        # None selected should not raise
        plot_pareto_front(all_pts, front_pts, None, out)
        assert (tmp_path / "nosel.png").exists()


# ---------------------------------------------------------------------------
# ParetoTracker.all_points tests
# ---------------------------------------------------------------------------
class TestParetoTrackerAllPoints:
    def test_all_points_includes_veto(self):
        tracker = ParetoTracker()
        tracker.add_point(
            run_id="R1",
            metrics={"latency_ns": 15000, "area_units": 60000, "power_mw": 600},
            synth_params={"unroll_factor": 16},
            status="VETO",
        )
        tracker.add_point(
            run_id="R2",
            metrics={"latency_ns": 8000, "area_units": 40000, "power_mw": 300},
            synth_params={"unroll_factor": 2},
            status="APPROVED",
        )
        all_pts = tracker.all_points
        assert len(all_pts) == 2
        assert any(p["run_id"] == "R1" for p in all_pts)
        # VETO point should not be in front
        front = tracker.front
        assert all(p["run_id"] != "R1" for p in front)


# ---------------------------------------------------------------------------
# No-plot flag test
# ---------------------------------------------------------------------------
class TestNoPlotFlag:
    def test_no_plot_skips_visualization(self):
        """plot=False runs without error and doesn't require matplotlib."""
        adapter = DummyHLSAdapter(noise_pct=0.0)
        policy = load_policy("policy/default_policy.yaml")
        params = SynthesisParams(unroll_factor=2, pipeline_depth=2)

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=3,
            strategy="shadow",
            plot=False,
        )
        assert result["final_status"] in ("CONVERGED", "MAX_ITERS_REACHED")
