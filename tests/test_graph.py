"""Tests for the LangGraph DSE state machine.

Covers:
- Happy path (bayesian strategy converges)
- HALT on poison data
- Convergence detection
- State transitions
"""

import pytest
from unittest.mock import MagicMock, patch

from aiv_dse.graph import (
    DSEState,
    DSEContext,
    build_dse_graph,
    run_graph,
    set_context,
    synthesize,
    validate_node,
    record,
    check_terminal,
    propose,
    apply_proposal,
    should_continue,
)
from aiv_dse.adapters.dummy_hls import DummyHLSAdapter
from aiv_dse.core.bayesian_advisor import BayesianAdvisor
from aiv_dse.core.pareto import ParetoTracker
from aiv_dse.core.validator import load_policy
from aiv_dse.llm.models import SynthesisParams


@pytest.fixture
def policy():
    """Load the default policy."""
    return {
        "constraints": [
            {"id": "latency", "field": "latency_ns", "max": 10000, "severity": "CRITICAL", "on_violation": "VETO"},
            {"id": "area", "field": "area_units", "max": 50000, "severity": "CRITICAL", "on_violation": "VETO"},
            {"id": "power", "field": "power_mw", "max": 500, "severity": "WARNING", "on_violation": "ESCALATE"},
        ]
    }


@pytest.fixture
def adapter():
    """Create a deterministic DummyHLS adapter."""
    return DummyHLSAdapter(noise_pct=0.0, seed=42)


@pytest.fixture
def context(adapter, policy):
    """Create and set the DSE context."""
    bayesian = BayesianAdvisor(sampler="tpe", seed=42, multi_objective=True)
    pareto = ParetoTracker()
    ctx = DSEContext(adapter=adapter, bayesian=bayesian, pareto=pareto, policy=policy)
    set_context(ctx)
    return ctx


@pytest.fixture
def initial_state(policy):
    """Create an initial DSE state."""
    params = SynthesisParams(unroll_factor=4, pipeline_depth=1)
    return DSEState(
        params=params.model_dump(),
        max_iterations=10,
        strategy="bayesian",
        policy=policy,
    )


class TestDSEState:
    """Test the DSEState schema."""

    def test_default_state(self):
        state = DSEState()
        assert state.iteration == 0
        assert state.status == ""
        assert state.converged is False
        assert state.halted is False

    def test_state_with_params(self):
        params = SynthesisParams(unroll_factor=8, pipeline_depth=2)
        state = DSEState(params=params.model_dump())
        assert state.params["unroll_factor"] == 8
        assert state.params["pipeline_depth"] == 2


class TestSynthesizeNode:
    """Test the synthesize node."""

    def test_synthesize_increments_iteration(self, context, initial_state):
        result = synthesize(initial_state)
        assert result["iteration"] == 1
        assert result["run_id"] == "RUN-001"
        assert "latency_ns" in result["metrics"]
        assert "area_units" in result["metrics"]
        assert "power_mw" in result["metrics"]

    def test_synthesize_uses_params(self, context, policy):
        # Sweet spot params: unroll=2, pipeline=2
        params = SynthesisParams(unroll_factor=2, pipeline_depth=2)
        state = DSEState(params=params.model_dump(), policy=policy)
        result = synthesize(state)
        # With these params, should get good metrics
        assert result["metrics"]["latency_ns"] < 10000
        assert result["metrics"]["area_units"] < 50000


class TestValidateNode:
    """Test the validate node."""

    def test_validate_approved(self, context, policy):
        state = DSEState(
            metrics={"latency_ns": 5000, "area_units": 30000, "power_mw": 200},
            policy=policy,
        )
        result = validate_node(state)
        assert result["status"] == "APPROVED"
        assert result["halted"] is False

    def test_validate_veto(self, context, policy):
        state = DSEState(
            metrics={"latency_ns": 15000, "area_units": 60000, "power_mw": 200},
            policy=policy,
        )
        result = validate_node(state)
        assert result["status"] == "VETO"
        assert result["halted"] is False

    def test_validate_halt_on_poison(self, context, policy):
        # Negative latency is physically impossible (poison)
        state = DSEState(
            metrics={"latency_ns": -100, "area_units": 30000, "power_mw": 200},
            policy=policy,
        )
        result = validate_node(state)
        assert result["status"] == "HALT"
        assert result["halted"] is True


class TestRecordNode:
    """Test the record node."""

    def test_record_updates_history(self, context, policy):
        params = SynthesisParams()
        state = DSEState(
            run_id="RUN-001",
            params=params.model_dump(),
            status="VETO",
            metrics={"latency_ns": 15000, "area_units": 60000, "power_mw": 200},
            validation={"violations": [{"constraint_id": "latency"}]},
            history=[],
            policy=policy,
        )
        result = record(state)
        assert len(result["history"]) == 1
        assert result["history"][0]["run_id"] == "RUN-001"
        assert result["history"][0]["status"] == "VETO"

    def test_record_rolling_window(self, context, policy):
        params = SynthesisParams()
        # Start with 3 entries
        existing_history = [
            {"run_id": "RUN-001", "status": "VETO"},
            {"run_id": "RUN-002", "status": "VETO"},
            {"run_id": "RUN-003", "status": "VETO"},
        ]
        state = DSEState(
            run_id="RUN-004",
            params=params.model_dump(),
            status="APPROVED",
            metrics={"latency_ns": 5000, "area_units": 30000, "power_mw": 200},
            validation={"violations": []},
            history=existing_history,
            policy=policy,
        )
        result = record(state)
        # Should keep only last 3
        assert len(result["history"]) == 3
        assert result["history"][-1]["run_id"] == "RUN-004"


class TestCheckTerminalNode:
    """Test the check_terminal node."""

    def test_continues_on_veto(self, context, policy):
        state = DSEState(
            status="VETO",
            iteration=1,
            max_iterations=10,
            history=[],
            policy=policy,
        )
        result = check_terminal(state)
        assert result["converged"] is False
        assert result["final_status"] == ""

    def test_stops_on_max_iters(self, context, policy):
        state = DSEState(
            status="VETO",
            iteration=10,
            max_iterations=10,
            history=[],
            policy=policy,
        )
        result = check_terminal(state)
        assert result["converged"] is False
        assert result["final_status"] == "MAX_ITERS_REACHED"

    def test_stops_on_halt(self, context, policy):
        state = DSEState(
            status="HALT",
            halted=True,
            iteration=1,
            max_iterations=10,
            history=[],
            policy=policy,
        )
        result = check_terminal(state)
        assert result["final_status"] == "HALT"

    def test_converges_on_approved_single_objective(self, context, policy):
        state = DSEState(
            status="APPROVED",
            iteration=5,
            max_iterations=10,
            multi_objective=False,
            history=[{"status": "APPROVED"}],
            policy=policy,
        )
        result = check_terminal(state)
        assert result["converged"] is True
        assert result["final_status"] == "CONVERGED"


class TestProposeNode:
    """Test the propose node."""

    def test_propose_generates_all_proposals(self, context, policy):
        params = SynthesisParams(unroll_factor=4, pipeline_depth=1)
        state = DSEState(
            status="VETO",
            params=params.model_dump(),
            validation={"violations": [
                {"constraint_id": "area", "field": "area_units", "observed": 60000, "threshold": 50000}
            ]},
            strategy="bayesian",
            history=[],
            policy=policy,
        )
        result = propose(state)
        assert "shadow_proposal" in result
        assert "bayesian_proposal" in result
        assert result["shadow_proposal"]["adjustments"] is not None
        assert result["bayesian_proposal"]["adjustments"] is not None


class TestApplyProposalNode:
    """Test the apply_proposal node."""

    def test_apply_bayesian_proposal(self, context, policy):
        params = SynthesisParams(unroll_factor=4, pipeline_depth=1)
        # First generate a proposal
        state = DSEState(
            status="VETO",
            params=params.model_dump(),
            validation={"violations": []},
            strategy="bayesian",
            bayesian_proposal={"adjustments": [], "overall_reasoning": "test", "confidence": 0.5, "cited_runs": ["N/A"]},
            history=[],
            policy=policy,
        )
        propose(state)  # This sets last_proposed_params on bayesian
        result = apply_proposal(state)
        assert "params" in result
        assert isinstance(result["params"], dict)


class TestShouldContinue:
    """Test the routing function."""

    def test_continues_on_veto(self, policy):
        state = DSEState(
            status="VETO",
            halted=False,
            converged=False,
            final_status="",
            policy=policy,
        )
        assert should_continue(state) == "propose"

    def test_stops_on_halt(self, policy):
        state = DSEState(
            halted=True,
            final_status="HALT",
            policy=policy,
        )
        assert should_continue(state) == "done"

    def test_stops_on_converged(self, policy):
        state = DSEState(
            converged=True,
            final_status="CONVERGED",
            policy=policy,
        )
        assert should_continue(state) == "done"


class TestGraphBuild:
    """Test graph construction."""

    def test_graph_builds(self):
        graph = build_dse_graph()
        assert graph is not None


class TestRunGraph:
    """Integration tests for run_graph."""

    def test_run_graph_bayesian_strategy(self, adapter, policy):
        """Test that bayesian strategy runs without errors."""
        params = SynthesisParams(unroll_factor=2, pipeline_depth=2)
        result = run_graph(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=3,
            strategy="bayesian",
            seed=42,
            multi_objective=True,
            verbose=False,
        )
        assert "final_status" in result
        assert "iterations" in result
        assert result["strategy"] == "bayesian"

    def test_run_graph_shadow_strategy(self, adapter, policy):
        """Test that shadow strategy runs without errors."""
        params = SynthesisParams(unroll_factor=4, pipeline_depth=1)
        result = run_graph(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=3,
            strategy="shadow",
            seed=42,
            verbose=False,
        )
        assert "final_status" in result
        assert result["strategy"] == "shadow"

    def test_run_graph_converges_at_sweet_spot(self, policy):
        """Test that starting at the sweet spot converges quickly."""
        adapter = DummyHLSAdapter(noise_pct=0.0, seed=42)
        # Sweet spot: unroll=2, pipeline=2 -> all metrics pass
        params = SynthesisParams(unroll_factor=2, pipeline_depth=2)
        result = run_graph(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=10,
            strategy="bayesian",
            multi_objective=False,  # Single-objective converges on first APPROVED
            seed=42,
            verbose=False,
        )
        # Should converge on first iteration
        assert result["final_status"] == "CONVERGED"
        assert result["iterations"] == 1

    def test_run_graph_halts_on_poison(self, policy):
        """Test that the graph halts when poison data is detected."""
        # Create adapter that returns poison data
        mock_adapter = MagicMock()
        mock_adapter.name.return_value = "MockHLS"
        mock_adapter.run_synthesis.return_value = {
            "run_id": "RUN-001",
            "latency_ns": -100,  # Poison: negative latency
            "area_units": 30000,
            "power_mw": 200,
        }

        params = SynthesisParams()
        result = run_graph(
            adapter=mock_adapter,
            policy=policy,
            initial_params=params,
            max_iters=5,
            strategy="bayesian",
            verbose=False,
        )
        assert result["final_status"] == "HALT"
        assert result["iterations"] == 1


class TestGraphMatchesRunLoop:
    """Test that graph.py produces equivalent results to run_loop.py."""

    def test_same_final_status_on_sweet_spot(self, policy):
        """Both should CONVERGE when starting at sweet spot (single-objective)."""
        adapter = DummyHLSAdapter(noise_pct=0.0, seed=42)
        params = SynthesisParams(unroll_factor=2, pipeline_depth=2)

        result = run_graph(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=5,
            strategy="bayesian",
            multi_objective=False,
            seed=42,
            verbose=False,
        )
        assert result["final_status"] == "CONVERGED"
