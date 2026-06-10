"""Tests for the closed-loop runner, all three strategies, and supporting modules.

All LLM calls are mocked. No API keys needed.
Bayesian tests use real Optuna (deterministic with seed).
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from aiv_dse.adapters.dummy_hls import DummyHLSAdapter
from aiv_dse.core.bayesian_advisor import BayesianAdvisor
from aiv_dse.core.convergence import check_convergence, compute_weighted_score
from aiv_dse.core.history import (
    append_full_history,
    get_tried_combos,
    load_full_history,
    search_history,
)
from aiv_dse.core.shadow_heuristic import shadow_propose
from aiv_dse.core.validator import ValidationResult, load_policy
from aiv_dse.llm.models import (
    JudgeVerdict,
    SpecConstraint,
    SpecPlan,
    SynthParamAdjustment,
    SynthParamProposal,
    SynthesisParams,
)
from aiv_dse.run_loop import _apply_proposal, explain_loop, run_loop


# ---------------------------------------------------------------------------
# _apply_proposal tests
# ---------------------------------------------------------------------------
class TestApplyProposal:
    def test_single_change(self):
        params = SynthesisParams(unroll_factor=4)
        proposal = SynthParamProposal(
            adjustments=[SynthParamAdjustment(
                param_name="unroll_factor",
                current_value=4.0,
                proposed_value=2.0,
                reasoning="test",
            )],
            overall_reasoning="test",
            confidence=0.9,
            cited_runs=["RUN-001"],
        )
        new_params = _apply_proposal(params, proposal)
        assert new_params.unroll_factor == 2

    def test_multiple_changes(self):
        params = SynthesisParams(unroll_factor=4, pipeline_depth=1)
        proposal = SynthParamProposal(
            adjustments=[
                SynthParamAdjustment(
                    param_name="unroll_factor",
                    current_value=4.0, proposed_value=2.0, reasoning="t",
                ),
                SynthParamAdjustment(
                    param_name="pipeline_depth",
                    current_value=1.0, proposed_value=3.0, reasoning="t",
                ),
            ],
            overall_reasoning="test",
            confidence=0.9,
            cited_runs=["RUN-001"],
        )
        new_params = _apply_proposal(params, proposal)
        assert new_params.unroll_factor == 2
        assert new_params.pipeline_depth == 3

    def test_unknown_param_ignored(self):
        params = SynthesisParams(unroll_factor=4)
        proposal = SynthParamProposal(
            adjustments=[SynthParamAdjustment(
                param_name="nonexistent",
                current_value=0.0, proposed_value=1.0, reasoning="t",
            )],
            overall_reasoning="test",
            confidence=0.9,
            cited_runs=["RUN-001"],
        )
        new_params = _apply_proposal(params, proposal)
        assert new_params.unroll_factor == 4  # unchanged


# ---------------------------------------------------------------------------
# Shadow heuristic tests
# ---------------------------------------------------------------------------
class TestShadowHeuristic:
    def test_area_violation_decreases_unroll(self):
        result = ValidationResult(
            status="VETO",
            violations=[{
                "constraint_id": "area",
                "field": "area_units",
                "observed": 70000,
                "threshold": 50000,
                "severity": "WARNING",
                "action": "ESCALATE",
            }],
        )
        params = SynthesisParams(unroll_factor=4)
        policy = {"constraints": [{"id": "area", "field": "area_units", "max": 50000}]}
        proposal = shadow_propose(result, params, policy)
        assert len(proposal.adjustments) == 1
        assert proposal.adjustments[0].proposed_value == 3.0  # 4 - 1

    def test_latency_violation_increases_unroll(self):
        result = ValidationResult(
            status="VETO",
            violations=[{
                "constraint_id": "latency",
                "field": "latency_ns",
                "observed": 15000,
                "threshold": 10000,
                "severity": "CRITICAL",
                "action": "VETO",
            }],
        )
        params = SynthesisParams(unroll_factor=4)
        policy = {"constraints": [{"id": "latency", "field": "latency_ns", "max": 10000}]}
        proposal = shadow_propose(result, params, policy)
        assert len(proposal.adjustments) == 1
        assert proposal.adjustments[0].proposed_value == 5.0  # 4 + 1

    def test_no_violations_no_changes(self):
        result = ValidationResult(status="APPROVED")
        params = SynthesisParams()
        proposal = shadow_propose(result, params, {})
        assert len(proposal.adjustments) == 0
        assert proposal.confidence == 1.0


# ---------------------------------------------------------------------------
# Bayesian advisor tests
# ---------------------------------------------------------------------------
class TestBayesianAdvisor:
    def test_observe_and_propose(self):
        bo = BayesianAdvisor(sampler="tpe", seed=42)
        policy = {"constraints": [
            {"id": "latency", "field": "latency_ns", "max": 10000},
            {"id": "area", "field": "area_units", "max": 50000},
            {"id": "power", "field": "power_mw", "max": 500},
        ]}
        params = SynthesisParams(unroll_factor=4)
        report = {"latency_ns": 12000, "area_units": 60000, "power_mw": 400}

        # Observe 3 points
        for i in range(3):
            bo.observe(params, report, policy)

        proposal = bo.propose(params)
        assert isinstance(proposal, SynthParamProposal)
        assert proposal.confidence > 0

    def test_cold_start_returns_valid(self):
        bo = BayesianAdvisor(sampler="tpe", seed=42)
        params = SynthesisParams()
        proposal = bo.propose(params)
        assert isinstance(proposal, SynthParamProposal)

    def test_violation_score_zero_for_feasible(self):
        report = {"latency_ns": 8000, "area_units": 40000, "power_mw": 300}
        policy = {"constraints": [
            {"id": "latency", "field": "latency_ns", "max": 10000},
            {"id": "area", "field": "area_units", "max": 50000},
            {"id": "power", "field": "power_mw", "max": 500},
        ]}
        score = BayesianAdvisor._compute_violation_score(report, policy)
        assert score == 0.0

    def test_violation_score_positive_for_infeasible(self):
        report = {"latency_ns": 15000, "area_units": 60000, "power_mw": 600}
        policy = {"constraints": [
            {"id": "latency", "field": "latency_ns", "max": 10000},
            {"id": "area", "field": "area_units", "max": 50000},
            {"id": "power", "field": "power_mw", "max": 500},
        ]}
        score = BayesianAdvisor._compute_violation_score(report, policy)
        assert score > 0


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------
class TestConvergence:
    def test_weighted_score(self):
        metrics = {"latency_ns": 8000, "area_units": 40000, "power_mw": 300}
        policy = {"constraints": [
            {"id": "latency", "field": "latency_ns", "max": 10000},
            {"id": "area", "field": "area_units", "max": 50000},
            {"id": "power", "field": "power_mw", "max": 500},
        ]}
        score = compute_weighted_score(metrics, policy)
        assert 0 < score < 1.0  # All below thresholds

    def test_convergence_detected(self):
        """3 APPROVED runs with stable scores -> converged."""
        state = {"history": [
            {"status": "APPROVED", "metrics": {"latency_ns": 8000, "area_units": 40000, "power_mw": 300}},
            {"status": "APPROVED", "metrics": {"latency_ns": 8010, "area_units": 40020, "power_mw": 301}},
            {"status": "APPROVED", "metrics": {"latency_ns": 8005, "area_units": 40010, "power_mw": 300}},
        ]}
        policy = {"constraints": [
            {"id": "latency", "field": "latency_ns", "max": 10000},
            {"id": "area", "field": "area_units", "max": 50000},
            {"id": "power", "field": "power_mw", "max": 500},
        ]}
        msg = check_convergence(state, policy)
        assert msg is not None
        assert "Converged" in msg

    def test_not_converged_if_veto(self):
        state = {"history": [
            {"status": "APPROVED", "metrics": {"latency_ns": 8000, "area_units": 40000, "power_mw": 300}},
            {"status": "VETO", "metrics": {"latency_ns": 15000, "area_units": 60000, "power_mw": 600}},
            {"status": "APPROVED", "metrics": {"latency_ns": 8000, "area_units": 40000, "power_mw": 300}},
        ]}
        policy = {"constraints": [
            {"id": "latency", "field": "latency_ns", "max": 10000},
        ]}
        msg = check_convergence(state, policy)
        assert msg is None


# ---------------------------------------------------------------------------
# History tests
# ---------------------------------------------------------------------------
class TestHistory:
    def test_append_and_load(self, tmp_path):
        path = str(tmp_path / "history.json")
        entry = {"run_id": "RUN-001", "status": "APPROVED", "violations": []}
        append_full_history(entry, path)
        history = load_full_history(path)
        assert len(history) == 1
        assert history[0]["run_id"] == "RUN-001"

    def test_search_by_constraint(self):
        history = [
            {"run_id": "R1", "violations": [{"constraint_id": "latency"}]},
            {"run_id": "R2", "violations": [{"constraint_id": "area"}]},
            {"run_id": "R3", "violations": [{"constraint_id": "latency"}]},
        ]
        results = search_history(history, "latency")
        assert len(results) == 2
        assert results[0]["run_id"] == "R1"

    def test_get_tried_combos(self):
        history = [
            {"run_id": "R1", "status": "VETO", "metrics": {"latency_ns": 15000},
             "synth_params": {"unroll_factor": 4}},
            {"run_id": "R2", "status": "APPROVED", "metrics": {"latency_ns": 8000},
             "synth_params": {"unroll_factor": 2}},
        ]
        combos = get_tried_combos(history)
        assert len(combos) == 2
        assert combos[0]["params"]["unroll_factor"] == 4


# ---------------------------------------------------------------------------
# Loop integration tests (mocked LLM)
# ---------------------------------------------------------------------------
class TestLoop:
    def test_immediate_approval(self):
        """If starting params are already feasible, converge once Pareto front stabilises."""
        adapter = DummyHLSAdapter(noise_pct=0.0)
        policy = load_policy("policy/default_policy.yaml")
        # Sweet spot: unroll=2, pipeline=2 -> APPROVED
        params = SynthesisParams(unroll_factor=2, pipeline_depth=2)

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=5,
            strategy="shadow",  # No LLM needed
        )
        assert result["final_status"] == "CONVERGED"
        # Multi-objective (default): needs 3 stable front updates to converge
        assert result["iterations"] == 3

    def test_shadow_converges_or_hits_max(self):
        """Shadow strategy should run without errors."""
        adapter = DummyHLSAdapter(noise_pct=0.0, seed=42)
        policy = load_policy("policy/default_policy.yaml")
        params = SynthesisParams(unroll_factor=4)

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=5,
            strategy="shadow",
        )
        assert result["final_status"] in ("CONVERGED", "MAX_ITERS_REACHED")

    def test_bayesian_runs_without_error(self):
        """Bayesian strategy should run without errors (no LLM needed)."""
        adapter = DummyHLSAdapter(noise_pct=0.0, seed=42)
        policy = load_policy("policy/default_policy.yaml")
        params = SynthesisParams(unroll_factor=4)

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=5,
            strategy="bayesian",
            bayesian_seed=42,
        )
        assert result["final_status"] in ("CONVERGED", "MAX_ITERS_REACHED")
        assert len(result["comparison_log"]) >= 1

    def test_max_iters_enforced(self):
        """Loop stops at max_iters even if not converged."""
        adapter = DummyHLSAdapter(noise_pct=0.0)
        policy = load_policy("policy/default_policy.yaml")
        # Start with params that violate constraints
        params = SynthesisParams(unroll_factor=16)

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=3,
            strategy="shadow",
        )
        assert result["iterations"] <= 3

    def test_post_hook_success(self, tmp_path):
        """Post-hook with exit 0 -> converged."""
        adapter = DummyHLSAdapter(noise_pct=0.0)
        policy = load_policy("policy/default_policy.yaml")
        params = SynthesisParams(unroll_factor=2, pipeline_depth=2)

        # Use a cross-platform command
        hook = "python -c \"print('ok')\""

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=3,
            strategy="shadow",
            post_hook=hook,
        )
        assert result["final_status"] == "CONVERGED"

    def test_comparison_log_populated(self):
        """Comparison log should have entries for non-approved iterations."""
        adapter = DummyHLSAdapter(noise_pct=0.0, seed=42)
        policy = load_policy("policy/default_policy.yaml")
        params = SynthesisParams(unroll_factor=8)

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=3,
            strategy="shadow",
            bayesian_seed=42,
        )
        # If there were non-approved iterations, log should exist
        if result["final_status"] != "CONVERGED" or result["iterations"] > 1:
            assert len(result["comparison_log"]) >= 1
            entry = result["comparison_log"][0]
            assert "shadow" in entry
            assert "bayesian" in entry


    def test_loop_with_phase4_params(self):
        """Loop should handle Phase 4 params without error."""
        adapter = DummyHLSAdapter(noise_pct=0.0, seed=42)
        policy = load_policy("policy/default_policy.yaml")

        params = SynthesisParams(
            unroll_factor=2,
            pipeline_depth=2,
            dpo_mode="DPO_AUTO_OPT",
            flatten=True,
            resource_sharing=True,
        )

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=3,
            strategy="shadow",
        )
        assert result["final_status"] in ("CONVERGED", "MAX_ITERS_REACHED")

    def test_loop_with_source_and_knowledge(self):
        """Loop runs with --source and --knowledge-dir (shadow, no LLM)."""
        adapter = DummyHLSAdapter(noise_pct=0.0, seed=42)
        policy = load_policy("policy/default_policy.yaml")
        params = SynthesisParams(unroll_factor=4)

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=3,
            strategy="shadow",
            source_path="samples/fft256_design.cpp",
            knowledge_dir="knowledge",
        )
        assert result["final_status"] in ("CONVERGED", "MAX_ITERS_REACHED")
        # Code advisory is None because shadow strategy doesn't invoke LLM
        assert result["code_advisory"] is None

    def test_loop_without_source_unchanged(self):
        """Existing loop behavior is unchanged without --source."""
        adapter = DummyHLSAdapter(noise_pct=0.0)
        policy = load_policy("policy/default_policy.yaml")
        params = SynthesisParams(unroll_factor=2, pipeline_depth=2)

        result = run_loop(
            adapter=adapter,
            policy=policy,
            initial_params=params,
            max_iters=3,
            strategy="shadow",
        )
        assert result["final_status"] == "CONVERGED"
        assert result["code_advisory"] is None


# ---------------------------------------------------------------------------
# New Pydantic model tests
# ---------------------------------------------------------------------------
class TestNewModels:
    def test_synthesis_params_defaults(self):
        p = SynthesisParams()
        assert p.unroll_factor == 4
        assert p.pipeline_depth == 1
        assert p.clock_period_ns == 10.0
        assert p.array_partition_factor == 1

    def test_synth_param_proposal_requires_cited_runs(self):
        with pytest.raises(Exception):
            SynthParamProposal(
                adjustments=[],
                overall_reasoning="test",
                confidence=0.5,
                cited_runs=[],
            )

    def test_judge_verdict_defaults(self):
        v = JudgeVerdict(agree=True, confidence=0.9)
        assert v.disagreements == []
        assert v.alternative_suggestion == ""

    def test_spec_plan_valid(self):
        plan = SpecPlan(
            constraints=[SpecConstraint(
                id="latency", field="latency_ns", max=10000,
                severity="CRITICAL", on_violation="VETO",
                reasoning="Spec says max 10us",
            )],
            initial_params=SynthesisParams(),
            reasoning="Based on spec analysis",
        )
        assert len(plan.constraints) == 1
        assert plan.warnings == []


# ---------------------------------------------------------------------------
# Spec planner tests
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Explain flag tests (Phase 12)
# ---------------------------------------------------------------------------
class TestExplainFlag:
    def test_explain_prints_steps(self, capsys):
        explain_loop()
        captured = capsys.readouterr().out
        assert "Step  1:" in captured
        assert "Step 13:" in captured

    def test_explain_has_all_steps(self, capsys):
        explain_loop()
        captured = capsys.readouterr().out
        for i in range(1, 14):
            assert f"Step {i:>2}:" in captured


# ---------------------------------------------------------------------------
# Spec planner tests
# ---------------------------------------------------------------------------
class TestSpecPlanner:
    def test_load_spec_txt(self, tmp_path):
        spec_file = tmp_path / "spec.txt"
        spec_file.write_text("Design: Test IP\nClock: 100 MHz")
        from aiv_dse.llm.spec_planner import load_spec
        text = load_spec(str(spec_file))
        assert "Test IP" in text
        assert "100 MHz" in text
