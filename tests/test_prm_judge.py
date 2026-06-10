"""Tests for C2: PRM-style judge (per-adjustment scoring).

Verifies the model schema and the partial-acceptance application logic.
The actual LLM call is mocked elsewhere (it follows the same pattern as
other LLM tests in this suite -- no real API calls).
"""

import pytest
from pydantic import ValidationError

from aiv_dse.llm.judge import apply_prm_verdict
from aiv_dse.llm.models import (
    AdjustmentScore,
    PRMJudgeVerdict,
    SynthParamAdjustment,
    SynthParamProposal,
)


def _make_proposal(*param_names: str) -> SynthParamProposal:
    return SynthParamProposal(
        adjustments=[
            SynthParamAdjustment(
                param_name=p,
                current_value=4,
                proposed_value=8,
                reasoning=f"adjust {p}",
            )
            for p in param_names
        ],
        overall_reasoning="multi-step proposal",
        confidence=0.9,
        cited_runs=["RUN-001"],
    )


def _make_verdict(scores: dict[str, bool], conf: float = 0.8) -> PRMJudgeVerdict:
    return PRMJudgeVerdict(
        scores=[
            AdjustmentScore(
                param_name=p,
                accept=ok,
                reasoning="per-step assessment",
                citation_verified=True,
            )
            for p, ok in scores.items()
        ],
        overall_reasoning="per-step rollup",
        overall_confidence=conf,
    )


class TestPRMVerdictModel:
    def test_schema_validation_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            PRMJudgeVerdict(
                scores=[],
                overall_reasoning="x",
                overall_confidence=0.5,
                unexpected_field="bad",
            )

    def test_all_accepted(self):
        v = _make_verdict({"unroll_factor": True, "pipeline_depth": True})
        assert v.all_accepted() is True
        assert v.any_accepted() is True

    def test_partial_accepted(self):
        v = _make_verdict({"unroll_factor": True, "pipeline_depth": False})
        assert v.all_accepted() is False
        assert v.any_accepted() is True
        assert v.accepted_param_names() == ["unroll_factor"]

    def test_none_accepted(self):
        v = _make_verdict({"unroll_factor": False})
        assert v.any_accepted() is False
        assert v.accepted_param_names() == []


class TestApplyPRMVerdict:
    """The core value: the loop applies only the verified-good adjustments."""

    def test_keeps_only_accepted_adjustments(self):
        proposal = _make_proposal("unroll_factor", "resource_sharing")
        verdict = _make_verdict({
            "unroll_factor": True,
            "resource_sharing": False,  # rejected -- conflicts with unroll
        })
        filtered = apply_prm_verdict(proposal, verdict)
        names = [a.param_name for a in filtered.adjustments]
        assert names == ["unroll_factor"]
        assert "resource_sharing" not in names

    def test_filtered_confidence_is_min_of_proposal_and_verdict(self):
        proposal = _make_proposal("unroll_factor")
        verdict = _make_verdict({"unroll_factor": True}, conf=0.6)
        filtered = apply_prm_verdict(proposal, verdict)
        # Original proposal was 0.9 confidence; PRM said 0.6 -> use the lower
        assert filtered.confidence == 0.6

    def test_cited_runs_preserved(self):
        proposal = _make_proposal("unroll_factor")
        verdict = _make_verdict({"unroll_factor": True})
        filtered = apply_prm_verdict(proposal, verdict)
        assert filtered.cited_runs == ["RUN-001"]

    def test_reasoning_records_filter_summary(self):
        proposal = _make_proposal("a", "b", "c")
        verdict = _make_verdict({"a": True, "b": False, "c": True})
        filtered = apply_prm_verdict(proposal, verdict)
        assert "kept 2/3 adjustments" in filtered.overall_reasoning

    def test_empty_when_nothing_accepted(self):
        proposal = _make_proposal("unroll_factor")
        verdict = _make_verdict({"unroll_factor": False})
        filtered = apply_prm_verdict(proposal, verdict)
        assert filtered.adjustments == []
