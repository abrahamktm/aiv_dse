import pytest
from pydantic import ValidationError

from aiv_dse.llm.models import ConstraintAdjustment, LLMProposal


def _valid_proposal_data():
    return {
        "adjustments": [
            {
                "constraint_id": "latency",
                "current_max": 10000,
                "proposed_max": 12000,
                "reasoning": "RUN-002 had latency=15000, relaxing to 12000 covers 80th percentile",
            }
        ],
        "overall_reasoning": "Based on RUN-001 and RUN-002 trends",
        "confidence": 0.85,
        "cited_runs": ["RUN-001", "RUN-002"],
    }


def test_proposal_schema_valid():
    """A well-formed proposal should parse without error."""
    data = _valid_proposal_data()
    proposal = LLMProposal(**data)
    assert proposal.confidence == 0.85
    assert len(proposal.adjustments) == 1
    assert proposal.adjustments[0].constraint_id == "latency"


def test_proposal_rejects_extra_fields():
    """extra='forbid' should reject unexpected fields."""
    data = _valid_proposal_data()
    data["unexpected_field"] = "should fail"
    with pytest.raises(ValidationError):
        LLMProposal(**data)


def test_adjustment_rejects_extra_fields():
    """ConstraintAdjustment also rejects extra fields."""
    with pytest.raises(ValidationError):
        ConstraintAdjustment(
            constraint_id="latency",
            current_max=10000,
            proposed_max=12000,
            reasoning="test",
            extra_field="bad",
        )


def test_confidence_must_be_in_range():
    """Confidence outside 0.0-1.0 should fail."""
    data = _valid_proposal_data()

    data["confidence"] = 1.5
    with pytest.raises(ValidationError):
        LLMProposal(**data)

    data["confidence"] = -0.1
    with pytest.raises(ValidationError):
        LLMProposal(**data)


def test_cited_runs_required_nonempty():
    """Empty cited_runs should fail validation."""
    data = _valid_proposal_data()
    data["cited_runs"] = []
    with pytest.raises(ValidationError):
        LLMProposal(**data)
