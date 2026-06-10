import os
import tempfile

from aiv_dse.core.validator import ValidationResult
from aiv_dse.llm.models import ConstraintAdjustment, LLMProposal
from aiv_dse.workflow.edr_writer import write_edr


def _sample_history():
    return [
        {
            "run_id": "RUN-001",
            "timestamp": "2025-01-01T00:00:00Z",
            "status": "APPROVED",
            "metrics": {"latency_ns": 8500, "area_units": 42000, "power_mw": 350},
            "violations": [],
        },
        {
            "run_id": "RUN-002",
            "timestamp": "2025-01-01T01:00:00Z",
            "status": "VETO",
            "metrics": {"latency_ns": 15000, "area_units": 62000, "power_mw": 620},
            "violations": [
                {"constraint_id": "latency", "field": "latency_ns",
                 "observed": 15000, "threshold": 10000, "severity": "CRITICAL",
                 "action": "VETO"},
            ],
        },
    ]


def _sample_proposal():
    return LLMProposal(
        adjustments=[
            ConstraintAdjustment(
                constraint_id="latency",
                current_max=10000,
                proposed_max=12000,
                reasoning="RUN-002 latency=15000; 12000 covers 80th percentile",
            ),
        ],
        overall_reasoning="Based on RUN-001 and RUN-002 trends",
        confidence=0.82,
        cited_runs=["RUN-001", "RUN-002"],
    )


def _sample_validation():
    return ValidationResult(
        status="VETO",
        violations=[
            {"constraint_id": "latency", "field": "latency_ns",
             "observed": 15000, "threshold": 10000, "severity": "CRITICAL",
             "action": "VETO"},
        ],
        reasons=["latency: latency_ns=15000 exceeds max 10000 (50% over, severity=CRITICAL)"],
        suggested_relaxations=["increase latency_ns max from 10000 to 16500"],
    )


def test_edr_contains_run_history():
    """EDR markdown should include the run history table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "edr_test.md")
        write_edr(
            run_history=_sample_history(),
            proposal=None,
            validation=_sample_validation(),
            output_path=path,
        )
        content = open(path, encoding="utf-8").read()
        assert "## Run History" in content
        assert "RUN-001" in content
        assert "RUN-002" in content
        assert "APPROVED" in content
        assert "VETO" in content


def test_edr_contains_violations():
    """EDR should list violations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "edr_test.md")
        write_edr(
            run_history=_sample_history(),
            proposal=None,
            validation=_sample_validation(),
            output_path=path,
        )
        content = open(path, encoding="utf-8").read()
        assert "## Violations" in content
        assert "latency" in content


def test_edr_contains_proposal():
    """EDR should include LLM proposal when present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "edr_test.md")
        write_edr(
            run_history=_sample_history(),
            proposal=_sample_proposal(),
            validation=_sample_validation(),
            output_path=path,
            human_decision="Accepted",
        )
        content = open(path, encoding="utf-8").read()
        assert "## LLM Proposal" in content
        assert "0.82" in content
        assert "latency" in content
        assert "12000" in content
        assert "## Decision" in content
        assert "Accepted" in content


def test_edr_no_proposal_section_when_none():
    """EDR should omit proposal section if no LLM was invoked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "edr_test.md")
        write_edr(
            run_history=_sample_history(),
            proposal=None,
            validation=_sample_validation(),
            output_path=path,
        )
        content = open(path, encoding="utf-8").read()
        assert "## LLM Proposal" not in content
