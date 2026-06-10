"""Tests for code advisor (mocked LLM calls)."""

import pytest
from unittest.mock import patch, MagicMock

from aiv_dse.core.validator import ValidationResult
from aiv_dse.llm.models import (
    CodeAdvisoryReport,
    CodeProfile,
    CodeSuggestion,
    KnowledgeChunk,
    LoopInfo,
    ArrayInfo,
    PragmaInfo,
    SynthesisParams,
)


@pytest.fixture
def sample_profile():
    return CodeProfile(
        file_path="test.cpp",
        total_lines=80,
        loops=[LoopInfo(line_number=10, loop_type="for", iteration_count=256)],
        arrays=[ArrayInfo(line_number=5, name="buf", element_type="int", dimensions=[256])],
        pragmas=[PragmaInfo(line_number=9, directive="#pragma HLS PIPELINE II=4", category="pipeline")],
    )


@pytest.fixture
def sample_result():
    return ValidationResult(
        status="VETO",
        violations=[{
            "constraint_id": "area",
            "field": "area_units",
            "observed": 60000,
            "threshold": 50000,
            "severity": "WARNING",
            "action": "ESCALATE",
        }],
    )


def test_advisory_report_schema_valid():
    """CodeAdvisoryReport validates correctly."""
    report = CodeAdvisoryReport(
        suggestions=[CodeSuggestion(
            category="pragma_insert",
            target_line=42,
            suggested_change="Add #pragma HLS ARRAY_PARTITION",
            reasoning="Array lacks partitioning",
            expected_impact="Reduce latency ~20%",
            priority="high",
        )],
        overall_assessment="Design has optimization opportunities",
        confidence=0.8,
        cited_metrics=["latency_ns"],
    )
    assert len(report.suggestions) == 1
    assert report.confidence == 0.8


def test_advisory_report_rejects_bad_confidence():
    """Confidence > 1.0 is rejected."""
    with pytest.raises(Exception):
        CodeAdvisoryReport(
            suggestions=[],
            overall_assessment="test",
            confidence=1.5,
        )


def test_format_code_context(sample_profile, sample_result):
    """Verify context string includes source code and profile."""
    from aiv_dse.llm.code_advisor import _format_code_context

    policy = {"constraints": [{"id": "area", "max": 50000, "field": "area_units"}]}
    state = {"history": []}
    params = SynthesisParams()

    context = _format_code_context(
        "void test() { int x[256]; }",
        sample_profile,
        policy,
        state,
        sample_result,
        params,
        knowledge_chunks=[KnowledgeChunk(text="DPO reduces area", source="test.md", score=0.9)],
    )

    assert "Source Code" in context
    assert "Code Profile" in context
    assert "test.cpp" in context
    assert "area" in context.lower()
    assert "Domain Knowledge" in context
    assert "DPO reduces area" in context


def test_code_suggestion_categories():
    """All valid categories work."""
    for cat in ("pragma_insert", "pragma_modify", "coding_style", "restructure"):
        s = CodeSuggestion(
            category=cat,
            target_line=1,
            suggested_change="test",
            reasoning="test",
            expected_impact="test",
        )
        assert s.category == cat


def test_code_suggestion_default_priority():
    """Default priority is medium."""
    s = CodeSuggestion(
        category="pragma_insert",
        target_line=1,
        suggested_change="test",
        reasoning="test",
        expected_impact="test",
    )
    assert s.priority == "medium"
