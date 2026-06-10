"""Tests for C1: Reflexion / lessons_learned.

Verifies that:
1. Judge rejections append lessons to state.
2. The advisor's prompt context includes the lessons.
3. Lessons are capped at MAX_LESSONS.
4. Backward-compat: state files without lessons_learned still load.
"""

import os
import tempfile

import pytest

from aiv_dse.core.state import (
    MAX_LESSONS,
    append_lesson,
    load_state,
    save_state,
)
from aiv_dse.core.validator import ValidationResult
from aiv_dse.llm.models import SynthesisParams
from aiv_dse.llm.synth_advisor import _format_synth_context


class TestAppendLesson:
    def test_first_lesson_creates_list(self):
        state = {"history": []}
        out = append_lesson(state, 1, "unroll 4->16", "over-correction")
        assert out["lessons_learned"][0]["iteration"] == 1
        assert out["lessons_learned"][0]["proposed_change"] == "unroll 4->16"
        assert "over-correction" in out["lessons_learned"][0]["rejection_reason"]

    def test_capped_at_max_lessons(self):
        state = {"history": [], "lessons_learned": []}
        for i in range(MAX_LESSONS + 5):
            append_lesson(state, i, f"change-{i}", "reason")
        assert len(state["lessons_learned"]) == MAX_LESSONS
        # The OLDEST lessons should be the ones dropped
        first_kept_iter = state["lessons_learned"][0]["iteration"]
        assert first_kept_iter == 5  # dropped 0..4


class TestStateBackwardCompat:
    def test_legacy_state_file_without_lessons_field_loads(self):
        """State files written before Reflexion still load (lessons_learned defaulted)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"history": [{"run_id": "RUN-001", "status": "VETO"}]}')
            path = f.name
        try:
            state = load_state(path)
            assert state["history"][0]["run_id"] == "RUN-001"
            # Phase C1: lessons_learned auto-added
            assert state["lessons_learned"] == []
        finally:
            os.unlink(path)


class TestAdvisorReadsLessons:
    """The whole point of Reflexion -- the advisor must SEE the lessons."""

    def test_lessons_appear_in_synth_context(self):
        state = {
            "history": [],
            "lessons_learned": [
                {
                    "iteration": 3,
                    "proposed_change": "unroll 4->16",
                    "rejection_reason": "over-correction; physics limit exceeded",
                },
            ],
        }
        ctx = _format_synth_context(
            policy={"constraints": []},
            state=state,
            result=ValidationResult(status="VETO", violations=[]),
            current_params=SynthesisParams(),
        )
        assert "Lessons from past rejections" in ctx
        assert "iter 3" in ctx
        assert "unroll 4->16" in ctx
        assert "over-correction" in ctx

    def test_empty_lessons_section_omitted(self):
        """When there are no lessons, the section should not pollute the prompt."""
        state = {"history": [], "lessons_learned": []}
        ctx = _format_synth_context(
            policy={"constraints": []},
            state=state,
            result=ValidationResult(status="VETO", violations=[]),
            current_params=SynthesisParams(),
        )
        assert "Lessons from past rejections" not in ctx
