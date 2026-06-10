"""Tests for Phase 4 extended synthesis parameters."""

import pytest
from pydantic import ValidationError

from aiv_dse.llm.models import SynthesisParams


def test_default_params_unchanged():
    """Existing code should get same defaults."""
    p = SynthesisParams()
    assert p.unroll_factor == 4
    assert p.pipeline_depth == 1
    assert p.clock_period_ns == 10.0
    assert p.array_partition_factor == 1


def test_phase4_defaults():
    """Phase 4 params should have sensible defaults."""
    p = SynthesisParams()
    assert p.clock_slack_ns == 0.0
    assert p.dpo_mode == "none"
    assert p.flatten is False
    assert p.inline is False
    assert p.loop_merge is False
    assert p.bitwidth_reduce is False
    assert p.resource_sharing is False


def test_phase4_custom_values():
    """Can set Phase 4 params explicitly."""
    p = SynthesisParams(
        clock_slack_ns=2.5,
        dpo_mode="DPO_AUTO_ALL",
        flatten=True,
        resource_sharing=True,
    )
    assert p.clock_slack_ns == 2.5
    assert p.dpo_mode == "DPO_AUTO_ALL"
    assert p.flatten is True
    assert p.resource_sharing is True


def test_dpo_mode_validation():
    """DPO mode must be one of valid options."""
    with pytest.raises(ValidationError):
        SynthesisParams(dpo_mode="INVALID_MODE")

    for mode in ("none", "DPO_AUTO_ALL", "DPO_AUTO_OPT", "DPO_AUTO_EXPR"):
        p = SynthesisParams(dpo_mode=mode)
        assert p.dpo_mode == mode


def test_clock_slack_range():
    """Clock slack should be constrained to reasonable range."""
    with pytest.raises(ValidationError):
        SynthesisParams(clock_slack_ns=-10.0)

    with pytest.raises(ValidationError):
        SynthesisParams(clock_slack_ns=100.0)

    p = SynthesisParams(clock_slack_ns=-4.9)
    assert p.clock_slack_ns == -4.9


def test_backward_compatibility():
    """Old code using only Phase 3 params should work unchanged."""
    p = SynthesisParams(unroll_factor=8, pipeline_depth=2, clock_period_ns=5.0)
    assert p.unroll_factor == 8
    assert p.dpo_mode == "none"
    assert p.flatten is False


def test_model_dump_includes_all_fields():
    """model_dump() should include Phase 4 fields."""
    p = SynthesisParams()
    d = p.model_dump()
    assert "clock_slack_ns" in d
    assert "dpo_mode" in d
    assert "flatten" in d
    assert "resource_sharing" in d
