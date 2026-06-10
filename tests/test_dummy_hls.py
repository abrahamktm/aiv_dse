"""Tests for the DummyHLS adapter."""

import pytest

from aiv_dse.adapters.dummy_hls import DummyHLSAdapter
from aiv_dse.llm.models import SynthesisParams


@pytest.fixture
def adapter():
    """Deterministic adapter (zero noise, fixed seed)."""
    return DummyHLSAdapter(noise_pct=0.0, seed=42)


@pytest.fixture
def default_params():
    return SynthesisParams()


def test_returns_report_dict(adapter, default_params):
    report = adapter.run_synthesis(default_params, "RUN-001")
    assert "run_id" in report
    assert "latency_ns" in report
    assert "area_units" in report
    assert "power_mw" in report
    assert report["run_id"] == "RUN-001"


def test_deterministic_with_zero_noise(default_params):
    a1 = DummyHLSAdapter(noise_pct=0.0)
    a2 = DummyHLSAdapter(noise_pct=0.0)
    r1 = a1.run_synthesis(default_params, "R1")
    r2 = a2.run_synthesis(default_params, "R2")
    assert r1["latency_ns"] == r2["latency_ns"]
    assert r1["area_units"] == r2["area_units"]
    assert r1["power_mw"] == r2["power_mw"]


def test_reproducible_with_seed(default_params):
    a1 = DummyHLSAdapter(noise_pct=5.0, seed=123)
    a2 = DummyHLSAdapter(noise_pct=5.0, seed=123)
    r1 = a1.run_synthesis(default_params, "R1")
    r2 = a2.run_synthesis(default_params, "R2")
    assert r1["latency_ns"] == r2["latency_ns"]
    assert r1["area_units"] == r2["area_units"]


def test_higher_unroll_lower_latency(adapter):
    p_low = SynthesisParams(unroll_factor=2)
    p_high = SynthesisParams(unroll_factor=8)
    r_low = adapter.run_synthesis(p_low, "R1")
    r_high = adapter.run_synthesis(p_high, "R2")
    assert r_high["latency_ns"] < r_low["latency_ns"]


def test_higher_unroll_higher_area(adapter):
    p_low = SynthesisParams(unroll_factor=2)
    p_high = SynthesisParams(unroll_factor=8)
    r_low = adapter.run_synthesis(p_low, "R1")
    r_high = adapter.run_synthesis(p_high, "R2")
    assert r_high["area_units"] > r_low["area_units"]


def test_higher_unroll_higher_power(adapter):
    p_low = SynthesisParams(unroll_factor=2)
    p_high = SynthesisParams(unroll_factor=8)
    r_low = adapter.run_synthesis(p_low, "R1")
    r_high = adapter.run_synthesis(p_high, "R2")
    assert r_high["power_mw"] > r_low["power_mw"]


def test_higher_pipeline_lower_latency(adapter):
    p1 = SynthesisParams(pipeline_depth=1)
    p2 = SynthesisParams(pipeline_depth=4)
    r1 = adapter.run_synthesis(p1, "R1")
    r2 = adapter.run_synthesis(p2, "R2")
    assert r2["latency_ns"] < r1["latency_ns"]


def test_sweet_spot_meets_default_policy(adapter):
    """unroll=2, pipeline=2 should meet all default constraints."""
    p = SynthesisParams(unroll_factor=2, pipeline_depth=2)
    r = adapter.run_synthesis(p, "R1")
    assert r["latency_ns"] <= 10000
    assert r["area_units"] <= 50000
    assert r["power_mw"] <= 500


def test_name(adapter):
    assert adapter.name() == "DummyHLS"


def test_noise_varies_output(default_params):
    a = DummyHLSAdapter(noise_pct=10.0, seed=None)
    r1 = a.run_synthesis(default_params, "R1")
    r2 = a.run_synthesis(default_params, "R2")
    # With 10% noise and no fixed seed, at least one metric should differ
    # (probabilistic, but extremely unlikely to be identical)
    differs = (
        r1["latency_ns"] != r2["latency_ns"]
        or r1["area_units"] != r2["area_units"]
        or r1["power_mw"] != r2["power_mw"]
    )
    assert differs


# --- Phase 4 tests ---

def test_dpo_reduces_area(adapter):
    """DPO_AUTO_ALL should reduce area compared to none."""
    p_none = SynthesisParams(dpo_mode="none")
    p_dpo = SynthesisParams(dpo_mode="DPO_AUTO_ALL")
    r_none = adapter.run_synthesis(p_none, "R1")
    r_dpo = adapter.run_synthesis(p_dpo, "R2")
    assert r_dpo["area_units"] < r_none["area_units"]


def test_resource_sharing_reduces_area(adapter):
    p_no = SynthesisParams(resource_sharing=False)
    p_yes = SynthesisParams(resource_sharing=True)
    r_no = adapter.run_synthesis(p_no, "R1")
    r_yes = adapter.run_synthesis(p_yes, "R2")
    assert r_yes["area_units"] < r_no["area_units"]


def test_flatten_reduces_latency(adapter):
    p_no = SynthesisParams(flatten=False)
    p_yes = SynthesisParams(flatten=True)
    r_no = adapter.run_synthesis(p_no, "R1")
    r_yes = adapter.run_synthesis(p_yes, "R2")
    assert r_yes["latency_ns"] < r_no["latency_ns"]
    assert r_yes["area_units"] > r_no["area_units"]


def test_positive_slack_improves_latency(adapter):
    p_tight = SynthesisParams(clock_slack_ns=0.0)
    p_loose = SynthesisParams(clock_slack_ns=2.0)
    r_tight = adapter.run_synthesis(p_tight, "R1")
    r_loose = adapter.run_synthesis(p_loose, "R2")
    assert r_loose["latency_ns"] < r_tight["latency_ns"]
