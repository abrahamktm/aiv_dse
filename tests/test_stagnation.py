from aiv_dse.core.stagnation import compute_deltas_vs_baseline, detect_stagnation
from aiv_dse.core.state import append_result
from aiv_dse.core.validator import ValidationResult


def _make_report(run_id, latency, area, power):
    return {"run_id": run_id, "latency_ns": latency, "area_units": area, "power_mw": power}


def _make_result(status="APPROVED"):
    return ValidationResult(status=status)


def test_no_stagnation_with_large_deltas():
    """Two runs with >5% deltas should not trigger stagnation."""
    state = {"history": []}
    state = append_result(state, _make_result(), _make_report("R-1", 1000, 40000, 400))
    state = append_result(state, _make_result(), _make_report("R-2", 800, 35000, 350))
    assert detect_stagnation(state, threshold_pct=5.0, window=2) is None


def test_no_stagnation_insufficient_history():
    """Fewer runs than window -> no stagnation."""
    state = {"history": []}
    state = append_result(state, _make_result(), _make_report("R-1", 1000, 40000, 400))
    assert detect_stagnation(state, window=3) is None


def test_stagnation_detected():
    """Three runs with <1% deltas should trigger stagnation."""
    state = {"history": []}
    state = append_result(state, _make_result(), _make_report("R-1", 1000, 40000, 400))
    state = append_result(state, _make_result(), _make_report("R-2", 1005, 40100, 401))
    state = append_result(state, _make_result(), _make_report("R-3", 1008, 40150, 402))
    msg = detect_stagnation(state, threshold_pct=2.0, window=3)
    assert msg is not None
    assert "Stagnation" in msg


def test_baseline_deltas():
    """Compute deltas vs a specific baseline run_id."""
    state = {"history": []}
    state = append_result(state, _make_result(), _make_report("R-1", 1000, 40000, 400))
    state = append_result(state, _make_result(), _make_report("R-2", 900, 44000, 380))
    state = append_result(state, _make_result(), _make_report("R-3", 800, 42000, 360))

    deltas = compute_deltas_vs_baseline(state, "R-1")
    assert deltas is not None
    # R-3 vs R-1: latency 800 vs 1000 = -20%, area 42000 vs 40000 = +5%
    assert deltas["latency_ns"] == -20.0
    assert deltas["area_units"] == 5.0
    assert deltas["power_mw"] == -10.0


def test_baseline_not_found():
    """Unknown baseline run_id returns None."""
    state = {"history": []}
    state = append_result(state, _make_result(), _make_report("R-1", 1000, 40000, 400))
    assert compute_deltas_vs_baseline(state, "NONEXISTENT") is None
