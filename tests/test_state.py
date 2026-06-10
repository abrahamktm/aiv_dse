from aiv_dse.core.state import append_result, compute_deltas, history_summary
from aiv_dse.core.validator import ValidationResult


def _make_report(run_id, latency, area, power):
    return {"run_id": run_id, "latency_ns": latency, "area_units": area, "power_mw": power}


def _make_result(status="APPROVED"):
    return ValidationResult(status=status)


def test_append_and_trim():
    state = {"history": []}
    for i in range(5):
        state = append_result(state, _make_result(), _make_report(f"R-{i}", 100, 200, 50))
    assert len(state["history"]) == 3
    assert state["history"][0]["run_id"] == "R-2"
    assert state["history"][-1]["run_id"] == "R-4"


def test_compute_deltas():
    state = {"history": []}
    state = append_result(state, _make_result(), _make_report("R-1", 1000, 40000, 400))
    state = append_result(state, _make_result(), _make_report("R-2", 900, 44000, 380))
    deltas = compute_deltas(state)
    assert deltas is not None
    assert deltas["latency_ns"] == -10.0       # 900 vs 1000 = -10%
    assert deltas["area_units"] == 10.0        # 44000 vs 40000 = +10%
    assert deltas["power_mw"] == -5.0          # 380 vs 400 = -5%


def test_compute_deltas_returns_none_if_single_run():
    state = {"history": []}
    state = append_result(state, _make_result(), _make_report("R-1", 1000, 40000, 400))
    assert compute_deltas(state) is None


def test_history_summary():
    state = {"history": []}
    state = append_result(state, _make_result("VETO"), _make_report("R-1", 1000, 40000, 400))
    state = append_result(state, _make_result("APPROVED"), _make_report("R-2", 900, 38000, 350))
    summary = history_summary(state)
    assert "2 runs" in summary
    assert "APPROVED" in summary
    assert "VETO" in summary


def test_history_summary_empty():
    assert history_summary({"history": []}) == "No runs recorded."
