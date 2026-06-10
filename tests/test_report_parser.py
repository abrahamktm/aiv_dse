import pytest

from aiv_dse.adapters.report_parser import load_report, validate_physics, PoisonDataError


def test_load_valid_report():
    report = load_report("samples/report_pass.json")
    assert report["run_id"] == "RUN-001"
    assert report["latency_ns"] == 8500
    assert report["area_units"] == 42000
    assert report["power_mw"] == 350


def test_valid_report_passes_physics():
    report = load_report("samples/report_pass.json")
    validate_physics(report)  # should not raise


def test_poison_raises():
    report = load_report("samples/poison_report.json")
    with pytest.raises(PoisonDataError, match="Poison data"):
        validate_physics(report)


def test_fail_report_passes_physics():
    """report_fail has bad metrics but they are physically possible (positive values)."""
    report = load_report("samples/report_fail.json")
    validate_physics(report)  # should not raise -- values are positive
