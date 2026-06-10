"""Tests for HLS report file parser."""

import pytest

from aiv_dse.adapters.rpt_parser import (
    ReportParseError,
    parse_all_reports,
    parse_area_rpt,
    parse_power_rpt,
    parse_synth_log,
    parse_timing_rpt,
)


def test_parse_timing_rpt():
    result = parse_timing_rpt("samples/rpt/timing.rpt")
    assert result["latency_ns"] == 8500.0
    assert result["clock_period_ns"] == 8.734
    assert result["slack_ns"] == 1.266


def test_parse_area_rpt():
    result = parse_area_rpt("samples/rpt/area.rpt")
    assert result["area_units"] == 42000
    assert result["breakdown"]["luts"] == 42000
    assert result["breakdown"]["ffs"] == 18500
    assert result["breakdown"]["bram"] == 8
    assert result["breakdown"]["dsp"] == 12


def test_parse_power_rpt():
    result = parse_power_rpt("samples/rpt/power.rpt")
    assert result["power_mw"] == 350.0
    assert result["breakdown"]["dynamic_mw"] == 280.0
    assert result["breakdown"]["static_mw"] == 70.0


def test_parse_synth_log():
    result = parse_synth_log("samples/rpt/synth.log")
    assert len(result["warnings"]) == 1
    assert "TIMING-101" in result["warnings"][0]
    assert len(result["suggestions"]) == 2
    assert "OPT-201" in result["suggestions"][0]
    assert result["exit_status"] == "SUCCESS"


def test_parse_all_reports():
    result = parse_all_reports("samples/rpt")
    assert result["latency_ns"] == 8500.0
    assert result["area_units"] == 42000
    assert result["power_mw"] == 350.0
    assert result["clock_period_ns"] == 8.734
    assert result["synthesis_status"] == "SUCCESS"


def test_missing_file_raises():
    with pytest.raises(ReportParseError):
        parse_timing_rpt("nonexistent.rpt")
