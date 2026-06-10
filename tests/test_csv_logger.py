"""Tests for CSV run logger."""

import csv

from aiv_dse.core.csv_logger import CSV_HEADERS, init_csv_log, log_run
from aiv_dse.llm.models import SynthesisParams


def test_init_csv_creates_file(tmp_path):
    path = tmp_path / "runs.csv"
    init_csv_log(str(path))

    assert path.exists()
    with open(path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert headers == CSV_HEADERS


def test_log_run_appends_row(tmp_path):
    path = tmp_path / "runs.csv"

    params = SynthesisParams(
        unroll_factor=8,
        dpo_mode="DPO_AUTO_ALL",
        flatten=True,
    )
    report = {"latency_ns": 7500, "area_units": 38000, "power_mw": 280}

    log_run(str(path), "RUN-001", "APPROVED", report, params)

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "RUN-001"
    assert rows[0]["status"] == "APPROVED"
    assert rows[0]["latency_ns"] == "7500"
    assert rows[0]["unroll_factor"] == "8"
    assert rows[0]["dpo_mode"] == "DPO_AUTO_ALL"
    assert rows[0]["flatten"] == "True"


def test_multiple_runs_append(tmp_path):
    path = tmp_path / "runs.csv"

    params = SynthesisParams()
    report = {"latency_ns": 9000, "area_units": 45000, "power_mw": 320}

    log_run(str(path), "RUN-001", "VETO", report, params)
    log_run(str(path), "RUN-002", "APPROVED", report, params)

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["run_id"] == "RUN-001"
    assert rows[1]["run_id"] == "RUN-002"
