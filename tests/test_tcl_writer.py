"""Tests for TCL config file writer."""

import os

from aiv_dse.adapters.tcl_writer import (
    backup_tcl_files,
    write_block_config,
    write_block_procs_tcl,
    write_project_tt2_tcl,
    write_tcl_config,
)
from aiv_dse.llm.models import SynthesisParams


def test_write_project_tt2_tcl(tmp_path):
    params = SynthesisParams(clock_period_ns=5.0, dpo_mode="DPO_AUTO_ALL")
    write_project_tt2_tcl(str(tmp_path), params)

    content = (tmp_path / "project.tt2.tcl").read_text()
    assert "set_attr clock_period {5.0}" in content
    assert "set_attr dpo_mode {DPO_AUTO_ALL}" in content


def test_write_block_config(tmp_path):
    params = SynthesisParams(
        unroll_factor=8,
        pipeline_depth=2,
        flatten=True,
        resource_sharing=True,
    )
    write_block_config(str(tmp_path), params)

    content = (tmp_path / "block.config").read_text()
    assert "set_directive_unroll -factor 8" in content
    assert "set_directive_pipeline -II 2" in content
    assert "set_directive_inline" in content
    assert "set_directive_resource_sharing" in content


def test_write_block_procs_tcl(tmp_path):
    params = SynthesisParams(clock_slack_ns=1.5)
    write_block_procs_tcl(str(tmp_path), params)

    content = (tmp_path / "block.procs.tcl").read_text()
    assert "set clock_slack {1.5}" in content


def test_backup_files(tmp_path):
    (tmp_path / "project.tt2.tcl").write_text("old content")
    (tmp_path / "block.config").write_text("old content")

    backups = backup_tcl_files(str(tmp_path))

    assert len(backups) == 2
    for backup_path in backups.values():
        assert os.path.exists(backup_path)
        assert "backup_" in backup_path


def test_write_tcl_config_integration(tmp_path):
    (tmp_path / "project.tt2.tcl").write_text("# old\n")

    params = SynthesisParams(
        unroll_factor=4,
        pipeline_depth=2,
        clock_period_ns=10.0,
        clock_slack_ns=0.5,
        dpo_mode="DPO_AUTO_OPT",
        flatten=True,
    )

    backups = write_tcl_config(params, str(tmp_path))
    assert len(backups) >= 1

    assert (tmp_path / "project.tt2.tcl").exists()
    assert (tmp_path / "block.config").exists()
    assert (tmp_path / "block.procs.tcl").exists()
