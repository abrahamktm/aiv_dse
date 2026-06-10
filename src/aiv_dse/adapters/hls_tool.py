"""HLS Tool adapter.

Integrates with real HLS tooling:
1. Writes params to TCL config files (project.tt2.tcl, block.config, block.procs.tcl)
2. Runs HLS tool via subprocess
3. Parses output .rpt files (timing.rpt, area.rpt, power.rpt, synth.log)
4. Returns standard report dict
"""

import os
import subprocess
from typing import Any, Dict

from aiv_dse.adapters.base import HLSAdapter
from aiv_dse.adapters.rpt_parser import parse_all_reports
from aiv_dse.adapters.tcl_writer import write_tcl_config
from aiv_dse.llm.models import SynthesisParams


class HLSToolAdapter(HLSAdapter):
    """Adapter for HLS tooling."""

    def __init__(
        self,
        project_dir: str,
        hls_bin: str = "hls_synth",
        report_subdir: str = "out/reports",
        timeout_seconds: int = 600,
    ):
        """
        Args:
            project_dir:     Path to HLS project (contains *.tcl files)
            hls_bin:         Name or path to HLS executable
            report_subdir:   Where reports are written (relative to project_dir)
            timeout_seconds: Max wait time for synthesis
        """
        self._project_dir = project_dir
        self._hls_bin = hls_bin
        self._report_subdir = report_subdir
        self._timeout = timeout_seconds

    def name(self) -> str:
        return "HLSTool"

    def run_synthesis(
        self, params: SynthesisParams, run_id: str
    ) -> Dict[str, Any]:
        """Run HLS synthesis with the given parameters.

        Raises:
            RuntimeError: If synthesis fails or reports cannot be parsed.
        """
        # 1. Write TCL config
        backups = write_tcl_config(params, self._project_dir)
        print(f"  [{self.name()}] Wrote TCL configs (backed up {len(backups)} files)")

        # 2. Run synthesis
        cmd = [self._hls_bin, "-batch", "-project", self._project_dir]
        print(f"  [{self.name()}] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self._project_dir,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"HLS binary not found: '{self._hls_bin}'. "
                f"Ensure the HLS tool is installed and on PATH."
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"HLS synthesis timed out after {self._timeout}s"
            ) from e

        if result.returncode != 0:
            raise RuntimeError(
                f"HLS synthesis failed (exit {result.returncode}):\n"
                f"stderr: {result.stderr[:500]}"
            )

        print(f"  [{self.name()}] Synthesis completed (exit 0)")

        # 3. Parse reports
        report_dir = os.path.join(self._project_dir, self._report_subdir)
        try:
            metrics = parse_all_reports(report_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse HLS reports from {report_dir}: {e}"
            ) from e

        # 4. Build standard report dict
        report = {
            "run_id": run_id,
            "latency_ns": metrics["latency_ns"],
            "area_units": metrics["area_units"],
            "power_mw": metrics["power_mw"],
            "unroll_factor": params.unroll_factor,
            "pipeline_depth": params.pipeline_depth,
            "clock_period_ns": params.clock_period_ns,
            "array_partition_factor": params.array_partition_factor,
            "clock_slack_ns": params.clock_slack_ns,
            "dpo_mode": params.dpo_mode,
            "flatten": params.flatten,
            "inline": params.inline,
            "loop_merge": params.loop_merge,
            "bitwidth_reduce": params.bitwidth_reduce,
            "resource_sharing": params.resource_sharing,
            "notes": (
                f"HLS: unroll={params.unroll_factor}, "
                f"pipeline={params.pipeline_depth}, dpo={params.dpo_mode}"
            ),
            "warnings": metrics.get("warnings", []),
            "suggestions": metrics.get("suggestions", []),
        }

        print(
            f"  [{self.name()}] Parsed: latency={report['latency_ns']}ns, "
            f"area={report['area_units']}, power={report['power_mw']}mW"
        )

        return report
