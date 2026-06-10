"""Simulated HLS adapter with realistic physics model.

The tradeoff triangle:
  - Higher unroll   -> lower latency, higher area + power
  - Higher pipeline -> lower latency, higher area (modest power increase)
  - Lower clock     -> lower latency (faster clock), higher power
  - Higher partition -> lower latency, higher area + power

With default policy (latency<=10000, area<=50000, power<=500):
  - Default params (unroll=4, pipe=1, clock=10, part=1):
      latency~10000, area~60000, power~300  -> VETO on area
  - Sweet spot (unroll=2, pipe=2, clock=10, part=1):
      latency~8165, area~39000, power~212   -> APPROVED
  - The loop must explore to find this.

Configurable noise for realism; seed for deterministic tests.
"""

import math
import random
from typing import Any, Dict, Optional

from aiv_dse.adapters.base import HLSAdapter
from aiv_dse.llm.models import SynthesisParams


class DummyHLSAdapter(HLSAdapter):
    """Simulated HLS tool with physics-based area/power/latency model."""

    def __init__(
        self,
        noise_pct: float = 5.0,
        seed: Optional[int] = None,
    ):
        self._noise_pct = noise_pct
        self._rng = random.Random(seed)

    def name(self) -> str:
        return "DummyHLS"

    def run_synthesis(
        self, params: SynthesisParams, run_id: str
    ) -> Dict[str, Any]:
        u = params.unroll_factor
        p = params.pipeline_depth
        c = params.clock_period_ns
        a = params.array_partition_factor
        slack = params.clock_slack_ns
        dpo = params.dpo_mode

        # --- Base physics model (Phase 3) ---
        # Latency: more parallelism (unroll, pipeline, partition) -> lower
        latency = 20000.0 / (
            math.sqrt(u) * math.log2(p + 1) * (10.0 / c) * math.sqrt(a)
        )

        # Area: more parallelism -> higher (partition has sub-linear cost)
        area = (
            15000.0
            * u
            * (1.0 + 0.3 * (p - 1))
            * math.pow(10.0 / c, 0.5)
            * math.pow(a, 0.7)
        )

        # Power: driven by unroll and clock; pipeline/partition have smaller effect
        power = (
            150.0
            * math.sqrt(u)
            * (1.0 + 0.1 * (p - 1))
            * math.pow(10.0 / c, 1.2)
            * math.pow(a, 0.4)
        )

        # --- Phase 4 adjustments ---

        # Clock slack: +slack -> easier timing (~1% latency improvement per ns)
        #              +slack -> more area (~2% increase per ns)
        latency *= (1.0 - slack * 0.01)
        area *= (1.0 + slack * 0.02)
        power *= (1.0 - slack * 0.015)

        # DPO modes: reduce area and power (more aggressive = more reduction)
        dpo_factor = {
            "none": 1.0,
            "DPO_AUTO_EXPR": 0.95,
            "DPO_AUTO_OPT": 0.88,
            "DPO_AUTO_ALL": 0.80,
        }.get(dpo, 1.0)
        area *= dpo_factor
        power *= dpo_factor

        # Flatten: better optimization -> lower latency, higher area
        if params.flatten:
            latency *= 0.95
            area *= 1.15

        # Inline: similar to flatten but less aggressive
        if params.inline:
            latency *= 0.93
            area *= 1.12

        # Loop merge: reduce latency for sequential loops
        if params.loop_merge:
            latency *= 0.90

        # Bitwidth reduction: smaller logic -> less area and power
        if params.bitwidth_reduce:
            area *= 0.85
            power *= 0.88

        # Resource sharing: share HW resources -> less area, slight power increase
        if params.resource_sharing:
            area *= 0.75
            power *= 1.05

        # Apply noise
        if self._noise_pct > 0:
            for name, val in [("latency", latency), ("area", area), ("power", power)]:
                noise = 1.0 + self._rng.uniform(
                    -self._noise_pct / 100, self._noise_pct / 100
                )
                if name == "latency":
                    latency = val * noise
                elif name == "area":
                    area = val * noise
                else:
                    power = val * noise

        return {
            "run_id": run_id,
            "latency_ns": round(latency),
            "area_units": round(area),
            "power_mw": round(power),
            "unroll_factor": u,
            "pipeline_depth": p,
            "clock_period_ns": c,
            "array_partition_factor": a,
            "clock_slack_ns": slack,
            "dpo_mode": dpo,
            "flatten": params.flatten,
            "inline": params.inline,
            "loop_merge": params.loop_merge,
            "bitwidth_reduce": params.bitwidth_reduce,
            "resource_sharing": params.resource_sharing,
            "notes": (
                f"DummyHLS: unroll={u}, pipeline={p}, clock={c}ns, "
                f"partition={a}, slack={slack}ns, dpo={dpo}"
            ),
        }
