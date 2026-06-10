"""Abstract base class for HLS tool adapters.

Any HLS tool implements this interface.
The loop doesn't care which tool runs -- it only sees the report dict.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from aiv_dse.llm.models import SynthesisParams


class HLSAdapter(ABC):
    """Interface that every HLS tool adapter must implement."""

    @abstractmethod
    def run_synthesis(
        self, params: SynthesisParams, run_id: str
    ) -> Dict[str, Any]:
        """Run synthesis with the given parameters.

        Args:
            params: Synthesis knobs (unroll, pipeline, clock, partition).
            run_id: Unique identifier for this run (e.g. "RUN-003").

        Returns:
            A report dict with at minimum:
              run_id, latency_ns, area_units, power_mw, unroll_factor, notes
        """

    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name (e.g. 'DummyHLS', 'HLSTool')."""
