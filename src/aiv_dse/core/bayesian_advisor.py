"""Bayesian optimization advisor using Optuna.

Uses TPE (Tree-structured Parzen Estimator) by default, or GP sampler.
Maintains its own Optuna study and proposes synthesis parameters.

TPE is fast and handles integer params natively.
GP (Gaussian Process) is classic Bayesian optimization with better
theoretical guarantees but slower per iteration.

Cold start (< 3 observations): Optuna handles this internally with
random sampling from the TPE prior.
"""

from typing import Any, Dict, List, Optional

import optuna

from aiv_dse.llm.models import SynthParamAdjustment, SynthParamProposal, SynthesisParams

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Full search space including Phase 4 knobs
_DISTRIBUTIONS = {
    "unroll_factor": optuna.distributions.IntDistribution(1, 64),
    "pipeline_depth": optuna.distributions.IntDistribution(1, 16),
    "clock_period_ns": optuna.distributions.FloatDistribution(1.0, 100.0),
    "array_partition_factor": optuna.distributions.IntDistribution(1, 32),
    "clock_slack_ns": optuna.distributions.FloatDistribution(-5.0, 50.0),
    "dpo_mode": optuna.distributions.CategoricalDistribution(
        ["none", "DPO_AUTO_ALL", "DPO_AUTO_OPT", "DPO_AUTO_EXPR"]
    ),
    "flatten": optuna.distributions.CategoricalDistribution([False, True]),
    "inline": optuna.distributions.CategoricalDistribution([False, True]),
    "loop_merge": optuna.distributions.CategoricalDistribution([False, True]),
    "bitwidth_reduce": optuna.distributions.CategoricalDistribution([False, True]),
    "resource_sharing": optuna.distributions.CategoricalDistribution([False, True]),
}


class BayesianAdvisor:
    """Optuna-based Bayesian optimizer for synthesis parameter search."""

    def __init__(
        self,
        sampler: str = "tpe",
        seed: Optional[int] = None,
        multi_objective: bool = False,
    ):
        self._multi_objective = multi_objective

        if multi_objective:
            self._sampler = optuna.samplers.NSGAIISampler(seed=seed)
            self._study = optuna.create_study(
                directions=["minimize", "minimize", "minimize"],
                sampler=self._sampler,
            )
        else:
            if sampler == "gp":
                self._sampler = optuna.samplers.GPSampler(seed=seed)
            else:
                self._sampler = optuna.samplers.TPESampler(seed=seed)

            self._study = optuna.create_study(
                direction="minimize",
                sampler=self._sampler,
            )
        self._n_observations = 0

    def observe(
        self,
        params: SynthesisParams,
        report: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> None:
        """Feed an observation to the Optuna study.

        Computes violation_score: sum of max(0, observed-threshold)/threshold.
        0.0 = all constraints pass. Higher = worse.
        """
        param_dict = {
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
        }

        if self._multi_objective:
            values = [
                float(report.get("latency_ns", 0)),
                float(report.get("area_units", 0)),
                float(report.get("power_mw", 0)),
            ]
        else:
            score = self._compute_violation_score(report, policy)
            values = [score]

        trial = optuna.trial.create_trial(
            params=param_dict,
            distributions=_DISTRIBUTIONS,
            values=values,
        )
        self._study.add_trial(trial)
        self._n_observations += 1

    def propose(
        self,
        current_params: SynthesisParams,
    ) -> SynthParamProposal:
        """Ask Optuna for the next point to evaluate.

        Returns a SynthParamProposal for compatibility with the loop.
        """
        trial = self._study.ask(_DISTRIBUTIONS)

        suggested = dict(trial.params)

        # Tell the study we won't actually evaluate this trial
        self._study.tell(trial, state=optuna.trial.TrialState.PRUNED)

        # Build the full proposed SynthesisParams (preserves types)
        self.last_proposed_params = SynthesisParams.model_validate(suggested)

        # Build adjustments for changed numeric params (for logging)
        adjustments = []
        current_dict = current_params.model_dump()
        _NUMERIC_FIELDS = {
            "unroll_factor", "pipeline_depth", "clock_period_ns",
            "array_partition_factor", "clock_slack_ns",
        }
        for pname, new_val in suggested.items():
            old_val = current_dict.get(pname)
            if old_val is not None and new_val != old_val and pname in _NUMERIC_FIELDS:
                adjustments.append(SynthParamAdjustment(
                    param_name=pname,
                    current_value=float(old_val),
                    proposed_value=float(new_val),
                    reasoning=f"Bayesian optimizer ({self._sampler.__class__.__name__}) suggestion.",
                ))

        return SynthParamProposal(
            adjustments=adjustments,
            overall_reasoning=(
                f"Bayesian optimizer proposal "
                f"({self._n_observations} observations, "
                f"sampler={self._sampler.__class__.__name__})."
            ),
            confidence=min(0.3 + 0.1 * self._n_observations, 0.95),
            cited_runs=["N/A"],
        )

    @staticmethod
    def _compute_violation_score(
        report: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> float:
        """Sum of normalized constraint violations. 0.0 = feasible."""
        score = 0.0
        for c in policy.get("constraints", []):
            field = c["field"]
            threshold = c["max"]
            observed = report.get(field)
            if observed is not None and threshold > 0:
                excess = max(0.0, observed - threshold) / threshold
                score += excess
        return score
