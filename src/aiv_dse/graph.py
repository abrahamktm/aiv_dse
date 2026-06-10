"""LangGraph state machine for the DSE optimization loop.

This module wraps the existing run_loop.py logic in a LangGraph state machine,
enabling checkpointing, replay, and branching.

Usage:
    python -m aiv_dse.graph --strategy bayesian --max-iters 10
    python -m aiv_dse.graph --strategy shadow --max-iters 5 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from aiv_dse.adapters.base import HLSAdapter
from aiv_dse.adapters.dummy_hls import DummyHLSAdapter
from aiv_dse.adapters.report_parser import PoisonDataError, validate_physics
from aiv_dse.core.bayesian_advisor import BayesianAdvisor
from aiv_dse.core.convergence import check_convergence, check_pareto_convergence, compute_weighted_score
from aiv_dse.core.csv_logger import log_run
from aiv_dse.core.history import append_full_history
from aiv_dse.core.pareto import ParetoTracker
from aiv_dse.core.shadow_heuristic import shadow_propose
from aiv_dse.core.stagnation import detect_stagnation
from aiv_dse.core.validator import ValidationResult, load_policy, validate
from aiv_dse.llm.models import SynthesisParams, SynthParamProposal
from aiv_dse.tracing import flush_traces

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_PATH = str(_PROJECT_ROOT / "policy" / "default_policy.yaml")
FULL_HISTORY_PATH = str(_PROJECT_ROOT / "out" / "full_history.json")
CSV_LOG_PATH = str(_PROJECT_ROOT / "out" / "runs.csv")


# ---------------------------------------------------------------------------
# State schema (Pydantic model for type safety)
# ---------------------------------------------------------------------------

class DSEState(BaseModel):
    """State schema for the LangGraph DSE optimization loop."""
    model_config = {"extra": "forbid"}

    # Core state
    run_id: str = Field(default="RUN-000", description="Current run identifier")
    params: Dict[str, Any] = Field(default_factory=dict, description="Current synthesis params as dict")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Latest metrics from synthesis")
    validation: Dict[str, Any] = Field(default_factory=dict, description="Validator output")
    status: str = Field(default="", description="APPROVED / VETO / ESCALATE / HALT")

    # Proposals
    shadow_proposal: Dict[str, Any] = Field(default_factory=dict)
    bayesian_proposal: Dict[str, Any] = Field(default_factory=dict)
    llm_proposal: Optional[Dict[str, Any]] = Field(default=None)
    selected_proposal: Dict[str, Any] = Field(default_factory=dict)

    # History and tracking
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Rolling window history (max 3)")
    lessons_learned: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Reflexion: judge-rejection lessons read by advisor next iteration")
    iteration: int = Field(default=0)
    max_iterations: int = Field(default=10)

    # Configuration
    strategy: str = Field(default="bayesian", description="shadow / bayesian / llm")
    multi_objective: bool = Field(default=True)
    use_judge: bool = Field(default=True)
    sdk_mode: str = Field(default="langchain")
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "latency_ns": 0.4, "area_units": 0.3, "power_mw": 0.3
    })

    # Terminal conditions
    converged: bool = Field(default=False)
    halted: bool = Field(default=False)
    final_status: str = Field(default="")

    # Adapter and advisor references (stored as names for serialization)
    adapter_name: str = Field(default="dummy")
    bayesian_sampler: str = Field(default="tpe")
    seed: Optional[int] = Field(default=None)

    # Policy (loaded once)
    policy: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runtime context (not serialized in state)
# ---------------------------------------------------------------------------

class DSEContext:
    """Runtime context holding non-serializable objects."""

    def __init__(
        self,
        adapter: HLSAdapter,
        bayesian: BayesianAdvisor,
        pareto: ParetoTracker,
        policy: Dict[str, Any],
    ):
        self.adapter = adapter
        self.bayesian = bayesian
        self.pareto = pareto
        self.policy = policy


# Global context (set before running the graph)
_context: Optional[DSEContext] = None


def set_context(ctx: DSEContext) -> None:
    """Set the runtime context for graph execution."""
    global _context
    _context = ctx


def get_context() -> DSEContext:
    """Get the runtime context."""
    if _context is None:
        raise RuntimeError("DSE context not set. Call set_context() before running the graph.")
    return _context


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def synthesize(state: DSEState) -> Dict[str, Any]:
    """Node 1: Run HLS synthesis with current params."""
    ctx = get_context()

    iteration = state.iteration + 1
    run_id = f"RUN-{iteration:03d}"

    # Convert params dict to SynthesisParams
    params = SynthesisParams.model_validate(state.params)

    # Run synthesis
    report = ctx.adapter.run_synthesis(params, run_id)

    return {
        "run_id": run_id,
        "metrics": report,
        "iteration": iteration,
    }


def validate_node(state: DSEState) -> Dict[str, Any]:
    """Node 2: Validate metrics against policy (includes poison check)."""
    ctx = get_context()
    report = state.metrics

    # Poison check
    is_poison = False
    try:
        validate_physics(report)
    except PoisonDataError:
        is_poison = True

    # Validate against policy
    result = validate(report, ctx.policy, is_poison=is_poison)

    return {
        "validation": result.to_dict(),
        "status": result.status,
        "halted": result.status == "HALT",
    }


def record(state: DSEState) -> Dict[str, Any]:
    """Node 3: Record history, update Pareto tracker, log to CSV."""
    ctx = get_context()
    params = SynthesisParams.model_validate(state.params)

    # Build history entry
    entry = {
        "run_id": state.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": state.status,
        "metrics": {
            "latency_ns": state.metrics.get("latency_ns"),
            "area_units": state.metrics.get("area_units"),
            "power_mw": state.metrics.get("power_mw"),
        },
        "violations": state.validation.get("violations", []),
        "synth_params": params.model_dump(),
    }

    # Update rolling window history (max 3)
    history = list(state.history)
    history.append(entry)
    if len(history) > 3:
        history = history[-3:]

    # Append to full history file
    append_full_history(entry, FULL_HISTORY_PATH)

    # Update Pareto tracker
    ctx.pareto.add_point(
        run_id=state.run_id,
        metrics={
            "latency_ns": state.metrics.get("latency_ns", 0),
            "area_units": state.metrics.get("area_units", 0),
            "power_mw": state.metrics.get("power_mw", 0),
        },
        synth_params=params.model_dump(),
        status=state.status,
    )

    # CSV log
    result = ValidationResult(
        status=state.status,
        violations=state.validation.get("violations", []),
    )
    log_run(CSV_LOG_PATH, state.run_id, state.status, state.metrics, params)

    # Feed observation to Bayesian advisor
    ctx.bayesian.observe(params, state.metrics, ctx.policy)

    return {"history": history}


def check_terminal(state: DSEState) -> Dict[str, Any]:
    """Node 4: Check for terminal conditions (convergence, max iters)."""
    ctx = get_context()

    # Already halted (poison)
    if state.halted:
        return {
            "converged": False,
            "final_status": "HALT",
        }

    # Max iterations reached
    if state.iteration >= state.max_iterations:
        return {
            "converged": False,
            "final_status": "MAX_ITERS_REACHED",
        }

    # Check convergence on APPROVED
    if state.status == "APPROVED":
        if state.multi_objective:
            pareto_conv = check_pareto_convergence(ctx.pareto)
            if pareto_conv:
                return {
                    "converged": True,
                    "final_status": "CONVERGED",
                }
        else:
            # Single-objective: APPROVED = converged
            history_state = {"history": state.history}
            conv_msg = check_convergence(history_state, ctx.policy, weights=state.weights)
            return {
                "converged": True,
                "final_status": "CONVERGED",
            }

    return {
        "converged": False,
        "final_status": "",
    }


def propose(state: DSEState) -> Dict[str, Any]:
    """Node 5: Get proposals from all strategies."""
    ctx = get_context()
    params = SynthesisParams.model_validate(state.params)
    result = ValidationResult(
        status=state.status,
        violations=state.validation.get("violations", []),
    )

    # Shadow proposal (always)
    shadow = shadow_propose(result, params, ctx.policy)

    # Bayesian proposal (always)
    bayesian = ctx.bayesian.propose(params)

    # LLM proposal (only if strategy=llm)
    llm_proposal = None
    if state.strategy == "llm":
        try:
            from aiv_dse.llm.config import LLMSettings
            from aiv_dse.llm.synth_advisor import propose_synth_params

            settings = LLMSettings.from_env()
            settings.sdk_mode = state.sdk_mode
            if state.sdk_mode == "anthropic":
                settings.provider = "anthropic"
                if settings.model_name == "gpt-4o-mini":
                    settings.model_name = "claude-sonnet-4-20250514"

            # Reflexion: pass past judge rejections so the advisor can learn from them
            history_state = {
                "history": state.history,
                "lessons_learned": state.lessons_learned,
            }
            llm_proposal = propose_synth_params(
                ctx.policy, history_state, result, params, settings, None
            )

            # Judge (if enabled)
            new_lessons = list(state.lessons_learned)
            if state.use_judge and llm_proposal:
                # PRM-style judge (opt-in via env var) scores adjustments individually
                if os.getenv("AIVDSE_USE_PRM_JUDGE", "0") == "1":
                    from aiv_dse.llm.judge import prm_judge_proposal, apply_prm_verdict
                    prm_verdict = prm_judge_proposal(
                        llm_proposal, ctx.policy, history_state, result, params, settings
                    )
                    if prm_verdict.any_accepted():
                        llm_proposal = apply_prm_verdict(llm_proposal, prm_verdict)
                        for s in prm_verdict.scores:
                            if not s.accept:
                                new_lessons.append({
                                    "iteration": state.iteration,
                                    "proposed_change": f"{s.param_name} change",
                                    "rejection_reason": s.reasoning,
                                })
                    else:
                        new_lessons.append({
                            "iteration": state.iteration,
                            "proposed_change": f"multi-param: {','.join(llm_proposal.cited_runs)}",
                            "rejection_reason": prm_verdict.overall_reasoning,
                        })
                        llm_proposal = None
                else:
                    from aiv_dse.llm.judge import judge_proposal
                    verdict = judge_proposal(
                        llm_proposal, ctx.policy, history_state, result, params, settings
                    )
                    if not verdict.agree:
                        # Reflexion: record the rejection for next iteration
                        change_summary = ", ".join(
                            f"{a.param_name} {a.current_value}->{a.proposed_value}"
                            for a in llm_proposal.adjustments[:3]
                        )
                        new_lessons.append({
                            "iteration": state.iteration,
                            "proposed_change": change_summary,
                            "rejection_reason": "; ".join(verdict.disagreements) or verdict.alternative_suggestion,
                        })
                        # Fall back to Bayesian
                        llm_proposal = None
            # Cap lessons to last MAX_LESSONS
            from aiv_dse.core.state import MAX_LESSONS
            if len(new_lessons) > MAX_LESSONS:
                new_lessons = new_lessons[-MAX_LESSONS:]
        except Exception:
            llm_proposal = None
            new_lessons = list(state.lessons_learned)
    else:
        new_lessons = list(state.lessons_learned)

    return {
        "shadow_proposal": shadow.model_dump(),
        "bayesian_proposal": bayesian.model_dump(),
        "llm_proposal": llm_proposal.model_dump() if llm_proposal else None,
        "lessons_learned": new_lessons,
    }


def apply_proposal(state: DSEState) -> Dict[str, Any]:
    """Node 6: Apply the selected strategy's proposal."""
    ctx = get_context()
    params = SynthesisParams.model_validate(state.params)

    # Select proposal based on strategy
    if state.strategy == "llm" and state.llm_proposal:
        selected = SynthParamProposal.model_validate(state.llm_proposal)
        # Apply adjustments
        new_params = _apply_proposal(params, selected)
    elif state.strategy == "bayesian":
        # Bayesian advisor stores the full proposed params
        new_params = ctx.bayesian.last_proposed_params
    elif state.strategy == "shadow":
        selected = SynthParamProposal.model_validate(state.shadow_proposal)
        new_params = _apply_proposal(params, selected)
    else:
        # Fallback to Bayesian
        new_params = ctx.bayesian.last_proposed_params

    return {
        "params": new_params.model_dump(),
        "selected_proposal": state.bayesian_proposal if state.strategy == "bayesian" else state.shadow_proposal,
    }


def _apply_proposal(params: SynthesisParams, proposal: SynthParamProposal) -> SynthesisParams:
    """Apply a proposal's adjustments to current params."""
    data = params.model_dump()
    for adj in proposal.adjustments:
        if adj.param_name in data:
            if adj.param_name in ("unroll_factor", "pipeline_depth", "array_partition_factor"):
                data[adj.param_name] = int(adj.proposed_value)
            else:
                data[adj.param_name] = adj.proposed_value
    return SynthesisParams.model_validate(data)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def should_continue(state: DSEState) -> Literal["propose", "done"]:
    """After check_terminal: continue proposing or finish?"""
    if state.halted or state.converged or state.final_status:
        return "done"
    return "propose"


def after_propose(state: DSEState) -> Literal["apply"]:
    """After propose: always apply."""
    return "apply"


def after_apply(state: DSEState) -> Literal["synthesize"]:
    """After apply: loop back to synthesize."""
    return "synthesize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_dse_graph() -> StateGraph:
    """Build and compile the DSE state machine."""

    # Create graph with state schema
    graph = StateGraph(DSEState)

    # Add nodes
    graph.add_node("synthesize", synthesize)
    graph.add_node("validate", validate_node)
    graph.add_node("record", record)
    graph.add_node("check_terminal", check_terminal)
    graph.add_node("propose", propose)
    graph.add_node("apply", apply_proposal)

    # Entry point
    graph.set_entry_point("synthesize")

    # Linear edges
    graph.add_edge("synthesize", "validate")
    graph.add_edge("validate", "record")
    graph.add_edge("record", "check_terminal")

    # Conditional edge after check_terminal
    graph.add_conditional_edges(
        "check_terminal",
        should_continue,
        {
            "propose": "propose",
            "done": END,
        }
    )

    # Propose -> Apply -> Synthesize (loop)
    graph.add_edge("propose", "apply")
    graph.add_edge("apply", "synthesize")

    return graph.compile()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_graph(
    adapter: HLSAdapter,
    policy: Dict[str, Any],
    initial_params: SynthesisParams,
    max_iters: int = 10,
    strategy: str = "bayesian",
    use_judge: bool = True,
    sdk_mode: str = "langchain",
    weights: Optional[Dict[str, float]] = None,
    bayesian_sampler: str = "tpe",
    seed: Optional[int] = None,
    multi_objective: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the DSE loop using the LangGraph state machine.

    Returns a summary dict compatible with run_loop.py output.
    """
    # Initialize advisors
    bayesian = BayesianAdvisor(
        sampler=bayesian_sampler,
        seed=seed,
        multi_objective=multi_objective,
    )
    pareto = ParetoTracker()

    # Set runtime context
    ctx = DSEContext(
        adapter=adapter,
        bayesian=bayesian,
        pareto=pareto,
        policy=policy,
    )
    set_context(ctx)

    # Build graph
    app = build_dse_graph()

    # Initial state
    initial_state = DSEState(
        params=initial_params.model_dump(),
        max_iterations=max_iters,
        strategy=strategy,
        use_judge=use_judge,
        sdk_mode=sdk_mode,
        weights=weights or {"latency_ns": 0.4, "area_units": 0.3, "power_mw": 0.3},
        multi_objective=multi_objective,
        policy=policy,
        seed=seed,
        bayesian_sampler=bayesian_sampler,
    )

    # Run the graph
    if verbose:
        print(f"\n=== Starting LangGraph DSE Loop ===")
        print(f"  Strategy:  {strategy}")
        print(f"  Adapter:   {adapter.name()}")
        print(f"  Max iters: {max_iters}")

    final_state = None
    for state in app.stream(initial_state.model_dump()):
        # state is a dict with node_name -> output
        node_name = list(state.keys())[0]
        node_output = state[node_name]

        if verbose and node_name == "synthesize":
            run_id = node_output.get("run_id", "?")
            iteration = node_output.get("iteration", 0)
            metrics = node_output.get("metrics", {})
            print(f"\n--- Iteration {iteration} ({run_id}) ---")
            print(f"  latency: {metrics.get('latency_ns')}, area: {metrics.get('area_units')}, power: {metrics.get('power_mw')}")

        if verbose and node_name == "validate":
            status = node_output.get("status", "?")
            print(f"  status: {status}")

        # Track final state
        final_state = node_output

    # Get final iteration count from context
    final_iteration = ctx.pareto.all_points[-1]["run_id"] if ctx.pareto.all_points else "RUN-000"
    iteration_num = int(final_iteration.split("-")[1]) if "-" in final_iteration else 0

    # Determine final status
    if final_state and isinstance(final_state, dict):
        final_status = final_state.get("final_status", "MAX_ITERS_REACHED")
    else:
        final_status = "MAX_ITERS_REACHED"

    if verbose:
        print(f"\n{'='*50}")
        print(f"=== Loop Summary ===")
        print(f"  Adapter:    {adapter.name()}")
        print(f"  Strategy:   {strategy}")
        print(f"  Iterations: {iteration_num}")
        print(f"  Status:     {final_status}")
        if multi_objective:
            print(f"  Pareto front: {pareto.front_size} points")
        print(f"{'='*50}")

    return {
        "final_status": final_status,
        "iterations": iteration_num,
        "strategy": strategy,
        "adapter": adapter.name(),
        "pareto_summary": pareto.summary() if multi_objective else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIV-DSE: LangGraph-based design space exploration"
    )
    parser.add_argument(
        "--strategy", default="bayesian", choices=["llm", "bayesian", "shadow"],
        help="Which advisor drives the loop",
    )
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--sdk", default="langchain", choices=["langchain", "anthropic"])
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--sampler", default="tpe", choices=["tpe", "gp"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-multi-objective", action="store_true")

    # Starting params
    parser.add_argument("--unroll", type=int, default=None)
    parser.add_argument("--pipeline", type=int, default=None)
    parser.add_argument("--clock", type=float, default=None)
    parser.add_argument("--partition", type=int, default=None)

    # Phase 4 directives
    parser.add_argument("--slack", type=float, default=None)
    parser.add_argument("--dpo", default=None,
                        choices=["none", "DPO_AUTO_ALL", "DPO_AUTO_OPT", "DPO_AUTO_EXPR"])
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--inline", action="store_true")
    parser.add_argument("--loop-merge", action="store_true")
    parser.add_argument("--bitwidth-reduce", action="store_true")
    parser.add_argument("--resource-sharing", action="store_true")

    # Weights
    parser.add_argument("--weight-latency", type=float, default=0.4)
    parser.add_argument("--weight-area", type=float, default=0.3)
    parser.add_argument("--weight-power", type=float, default=0.3)

    args = parser.parse_args()

    # Build adapter
    adapter = DummyHLSAdapter(noise_pct=5.0, seed=args.seed)

    # Load policy
    policy = load_policy(POLICY_PATH)

    # Build initial params
    param_kwargs: Dict[str, Any] = {}
    if args.unroll is not None:
        param_kwargs["unroll_factor"] = args.unroll
    if args.pipeline is not None:
        param_kwargs["pipeline_depth"] = args.pipeline
    if args.clock is not None:
        param_kwargs["clock_period_ns"] = args.clock
    if args.partition is not None:
        param_kwargs["array_partition_factor"] = args.partition
    if args.slack is not None:
        param_kwargs["clock_slack_ns"] = args.slack
    if args.dpo is not None:
        param_kwargs["dpo_mode"] = args.dpo
    if args.flatten:
        param_kwargs["flatten"] = True
    if args.inline:
        param_kwargs["inline"] = True
    if args.loop_merge:
        param_kwargs["loop_merge"] = True
    if args.bitwidth_reduce:
        param_kwargs["bitwidth_reduce"] = True
    if args.resource_sharing:
        param_kwargs["resource_sharing"] = True

    initial_params = SynthesisParams(**param_kwargs)

    weights = {
        "latency_ns": args.weight_latency,
        "area_units": args.weight_area,
        "power_mw": args.weight_power,
    }

    result = run_graph(
        adapter=adapter,
        policy=policy,
        initial_params=initial_params,
        max_iters=args.max_iters,
        strategy=args.strategy,
        use_judge=not args.no_judge,
        sdk_mode=args.sdk,
        weights=weights,
        bayesian_sampler=args.sampler,
        seed=args.seed,
        multi_objective=not args.no_multi_objective,
    )

    # Flush any pending Langfuse traces
    flush_traces()


if __name__ == "__main__":
    main()
