"""Stage 3 closed-loop CLI runner.

Usage:
    python -m aiv_dse.run_loop                                        # LLM-driven (default)
    python -m aiv_dse.run_loop --spec specs/ip_spec_example.txt       # LLM reads spec first
    python -m aiv_dse.run_loop --strategy bayesian --max-iters 15     # Optuna TPE
    python -m aiv_dse.run_loop --strategy bayesian --sampler gp       # Optuna GP
    python -m aiv_dse.run_loop --strategy shadow --max-iters 20       # heuristic-driven
    python -m aiv_dse.run_loop --sdk anthropic --no-judge             # direct SDK, no judge
    python -m aiv_dse.run_loop --unroll 8 --pipeline 2                # custom start
    python -m aiv_dse.run_loop --post-hook "make test_rtl"            # run command on APPROVED

--strategy selects which advisor DRIVES the loop.
ALL THREE strategies are logged every iteration regardless.
"""

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiv_dse.adapters.base import HLSAdapter
from aiv_dse.adapters.dummy_hls import DummyHLSAdapter
from aiv_dse.adapters.report_parser import PoisonDataError, validate_physics
from aiv_dse.core.bayesian_advisor import BayesianAdvisor
from aiv_dse.core.convergence import check_convergence, check_pareto_convergence, compute_weighted_score
from aiv_dse.core.pareto import ParetoTracker
from aiv_dse.core.csv_logger import log_run
from aiv_dse.core.history import append_full_history
from aiv_dse.core.shadow_heuristic import shadow_propose
from aiv_dse.core.stagnation import detect_stagnation
from aiv_dse.core.validator import ValidationResult, load_policy, validate
from aiv_dse.llm.models import SynthParamProposal, SynthesisParams
from aiv_dse.tracing import flush_traces

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_PATH = str(_PROJECT_ROOT / "policy" / "default_policy.yaml")
FULL_HISTORY_PATH = str(_PROJECT_ROOT / "out" / "full_history.json")
CSV_LOG_PATH = str(_PROJECT_ROOT / "out" / "runs.csv")


def _apply_proposal(
    params: SynthesisParams,
    proposal: SynthParamProposal,
) -> SynthesisParams:
    """Apply a proposal's adjustments to current params."""
    data = params.model_dump()
    for adj in proposal.adjustments:
        if adj.param_name in data:
            # Coerce to int for integer fields
            if adj.param_name in ("unroll_factor", "pipeline_depth", "array_partition_factor"):
                data[adj.param_name] = int(adj.proposed_value)
            else:
                data[adj.param_name] = adj.proposed_value
    return SynthesisParams.model_validate(data)


def _print_report(report: Dict[str, Any], iteration: int) -> None:
    print(f"\n--- Iteration {iteration} ---")
    print(f"  run_id:     {report.get('run_id')}")
    print(f"  latency_ns: {report.get('latency_ns')}")
    print(f"  area_units: {report.get('area_units')}")
    print(f"  power_mw:   {report.get('power_mw')}")
    params_str = ", ".join(
        f"{k}={report.get(k)}"
        for k in ("unroll_factor", "pipeline_depth", "clock_period_ns", "array_partition_factor")
        if report.get(k) is not None
    )
    if params_str:
        print(f"  params:     {params_str}")


def _print_result(result: ValidationResult) -> None:
    print(f"  status:     {result.status}")
    if result.violations:
        for v in result.violations:
            pct = ((v["observed"] - v["threshold"]) / v["threshold"]) * 100
            print(f"    {v['constraint_id']}: {v['observed']} > {v['threshold']} ({pct:.0f}% over)")


def _print_proposal(label: str, proposal: SynthParamProposal) -> None:
    if proposal.adjustments:
        changes = ", ".join(
            f"{a.param_name}: {a.current_value}->{a.proposed_value}"
            for a in proposal.adjustments
        )
    else:
        changes = "no changes"
    print(f"  [{label}] {changes} (conf={proposal.confidence:.2f})")


def _build_history_entry(
    report: Dict[str, Any],
    result: ValidationResult,
    params: SynthesisParams,
) -> Dict[str, Any]:
    return {
        "run_id": report.get("run_id", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": result.status,
        "metrics": {
            "latency_ns": report.get("latency_ns"),
            "area_units": report.get("area_units"),
            "power_mw": report.get("power_mw"),
        },
        "violations": result.violations,
        "synth_params": params.model_dump(),
    }


def _print_code_suggestions(advisory: Any) -> None:
    """Print code advisory suggestions."""
    if not advisory or not advisory.suggestions:
        return
    print(f"\n  === Code Suggestions ({len(advisory.suggestions)}) ===")
    print(f"  Assessment: {advisory.overall_assessment}")
    print(f"  Confidence: {advisory.confidence:.2f}")
    for i, s in enumerate(advisory.suggestions, 1):
        print(f"  [{i}] [{s.priority.upper()}] {s.category} @ line {s.target_line}")
        print(f"      {s.suggested_change}")
        print(f"      Impact: {s.expected_impact}")


def run_loop(
    adapter: HLSAdapter,
    policy: Dict[str, Any],
    initial_params: SynthesisParams,
    max_iters: int = 10,
    strategy: str = "llm",
    use_judge: bool = True,
    sdk_mode: str = "langchain",
    post_hook: Optional[str] = None,
    spec_summary: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    bayesian_sampler: str = "tpe",
    bayesian_seed: Optional[int] = None,
    source_path: Optional[str] = None,
    knowledge_dir: Optional[str] = None,
    rebuild_knowledge: bool = False,
    summarize_knowledge: bool = False,
    multi_objective: bool = True,
    plot: bool = True,
    auto_relax: bool = False,
    relax_step_pct: float = 10.0,
    max_relax_iters: int = 3,
) -> Dict[str, Any]:
    """Run the closed-loop DSE.

    Returns a summary dict with iteration count, final status,
    and strategy comparison log.
    """
    from aiv_dse.llm.config import LLMSettings

    params = copy.deepcopy(initial_params)
    state: Dict[str, Any] = {"history": []}

    # Phase 5: Code analysis (pre-loop, one-time)
    code_profile = None
    source_code = None
    if source_path:
        from aiv_dse.core.code_analyzer import analyze_source
        code_profile = analyze_source(source_path)
        with open(source_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        print(f"  Code profile: {len(code_profile.loops)} loops, "
              f"{len(code_profile.arrays)} arrays, "
              f"{len(code_profile.pragmas)} pragmas, "
              f"{len(code_profile.functions)} functions")

    # Phase 5: Knowledge retriever (pre-loop, one-time)
    knowledge_retriever = None
    if knowledge_dir:
        from aiv_dse.core.knowledge_retriever import KnowledgeRetriever
        if rebuild_knowledge:
            kr = KnowledgeRetriever.__new__(KnowledgeRetriever)
            kr._knowledge_dir = knowledge_dir
            kr._cache_dir = knowledge_dir + "/.cache"
            kr._chunks = []
            kr._index = {}
            kr._loaded = False
            llm_settings = None
            if summarize_knowledge and strategy == "llm":
                llm_settings = LLMSettings.from_env()
                llm_settings.sdk_mode = sdk_mode
            kr.build_index(summarize=summarize_knowledge, settings=llm_settings)
            knowledge_retriever = kr
        else:
            knowledge_retriever = KnowledgeRetriever(knowledge_dir)
        print(f"  Knowledge: {knowledge_retriever.chunk_count} chunks indexed")

    # Initialize strategies
    bayesian = BayesianAdvisor(
        sampler=bayesian_sampler, seed=bayesian_seed,
        multi_objective=multi_objective,
    )

    # Pareto tracker (always created; only used meaningfully when multi_objective)
    pareto = ParetoTracker()

    # Strategy comparison log
    comparison_log: List[Dict[str, Any]] = []

    # LLM settings (only needed if strategy=llm or for logging)
    settings = None
    if strategy == "llm":
        settings = LLMSettings.from_env()
        settings.sdk_mode = sdk_mode
        if sdk_mode == "anthropic":
            settings.provider = "anthropic"
            if settings.model_name == "gpt-4o-mini":
                settings.model_name = "claude-sonnet-4-20250514"

    final_status = "MAX_ITERS_REACHED"
    converged_iter = None
    code_advisory = None

    for iteration in range(1, max_iters + 1):
        run_id = f"RUN-{iteration:03d}"

        # 1. Run synthesis
        report = adapter.run_synthesis(params, run_id)
        _print_report(report, iteration)

        # 2. Poison check
        try:
            validate_physics(report)
        except PoisonDataError as e:
            print(f"  HALT: Poison detected -- {e}")
            final_status = "HALT"
            break

        # 3. Validate against policy
        result = validate(report, policy)
        _print_result(result)

        # 4. Build history entry (includes synth_params)
        entry = _build_history_entry(report, result, params)
        state["history"].append(entry)
        if len(state["history"]) > 3:
            state["history"] = state["history"][-3:]

        # 5. Full history (never trimmed)
        append_full_history(entry, FULL_HISTORY_PATH)

        # 5a. Pareto tracker
        pareto.add_point(
            run_id=report.get("run_id", "unknown"),
            metrics={
                "latency_ns": report.get("latency_ns", 0),
                "area_units": report.get("area_units", 0),
                "power_mw": report.get("power_mw", 0),
            },
            synth_params=params.model_dump(),
            status=result.status,
        )

        # 5b. CSV log (append-only, all params + metrics)
        log_run(CSV_LOG_PATH, run_id, result.status, report, params)

        # 6. Feed observation to Bayesian advisor
        bayesian.observe(params, report, policy)

        # 7. If APPROVED
        if result.status == "APPROVED":
            score = compute_weighted_score(
                report, policy, weights
            )
            print(f"  APPROVED (weighted score: {score:.3f})")

            # Post-hook
            if post_hook:
                print(f"  Running post-hook: {post_hook}")
                hook_result = subprocess.run(
                    post_hook, shell=True, capture_output=True, text=True
                )
                if hook_result.returncode != 0:
                    print(f"  POST_HOOK_FAIL (exit {hook_result.returncode})")
                    print(f"  stderr: {hook_result.stderr.strip()}")
                    final_status = "POST_HOOK_FAIL"
                    continue  # Don't count as converged, keep iterating
                else:
                    print(f"  Post-hook OK")

            if multi_objective:
                # Multi-objective: keep growing the Pareto front
                print(f"  Pareto front size: {pareto.front_size}")
                pareto_conv = check_pareto_convergence(pareto)
                if pareto_conv:
                    print(f"  {pareto_conv}")
                    final_status = "CONVERGED"
                    converged_iter = iteration
                    break
                # Don't break — continue iterating to grow the front
            else:
                # Single-objective: original behavior
                conv_msg = check_convergence(state, policy, weights=weights)
                if conv_msg:
                    print(f"  {conv_msg}")

                final_status = "CONVERGED"
                converged_iter = iteration
                break

        # 8. Check stagnation
        stag_msg = detect_stagnation(state)
        if stag_msg:
            print(f"  Warning: {stag_msg}")

        # 9. Get proposals from ALL THREE strategies
        # Shadow (always runs)
        shadow_proposal = shadow_propose(result, params, policy)
        _print_proposal("Shadow", shadow_proposal)

        # Bayesian (always runs)
        bayesian_proposal = bayesian.propose(params)
        _print_proposal("Bayesian", bayesian_proposal)

        # Phase 5: Retrieve knowledge chunks for this iteration
        knowledge_chunks = []
        if knowledge_retriever:
            from aiv_dse.core.knowledge_retriever import KnowledgeRetriever
            query = KnowledgeRetriever.build_query_from_violations(result, params)
            knowledge_chunks = knowledge_retriever.retrieve(query, top_k=3)

        # LLM (only if strategy=llm and settings available)
        llm_proposal = None
        if strategy == "llm" and settings:
            try:
                from aiv_dse.llm.synth_advisor import propose_synth_params
                llm_proposal = propose_synth_params(
                    policy, state, result, params, settings, spec_summary,
                    knowledge_chunks=knowledge_chunks or None,
                )
                _print_proposal("LLM", llm_proposal)

                # 10. Judge (if enabled)
                if use_judge:
                    from aiv_dse.llm.judge import judge_proposal
                    verdict = judge_proposal(
                        llm_proposal, policy, state, result, params, settings
                    )
                    if verdict.agree:
                        print(f"  Judge: AGREE (conf={verdict.confidence:.2f})")
                    else:
                        print(f"  Judge: DISAGREE -- {'; '.join(verdict.disagreements)}")
                        # On disagreement, fall back to Bayesian
                        print(f"  Escalating: using Bayesian proposal instead.")
                        llm_proposal = None

            except Exception as e:
                print(f"  LLM error: {e}")
                print(f"  Falling back to Bayesian proposal.")

        # Phase 5: Code advisory (only with source + LLM)
        code_advisory = None
        if source_code and code_profile and strategy == "llm" and settings:
            try:
                from aiv_dse.llm.code_advisor import advise_code_changes
                code_advisory = advise_code_changes(
                    source_code, code_profile, policy, state, result,
                    params, settings, knowledge_chunks or None,
                )
                _print_code_suggestions(code_advisory)

                if use_judge and code_advisory and code_advisory.suggestions:
                    from aiv_dse.llm.judge import judge_code_advisory
                    code_verdict = judge_code_advisory(
                        code_advisory, code_profile, policy, state,
                        result, params, settings,
                    )
                    if code_verdict.agree:
                        print(f"  Code Judge: AGREE (conf={code_verdict.confidence:.2f})")
                    else:
                        print(f"  Code Judge: DISAGREE -- {'; '.join(code_verdict.disagreements)}")
            except Exception as e:
                print(f"  Code advisor error: {e}")

        # Log comparison
        comparison_log.append({
            "iteration": iteration,
            "shadow": shadow_proposal.model_dump(),
            "bayesian": bayesian_proposal.model_dump(),
            "llm": llm_proposal.model_dump() if llm_proposal else None,
        })

        # 12. Apply the SELECTED strategy's proposal
        if strategy == "llm" and llm_proposal:
            selected = llm_proposal
            selected_label = "LLM"
        elif strategy == "bayesian":
            selected = bayesian_proposal
            selected_label = "Bayesian"
        elif strategy == "shadow":
            selected = shadow_proposal
            selected_label = "Shadow"
        else:
            # LLM failed or not available, fall back to Bayesian
            selected = bayesian_proposal
            selected_label = "Bayesian (fallback)"

        print(f"  Applying: {selected_label}")
        # Bayesian advisor builds a full SynthesisParams (preserves types);
        # other strategies use float-based adjustments via _apply_proposal.
        if selected_label in ("Bayesian", "Bayesian (fallback)") and hasattr(bayesian, "last_proposed_params"):
            params = bayesian.last_proposed_params
        else:
            params = _apply_proposal(params, selected)

    # Pareto selection (multi-objective only)
    pareto_selection = None
    if multi_objective and pareto.front_size > 0:
        pareto_selection = pareto.select_by_weights(policy, weights)
        if pareto_selection:
            print(f"\n  Pareto selection (by weights): {pareto_selection['run_id']}")
            print(f"    metrics: {pareto_selection['metrics']}")

        # Pareto visualization
        if plot:
            try:
                from aiv_dse.core.visualize import plot_pareto_front
                plot_path = str(_PROJECT_ROOT / "out" / "pareto_front.png")
                plot_pareto_front(
                    all_points=pareto.all_points,
                    front_points=pareto.front,
                    selected_point=pareto_selection,
                    output_path=plot_path,
                )
                print(f"  Pareto plot saved: {plot_path}")
            except ImportError:
                print("  Pareto plot skipped (matplotlib not installed)")
            except Exception as e:
                print(f"  Pareto plot failed: {e}")

    # Final summary
    print(f"\n{'='*50}")
    print(f"=== Loop Summary ===")
    print(f"  Adapter:    {adapter.name()}")
    print(f"  Strategy:   {strategy}")
    print(f"  Iterations: {converged_iter or max_iters}")
    print(f"  Status:     {final_status}")
    if converged_iter:
        print(f"  Converged at iteration {converged_iter}")
    if multi_objective:
        print(f"  Pareto front: {pareto.front_size} points")
    print(f"{'='*50}")

    # Constraint relaxation analysis (Phase 13)
    relaxation_report = None
    if final_status == "MAX_ITERS_REACHED":
        from aiv_dse.core.constraint_relaxer import analyze_and_relax
        from aiv_dse.core.history import load_full_history

        full_history = load_full_history(FULL_HISTORY_PATH)
        rel_report = analyze_and_relax(
            full_history, policy,
            step_pct=relax_step_pct,
            auto_relax=auto_relax,
        )

        if rel_report.unreachable:
            print(f"\n  === Unreachable Constraints ===")
            for u in rel_report.unreachable:
                print(f"    {u.constraint_id}: threshold={u.current_threshold}, "
                      f"closest={u.closest_observed}, gap={u.gap_pct:.1f}%, "
                      f"consecutive_vetos={u.consecutive_vetos}")

            if auto_relax and rel_report.relaxed_policy and max_relax_iters > 0:
                print(f"\n  Auto-relaxing constraints:")
                for cid in rel_report.relaxed_constraints:
                    for c in rel_report.relaxed_policy.get("constraints", []):
                        if c["id"] == cid:
                            print(f"    {cid}: new max = {c['max']:.1f}")

                print(f"  Re-running loop with relaxed policy "
                      f"(remaining relax iters: {max_relax_iters - 1})")
                relaxed_result = run_loop(
                    adapter=adapter,
                    policy=rel_report.relaxed_policy,
                    initial_params=initial_params,
                    max_iters=max_iters,
                    strategy=strategy,
                    use_judge=use_judge,
                    sdk_mode=sdk_mode,
                    post_hook=post_hook,
                    spec_summary=spec_summary,
                    weights=weights,
                    bayesian_sampler=bayesian_sampler,
                    bayesian_seed=bayesian_seed,
                    source_path=source_path,
                    knowledge_dir=knowledge_dir,
                    rebuild_knowledge=rebuild_knowledge,
                    summarize_knowledge=summarize_knowledge,
                    multi_objective=multi_objective,
                    plot=plot,
                    auto_relax=auto_relax,
                    relax_step_pct=relax_step_pct,
                    max_relax_iters=max_relax_iters - 1,
                )
                relaxed_result["relaxation_applied"] = True
                relaxed_result["relaxation_report"] = {
                    "unreachable": [
                        {
                            "constraint_id": u.constraint_id,
                            "field": u.field,
                            "current_threshold": u.current_threshold,
                            "closest_observed": u.closest_observed,
                            "consecutive_vetos": u.consecutive_vetos,
                            "gap_pct": u.gap_pct,
                            "suggested_threshold": u.suggested_threshold,
                        }
                        for u in rel_report.unreachable
                    ],
                    "relaxed_constraints": rel_report.relaxed_constraints,
                }
                return relaxed_result

        relaxation_report = {
            "unreachable": [
                {
                    "constraint_id": u.constraint_id,
                    "field": u.field,
                    "current_threshold": u.current_threshold,
                    "closest_observed": u.closest_observed,
                    "consecutive_vetos": u.consecutive_vetos,
                    "gap_pct": u.gap_pct,
                    "suggested_threshold": u.suggested_threshold,
                }
                for u in rel_report.unreachable
            ] if rel_report.unreachable else [],
            "relaxed_constraints": rel_report.relaxed_constraints,
        }

    return {
        "final_status": final_status,
        "iterations": converged_iter or max_iters,
        "strategy": strategy,
        "adapter": adapter.name(),
        "comparison_log": comparison_log,
        "code_advisory": code_advisory.model_dump() if code_advisory else None,
        "pareto_summary": pareto.summary() if multi_objective else None,
        "pareto_selection": pareto_selection,
        "relaxation_report": relaxation_report,
    }


def explain_loop() -> None:
    """Print a numbered description of each stage in the iteration loop."""
    steps = [
        "Load policy: read the YAML policy file and extract constraint thresholds.",
        "Initialise parameters: build the starting SynthesisParams (from CLI or spec plan).",
        "Run synthesis: call the HLS adapter with current params to produce a metrics report.",
        "Poison check: run validate_physics() to detect anomalous or fabricated data.",
        "Validate: compare report metrics against every policy constraint (APPROVED / VETO / ESCALATE).",
        "Record history: append the run entry to the rolling window and the full history file.",
        "Convergence check: if APPROVED, compute weighted score and check Pareto front stability.",
        "Stagnation detection: warn if the last N runs show no improvement.",
        "Propose (Shadow): heuristic adjusts unroll_factor by +/-1 toward the most-violated constraint.",
        "Propose (Bayesian): Optuna TPE/GP suggests the next parameter combination.",
        "Propose (LLM): language model reasons over history and proposes multi-dimensional changes.",
        "Judge: a second LLM cross-checks the proposal; on disagreement, fall back to Bayesian.",
        "Apply: the selected strategy's proposal becomes the next iteration's SynthesisParams.",
    ]
    print("AIV-DSE Iteration Loop -- 13 Steps\n")
    for i, step in enumerate(steps, 1):
        print(f"  Step {i:>2}: {step}")
    print("\nAfter the loop ends:")
    print("  - Pareto front selection by weights (multi-objective mode)")
    print("  - Final summary: adapter, strategy, iterations, status, front size")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIV-DSE Stage 3: Closed-loop design space exploration"
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Print a numbered description of each loop stage and exit",
    )
    parser.add_argument(
        "--backend", default="loop", choices=["loop", "graph"],
        help="Execution backend: 'loop' (legacy while loop) or 'graph' (LangGraph state machine)",
    )
    parser.add_argument(
        "--strategy", default="llm", choices=["llm", "bayesian", "shadow"],
        help="Which advisor drives the loop (all three are logged)",
    )
    parser.add_argument(
        "--max-iters", type=int, default=10,
        help="Maximum iterations before stopping",
    )
    parser.add_argument(
        "--sdk", default="langchain", choices=["langchain", "anthropic"],
        help="SDK mode for LLM calls",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Disable LLM-as-judge cross-check",
    )
    parser.add_argument(
        "--adapter", default="dummy", choices=["dummy", "hls_tool"],
        help="HLS tool adapter",
    )
    parser.add_argument("--project-dir", default=".", help="HLS project directory")
    parser.add_argument("--spec", default=None, help="Path to IP spec file (.txt or .pdf)")
    parser.add_argument("--post-hook", default=None, help="Command to run after APPROVED")
    parser.add_argument("--sampler", default="tpe", choices=["tpe", "gp"],
                        help="Bayesian sampler type")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # Starting params (Phase 3)
    parser.add_argument("--unroll", type=int, default=None)
    parser.add_argument("--pipeline", type=int, default=None)
    parser.add_argument("--clock", type=float, default=None)
    parser.add_argument("--partition", type=int, default=None)

    # Phase 4 HLS directives
    parser.add_argument("--slack", type=float, default=None, help="Clock slack (ns)")
    parser.add_argument("--dpo", default=None,
                        choices=["none", "DPO_AUTO_ALL", "DPO_AUTO_OPT", "DPO_AUTO_EXPR"],
                        help="Datapath optimization mode")
    parser.add_argument("--flatten", action="store_true", help="Enable flatten")
    parser.add_argument("--inline", action="store_true", help="Enable inline")
    parser.add_argument("--loop-merge", action="store_true", help="Enable loop merge")
    parser.add_argument("--bitwidth-reduce", action="store_true", help="Enable bitwidth reduction")
    parser.add_argument("--resource-sharing", action="store_true", help="Enable resource sharing")

    # Phase 5: Code analysis + RAG
    parser.add_argument("--source", default=None,
                        help="Path to SystemC/C++ source file for code-aware analysis")
    parser.add_argument("--knowledge-dir", default=None,
                        help="Path to knowledge directory (default: knowledge/)")
    parser.add_argument("--rebuild-knowledge", action="store_true",
                        help="Force rebuild the knowledge index cache")
    parser.add_argument("--summarize-knowledge", action="store_true",
                        help="Compress knowledge chunks via LLM at build time")
    parser.add_argument("--no-multi-objective", action="store_true",
                        help="Disable multi-objective Pareto optimization (use single-objective)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip Pareto front PNG generation")

    # Phase 13: Constraint relaxation
    parser.add_argument("--auto-relax", action="store_true",
                        help="Automatically relax unreachable constraints and re-run")
    parser.add_argument("--relax-step-pct", type=float, default=10.0,
                        help="Percentage to increase thresholds per relaxation step (default: 10)")
    parser.add_argument("--max-relax-iters", type=int, default=3,
                        help="Maximum number of relaxation re-runs (default: 3)")

    # Weights
    parser.add_argument("--weight-latency", type=float, default=0.4)
    parser.add_argument("--weight-area", type=float, default=0.3)
    parser.add_argument("--weight-power", type=float, default=0.3)

    args = parser.parse_args()

    if args.explain:
        explain_loop()
        return

    # Delegate to LangGraph backend if requested
    if args.backend == "graph":
        from aiv_dse.graph import run_graph

        # Build adapter for graph backend
        if args.adapter == "hls_tool":
            from aiv_dse.adapters.hls_tool import HLSToolAdapter
            adapter = HLSToolAdapter(args.project_dir)
        else:
            adapter = DummyHLSAdapter(noise_pct=5.0, seed=args.seed)

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

        run_graph(
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
        return

    # Build adapter
    if args.adapter == "hls_tool":
        from aiv_dse.adapters.hls_tool import HLSToolAdapter
        adapter: HLSAdapter = HLSToolAdapter(args.project_dir)
    else:
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

    # Spec planning (pre-loop)
    spec_summary = None
    if args.spec:
        from aiv_dse.llm.spec_planner import load_spec, plan_from_spec
        from aiv_dse.llm.config import LLMSettings

        print(f"=== Reading IP spec: {args.spec} ===")
        spec_text = load_spec(args.spec)
        spec_summary = spec_text[:500]  # Summary for loop context

        settings = LLMSettings.from_env()
        settings.sdk_mode = args.sdk
        if args.sdk == "anthropic":
            settings.provider = "anthropic"
            if settings.model_name == "gpt-4o-mini":
                settings.model_name = "claude-sonnet-4-20250514"

        print("  LLM analyzing spec...")
        spec_plan = plan_from_spec(spec_text, settings)

        print(f"\n=== Spec Plan ===")
        print(f"  Reasoning: {spec_plan.reasoning}")
        print(f"  Constraints:")
        for c in spec_plan.constraints:
            print(f"    - {c.id}: max {c.max} {c.field} ({c.severity}, {c.on_violation})")
            print(f"      Reason: {c.reasoning}")
        print(f"  Initial params:")
        ip = spec_plan.initial_params
        print(f"    unroll={ip.unroll_factor}, pipeline={ip.pipeline_depth}, "
              f"clock={ip.clock_period_ns}ns, partition={ip.array_partition_factor}")
        if spec_plan.warnings:
            print(f"  Warnings:")
            for w in spec_plan.warnings:
                print(f"    - {w}")

        # HITL review of spec plan
        print("\n  Review the spec plan above.")
        choice = input("  Accept spec plan (a), use defaults (d), or quit (q)? ").strip().lower()
        if choice == "q":
            print("  Aborted.")
            return
        elif choice == "a":
            # Use spec plan's params and build policy from constraints
            initial_params = spec_plan.initial_params
            # Override policy constraints with spec-derived ones
            policy["constraints"] = [c.model_dump() for c in spec_plan.constraints]
            print("  Using spec-derived constraints and params.")
        else:
            print("  Using default policy and params.")

    # Weights
    weights = {
        "latency_ns": args.weight_latency,
        "area_units": args.weight_area,
        "power_mw": args.weight_power,
    }

    print(f"\n=== Starting loop ===")
    print(f"  Strategy:  {args.strategy}")
    print(f"  Adapter:   {adapter.name()}")
    print(f"  Max iters: {args.max_iters}")
    p = initial_params
    print(f"  Params:    unroll={p.unroll_factor}, pipeline={p.pipeline_depth}, "
          f"clock={p.clock_period_ns}ns, partition={p.array_partition_factor}")
    print(f"  Weights:   latency={weights['latency_ns']}, area={weights['area_units']}, "
          f"power={weights['power_mw']}")

    result = run_loop(
        adapter=adapter,
        policy=policy,
        initial_params=initial_params,
        max_iters=args.max_iters,
        strategy=args.strategy,
        use_judge=not args.no_judge,
        sdk_mode=args.sdk,
        post_hook=args.post_hook,
        spec_summary=spec_summary,
        weights=weights,
        bayesian_sampler=args.sampler,
        bayesian_seed=args.seed,
        source_path=args.source,
        knowledge_dir=args.knowledge_dir,
        rebuild_knowledge=args.rebuild_knowledge,
        summarize_knowledge=args.summarize_knowledge,
        multi_objective=not args.no_multi_objective,
        plot=not args.no_plot,
        auto_relax=args.auto_relax,
        relax_step_pct=args.relax_step_pct,
        max_relax_iters=args.max_relax_iters,
    )

    # Flush any pending Langfuse traces
    flush_traces()


if __name__ == "__main__":
    main()
