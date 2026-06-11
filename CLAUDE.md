# AIV-DSE -- Claude Code Context

Project: Agentic Design Space Exploration for HLS tools.
Sister project to AIV-DE (one-shot governance) -- this one does iterative exploration.

This is AIV-DSE, an agentic HLS design space exploration framework.
See PROJECT_BRIEF.md for the design contract (vision, hard rules, design principles).

## Current priorities (in order)
1. ~~Migrate while loop to LangGraph (src/aiv_dse/graph.py)~~ **DONE** (Phase 7)
2. ~~Add Gradio UI (app.py)~~ **DONE** (deploy to HF Spaces when ready)
3. ~~Add Langfuse tracing (src/aiv_dse/tracing.py)~~ **DONE**
4. ~~LLM-layer upgrades + CI + benchmark + FIR sample~~ **DONE** (Phase 8)
5. ~~Automatic constraint regression (`--auto-relax` flow)~~ **DONE**
6. ~~`--explain` CLI flag (step trace without execution)~~ **DONE**
7. Build MCP server (src/aiv_dse/mcp_server.py)
8. Replace TF-IDF with ChromaDB RAG (src/aiv_dse/rag.py)
9. Screenshot LLM-as-judge trace for README
10. Build golden dataset + RAGAS evaluation


## Current State: Phase 8 Complete

**218 tests passing, 3 skipped** (matplotlib display tests, expected in CI).
Run: `$env:PYTHONPATH='src'; python -m pytest tests/ -v`

### Completed Phases

- **Stage 0**: Project scaffold (pyproject.toml, src layout, .env.example)
- **Stage 1**: Deterministic validation (report_parser, validator, state tracking)
- **Stage 2**: LLM constraint advisor + HITL + EDR writer (dual SDK: LangChain + Anthropic)
- **Stage 3**: Closed-loop with 3 strategies (shadow heuristic, Bayesian/Optuna TPE, LLM + judge + spec planner)
- **Phase 4**: Full HLS directive set (7 new SynthesisParams knobs), .rpt parsing, TCL config writer, CSV logger, extended DummyHLS physics
- **Phase 5**: SystemC code-aware advisor (regex static analysis, RAG knowledge retriever with TF-IDF, code suggestions)
- **Phase 6**: Multi-objective Pareto front (NSGA-II sampler, Pareto dominance tracking, frontier convergence, weight-based selection)
- **Phase 7**: LangGraph migration (`src/aiv_dse/graph.py`, cyclic state machine, `--backend graph|loop` CLI switch)
- **Phase 8**: LLM-layer upgrades — Anthropic prompt caching on advisor calls, opt-in extended thinking on judge (`AIVDSE_JUDGE_THINKING=1`), cross-provider judge (Claude advisor + Gemini judge by default; OpenAI fallback), Reflexion (`lessons_learned` in state, capped at `MAX_LESSONS=10`), PRM-style judge (`AIVDSE_USE_PRM_JUDGE=1`, per-adjustment partial acceptance) — plus `scripts/benchmark.py`, GitHub Actions CI (`.github/workflows/test.yml`), 32-tap FIR sample
- **Phase 18 partial**: `--explain` CLI flag shipped; Mermaid/ASCII orchestration diagrams still open
- **Phase 13**: Automatic constraint regression (`src/aiv_dse/core/constraint_relaxer.py`, `--auto-relax / --relax-step-pct / --max-relax-iters` CLI flags)

---

## File Map

```
src/aiv_dse/
  adapters/
    base.py              # ABC: HLSAdapter (run_synthesis, name)
    dummy_hls.py         # Simulated HLS with physics model (all 11 knobs)
    hls_tool.py          # Real HLS tool adapter (write TCL -> run -> parse reports)
    report_parser.py     # Stage 1 JSON report parser
    rpt_parser.py        # Phase 4: HLS .rpt file parsers (timing, area, power, synth.log)
    tcl_writer.py        # Phase 4: TCL config generator (project.tt2.tcl, block.config, block.procs.tcl)
  core/
    validator.py         # Policy-as-code: validate(report, policy) -> ValidationResult
    state.py             # Rolling window state (MAX_HISTORY=3, deltas, lessons_learned)
    stagnation.py        # Stagnation detection across runs
    shadow_heuristic.py  # Deterministic benchmark strategy (adjust unroll by 1)
    bayesian_advisor.py  # Optuna TPE/GP/NSGA-II optimizer (11-dim search space, single/multi-objective)
    convergence.py       # Weighted scoring + convergence detection + Pareto convergence
    pareto.py            # Phase 6: Pareto dominance, front computation, ParetoTracker
    history.py           # Full history (never trimmed), combo tracking
    csv_logger.py        # Phase 4: Append-only CSV run logger
    code_analyzer.py     # Phase 5: Regex static analysis of SystemC/C++ source
    knowledge_retriever.py # Phase 5: TF-IDF RAG retriever with persistent cache
    constraint_relaxer.py  # Phase 13: Detect unreachable constraints + auto-relax thresholds
    visualize.py           # Matplotlib plots (convergence, Pareto front)
  llm/
    config.py            # LLMSettings, get_llm(), get_anthropic_client() (Claude/OpenAI/Gemini)
    models.py            # Pydantic models (SynthesisParams, SynthParamProposal, JudgeVerdict, PRMJudgeVerdict, LessonLearned, SpecPlan, etc.)
    prompt_formatter.py  # format_context() for LLM prompts
    constraint_advisor.py # Stage 2: LLM proposes constraint threshold changes
    synth_advisor.py     # Stage 3: LLM proposes synthesis param changes (prompt caching enabled)
    judge.py             # LLM-as-judge cross-check; cross-provider, opt-in extended thinking, PRM-style scoring
    code_advisor.py      # Phase 5: LLM code-level suggestions (priority/category/impact)
    spec_planner.py      # IP spec reader (txt/pdf) -> SpecPlan
  workflow/
    hitl.py              # Human-in-the-loop review
    edr_writer.py        # Engineering Decision Record writer
  tracing.py             # Langfuse observability (@observe decorator, trace helpers)
  run_stage1.py          # CLI for Stage 1
  run_stage2.py          # CLI for Stage 2
  run_loop.py            # CLI for Stage 3+ closed loop (--backend loop|graph, --auto-relax, --explain)
  graph.py               # Phase 7: LangGraph state machine

scripts/benchmark.py            # Phase 8: reproducible strategy comparison (shadow vs Bayesian vs LLM)
.github/workflows/test.yml      # Phase 8: pytest on push/PR
app.py                          # Gradio UI (report validation + history; deployable to HF Spaces)

policy/default_policy.yaml   # latency<=10000, area<=50000, power<=500, unroll<=16
samples/                     # report_pass.json, report_fail.json, poison_report.json, fir_filter_design.cpp
samples/rpt/                 # Synthetic HLS timing.rpt, area.rpt, power.rpt, synth.log
specs/ip_spec_example.txt    # Sample FFT-256 spec
specs/ip_spec_fir.txt        # Sample 32-tap FIR spec

tests/                       # 218 tests passed + 3 skipped, all mocked, no API keys needed
```

---

## Key Design Decisions

1. **SynthesisParams** (11 knobs): unroll_factor, pipeline_depth, clock_period_ns, array_partition_factor, clock_slack_ns, dpo_mode, flatten, inline, loop_merge, bitwidth_reduce, resource_sharing
2. **Policy** (SPEC) stays fixed; SynthesisParams (KNOBS) are what the loop tunes
3. **Shadow heuristic** is a benchmark (not fallback) -- proves LLM adds value by comparing convergence
4. **LLM-as-judge** evaluates ALL LLM output (advisor + spec planner), not just some
5. **Dual SDK**: Every LLM call has both LangChain and direct Anthropic SDK paths
6. **Pydantic `extra="forbid"`** on all LLM-facing models
7. **Bayesian advisor** uses `last_proposed_params` (full SynthesisParams) instead of float-encoded adjustments to preserve categorical/boolean types
8. **Multi-objective is default** (`multi_objective=True`). Uses NSGA-II sampler, Pareto front tracking with deduplication, and frontier convergence (front size stable for 3 updates). Disable with `--no-multi-objective`

---

## DummyHLS Physics Model

Sweet spot for default policy: `unroll=2, pipeline=2` -> latency~8165, area~39000, power~212 -> APPROVED

Phase 4 multipliers:
- `clock_slack`: latency *= (1 - slack*0.01), area *= (1 + slack*0.02)
- `DPO_AUTO_ALL`: 0.80x area/power, `DPO_AUTO_OPT`: 0.88x, `DPO_AUTO_EXPR`: 0.95x
- `flatten`: 0.95x latency, 1.15x area
- `inline`: 0.93x latency, 1.12x area
- `loop_merge`: 0.90x latency
- `bitwidth_reduce`: 0.85x area, 0.88x power
- `resource_sharing`: 0.75x area, 1.05x power

---

## Next Phases (Roadmap)

> Numbering reflects shipping order, not the original plan numbers. Earlier numbers
> were reused — Phase 7 became LangGraph and Phase 8 became LLM-layer upgrades, so
> the original "Phase 7 = batch/parallel" and "Phase 8 = HLS CI/CD" ideas live on
> below at higher numbers, with their original details preserved.

### Phase 9: Run Checkpoint & Resume (DIY JSON, no LangGraph dep)
- Serialize the full `DSEState` (iteration, history, lessons_learned, ParetoTracker points, current SynthesisParams) to `out/checkpoint.json` after each iteration in the `record` node
- `--resume` CLI flag hydrates `DSEState` from the file when present and continues from the saved iteration
- Stdlib only (`json` + `Pydantic.model_dump()`); no `langgraph-checkpoint-sqlite` dep so it works on both `loop` and `graph` backends and survives any future LangGraph migration
- Solves the stop-on-day-1 / resume-on-day-2 case in-run
- ~3–4 hr including tests

### Phase 10: Warm-Start from Run History
- Read past `out/*.csv` rows at startup, filter to APPROVED runs, dedup, feed as prior trials into the Bayesian study
- Optimizer-agnostic via a small `HistorySeeder` Protocol: ship `OptunaSeeder` (`study.add_trial(FixedTrial(params, values))`) now; `BoTorchSeeder` slot reserved for the eventual BoTorch migration
- New CLI flag `--warm-start-from out/runs.csv` (or auto-discover when present)
- Cross-run / cross-IP knowledge transfer at the optimizer-prior layer; complementary to but distinct from Phase 9's same-run resume
- ~½ day including tests

### Phase 11: MCP Server
- Expose the DSE loop as MCP tools (`src/aiv_dse/mcp_server.py`)
- Lets external agents drive synthesis runs / read state / inspect the Pareto front

### Phase 12: ChromaDB RAG
- Replace TF-IDF retriever (`knowledge_retriever.py`) with ChromaDB (`src/aiv_dse/rag.py`)
- Vector embeddings for better recall on SystemC/HLS knowledge

### Phase 13: Golden Dataset + RAGAS Evaluation
- Curated golden dataset of (IP, constraints, expected best-params) tuples
- RAGAS-style metrics for retriever + advisor quality
- Regression detection if eval scores drop

### Phase 14: Batch/Parallel Exploration
- Run N synthesis jobs in parallel (thread pool or subprocess pool)
- Bayesian batch acquisition (q-EI, q-UCB)
- Progress dashboard

### Phase 15: Real HLS Tool CI/CD Integration
- GitHub Actions / Jenkins pipeline integration for actual HLS runs (current CI only runs pytest)
- Automatic nightly DSE runs
- Regression detection (alert if metrics regress from baseline)

### Phase 16: Transfer Learning Across IPs (full meta-model)
- Train a meta-model on CSV logs from multiple IPs (Phase 10 already covers the lightweight warm-start; this is the heavier follow-up)
- IP similarity scoring (cosine over a feature vector built from `SpecPlan`)
- Pareto-informed prior weighting in the Bayesian acquisition function

### Phase 17: Dashboard / Web UI (extensions on top of `app.py`)
- Real-time convergence plots
- Interactive constraint tuning
- Pareto front explorer
- LLM-as-judge trace screenshot for README

### Phase 18: Workflow Documentation & Diagrams
- Mermaid orchestration diagram in README (renders in GitHub/Confluence/VS Code)
- ASCII sequence diagram in CLAUDE.md for developer onboarding
- Per-iteration message flow diagram (strategy → judge → adapt cycle)
- `--explain` CLI flag already shipped — diagrams are the remaining piece

### Phase 19: Constraint-Regression Polish (core auto-relax already shipped; this is the tail)
- Pareto-front-derived "feasible value" suggestions in the relaxation report
  ("area constraint unreachable, closest was X vs target Y")
- Compress `_format_synth_context` history to 1-line-per-entry summaries (~40 tokens vs ~180)

### Phase 20: Production Hardening
- Retry logic for LLM calls (exponential backoff)
- Token budget tracking
- Rate limiting
- Structured logging (JSON) + OpenTelemetry traces

---

## Quick Commands

```powershell
cd aiv_dse
$env:PYTHONPATH='src'

# Run all tests
python -m pytest tests/ -v

# Demo: LangGraph backend (recommended)
python -m aiv_dse.run_loop --backend graph --strategy bayesian --max-iters 10 --seed 42

# Demo: legacy while-loop backend
python -m aiv_dse.run_loop --backend loop --strategy shadow --max-iters 10

# Demo: direct LangGraph entry point
python -m aiv_dse.graph --strategy bayesian --dpo DPO_AUTO_ALL --flatten --max-iters 10 --seed 42

# Demo: single-objective mode (legacy)
python -m aiv_dse.run_loop --strategy bayesian --no-multi-objective --max-iters 5 --seed 42

# Demo: LLM (needs .env with ANTHROPIC_API_KEY)
python -m aiv_dse.run_loop --strategy llm --sdk anthropic --max-iters 5

# Demo: with Langfuse tracing (needs LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY)
$env:AIVDSE_USE_LANGFUSE='1'; python -m aiv_dse.run_loop --strategy llm --max-iters 5
```

---

## Dependencies

Core (in `requirements.txt`, installed in CI):
```
pyyaml, pytest, langchain>=0.3, langchain-openai>=0.2, langchain-anthropic>=0.3,
langgraph>=0.2, anthropic>=0.39, pydantic>=2.0, python-dotenv, optuna>=3.0,
pdfplumber>=0.10, matplotlib>=3.7
```

Optional / lazy-imported (not required to import the package or run tests):
- `langfuse` — observability (`AIVDSE_USE_LANGFUSE=1`)
- `langchain_google_genai` — Gemini judge (only when `provider="google"`)
- `gradio` — `app.py` web UI

## Rules
- Never break existing tests
- Keep backward compatibility (legacy run_loop.py still works)
- All new features must be opt-in via env vars or CLI flags
- All data is synthetic (no proprietary IP)
