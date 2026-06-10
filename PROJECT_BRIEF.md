# AIV-DSE Project Brief

## Vision

A human-in-the-loop agentic framework for hardware design space exploration.
The system helps engineers find the right area/power/latency tradeoff by
iteratively validating synthesis results against constraints, tracking metric
trends, and (in Stage 2+) using an LLM to propose constraint adjustments
grounded in actual run data.

## Hard rules

- **IP-blind.** No proprietary vendor tool integration required.
  Uses synthetic data in /samples for demos and tests.
- **No hallucination.** The LLM must NEVER assume or fabricate metrics. All
  reasoning must cite actual values from prior runs stored in state.json.
  If there is no prior run data, the LLM says so explicitly.
- **Governance-first.** Deterministic validators have authority over LLM proposals.
  The LLM proposes, the validator disposes.
- **Human steering.** The human can review, modify, add, or reject any constraint
  or priority the workflow suggests. The workflow adapts to human input, not the
  other way around.

## Stages

### Stage 1: Deterministic Governance Core (no LLM)
- Policy-as-code (YAML constraints with severity and actions)
- Report parser with physics validation and poison detection
- Validator: compare metrics to thresholds, return structured result
- State history: last 3 runs, delta tracking, convergence detection
- pytest test suite

### Stage 2: LLM-Powered Exploration (future)
- Policy-to-prompt bridge (format constraints for LLM context)
- Pydantic models for structured LLM output (Action DSL + reasoning)
- LLM proposes constraint adjustments with reasoning grounded in run history
  ("reduce unroll from 16 to 8 because run RUN-002 showed area=62000, which is
  24% over the 50000 budget, and run RUN-001 with unroll=4 had area=42000")
- Confidence field in LLM output -- low confidence triggers HITL review
- HITL review step: human approves/modifies/rejects proposed constraints
- Tool calling: LLM can invoke validator and state-query tools directly
  (read-only -- it can check "what was area in RUN-001?" but cannot write state)
- EDR (Engineering Decision Record) writer
- Model-agnostic prompts (work with Claude, GPT, Gemini via LangChain)

### Stage 3: Advanced features (future)
- Weighted tradeoff scoring (configurable area/latency/power weights)
- Curated attribute reference (extracted from docs, stored in repo)
- Convergence detection ("last 3 runs <2% delta -> suggest stopping")
- Rollback to prior constraint set from state history
- LLM-as-judge: cross-check primary LLM's reasoning with a second LLM
  (different provider). If they disagree on the proposed action, escalate
  to human. Demonstrates adversarial validation, not single-model trust.

### Phase 4: Full HLS Tool Integration
- 7 new synthesis directives (clock_slack, DPO, flatten, inline, loop_merge, bitwidth_reduce, resource_sharing)
- Real .rpt file parsing (timing, area, power, synth log)
- TCL config generation for HLS tools
- CSV run logger for post-hoc analysis

### Phase 5: SystemC Code-Aware Advisor
- Regex-based static analysis of SystemC/C++ source (loops, arrays, pragmas, functions)
- RAG knowledge retriever with TF-IDF, multi-source ingestion, persistent caching
- LLM code suggestions with priority, category, and expected impact

### Phase 6: Multi-Objective Pareto Front
- True multi-objective optimization across latency, area, power (replaces single-score)
- Pareto dominance tracking and front computation with deduplication
- NSGA-II sampler (Optuna) for multi-objective Bayesian optimization
- Frontier convergence: front size stable for N updates = converged
- Weight-based selection from the Pareto front
- Multi-objective is enabled by default; disable with `--no-multi-objective`

### Future Phases
- **Phase 7**: Batch/parallel exploration (thread pool, batch acquisition)
- **Phase 8**: Real HLS tool CI/CD integration (GitHub Actions, regression detection)
- **Phase 9**: Transfer learning across IPs (meta-model, warm-start)
- **Phase 10**: LangGraph migration (state machine, checkpointing)
- **Phase 11**: Dashboard / Web UI (Streamlit, Pareto front explorer)
- **Phase 12**: Workflow documentation & diagrams
- **Phase 13**: Automatic constraint regression
- **Phase 14**: Production hardening (retry logic, token budget, OpenTelemetry)

## Design principles

1. **LLM proposes, tools dispose.** Same as AIV-DE.
2. **All reasoning is evidence-based.** The LLM cites run_id, metric values,
   and deltas from state.json. No unsupported claims.
3. **Generic policy evaluation.** The validator iterates a constraint list from
   YAML instead of hardcoding field names. Adding a new constraint = one YAML
   entry, zero code changes.
4. **State is the source of truth.** Every run appends to state.json. The LLM
   reads state, never writes it. Only the validator writes state.
5. **Human has final authority.** The workflow suggests, the human decides.

## FAQ

**How is this different from AIV-DE?**

AIV-DE makes a one-shot architecture decision. AIV-DSE runs an iterative
optimization loop -- each run produces new metrics, the validator checks
constraints, state tracks deltas, and the system decides whether to continue
or halt. The human steers the exploration, not just escalation.

**How does the system prevent hallucination?**

The LLM's structured output includes a reasoning field that must reference
actual run_ids and metric values from state.json. The validator independently
checks all claims. If the LLM cites a metric that doesn't match state, the
validator catches it. The LLM never writes to state -- only the deterministic
validator does.

**How would this scale?**

Each IP exploration is independent -- trivially parallelizable. State is
JSON per IP, no shared database. Policy YAML is version-controlled so teams
can have different constraints per project.
