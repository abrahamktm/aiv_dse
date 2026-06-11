# AI-Driven Hardware Design Space Exploration

![tests](https://github.com/abrahamktm/aiv_dse/actions/workflows/test.yml/badge.svg)

An AI agent that automatically tunes hardware chip designs — running experiments,
checking results, and proposing better settings until the design meets all targets.

**Key features:**
- **Autonomous loop**: Runs dozens of experiments without human babysitting
- **Three search strategies**: Dumb heuristic, Bayesian math, and LLM reasoning — all compared
- **Adversarial validation**: Two LLMs *from different providers* cross-check each other (Claude advisor + Gemini judge by default). Disagreement triggers human review
- **PRM-style judge**: Scores each adjustment independently so good parts of a proposal are kept even when other parts are rejected
- **Reflexion self-improvement**: When the judge rejects a proposal, the advisor reads that rejection on the next iteration so it doesn't repeat the same mistake
- **Multi-objective Pareto front**: Finds the best tradeoffs instead of collapsing everything to one score
- **Governance-first**: AI proposes, deterministic code decides. AI never writes final state directly
- **LangGraph state machine**: Production-ready cyclic workflow with checkpointing
- **Prompt caching + extended thinking**: cost and quality optimizations on LLM calls
- **Empirical benchmark**: A reproducible script proves which search strategy beats the others on the synthetic landscape
- **218 tests**, all mocked. No API keys needed for CI
- **Langfuse observability**: Full LLM call tracing
- **Gradio UI**: Web interface for report validation and history

---

## For Beginners: What Problem Does This Solve?

### The hardware design tuning problem

When engineers design a chip or hardware block, they have to choose dozens of settings
(called **directives** or **knobs**). Each knob affects three competing goals:

| Goal | What it means | Example target |
|------|--------------|----------------|
| **Latency** | How fast the chip runs | Under 10,000 ns |
| **Area** | How much chip space it uses | Under 50,000 units |
| **Power** | How much power it consumes | Under 500 mW |

The problem: **improving one metric often worsens another.** For example:
- Unrolling a loop 8x makes it faster (lower latency) but uses more chip area
- Reducing clock speed reduces area but increases latency
- There is no single "best" setting — it depends on what matters most to you

Manually exploring all combinations is slow, expensive, and error-prone. This project
automates that process using an AI agent.

---

## How It Works: The Agentic Loop

Instead of guessing once, AIV-DSE runs an **iterative loop** — like a scientist
running controlled experiments and adjusting based on results.

```
                    ┌─────────────────────────────────────────┐
                    │           THE DSE LOOP                  │
                    │                                         │
  You provide:      │  1. Run synthesis with current settings │
  - Initial params  │         ↓                               │
  - Policy targets  │  2. Parse the results report            │
                    │         ↓                               │
                    │  3. Validate: did we meet all targets?  │
                    │         ↓                               │
                    │  4. If yes → add to Pareto front        │
                    │     If no  → figure out what to change  │
                    │         ↓                               │
                    │  5. Three strategies propose new params │
                    │     (Shadow, Bayesian, LLM)             │
                    │         ↓                               │
                    │  6. LLM Judge cross-checks the proposal │
                    │         ↓                               │
                    │  7. Apply best proposal → go to step 1  │
                    │                                         │
                    │  STOP when: converged / max iters /     │
                    │             critical error detected      │
                    └─────────────────────────────────────────┘
```

Each iteration, the system learns from the previous run. Over 10–20 iterations
it zero-in on a design that satisfies all constraints — or tells you it's impossible
and what the closest achievable values were.

---

## Architecture Deep Dive: Why Each Decision Was Made

### Decision 1: Why a loop at all? (vs. one-shot AI)

You could just ask an LLM: *"What HLS settings should I use?"* and apply the answer.
The problem: LLMs hallucinate, and hardware physics are complex. The first guess is
usually wrong.

The loop approach uses **real synthesis feedback** to correct mistakes:

```
Iteration 1: unroll=4, pipeline=2  →  latency=12500 ns  ← TOO SLOW (target: <10000)
Iteration 2: unroll=8, pipeline=3  →  latency=9800  ns  ← PASS
                                       area=52000 units  ← TOO BIG (target: <50000)
Iteration 3: unroll=6, pipeline=3  →  latency=9200  ns  ← PASS
                                       area=47000 units  ← PASS
                                       power=310 mW      ← PASS → APPROVED!
```

**Why it matters:** The AI uses actual synthesis results, not guesses. Each run gives
it more information to work with.

---

### Decision 2: Why three strategies? (Shadow / Bayesian / LLM)

No single strategy wins every time. Running all three lets the system compare them:

| Strategy | What it does | Best for |
|----------|-------------|---------|
| **Shadow (heuristic)** | Simple rule: if latency is violated, increase unroll by 1 | Baseline benchmark — proves the others add value |
| **Bayesian (Optuna)** | Builds a statistical model of the search space; focuses on promising regions | When you have 5+ runs of data and want math-guided search |
| **LLM advisor** | Reads the full run history and reasons about multi-dimensional tradeoffs | When constraints are interrelated in non-obvious ways |

The shadow heuristic is intentionally "dumb" — it's a benchmark. If the LLM doesn't
outperform a one-line rule, the LLM isn't adding real value.

```
After 10 runs:
  Shadow converged at iteration 8   (unroll went 4→5→6→7→... slowly)
  Bayesian converged at iteration 5  (jumped to unroll=6, pipeline=3 early)
  LLM converged at iteration 4      (combined unroll + loop_merge in one shot)
```

---

### Decision 3: Why LLM-as-Judge? (Adversarial validation)

When an LLM proposes a parameter change, how do you know if the reasoning is sound?
You ask a second LLM to check it — from a skeptical angle.

```
PRIMARY LLM (Claude):
  "Reduce unroll from 8 to 4. Area is 52000, 4% over budget.
   This will bring area down by ~25% at cost of ~10% latency increase.
   Latency has 15% headroom, so this is safe."

JUDGE LLM (cross-checks):
  "I agree the direction is right. However, the latency estimate is
   optimistic — unroll reduction also affects pipeline depth interaction.
   Recommend also reducing pipeline from 3 to 2 to avoid timing closure risk."

OUTCOME: Disagreement → proposal escalated to human review
```

**Why it matters:** A single LLM can be confidently wrong. Two LLMs disagree when
the reasoning is shaky — that disagreement is a signal to involve a human.

The governance rule: **the deterministic validator always has final say.** The LLM
can propose whatever it wants, but the validator checks the actual numbers. LLMs
never directly write state.

---

### Decision 4: Why a Pareto Front? (Multi-objective optimization)

With three competing goals, there is no single "best" design. There is a set of
designs where you can't improve one metric without worsening another. That set is
called the **Pareto front**.

**Example with three designs:**

```
Design A:  latency=8000, area=48000, power=480  (fast, small, near power limit)
Design B:  latency=9200, area=42000, power=310  (slightly slower, much smaller, low power)
Design C:  latency=7500, area=55000, power=290  (fastest, but too big — violates area)

Design C is dominated by nothing — it's fast and low power — but violates area.
Design A and B are both on the Pareto front: neither is strictly better than the other.

If power matters most to you → pick Design B
If latency matters most → pick Design A
```

Without a Pareto front, a single-score optimizer might collapse all three into
`score = 0.4*latency + 0.3*area + 0.3*power` and return one answer. But that
answer depends entirely on your weights — and you might not know your weights upfront.

**The loop keeps the full Pareto front and lets you choose at the end.**

```powershell
# Weight-based selection at the end (latency priority)
python -m aiv_dse.run_loop --weight-latency 0.7 --weight-area 0.15 --weight-power 0.15
```

The NSGA-II sampler (used by the Bayesian strategy) is designed specifically to
explore multiple objectives simultaneously — it won't sacrifice all area gains just
to improve latency by 1%.

---

### Decision 5: Why LangGraph? (State machine over while loop)

The original implementation was a Python `while` loop. LangGraph replaces it with
an explicit **state machine** — a graph where each node is a step and edges define
what happens next.

```
while loop (before):               LangGraph graph (after):
─────────────────────              ──────────────────────────────
while not done:                    synthesize ──→ validate ──→ record
  run_synthesis()                       ↑                        ↓
  validate()                            │                  check_terminal
  propose()                             │                   /          \
  apply()                           apply ←── propose    END           END
```

**Why it matters for production:**
- **Checkpointing**: if the run crashes at iteration 7, restart from iteration 7 — not 0
- **Replay**: re-run any iteration with different parameters for debugging
- **Visibility**: the graph structure is explicit — you can see all possible paths
- **Testing**: each node is a pure function, easy to unit test in isolation

---

### Decision 6: What is "Governance-First"?

This is the core safety principle of the whole system:

```
WITHOUT governance:                WITH governance (this project):
────────────────────               ────────────────────────────────
LLM sees bad result                LLM sees bad result
LLM decides what to change         LLM PROPOSES what to change
LLM writes new config              Deterministic validator CHECKS the proposal
                                   Validator either applies or rejects it
                                   LLM never writes state directly
```

**Why it matters:** LLMs can be confidently wrong about numbers. The deterministic
validator uses hard-coded policy rules (`area must be < 50000`) — no LLM can override
that. This makes the system auditable and safe for production use.

---

## Full Architecture (Component Map)

```
┌─────────────────────────────────────────────────────────────────┐
│                         AIV-DSE SYSTEM                          │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ IP Spec  │───▶│ Spec Planner │───▶│  Initial Params    │    │
│  │ (txt/pdf)│    │   (LLM)      │    │  (SynthesisParams) │    │
│  └──────────┘    └──────────────┘    └─────────┬──────────┘    │
│                                                │               │
│              ┌─────────────────────────────────▼──────┐        │
│              │           LANGGRAPH STATE MACHINE       │        │
│              │                                        │        │
│              │  ┌──────────┐   ┌──────────────────┐  │        │
│              │  │Synthesize│──▶│     Validate      │  │        │
│              │  │(HLS tool)│   │  (policy checker) │  │        │
│              │  └────▲─────┘   └────────┬─────────┘  │        │
│              │       │                  ▼             │        │
│              │  ┌────┴─────┐   ┌──────────────────┐  │        │
│              │  │  Apply   │   │      Record       │  │        │
│              │  │  params  │   │ (history, Pareto, │  │        │
│              │  └────▲─────┘   │    CSV log)       │  │        │
│              │       │         └────────┬─────────┘  │        │
│              │  ┌────┴─────┐            ▼             │        │
│              │  │ Propose  │   ┌──────────────────┐  │        │
│              │  │ params   │◀──│  Check Terminal  │  │        │
│              │  └──────────┘   │(converged/halt?) │  │        │
│              │                 └────────┬─────────┘  │        │
│              └─────────────────────────┼────────────┘        │
│                                        │ END                   │
│  ┌─────────────────────────────────────▼──────────────────┐   │
│  │                    PROPOSE NODE                         │   │
│  │                                                        │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │   │
│  │  │   Shadow    │  │   Bayesian   │  │     LLM     │  │   │
│  │  │ (heuristic) │  │  (NSGA-II)   │  │  (advisor)  │  │   │
│  │  └─────────────┘  └──────────────┘  └──────┬──────┘  │   │
│  │                                             │          │   │
│  │                                    ┌────────▼───────┐  │   │
│  │                                    │   LLM Judge    │  │   │
│  │                                    │ (cross-check)  │  │   │
│  │                                    └────────────────┘  │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────┐   ┌─────────────────┐                   │
│  │  Langfuse traces │   │   Gradio UI     │                   │
│  │  (observability) │   │  (web interface)│                   │
│  └──────────────────┘   └─────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## What's in this repo (beginner-friendly file map)

A plain-English tour of where things live. Skip to **Quick Start** if you already
know the layout.

### Source code — `src/aiv_dse/`

The main brain of the project.

| Folder / file | What it does |
|---|---|
| `adapters/` | Plug-ins for hardware synthesis tools. `dummy_hls.py` is a fake simulator (used for tests and demos — no real tool needed). `hls_tool.py` is the shape of a real-tool adapter. `report_parser.py` and `rpt_parser.py` read the output files HLS tools produce. `tcl_writer.py` writes the config files HLS tools consume. |
| `core/` | The decision-making logic — runs without any AI. `validator.py` checks results against the policy. `state.py` and `history.py` remember what happened in past iterations. `pareto.py` tracks the best tradeoffs. `bayesian_advisor.py` uses statistics to pick the next experiment. `shadow_heuristic.py` is a simple "dumb" baseline to compare the AI against. `constraint_relaxer.py` notices when a target is impossible and suggests loosening it. `code_analyzer.py` + `knowledge_retriever.py` scan SystemC source code and pull relevant HLS tips. `csv_logger.py` writes every run to a CSV. `visualize.py` plots the Pareto front. |
| `llm/` | Everything AI-related. `config.py` picks which provider (Claude / OpenAI / Gemini). `models.py` defines the strict shapes AI responses must take. `prompt_formatter.py` builds the prompt. `constraint_advisor.py` + `synth_advisor.py` + `code_advisor.py` ask the AI for suggestions. `judge.py` is the **second AI** that double-checks the first one. `spec_planner.py` reads an IP spec (txt/pdf) and turns it into starting parameters. |
| `workflow/` | Human-in-the-loop bits. `hitl.py` asks the user to review when AIs disagree. `edr_writer.py` writes the "Engineering Decision Record" — a paper trail of why each decision was made. |
| `graph.py` | The conductor. Wires synthesize → validate → record → decide-what-next → apply → loop, using LangGraph as the state machine. |
| `run_loop.py` | Command-line entry point for the full closed-loop exploration. `python -m aiv_dse.run_loop ...` |
| `run_stage1.py`, `run_stage2.py` | Older / smaller CLIs that only run early stages (deterministic validation, single-LLM advisor). |
| `tracing.py` | Optional observability — pipes every AI call to Langfuse if you set the env vars. |

### Top-level files

| File / folder | What it does |
|---|---|
| `app.py` | Web UI built with Gradio. Run `python app.py` to use the system in a browser instead of from the command line. |
| `scripts/benchmark.py` | Reproducible head-to-head comparison of the three strategies (shadow vs Bayesian vs LLM) on synthetic data. |
| `policy/default_policy.yaml` | The constraint rules — your latency / area / power targets. Edit this to change what counts as "passing". |
| `samples/` | Example synthesis reports (`report_pass.json`, `report_fail.json`, `poison_report.json`), sample HLS reports under `samples/rpt/`, and a SystemC source example `fir_filter_design.cpp`. Lets you run everything without a real HLS tool. |
| `specs/` | Example IP specifications in plain text — `ip_spec_example.txt` (FFT-256) and `ip_spec_fir.txt` (32-tap FIR). The spec planner reads these. |
| `tests/` | The automated test suite (218+ tests). All mocked, no API keys required. Run with `pytest tests/`. |
| `knowledge/` | Hand-curated HLS tips/docs that the RAG retriever indexes. |
| `requirements.txt`, `pyproject.toml` | Python dependencies and package metadata. |
| `README.md` | What you're reading. The "what is this / how to use" doc. |
| `CLAUDE.md` | Developer / agent-facing notes — file map with shorter jargon-y descriptions, current phase status, dev quick-commands. |
| `PROJECT_BRIEF.md` | The design contract — vision, hard rules ("LLM proposes, validator disposes"), and design principles. |
| `.github/workflows/test.yml` | GitHub Actions config — runs the test suite on every push and PR. |
| `LICENSE` | MIT. |

---

## Quick Start

### Setup

```powershell
cd aiv_dse
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env   # edit with your API keys
```

### Run without API keys (simulated HLS)

```powershell
$env:PYTHONPATH='src'

# Validate a single report
python -m aiv_dse.run_stage1 samples/report_pass.json     # APPROVED
python -m aiv_dse.run_stage1 samples/report_fail.json     # VETO
python -m aiv_dse.run_stage1 samples/poison_report.json   # HALT

# Run the full loop (Bayesian strategy, no LLM calls needed)
python -m aiv_dse.run_loop --backend graph --strategy bayesian --max-iters 10 --seed 42

# See all loop stages explained
python -m aiv_dse.run_loop --explain
```

### Run with LLM (needs `ANTHROPIC_API_KEY` in `.env`)

```powershell
$env:PYTHONPATH='src'

# LLM-driven loop with LLM-as-judge
python -m aiv_dse.run_loop --strategy llm --sdk anthropic --max-iters 10

# LLM reads IP spec first, then runs loop
python -m aiv_dse.run_loop --spec specs/ip_spec_example.txt --sdk anthropic
```

### Gradio web UI

```powershell
pip install gradio
python app.py
# Open http://localhost:7860
```

---

## How it differs from AIV-DE (sister project)

| | AIV-DE | AIV-DSE |
|---|---|---|
| Pattern | One-shot decision pipeline | Iterative exploration loop |
| Agents | Sequential: analyst → architect → validator → writer | Cyclic: synthesize → validate → propose → apply |
| State | Single-run trace | Multi-run history with Pareto front |
| Human role | Escalation target (HITL on failure) | Active steering at each iteration |
| Output | Architecture Decision Record (ADR) | Engineering Decision Record (EDR) |

---

## Phase 4: HLS Directives (11 Tunable Knobs)

| Knob | CLI flag | Effect |
|------|----------|--------|
| `unroll_factor` | `--unroll` | Unroll loops: faster but more area |
| `pipeline_depth` | `--pipeline` | Pipeline stages: throughput vs. latency tradeoff |
| `clock_period_ns` | `--clock` | Clock target: slower clock = easier timing closure |
| `array_partition_factor` | `--partition` | Memory access parallelism |
| `clock_slack_ns` | `--slack` | Timing margin: positive = relaxed |
| `dpo_mode` | `--dpo` | Datapath optimization: `DPO_AUTO_ALL` / `OPT` / `EXPR` |
| `flatten` | `--flatten` | Flatten hierarchy: lower latency, higher area |
| `inline` | `--inline` | Inline functions: similar to flatten, less aggressive |
| `loop_merge` | `--loop-merge` | Merge adjacent loops: lower latency |
| `bitwidth_reduce` | `--bitwidth-reduce` | Auto bitwidth: less area + power |
| `resource_sharing` | `--resource-sharing` | Share HW resources: less area, slight power increase |

```powershell
$env:PYTHONPATH='src'

# All directives enabled (Bayesian will explore their interactions)
python -m aiv_dse.run_loop --strategy bayesian --dpo DPO_AUTO_OPT --flatten --inline --loop-merge --bitwidth-reduce --resource-sharing --max-iters 15
```

### Report Parsing

Parses HLS tool output files:
- `timing.rpt` → latency_ns, clock_period_ns, slack_ns
- `area.rpt` → area_units, breakdown (LUTs, FFs, BRAM, DSP)
- `power.rpt` → power_mw, breakdown (dynamic, static)
- `synth.log` → warnings, suggestions, exit_status

### TCL Config Writer

Generates HLS-compatible config files from `SynthesisParams`:
- `project.tt2.tcl` -- clock period, DPO mode
- `block.config` -- unroll, pipeline, partition, flatten, inline, loop_merge, bitwidth_reduce, resource_sharing
- `block.procs.tcl` -- clock slack

### CSV Run Logger

Every run appends one row to `out/runs.csv` with all params + metrics + status. Useful for:
- Post-hoc analysis in pandas / Excel
- Training data for transfer learning (Phase 9)
- Audit trail of every synthesis attempt

---

## Phase 5: Code-Aware Advisory (RAG + Static Analysis)

So far the loop only tunes **knobs** (`unroll`, `pipeline`, etc.) — it can't suggest changes
to the actual SystemC/C++ source code. Phase 5 adds three components that work together
to give code-level suggestions like *"add `#pragma HLS PIPELINE II=1` at line 47."*

### Component 1: Static Code Analyzer (`core/code_analyzer.py`)

Regex-based parser for SystemC/C++ source files. Extracts a structured profile:

```python
CodeProfile:
  - loops:       [for/while loops with line number, nesting depth, iteration count]
  - arrays:      [arrays with name, dimensions, element type, partition status]
  - pragmas:     [existing HLS pragmas with line number and directive]
  - functions:   [function names, call graph, which is top-level]
  - memory_access_pattern: "sequential" / "random" / "strided"
```

Example: pointed at `samples/fft256_design.cpp`, it finds 4 loops, 6 arrays, the
existing `#pragma HLS PIPELINE II=4`, and the butterfly function call graph.

### Component 2: Knowledge Retriever (TF-IDF RAG)

Indexes everything in the `knowledge/` directory (HLS directive references, optimization
strategies, your own notes) and supports semantic-ish lookup. Uses **TF-IDF** today —
will be replaced with **ChromaDB embeddings** as a planned upgrade.

```python
retriever = KnowledgeRetriever()
chunks = retriever.retrieve("how to reduce area without hurting latency", k=3)
# Returns top-3 most relevant chunks from knowledge/ as context for the LLM
```

The chunking strategy splits docs by markdown headers so each chunk is one self-contained
concept (e.g. "DPO modes" or "Array partitioning"), not arbitrary character counts.

### Component 3: Code Advisor (`llm/code_advisor.py`)

Takes the **code profile + retrieved knowledge chunks + current violations** and asks an
LLM to suggest specific code changes. Suggestions are categorised:

| Category | Example |
|----------|---------|
| `pragma_insert` | "Add `#pragma HLS PIPELINE II=1` at line 47" |
| `pragma_modify` | "Change `II=4` to `II=1` on line 63 (needs array partitioning)" |
| `coding_style` | "Replace `std::vector` with fixed-size array — not synthesizable" |
| `restructure` | "Split nested 2D loop at line 89 into two 1D loops" |

Every suggestion includes target line number, expected impact on latency/area/power, and
priority. **All advisory only** — never auto-applied. Goes through the same LLM-as-judge
cross-check as the synthesis parameter advisor.

### Why this matters

A pure-knob tuner has a ceiling: if your code doesn't expose the right structure, no amount
of `unroll`/`pipeline` tuning will hit the latency target. The code advisor identifies
*structural* fixes (missing pragmas, wrong loop order, unpartitioned arrays) that the
knob tuner can't reach.

---

## Phase 6: Multi-Objective Pareto Front

By default the loop tracks a Pareto front rather than optimizing a single score.

### How convergence works

1. Each APPROVED design is added to the Pareto front
2. A design is removed if a newer design dominates it on all three objectives
3. The loop stops when the front size is **stable for 3 consecutive updates** — meaning no better tradeoffs are being found

### Selecting a winner

After the loop finishes, the best point on the front is picked using weights:

```powershell
# Latency is most important
python -m aiv_dse.run_loop --strategy bayesian --weight-latency 0.7 --weight-area 0.15 --weight-power 0.15

# All objectives equal
python -m aiv_dse.run_loop --strategy bayesian --max-iters 10 --seed 42

# Legacy: single-objective mode (collapses to one score)
python -m aiv_dse.run_loop --strategy bayesian --no-multi-objective --max-iters 5
```

---

## Phase 8: LLM Layer Upgrades

Five concrete upgrades to the LLM layer for cost, latency, and quality. Each
is opt-in via env vars or works transparently — none break the default loop
behaviour.

### 1. Prompt caching — `cache_control={"type":"ephemeral"}`

**Plain English:** the LLM's instruction prompt (~30-60 lines) used to be sent in
full on every call. Anthropic added a feature to cache it for ~5 minutes — same
prompt, no re-sending, no re-charging.

**Effect:**
- ~90% reduction in input token cost on repeated calls
- ~2x latency improvement (cached system prompt skips re-processing)

**Status:** enabled by default in all 5 advisors' Anthropic SDK paths
(`judge.py`, `constraint_advisor.py`, `synth_advisor.py`, `code_advisor.py`,
`spec_planner.py`). No config needed.

### 2. Extended thinking on the judge — reasoning before the verdict

**Plain English:** modern Claude models support an "extended thinking" mode
where they think internally for N tokens before producing the final answer.
For adversarial cross-checking (the judge), this dramatically improves catch
rate on subtle reasoning flaws — like a chess player who actually pauses to
look 5 moves ahead.

**Trade-off:** higher cost and latency per judge call. Only worth it on the
judge (where every catch saves a synthesis run), not on the proposer.

**How to enable:**
```bash
$env:AIVDSE_JUDGE_THINKING='1'                # enable extended thinking on judge
$env:AIVDSE_JUDGE_THINKING_BUDGET='2048'      # thinking budget tokens (default 2048)
python -m aiv_dse.run_loop --strategy llm --max-iters 10
```

### 3. Adversarial judge **across providers** (Claude advisor → Gemini judge)

**Plain English:** the README always claimed *"two LLMs from different providers
cross-check each other"* — but until now, the judge was using the *same*
provider as the advisor. Now it's actually wired up.

**How it works:**
- Default: when `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` are both set,
  the advisor uses Claude and the judge uses Gemini Flash 2.0
- Falls back gracefully to OpenAI if Gemini isn't configured
- Falls back to same-provider when only one API key is set (legacy behaviour)
- Explicit override via `AIVDSE_JUDGE_PROVIDER=openai|anthropic|google`

**Why it matters:** two LLMs from the same family share blind spots — they
were trained on similar data and make correlated mistakes. Two LLMs from
different families give genuine adversarial diversity. *This is the bug
fix on the project's central claim.*

### 4. Reflexion — the advisor reads past rejections

**Plain English:** when the judge rejects a proposal, that rejection used
to be thrown away after the loop fell back to Bayesian. Now it's stored
as a *lesson learned* in state, and the advisor reads all recent lessons
on its next turn:

```
## Lessons from past rejections (avoid repeating these)
- iter 3: proposed 'unroll 4->16' -- rejected because: over-correction; physics limit exceeded
- iter 5: proposed 'resource_sharing=true' -- rejected because: conflicts with unroll>4
```

**Effect:** the advisor stops repeating the same mistake every iteration.
This is the [Reflexion](https://arxiv.org/abs/2303.11366) pattern —
self-improvement by reading past failures.

**Status:** enabled by default. Lessons are capped at the last 10 to keep
prompts bounded. Backward compatible — old state files without
`lessons_learned` still load.

### 5. PRM-style judge — score each adjustment independently

**Plain English:** the standard judge returned a single yes/no on the
whole proposal. If the advisor proposed two adjustments and one was good
+ one was bad, the standard judge rejected both. **Wasteful.**

The **PRM-style judge** (Process Reward Model) scores each adjustment
*independently*:

```
Adjustment 1 (unroll 8→4): grounded, addresses violation. ACCEPT.
Adjustment 2 (resource_sharing): contradicts knowledge_chunk_3. REJECT.
Apply partial: unroll only.
```

**Effects:**
- Good adjustments aren't thrown away when bundled with bad ones
- Citation hallucinations are caught at the per-step level
- Rejection reasons become more specific, feeding Reflexion's `lessons_learned`
  with richer signal

**How to enable:**
```bash
$env:AIVDSE_USE_PRM_JUDGE='1'
python -m aiv_dse.run_loop --backend graph --strategy llm --max-iters 10
```

When the env var is unset, the loop uses the standard binary judge (legacy
behaviour preserved).

---

## Empirical Benchmark: does the LLM strategy actually help?

`scripts/benchmark.py` runs each search strategy across N random seeds on
the synthetic `DummyHLSAdapter` landscape and reports which one converges
most efficiently.

### What this proves and what it doesn't

**Proves:** Given the same synthetic design landscape, which search algorithm
finds the sweet spot fastest.

**Does NOT prove:** Anything about real Cadence Stratus / Vivado HLS
performance. For that you would need a licensed HLS tool, many real IPs,
and engineer time. That's Phase 8 (real HLS CI/CD) on the roadmap.

### Running it

```powershell
$env:PYTHONPATH='src'
python scripts/benchmark.py --runs 5 --max-iters 25
```

### Example output (5 runs/strategy, deterministic noise=0)

```
| Strategy   | Runs |  Mean  |  Std  | Avg Front | Success | Time   |
|------------|------|--------|-------|-----------|---------|--------|
| shadow     |    5 |   25.0 |   0.0 |       0.0 |      0% |   0.1s |
| bayesian   |    5 |   25.0 |   0.0 |       0.2 |     20% |   0.1s |
| llm        |    5 |   25.0 |   0.0 |       0.0 |      0% |   0.0s |
```

Read this as: **on this synthetic landscape with a 25-iteration budget,
Bayesian finds a feasible design 20% of the time; the shadow heuristic
gets stuck oscillating between values and never finds one; the LLM
strategy without an API key falls back to shadow logic** (so its line is
identical when no key is configured).

This is the kind of honest empirical evidence the README used to claim
without showing — now backed by a reproducible script.

---

## Sample Designs

Two synthetic SystemC designs are bundled, demonstrating the system isn't
tied to one design type:

| Sample | What it is | Spec | Why it's useful |
|--------|-----------|------|----------------|
| `samples/fft256_design.cpp` | 256-point FFT (radix-2, 8 stages) | `specs/ip_spec_example.txt` | Tests on a compute-heavy, butterfly-parallel design |
| `samples/fir_filter_design.cpp` | 32-tap FIR filter | `specs/ip_spec_fir.txt` | Tests on a streaming MAC-heavy design with a tighter latency budget |

Run the loop with either spec:

```powershell
python -m aiv_dse.run_loop --spec specs/ip_spec_example.txt --strategy llm
python -m aiv_dse.run_loop --spec specs/ip_spec_fir.txt --strategy bayesian
```

---

## Supporting Workflow Components

These components don't appear in the headline architecture, but they're what make the loop
production-grade rather than a toy.

### IP Spec Planner (`llm/spec_planner.py`)

Before the loop even starts, you can hand it an **IP specification** (a `.txt` or `.pdf` file
describing the design — throughput target, memory budget, interface protocol). An LLM reads
the spec and proposes:

- **Initial policy constraints** (latency / area / power thresholds derived from the spec)
- **Initial synthesis parameters** as a reasonable starting point
- **Warnings** about architectural limitations the spec implies (e.g. *"BRAM bottleneck
  at unroll > 8"*)

The plan goes through judge + HITL review before iteration begins, so a bad initial guess
gets caught immediately rather than wasting 10 synthesis runs.

```powershell
python -m aiv_dse.run_loop --spec specs/ip_spec_example.txt --sdk anthropic
```

### Human-In-The-Loop Review (`workflow/hitl.py`)

The HITL step appears at three decision points:

1. **After spec planning** — review/edit the LLM's proposed starting constraints
2. **When judge disagrees with advisor** — human breaks the tie
3. **When confidence is low** (configurable threshold) — human approves before applying

Each review presents a structured diff (proposed change → current state) and asks for
**approve / modify / reject**. The loop pauses; nothing applies until the human acts.

### Engineering Decision Record (`workflow/edr_writer.py`)

At the end of each session, an **EDR** is written documenting:
- All runs attempted (params + metrics + status)
- Every proposal made (by which strategy)
- Every judge verdict (agree / disagree + reasoning)
- Every human override
- Final converged design (if any) + the alternatives on the Pareto front

This is the audit trail. For regulated industries (automotive, medical, aerospace) where
HLS-generated hardware needs traceability, the EDR is the "why did you choose these
parameters" answer.

### Stagnation Detection (`core/stagnation.py`)

After every run, the loop checks: *are the last N runs giving essentially identical metrics
despite parameter changes?* If yes, the strategy is **stuck** — likely exploring a flat
region of the design space. Stagnation triggers:

- A judge re-evaluation with stricter scrutiny
- Optional strategy swap (e.g. shadow → bayesian)
- Notification to HITL: *"5 runs delta < 2%, suggest stopping or pivoting"*

### Convergence Detection (`core/convergence.py`)

Two convergence modes:

| Mode | Criterion |
|------|-----------|
| **Single-objective** | Weighted score stable within 2% for 3 consecutive APPROVED runs |
| **Multi-objective** | Pareto front size stable for 3 consecutive front updates |

The multi-objective mode is default — single-objective collapses three metrics into one
weighted score and was the legacy path before Phase 6.

### Automatic Constraint Relaxation (`core/constraint_relaxer.py`)

What if a constraint is *physically unreachable* — no parameter combination ever satisfies
it? Auto-relax detects this: **if the same constraint VETOes N consecutive runs**, it's
flagged as likely infeasible. The system:

- Reports the closest achievable value observed
- Optionally relaxes the threshold by a configurable percentage and retries
- Notifies HITL: *"area constraint unreachable; closest was 51,300 vs target 50,000"*

```powershell
python -m aiv_dse.run_loop --auto-relax --relax-step-pct 5 --max-relax-iters 3
```

This prevents the loop from running 20 iterations chasing an impossible target.

---

## Iteration Loop Flowchart

```mermaid
flowchart TD
    Start([Start]) --> Synthesis[Run Synthesis]
    Synthesis --> Poison{Poison?}
    Poison -- Yes --> HALT([HALT])
    Poison -- No --> Validate[Validate against policy]
    Validate --> Status{Status?}
    Status -- APPROVED --> Converged{Converged?}
    Converged -- Yes --> DONE([DONE])
    Converged -- No --> Propose
    Status -- VETO --> Stagnation{Stagnation?}
    Stagnation --> Propose[Propose next params]
    Propose --> Strategies[Shadow / Bayesian / LLM]
    Strategies --> Judge[Judge cross-check]
    Judge --> Apply[Apply selected proposal]
    Apply --> Synthesis
```

---

## LangGraph State Machine

### Graph structure

```
synthesize -> validate -> record -> check_terminal -> propose -> apply -> synthesize
                                          |
                                          v
                                         END (convergence / halt / max_iters)
```

### Node reference

| Node | What it does |
|------|-------------|
| `synthesize` | Run HLS synthesis with current params |
| `validate` | Check metrics against policy (includes poison detection) |
| `record` | Update history, Pareto tracker, CSV log |
| `check_terminal` | Check convergence, max iterations, halt conditions |
| `propose` | Run all three strategies; judge cross-checks the LLM proposal |
| `apply` | Apply the selected strategy's proposal to state |

### Backends

```powershell
$env:PYTHONPATH='src'

# LangGraph backend (recommended for production)
python -m aiv_dse.run_loop --backend graph --strategy bayesian --max-iters 10

# Direct LangGraph entry point
python -m aiv_dse.graph --strategy bayesian --max-iters 10 --seed 42

# Legacy while-loop backend (still supported)
python -m aiv_dse.run_loop --backend loop --strategy bayesian --max-iters 10
```

---

## Gradio Web UI

```powershell
pip install gradio
python app.py
# Opens at http://localhost:7860
```

Features:
- Paste a JSON report, get APPROVED / VETO / ESCALATE / HALT with violation details
- View recent run history in a table
- Pre-loaded example reports for quick testing

### Deploy to Hugging Face Spaces

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create aiv-dse --type space --space-sdk gradio
```

---

## Langfuse Observability

All LLM calls are traced. Add to `.env` to enable:

```env
AIVDSE_USE_LANGFUSE=1
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Traced functions

| Function | Description |
|----------|-------------|
| `propose_adjustments` | LLM proposes constraint changes |
| `propose_synth_params` | LLM proposes synthesis param changes |
| `judge_proposal` | Second LLM cross-checks the proposal |
| `judge_code_advisory` | Second LLM cross-checks code suggestions |
| `plan_from_spec` | LLM reads IP spec and proposes initial config |
| `advise_code_changes` | LLM suggests code-level optimizations |

Tracing is opt-in. When `AIVDSE_USE_LANGFUSE=0`, the `@observe` decorator is a no-op.

---

## Tests

```powershell
$env:PYTHONPATH='src'

# All tests (218 tests, no API key needed)
python -m pytest tests/ -v

# By stage/phase
python -m pytest tests/test_report_parser.py tests/test_validator.py tests/test_state.py -v        # Stage 1
python -m pytest tests/test_stagnation.py tests/test_llm_models.py tests/test_edr_writer.py -v    # Stage 2
python -m pytest tests/test_dummy_hls.py tests/test_loop.py -v                                     # Stage 3
python -m pytest tests/test_extended_params.py tests/test_rpt_parser.py tests/test_tcl_writer.py tests/test_csv_logger.py -v  # Phase 4
python -m pytest tests/test_pareto.py -v                                                            # Phase 6
python -m pytest tests/test_graph.py -v                                                             # Phase 7 (LangGraph)
python -m pytest tests/test_tracing.py -v                                                           # Tracing
python -m pytest tests/test_judge_provider.py tests/test_judge_thinking.py tests/test_reflexion.py tests/test_prm_judge.py -v   # Phase 8 (LLM upgrades)
python -m pytest tests/test_benchmark.py -v                                                          # Benchmark smoke
```

---

## Roadmap & Future Improvements

Items planned but not yet implemented are marked 🔜; items already shipped
are marked ✅. The current codebase covers Phases 1-8 plus LangGraph, Gradio
UI, Langfuse tracing, Pareto-front multi-objective optimization, and an
empirical benchmark script.

### High-impact LLM / agent upgrades

| # | Item | Effort | Notes | Status |
|---|------|--------|-------|--------|
| 1 | **Prompt caching** | 1 hr | Add `cache_control={"type": "ephemeral"}` to system prompts. ~90% cost reduction and ~2x latency on repeat calls | ✅ |
| 2 | **Extended thinking on judge** | 30 min | Enable reasoning mode on adversarial judge — biggest quality lever per dollar | ✅ |
| 3 | **Adversarial judge across providers** | 1 hr | Claude as advisor, Gemini (or OpenAI) as judge. The original "two LLMs from different providers" promise, now actually wired up | ✅ |
| 4 | **Reflexion / self-improvement** | ½ day | When judge rejects a proposal, append rejection reason to a `lessons_learned` log that the advisor reads next iteration | ✅ |
| 5 | **PRM-style judge** | 1 day | Score each *adjustment* independently rather than the full proposal — enables partial proposal acceptance + catches citation hallucinations | ✅ |
| 6 | **MCP server** | 1 day | Expose validate / Pareto-front / code-analyze tools over Model Context Protocol so Claude Code, Cursor, etc. can call them by natural language | 🔜 |
| 7 | **Pydantic AI migration** | 2–3 days | Replace dual LangChain + raw Anthropic paths with a single Pydantic AI agent layer. Type-safe, provider-agnostic, ~600 lines less code | 🔜 |
| 8 | **PDF parsing via multimodal LLM** | 2 hr | Drop `pdfplumber` — Claude / GPT / Gemini accept PDFs natively now | 🔜 |
| 9 | **Anthropic Batch API for golden-dataset eval** | 2 hr | 50% cost discount on non-interactive evaluation runs | 🔜 |

### RAG / knowledge layer upgrades

| # | Item | Effort | Notes | Status |
|---|------|--------|-------|--------|
| 10 | **ChromaDB drop-in** | ½ day | Replace TF-IDF with sentence-transformers + ChromaDB. Same interface, better recall | 🔜 |
| 11 | **Hybrid search (BM25 + dense)** | 1 day | Modern RAG stack: combine sparse + dense retrieval | 🔜 |
| 12 | **Cross-encoder re-ranking** | ½ day | Cohere Rerank v3 or BGE-reranker-v2 on top-k results | 🔜 |
| 13 | **Late-interaction (ColBERT) for fine-grained matching** | 1 day | Premium quality on small corpora like ours | 🔜 |

### Optimisation algorithm upgrades

| # | Item | Effort | Notes | Status |
|---|------|--------|-------|--------|
| 14 | **BoTorch for multi-objective BO** | 1–2 days | Replace Optuna NSGA-II with BoTorch's qEHVI — better empirical performance on small-budget mixed-categorical problems | 🔜 |
| 15 | **Quality-Diversity (MAP-Elites)** | 2 days | Beyond Pareto front — explicitly explore the design space's diversity | 🔜 |

### Honesty & credibility gaps

| # | Item | Effort | Notes | Status |
|---|------|--------|-------|--------|
| 16 | **Empirical "does the LLM actually help?" benchmark** | 1 day | Compare convergence speed of shadow vs. Bayesian vs. LLM across N seeds. See `scripts/benchmark.py` and the Empirical Benchmark section above | ✅ |
| 17 | **GitHub Actions CI** | 30 min | Green badge in README, automatic test runs on every PR (`.github/workflows/test.yml`) | ✅ |
| 18 | **Diversify sample designs (FIR, matmul, CORDIC)** | 1 hr | Added a 32-tap FIR sample alongside FFT-256. Matmul and CORDIC remain planned | ✅ |
| 20 | **"Known Limitations" README section** | 1 hr | Honesty about where the system fails reads as engineering maturity | 🔜 |
| 21 | **Cost/token tracking** | 2 hr | Expected of production LLM apps — display in Gradio UI | 🔜 |
| 22 | **Interactive Pareto visualisation in Gradio** | ½ day | Plotly 3D scatter; the matplotlib code already exists in `core/visualize.py` | 🔜 |

### Architecture refactors

| # | Item | Effort | Notes | Status |
|---|------|--------|-------|--------|
| 23 | **Knob registry (YAML-driven)** | 1 day | Currently 11 knobs are hardcoded across 7 files. A registry collapses adding a knob to one YAML entry | 🔜 |
| 24 | **Async / parallel synthesis runs** | 1 day | Phase 7 on the original roadmap. Optuna supports parallel evaluation | 🔜 |

---

## Privacy

- No LangSmith or external telemetry
- Prompts contain only synthetic data (metrics + policy thresholds)
- All logging is local opt-in (`AIVDSE_LOG_LLM_IO=1` writes to `out/debug_llm/`)
- Tests mock all LLM calls -- no real API calls in test suite
