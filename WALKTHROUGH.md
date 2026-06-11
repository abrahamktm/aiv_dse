# AIV-DSE Walkthrough — Run It in 5 Minutes

A step-by-step tour from clone to running synthesis to seeing AI-driven
optimisation in action. No API keys needed for the main demo — everything below
runs against a synthetic HLS simulator so you can see the full loop with zero
external dependencies.

> **Audience:** anyone who's read the README opening and wants to actually try
> the thing.

---

## What you'll see

By the end of this walkthrough you will have:

1. ✅ Installed the project locally
2. ✅ Watched an AI agent run 5–10 synthesis experiments in a loop
3. ✅ Inspected the output (state file, CSV log, Pareto front)
4. ✅ Compared three different search strategies head-to-head
5. ✅ Optionally — added an LLM and watched two AIs cross-check each other
6. ✅ Opened the Gradio web UI

Estimated time: **5 minutes** for steps 1–4. Add 5 more for the optional LLM
section.

---

## Prerequisites

- **Python 3.12** (`python --version` should report 3.12.x)
- **git**
- **PowerShell** on Windows, or any POSIX shell on macOS/Linux — the commands
  below show the PowerShell flavour. Bash equivalents are noted where they differ.

No API keys required for steps 1–6. An `ANTHROPIC_API_KEY` is needed only for
step 7 (optional LLM strategy).

---

## Step 1 — Clone and install

```powershell
git clone https://github.com/abrahamktm/aiv_dse.git
cd aiv_dse
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

That's it. The `aiv_dse` package is now importable from `src/`.

---

## Step 2 — Sanity check (run the tests)

```powershell
$env:PYTHONPATH='src'                # macOS/Linux: export PYTHONPATH=src
python -m pytest tests/ -q
```

**Expected output:**

```
......................................................................... [ 33%]
......................................................................... [ 66%]
.........................................                                  [100%]
219 passed, 3 skipped in 8.45s
```

> 3 skips are matplotlib display tests (expected). If you see any failures,
> check Python version (must be 3.12) and try `pip install --upgrade -r requirements.txt`.

---

## Step 3 — Your first run (no API key, synthetic HLS)

This runs the closed-loop exploration against `DummyHLSAdapter`, a synthetic
simulator that mimics real HLS physics (loop unrolling reduces latency but
inflates area, etc.). The **Bayesian** strategy uses Optuna's TPE sampler to
pick the next experiment intelligently.

```powershell
python -m aiv_dse.run_loop `
    --backend graph `
    --strategy bayesian `
    --max-iters 8 `
    --seed 42
```

(Bash: replace `` ` `` line continuation with `\`.)

**What you'll see:** a stream of iteration output, something like —

```
[iter 1] params: unroll=4, pipeline=2, clock_period=10.0
         metrics: latency=10200ns area=43000 power=280mW
         status: VETO (latency exceeds 10000ns budget)

[iter 2] params: unroll=8, pipeline=2, clock_period=10.0
         metrics: latency=8500ns area=51000 power=320mW
         status: VETO (area exceeds 50000 budget)

[iter 3] params: unroll=2, pipeline=2, clock_period=10.0
         metrics: latency=8165ns area=39000 power=212mW
         status: APPROVED ✓

[CONVERGED] Bayesian found the sweet spot in 3 iterations.
Pareto front size: 1
```

Bayesian found the "sweet spot" (`unroll=2, pipeline=2`) in just 3 iterations.
The seed value (`42`) makes the run reproducible — try it again and you'll get
the identical trajectory.

---

## Step 4 — What just happened?

The loop ran one iteration of this state machine for each experiment:

```
┌──────────────┐   ┌──────────────┐   ┌────────────────┐
│  Synthesize  │──▶│   Validate   │──▶│     Record     │
│ (DummyHLS)   │   │ (policy YAML)│   │ (state, CSV,   │
└──────────────┘   └──────────────┘   │  Pareto front) │
        ▲                              └────────┬───────┘
        │                                       │
        │          ┌────────────────┐           │
        └──────────│   Apply new    │◀──────────│
                   │     params     │           │
                   └────────────────┘           ▼
                                       ┌────────────────┐
                                       │ Propose params │
                                       │   (Bayesian)   │
                                       └────────────────┘
```

**Files produced:**

| File | What's in it |
|---|---|
| `state.json` | Last 3 runs (rolling window). What the validator uses to compute deltas; what the LLM would read for working context. |
| `out/runs.csv` | Every run ever, append-only. The durable archive — all 11 synthesis knobs + 3 metrics + status per row. |
| Console output | The human-readable trace |

Open `out/runs.csv` in Excel / pandas — you'll see one row per iteration with
every knob value, every metric, and the validator's verdict. This file is the
ground truth; `state.json` is just the loop's working memory.

---

## Step 5 — Compare strategies head-to-head

The project ships with three search strategies. You just ran Bayesian. Try the
"shadow" strategy — a deliberately dumb heuristic that adjusts unroll by 1 at a
time. It exists as a **benchmark** to prove the smarter strategies actually add
value.

```powershell
python -m aiv_dse.run_loop --backend graph --strategy shadow --max-iters 10
```

Shadow will plod through the search space linearly. Note how many more
iterations it takes (often 8–10) to find an approved point vs Bayesian's 3.

There's also a reproducible benchmark script that compares all three across
multiple seeds:

```powershell
python scripts/benchmark.py
```

**Expected output:**

```
Strategy          Avg iters    Success rate    Pareto front
shadow            ~9.5         0%              -
bayesian          ~3.8         20%             1.8 points
llm (fallback)    ~5.2         0%              -
```

(The LLM strategy with no API key falls back to Bayesian under the hood, hence
the 0% success — see step 7 to enable the real LLM path.)

---

## Step 6 — Open the Gradio web UI

```powershell
pip install gradio
python app.py
```

A browser tab opens at `http://localhost:7860`. You can:

- Validate uploaded synthesis reports against a policy
- Browse run history across past invocations
- See the Pareto front visualised

---

## Step 7 — Optional: bring in an actual LLM

This step needs an API key. The default configuration uses **Claude as the
advisor** and **Gemini as the judge** — two LLMs from different providers,
cross-checking each other. If only one key is available, the system falls back
gracefully.

Create `.env` in the repo root:

```env
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...               # optional — enables cross-provider judge
```

Then run:

```powershell
python -m aiv_dse.run_loop --strategy llm --sdk anthropic --max-iters 5
```

**What's different from steps 3–5:** every proposal now goes through *two*
LLMs. The advisor proposes a parameter change with a reasoning trace; the
judge cross-examines the reasoning. If they disagree, the proposal is
escalated for human review (the `Human-in-the-Loop` step in `workflow/hitl.py`).

Look for the `[JUDGE]` lines in the output — they show the second LLM's
verdict on the first's proposal.

---

## Step 8 — Where to look next

| If you want to... | Read |
|---|---|
| Understand the architecture decisions | README `## Architecture Deep Dive` |
| See every file's purpose | README `## What's in this repo` |
| Add a new synthesis knob | `src/aiv_dse/llm/models.py` (`SynthesisParams`) and the YAML in `policy/` |
| Adjust the constraint targets | `policy/default_policy.yaml` |
| Use your own IP spec | drop a `.txt` or `.pdf` in `specs/` and pass `--spec specs/your_file.txt` |
| Deploy to Hugging Face Spaces | README `## Gradio Web UI` → "Deploy" section |
| Add a new LLM provider | `src/aiv_dse/llm/config.py` |

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: aiv_dse` | Forgot `$env:PYTHONPATH='src'` (or `export PYTHONPATH=src` on POSIX) |
| All runs return `VETO` forever | Constraints in `policy/default_policy.yaml` are too tight for the `DummyHLSAdapter` physics — relax or use `--auto-relax` flag |
| `ImportError: gradio` when running `app.py` | `pip install gradio` (intentionally optional; not in `requirements.txt`) |
| Tests fail on a fresh clone | `pip install --upgrade -r requirements.txt` — your local cache may have stale deps |

---

That's the whole flow: clone → install → run → inspect → compare → (optionally
add LLM) → open UI. Total wall-clock time on first try: ~5 minutes.
