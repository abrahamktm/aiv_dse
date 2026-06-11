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

## Scope

This document is the **design contract** (vision, rules, principles). For what's
currently shipped vs. planned, see `README.md` (feature list, commands) and
`CLAUDE.md` (file map, current priorities, dev notes). Phase numbering and
implementation status live there, not here, to keep this document evergreen.

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
