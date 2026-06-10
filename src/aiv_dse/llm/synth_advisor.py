"""LLM-powered synthesis parameter advisor.

Proposes synthesis knob changes (unroll, pipeline, clock, partition)
based on run history and violations. Policy thresholds are FIXED (spec).
Only the knobs are tuned.

Two implementations side by side (same pattern as constraint_advisor.py):
  1. _propose_via_langchain()  -- LangChain .with_structured_output()
  2. _propose_via_anthropic()  -- Direct Anthropic SDK with tool_use
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aiv_dse.core.state import METRIC_FIELDS
from aiv_dse.core.validator import ValidationResult
from aiv_dse.llm.config import LLMSettings, get_llm
from aiv_dse.llm.models import KnowledgeChunk, SynthParamProposal, SynthesisParams
from aiv_dse.tracing import observe

SYSTEM_PROMPT = """\
You are a hardware synthesis parameter tuning advisor. Your job is to propose
changes to synthesis KNOBS (unroll_factor, pipeline_depth, clock_period_ns,
array_partition_factor) to move the design toward feasibility.

Key tradeoffs:
  - Higher unroll_factor: lower latency, higher area and power
  - Higher pipeline_depth: lower latency, moderately higher area
  - Lower clock_period_ns (faster clock): lower latency, higher power
  - Higher array_partition_factor: lower latency, higher area and power

Rules:
1. Policy constraints (latency, area, power thresholds) are FIXED. Do not
   suggest changing them. Only tune the synthesis parameters.
2. Change at most 2 parameters per iteration for observability.
3. MUST cite actual run_ids and metric values from the history.
4. If latency is violated: consider increasing parallelism (unroll, pipeline).
5. If area or power is violated: consider decreasing parallelism.
6. If multiple constraints conflict: find a balanced tradeoff.
7. Your response must conform exactly to the requested JSON schema.
"""


def _format_synth_context(
    policy: Dict[str, Any],
    state: Dict[str, Any],
    result: ValidationResult,
    current_params: SynthesisParams,
    spec_summary: Optional[str] = None,
    knowledge_chunks: Optional[List["KnowledgeChunk"]] = None,
) -> str:
    """Build prompt context with current params + constraints + history."""
    sections = []

    # IP Specification summary (if available)
    if spec_summary:
        sections.append("## IP Specification Summary")
        sections.append(spec_summary)
        sections.append("")

    # Current synthesis parameters
    sections.append("## Current synthesis parameters")
    p = current_params.model_dump()
    for k, v in p.items():
        sections.append(f"- {k}: {v}")

    # Policy constraints (FIXED)
    sections.append("")
    sections.append("## Policy constraints (FIXED -- do not change)")
    for c in policy.get("constraints", []):
        sections.append(
            f"- {c['id']}: max {c['max']} {c['field']} "
            f"({c.get('severity', 'WARNING')})"
        )

    # Run history
    history = state.get("history", [])
    if history:
        sections.append("")
        sections.append(f"## Run history (last {len(history)})")
        for entry in history:
            m = entry.get("metrics", {})
            sp = entry.get("synth_params", {})
            metric_parts = [f"{f}={m.get(f)}" for f in METRIC_FIELDS if m.get(f) is not None]
            param_parts = [f"{k}={v}" for k, v in sp.items()] if sp else []
            line = f"{entry['run_id']}: {', '.join(metric_parts)} -> {entry['status']}"
            if param_parts:
                line += f" (params: {', '.join(param_parts)})"
            sections.append(line)

    # Latest violations
    if result.violations:
        sections.append("")
        sections.append("## Latest violations")
        for v in result.violations:
            threshold = v["threshold"]
            observed = v["observed"]
            pct_over = ((observed - threshold) / threshold) * 100
            sections.append(
                f"- {v['constraint_id']}: {observed} > {threshold} "
                f"({pct_over:.0f}% over, {v['severity']})"
            )

    # Reflexion: lessons from past judge rejections
    # The advisor reads these so it doesn't repeat the same mistakes that
    # the judge previously rejected.
    lessons = state.get("lessons_learned", [])
    if lessons:
        sections.append("")
        sections.append("## Lessons from past rejections (avoid repeating these)")
        for l in lessons:
            sections.append(
                f"- iter {l['iteration']}: proposed '{l['proposed_change']}' "
                f"-- rejected because: {l['rejection_reason']}"
            )

    # Domain knowledge (Phase 5 RAG)
    if knowledge_chunks:
        sections.append("")
        sections.append("## Domain Knowledge")
        for chunk in knowledge_chunks:
            sections.append(f"[Source: {chunk.source}]")
            sections.append(chunk.text)
            sections.append("")

    return "\n".join(sections)


def _log_io(settings: LLMSettings, prompt: str, response: str) -> None:
    os.makedirs(settings.log_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(settings.log_dir, f"synth_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "prompt": prompt, "response": response}, f, indent=2)


# ---------------------------------------------------------------------------
# Path 1: LangChain
# ---------------------------------------------------------------------------
def _propose_via_langchain(
    context: str,
    settings: LLMSettings,
) -> SynthParamProposal:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(SynthParamProposal)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    last_error = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            return structured_llm.invoke(messages)
        except Exception as e:
            last_error = e
            if attempt < settings.max_retries:
                continue

    raise RuntimeError(
        f"LangChain synth_advisor failed after {settings.max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Path 2: Direct Anthropic SDK
# ---------------------------------------------------------------------------
_SYNTH_TOOL_SCHEMA = {
    "name": "propose_synth_params",
    "description": "Propose synthesis parameter changes based on run history.",
    "input_schema": {
        "type": "object",
        "properties": {
            "adjustments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "param_name": {"type": "string"},
                        "current_value": {"type": "number"},
                        "proposed_value": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["param_name", "current_value", "proposed_value", "reasoning"],
                },
            },
            "overall_reasoning": {"type": "string"},
            "confidence": {"type": "number"},
            "cited_runs": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["adjustments", "overall_reasoning", "confidence", "cited_runs"],
    },
}


def _propose_via_anthropic(
    context: str,
    settings: LLMSettings,
) -> SynthParamProposal:
    import anthropic

    client = anthropic.Anthropic()

    last_error = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            response = client.messages.create(
                model=settings.model_name,
                max_tokens=1024,
                # Prompt caching: ~90% input-cost reduction on repeats
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": context}],
                tools=[_SYNTH_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "propose_synth_params"},
            )

            tool_block = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_block = block
                    break

            if tool_block is None:
                raise ValueError("No tool_use block in Anthropic response")

            return SynthParamProposal.model_validate(tool_block.input)

        except Exception as e:
            last_error = e
            if attempt < settings.max_retries:
                continue

    raise RuntimeError(
        f"Anthropic synth_advisor failed after {settings.max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@observe(name="synth_advisor")
def propose_synth_params(
    policy: Dict[str, Any],
    state: Dict[str, Any],
    result: ValidationResult,
    current_params: SynthesisParams,
    settings: LLMSettings,
    spec_summary: Optional[str] = None,
    knowledge_chunks: Optional[List[KnowledgeChunk]] = None,
) -> SynthParamProposal:
    """Propose synthesis parameter changes using the configured LLM.

    Args:
        policy:           Parsed policy YAML dict.
        state:            Current state dict with run history.
        result:           Latest ValidationResult.
        current_params:   Current synthesis knobs.
        settings:         LLM configuration.
        spec_summary:     Optional IP spec summary for domain context.
        knowledge_chunks: Retrieved domain knowledge chunks (optional).

    Returns:
        SynthParamProposal with adjustments, reasoning, confidence, cited_runs.
    """
    context = _format_synth_context(
        policy, state, result, current_params, spec_summary, knowledge_chunks
    )

    if settings.sdk_mode == "anthropic":
        proposal = _propose_via_anthropic(context, settings)
    else:
        proposal = _propose_via_langchain(context, settings)

    if settings.log_llm_io:
        _log_io(settings, context, proposal.model_dump_json(indent=2))

    return proposal
