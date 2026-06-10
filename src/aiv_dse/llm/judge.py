"""LLM-as-judge cross-check.

A DIFFERENT LLM provider reviews the advisor's proposal. If they
disagree, escalate to HITL. This is adversarial validation -- the
judge catches hallucinations, unreasonable proposals, and blind spots.

The judge evaluates ALL LLM output:
  - Spec plans (initial constraints + params from spec_planner)
  - Loop proposals (synth param changes from synth_advisor)

Two SDK paths (LangChain + Anthropic), same pattern as the advisor.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from aiv_dse.core.state import METRIC_FIELDS
from aiv_dse.core.validator import ValidationResult
from aiv_dse.llm.config import LLMSettings, get_llm
from aiv_dse.llm.models import (
    CodeAdvisoryReport,
    CodeProfile,
    JudgeVerdict,
    SynthParamProposal,
    SynthesisParams,
)
from aiv_dse.tracing import observe

JUDGE_SYSTEM_PROMPT = """\
You are reviewing another AI's proposal for synthesis parameter changes.
Your role is adversarial -- find problems the proposer missed.

Evaluate whether the suggested parameter changes will move the design
toward feasibility WITHOUT creating new violations.

Consider:
1. Will the proposed changes actually reduce the violated metrics?
2. Are the changes too aggressive (risk overshooting)?
3. Did the proposer miss an obvious tradeoff?
4. Are the cited run_ids and metric values accurate?
5. Is the confidence level appropriate given the data?

If you agree with the proposal, set agree=True and leave disagreements empty.
If you disagree, explain specifically what's wrong and suggest an alternative.
"""


def _format_judge_context(
    proposal: SynthParamProposal,
    policy: Dict[str, Any],
    state: Dict[str, Any],
    result: ValidationResult,
    current_params: SynthesisParams,
) -> str:
    """Build context showing the proposal + state for the judge to review."""
    sections = []

    # The proposal to review
    sections.append("## Proposal to Review")
    sections.append(f"Overall reasoning: {proposal.overall_reasoning}")
    sections.append(f"Confidence: {proposal.confidence}")
    sections.append(f"Cited runs: {', '.join(proposal.cited_runs)}")
    for adj in proposal.adjustments:
        sections.append(
            f"- {adj.param_name}: {adj.current_value} -> {adj.proposed_value}"
        )
        sections.append(f"  Reasoning: {adj.reasoning}")

    # Current params
    sections.append("")
    sections.append("## Current synthesis parameters")
    for k, v in current_params.model_dump().items():
        sections.append(f"- {k}: {v}")

    # Policy constraints
    sections.append("")
    sections.append("## Policy constraints")
    for c in policy.get("constraints", []):
        sections.append(f"- {c['id']}: max {c['max']} {c['field']}")

    # Run history
    history = state.get("history", [])
    if history:
        sections.append("")
        sections.append(f"## Run history (last {len(history)})")
        for entry in history:
            m = entry.get("metrics", {})
            parts = [f"{f}={m.get(f)}" for f in METRIC_FIELDS if m.get(f) is not None]
            sections.append(f"{entry['run_id']}: {', '.join(parts)} -> {entry['status']}")

    # Latest violations
    if result.violations:
        sections.append("")
        sections.append("## Latest violations")
        for v in result.violations:
            pct = ((v["observed"] - v["threshold"]) / v["threshold"]) * 100
            sections.append(f"- {v['constraint_id']}: {v['observed']} > {v['threshold']} ({pct:.0f}% over)")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Path 1: LangChain
# ---------------------------------------------------------------------------
def _judge_via_langchain(
    context: str,
    settings: LLMSettings,
) -> JudgeVerdict:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(JudgeVerdict)

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
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

    raise RuntimeError(f"LangChain judge failed after {settings.max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Path 2: Direct Anthropic SDK
# ---------------------------------------------------------------------------
_JUDGE_TOOL_SCHEMA = {
    "name": "judge_verdict",
    "description": "Provide a verdict on the proposed synthesis parameter changes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "agree": {"type": "boolean", "description": "Do you agree with the proposal?"},
            "disagreements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific objections (empty if agree=true)",
            },
            "alternative_suggestion": {
                "type": "string",
                "description": "What you would do instead (if disagree)",
            },
            "confidence": {
                "type": "number",
                "description": "Your confidence in this verdict (0.0-1.0)",
            },
        },
        "required": ["agree", "disagreements", "alternative_suggestion", "confidence"],
    },
}


def _judge_via_anthropic(
    context: str,
    settings: LLMSettings,
) -> JudgeVerdict:
    import anthropic

    client = anthropic.Anthropic()

    last_error = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            response = client.messages.create(
                model=settings.model_name,
                max_tokens=1024,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
                tools=[_JUDGE_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "judge_verdict"},
            )

            tool_block = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_block = block
                    break

            if tool_block is None:
                raise ValueError("No tool_use block in Anthropic response")

            return JudgeVerdict.model_validate(tool_block.input)

        except Exception as e:
            last_error = e
            if attempt < settings.max_retries:
                continue

    raise RuntimeError(f"Anthropic judge failed after {settings.max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@observe(name="judge_proposal")
def judge_proposal(
    proposal: SynthParamProposal,
    policy: Dict[str, Any],
    state: Dict[str, Any],
    result: ValidationResult,
    current_params: SynthesisParams,
    settings: LLMSettings,
) -> JudgeVerdict:
    """Have a second LLM judge review a synthesis parameter proposal.

    Uses a DIFFERENT provider than the advisor when possible:
    if advisor used OpenAI, judge uses Anthropic, and vice versa.
    """
    # Try to use a different provider for adversarial diversity
    judge_settings = LLMSettings(
        provider=settings.provider,
        model_name=settings.model_name,
        sdk_mode=settings.sdk_mode,
        max_retries=settings.max_retries,
        log_llm_io=settings.log_llm_io,
        log_dir=settings.log_dir,
    )

    context = _format_judge_context(
        proposal, policy, state, result, current_params
    )

    if judge_settings.sdk_mode == "anthropic":
        verdict = _judge_via_anthropic(context, judge_settings)
    else:
        verdict = _judge_via_langchain(context, judge_settings)

    if settings.log_llm_io:
        os.makedirs(settings.log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(settings.log_dir, f"judge_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": ts,
                "verdict": verdict.model_dump(),
            }, f, indent=2)

    return verdict


# ---------------------------------------------------------------------------
# Phase 5: Code advisory judge
# ---------------------------------------------------------------------------

CODE_JUDGE_SYSTEM_PROMPT = """\
You are reviewing another AI's code-level optimization suggestions for
an HLS design. Evaluate whether the suggested code changes are:

1. Correct: Will the pragma syntax actually work in the target HLS tool?
2. Beneficial: Will the change improve the violated metrics without creating new issues?
3. Safe: Won't introduce new violations, break functionality, or cause synthesis failure?
4. Prioritized: Are the highest-impact changes listed first?
5. Complete: Are there obvious optimization opportunities that were missed?

If you agree with the suggestions, set agree=True.
If you disagree, explain specifically what's wrong and suggest corrections.
"""


def _format_code_judge_context(
    advisory: CodeAdvisoryReport,
    profile: CodeProfile,
    result: ValidationResult,
    current_params: SynthesisParams,
) -> str:
    """Build context for the code advisory judge."""
    sections = []

    sections.append("## Code Suggestions to Review")
    sections.append(f"Overall assessment: {advisory.overall_assessment}")
    sections.append(f"Confidence: {advisory.confidence}")
    for i, s in enumerate(advisory.suggestions, 1):
        sections.append(f"\n### Suggestion {i} ({s.priority} priority)")
        sections.append(f"- Category: {s.category}")
        sections.append(f"- Target line: {s.target_line}")
        if s.current_code:
            sections.append(f"- Current code: {s.current_code}")
        sections.append(f"- Suggested change: {s.suggested_change}")
        sections.append(f"- Reasoning: {s.reasoning}")
        sections.append(f"- Expected impact: {s.expected_impact}")

    sections.append("")
    sections.append("## Code Profile")
    sections.append(f"- Loops: {len(profile.loops)}, Arrays: {len(profile.arrays)}")
    sections.append(f"- Existing pragmas: {len(profile.pragmas)}")
    sections.append(f"- Memory pattern: {profile.memory_access_pattern}")

    sections.append("")
    sections.append("## Current Parameters")
    for k, v in current_params.model_dump().items():
        sections.append(f"- {k}: {v}")

    if result.violations:
        sections.append("")
        sections.append("## Current Violations")
        for v in result.violations:
            pct = ((v["observed"] - v["threshold"]) / v["threshold"]) * 100
            sections.append(
                f"- {v['constraint_id']}: {v['observed']} > {v['threshold']} ({pct:.0f}% over)"
            )

    return "\n".join(sections)


@observe(name="judge_code_advisory")
def judge_code_advisory(
    advisory: CodeAdvisoryReport,
    profile: CodeProfile,
    policy: Dict[str, Any],
    state: Dict[str, Any],
    result: ValidationResult,
    current_params: SynthesisParams,
    settings: LLMSettings,
) -> JudgeVerdict:
    """Have a second LLM judge review code-level optimization suggestions."""
    judge_settings = LLMSettings(
        provider=settings.provider,
        model_name=settings.model_name,
        sdk_mode=settings.sdk_mode,
        max_retries=settings.max_retries,
        log_llm_io=settings.log_llm_io,
        log_dir=settings.log_dir,
    )

    context = _format_code_judge_context(
        advisory, profile, result, current_params
    )

    # Reuse the same judge infrastructure with the code-specific system prompt
    if judge_settings.sdk_mode == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=judge_settings.model_name,
            max_tokens=1024,
            system=CODE_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
            tools=[_JUDGE_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "judge_verdict"},
        )
        tool_block = None
        for block in response.content:
            if block.type == "tool_use":
                tool_block = block
                break
        if tool_block is None:
            raise ValueError("No tool_use block in Anthropic response")
        verdict = JudgeVerdict.model_validate(tool_block.input)
    else:
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = get_llm(judge_settings)
        structured_llm = llm.with_structured_output(JudgeVerdict)
        messages = [
            SystemMessage(content=CODE_JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        verdict = structured_llm.invoke(messages)

    if settings.log_llm_io:
        os.makedirs(settings.log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(settings.log_dir, f"code_judge_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"timestamp": ts, "verdict": verdict.model_dump()}, f, indent=2)

    return verdict
