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
from aiv_dse.llm.config import LLMSettings, get_llm, get_judge_settings
from aiv_dse.llm.models import (
    AdjustmentScore,
    CodeAdvisoryReport,
    CodeProfile,
    JudgeVerdict,
    PRMJudgeVerdict,
    SynthParamProposal,
    SynthesisParams,
)
from aiv_dse.tracing import observe


def _thinking_enabled() -> bool:
    """Whether extended-thinking mode is enabled on the judge (opt-in via env var)."""
    return os.getenv("AIVDSE_JUDGE_THINKING", "0") == "1"


def _thinking_budget() -> int:
    """Token budget for extended thinking on the judge."""
    return int(os.getenv("AIVDSE_JUDGE_THINKING_BUDGET", "2048"))


def _build_system_blocks(prompt: str) -> list:
    """Build system prompt as a list with prompt caching enabled.

    Prompt caching cuts ~90% off the input cost and ~2x latency on repeat calls
    by sharing the system prompt across requests within the cache TTL.
    Only applies in the direct Anthropic SDK path.
    """
    return [{
        "type": "text",
        "text": prompt,
        "cache_control": {"type": "ephemeral"},
    }]

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

    # Extended thinking + forced tool_choice is mutually exclusive.
    # When thinking is enabled, switch to auto tool_choice + instruct in system prompt.
    if _thinking_enabled():
        system_prompt = JUDGE_SYSTEM_PROMPT + "\n\nYou MUST call the judge_verdict tool to return your verdict."
        tool_choice = {"type": "auto"}
        max_tokens = 4096
        extra_kwargs = {
            "thinking": {"type": "enabled", "budget_tokens": _thinking_budget()},
            "temperature": 1.0,
        }
    else:
        system_prompt = JUDGE_SYSTEM_PROMPT
        tool_choice = {"type": "tool", "name": "judge_verdict"}
        max_tokens = 1024
        extra_kwargs = {}

    last_error = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            response = client.messages.create(
                model=settings.model_name,
                max_tokens=max_tokens,
                system=_build_system_blocks(system_prompt),
                messages=[{"role": "user", "content": context}],
                tools=[_JUDGE_TOOL_SCHEMA],
                tool_choice=tool_choice,
                **extra_kwargs,
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
    # Use a DIFFERENT provider for adversarial diversity when possible.
    # get_judge_settings() prefers an "opposite" provider (e.g. anthropic advisor
    # → google judge) when that provider's API key is configured. Falls back to
    # same provider when only one is configured.
    judge_settings = get_judge_settings(settings)

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
    # Use a different provider for adversarial diversity (same logic as judge_proposal)
    judge_settings = get_judge_settings(settings)

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
            system=_build_system_blocks(CODE_JUDGE_SYSTEM_PROMPT),
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


# ---------------------------------------------------------------------------
# Phase B5: PRM-style judge (per-adjustment scoring)
# ---------------------------------------------------------------------------

PRM_JUDGE_SYSTEM_PROMPT = """\
You are reviewing another AI's proposal for synthesis parameter changes,
scoring EACH ADJUSTMENT INDEPENDENTLY rather than the proposal as a whole.

For each adjustment in the proposal:
1. Decide whether THAT SPECIFIC adjustment should be applied (accept=true/false).
2. Verify the cited run_id and metric values match the actual run history
   (citation_verified=true if accurate, false if hallucinated).
3. Give a brief reasoning for the per-step decision.

This step-by-step scoring lets the loop apply the GOOD parts of a proposal
and reject the BAD parts -- unlike a single yes/no verdict that throws away
the whole proposal when only one adjustment is wrong.

Return a verdict scoring every adjustment listed in the proposal context.
"""


_PRM_JUDGE_TOOL_SCHEMA = {
    "name": "prm_judge_verdict",
    "description": "Score each adjustment in the proposal independently.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "param_name": {"type": "string"},
                        "accept": {"type": "boolean"},
                        "reasoning": {"type": "string"},
                        "citation_verified": {"type": "boolean"},
                    },
                    "required": ["param_name", "accept", "reasoning", "citation_verified"],
                },
            },
            "overall_reasoning": {"type": "string"},
            "overall_confidence": {"type": "number"},
        },
        "required": ["scores", "overall_reasoning", "overall_confidence"],
    },
}


def _prm_judge_via_anthropic(
    context: str,
    settings: LLMSettings,
) -> PRMJudgeVerdict:
    import anthropic

    client = anthropic.Anthropic()

    last_error = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            response = client.messages.create(
                model=settings.model_name,
                max_tokens=2048,
                system=_build_system_blocks(PRM_JUDGE_SYSTEM_PROMPT),
                messages=[{"role": "user", "content": context}],
                tools=[_PRM_JUDGE_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "prm_judge_verdict"},
            )

            tool_block = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_block = block
                    break

            if tool_block is None:
                raise ValueError("No tool_use block in Anthropic response")

            return PRMJudgeVerdict.model_validate(tool_block.input)

        except Exception as e:
            last_error = e
            if attempt < settings.max_retries:
                continue

    raise RuntimeError(
        f"Anthropic PRM judge failed after {settings.max_retries} attempts: {last_error}"
    )


def _prm_judge_via_langchain(
    context: str,
    settings: LLMSettings,
) -> PRMJudgeVerdict:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(PRMJudgeVerdict)

    messages = [
        SystemMessage(content=PRM_JUDGE_SYSTEM_PROMPT),
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
        f"LangChain PRM judge failed after {settings.max_retries} attempts: {last_error}"
    )


@observe(name="prm_judge_proposal")
def prm_judge_proposal(
    proposal: SynthParamProposal,
    policy: Dict[str, Any],
    state: Dict[str, Any],
    result: ValidationResult,
    current_params: SynthesisParams,
    settings: LLMSettings,
) -> PRMJudgeVerdict:
    """PRM-style judge: score each adjustment independently.

    Unlike judge_proposal (binary yes/no on the whole proposal), this returns
    a per-adjustment score so the loop can apply partial proposals --
    keeping good adjustments and dropping bad ones.

    Uses a different provider than the advisor when possible (same logic
    as the standard judge).
    """
    judge_settings = get_judge_settings(settings)

    context = _format_judge_context(
        proposal, policy, state, result, current_params
    )

    if judge_settings.sdk_mode == "anthropic":
        verdict = _prm_judge_via_anthropic(context, judge_settings)
    else:
        verdict = _prm_judge_via_langchain(context, judge_settings)

    if settings.log_llm_io:
        os.makedirs(settings.log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(settings.log_dir, f"prm_judge_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": ts,
                "verdict": verdict.model_dump(),
            }, f, indent=2)

    return verdict


def apply_prm_verdict(
    proposal: SynthParamProposal,
    verdict: PRMJudgeVerdict,
) -> SynthParamProposal:
    """Build a new SynthParamProposal containing only the accepted adjustments.

    Used by the loop when PRM mode is enabled: instead of rejecting a whole
    proposal on any disagreement, apply only the per-step-accepted adjustments.
    """
    accepted_names = set(verdict.accepted_param_names())
    filtered = [a for a in proposal.adjustments if a.param_name in accepted_names]

    return SynthParamProposal(
        adjustments=filtered,
        overall_reasoning=(
            f"Filtered by PRM judge: kept {len(filtered)}/{len(proposal.adjustments)} "
            f"adjustments. Original reasoning: {proposal.overall_reasoning}"
        ),
        confidence=min(proposal.confidence, verdict.overall_confidence),
        cited_runs=proposal.cited_runs,
    )
