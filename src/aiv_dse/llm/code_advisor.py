"""LLM-powered SystemC code advisor.

Analyzes source code alongside synthesis metrics and retrieved domain knowledge
to suggest pragma insertions, pragma modifications, and coding style improvements.

Suggestions are advisory only -- never auto-applied.

Two implementations side by side:
  1. _advise_via_langchain()  -- LangChain .with_structured_output()
  2. _advise_via_anthropic()  -- Direct Anthropic SDK with tool_use
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aiv_dse.core.state import METRIC_FIELDS
from aiv_dse.core.validator import ValidationResult
from aiv_dse.llm.config import LLMSettings, get_llm
from aiv_dse.llm.models import (
    CodeAdvisoryReport,
    CodeProfile,
    KnowledgeChunk,
    SynthesisParams,
)
from aiv_dse.tracing import observe


SYSTEM_PROMPT = """\
You are an HLS code optimization advisor. You analyze SystemC/C++ source code
alongside synthesis metrics to suggest code-level improvements.

Your suggestions fall into these categories:
1. pragma_insert: Add missing HLS pragmas (PIPELINE, UNROLL, ARRAY_PARTITION)
2. pragma_modify: Change existing pragma parameters (e.g., reduce II from 4 to 1)
3. coding_style: Improve code structure for HLS (fixed bounds, no dynamic alloc)
4. restructure: Refactor code (split 2D loops, extract functions, reorder ops)

Rules:
1. Reference specific line numbers from the code profile.
2. Explain the expected impact on latency, area, and power for each suggestion.
3. Order suggestions by priority (highest impact first).
4. Do not suggest changes that conflict with existing pragmas unless justified.
5. Consider the current synthesis metrics when prioritizing.
6. For array partitioning: match the partition factor to the loop unroll/pipeline.
7. For pipeline II reduction: ensure arrays accessed in the loop are partitioned.
8. Be conservative with area predictions -- pragma additions usually increase area.
9. Your response must conform exactly to the requested JSON schema.
"""


def _format_code_context(
    source_code: str,
    profile: CodeProfile,
    policy: Dict[str, Any],
    state: Dict[str, Any],
    result: ValidationResult,
    current_params: SynthesisParams,
    knowledge_chunks: Optional[List[KnowledgeChunk]] = None,
) -> str:
    """Build prompt context with source code, profile, metrics, and knowledge."""
    sections = []

    # Source code (truncated)
    lines = source_code.split("\n")
    if len(lines) > 200:
        truncated = "\n".join(lines[:200])
        sections.append(f"## Source Code (first 200 of {len(lines)} lines)")
        sections.append(truncated)
    else:
        sections.append("## Source Code")
        sections.append(source_code)

    # Code profile summary
    sections.append("")
    sections.append("## Code Profile")
    sections.append(f"- File: {profile.file_path}")
    sections.append(f"- Total lines: {profile.total_lines}")
    sections.append(f"- Memory pattern: {profile.memory_access_pattern}")

    if profile.loops:
        sections.append(f"- Loops ({len(profile.loops)}):")
        for loop in profile.loops:
            iters = f", {loop.iteration_count} iters" if loop.iteration_count else ""
            pragmas = []
            if loop.has_pipeline_pragma:
                pragmas.append("PIPELINE")
            if loop.has_unroll_pragma:
                pragmas.append("UNROLL")
            pragma_str = f" [{', '.join(pragmas)}]" if pragmas else " [no pragmas]"
            sections.append(
                f"  Line {loop.line_number}: {loop.loop_type}{iters}, "
                f"depth={loop.nesting_depth}{pragma_str}"
            )

    if profile.arrays:
        sections.append(f"- Arrays ({len(profile.arrays)}):")
        for arr in profile.arrays:
            dims = "x".join(str(d) for d in arr.dimensions)
            part = " [PARTITIONED]" if arr.has_partition_pragma else " [no partition]"
            sections.append(
                f"  Line {arr.line_number}: {arr.element_type} {arr.name}[{dims}]{part}"
            )

    if profile.pragmas:
        sections.append(f"- Existing pragmas ({len(profile.pragmas)}):")
        for p in profile.pragmas:
            sections.append(f"  Line {p.line_number}: {p.directive}")

    if profile.functions:
        sections.append(f"- Functions ({len(profile.functions)}):")
        for func in profile.functions:
            top = " [TOP]" if func.is_top_level else ""
            calls = f" -> calls: {', '.join(func.calls)}" if func.calls else ""
            sections.append(f"  Line {func.line_number}: {func.name}(){top}{calls}")

    # Current synthesis parameters
    sections.append("")
    sections.append("## Current Synthesis Parameters")
    for k, v in current_params.model_dump().items():
        sections.append(f"- {k}: {v}")

    # Policy constraints
    sections.append("")
    sections.append("## Policy Constraints")
    for c in policy.get("constraints", []):
        sections.append(f"- {c['id']}: max {c['max']} {c['field']}")

    # Latest violations
    if result.violations:
        sections.append("")
        sections.append("## Current Violations")
        for v in result.violations:
            threshold = v["threshold"]
            observed = v["observed"]
            pct_over = ((observed - threshold) / threshold) * 100
            sections.append(
                f"- {v['constraint_id']}: {observed} > {threshold} "
                f"({pct_over:.0f}% over)"
            )

    # Domain knowledge chunks
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
    path = os.path.join(settings.log_dir, f"code_advisor_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "prompt": prompt, "response": response}, f, indent=2)


# ---------------------------------------------------------------------------
# Path 1: LangChain
# ---------------------------------------------------------------------------
def _advise_via_langchain(
    context: str,
    settings: LLMSettings,
) -> CodeAdvisoryReport:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(CodeAdvisoryReport)

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
        f"LangChain code_advisor failed after {settings.max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Path 2: Direct Anthropic SDK
# ---------------------------------------------------------------------------
_CODE_TOOL_SCHEMA = {
    "name": "advise_code_changes",
    "description": "Suggest code-level HLS optimizations (pragmas, style, restructuring).",
    "input_schema": {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "target_line": {"type": "integer"},
                        "current_code": {"type": "string"},
                        "suggested_change": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "expected_impact": {"type": "string"},
                        "priority": {"type": "string"},
                    },
                    "required": [
                        "category", "target_line", "suggested_change",
                        "reasoning", "expected_impact",
                    ],
                },
            },
            "overall_assessment": {"type": "string"},
            "confidence": {"type": "number"},
            "cited_metrics": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["suggestions", "overall_assessment", "confidence"],
    },
}


def _advise_via_anthropic(
    context: str,
    settings: LLMSettings,
) -> CodeAdvisoryReport:
    import anthropic

    client = anthropic.Anthropic()

    last_error = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            response = client.messages.create(
                model=settings.model_name,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
                tools=[_CODE_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "advise_code_changes"},
            )

            tool_block = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_block = block
                    break

            if tool_block is None:
                raise ValueError("No tool_use block in Anthropic response")

            return CodeAdvisoryReport.model_validate(tool_block.input)

        except Exception as e:
            last_error = e
            if attempt < settings.max_retries:
                continue

    raise RuntimeError(
        f"Anthropic code_advisor failed after {settings.max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@observe(name="code_advisor")
def advise_code_changes(
    source_code: str,
    profile: CodeProfile,
    policy: Dict[str, Any],
    state: Dict[str, Any],
    result: ValidationResult,
    current_params: SynthesisParams,
    settings: LLMSettings,
    knowledge_chunks: Optional[List[KnowledgeChunk]] = None,
) -> CodeAdvisoryReport:
    """Analyze source code and suggest code-level HLS optimizations.

    Args:
        source_code:      Raw source code string.
        profile:          CodeProfile from static analysis.
        policy:           Parsed policy YAML dict.
        state:            Current state dict with run history.
        result:           Latest ValidationResult.
        current_params:   Current synthesis knobs.
        settings:         LLM configuration.
        knowledge_chunks: Retrieved domain knowledge (optional).

    Returns:
        CodeAdvisoryReport with suggestions, assessment, confidence.
    """
    context = _format_code_context(
        source_code, profile, policy, state, result,
        current_params, knowledge_chunks,
    )

    if settings.sdk_mode == "anthropic":
        advisory = _advise_via_anthropic(context, settings)
    else:
        advisory = _advise_via_langchain(context, settings)

    if settings.log_llm_io:
        _log_io(settings, context, advisory.model_dump_json(indent=2))

    return advisory
