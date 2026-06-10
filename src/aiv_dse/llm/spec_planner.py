"""IP-spec-driven pre-loop planning.

The LLM reads an IP specification (text or PDF) and proposes:
  - Initial policy constraints (latency, area, power thresholds)
  - Initial synthesis parameters (unroll, pipeline, clock, partition)
  - IP type classification (FIR, FFT, CORDIC, etc.)
  - Warnings about design-specific limitations

The spec plan goes through judge + HITL review before the loop starts.

For PDF input: uses pdfplumber to extract text page by page.
For txt/md: reads directly.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from aiv_dse.llm.config import LLMSettings, get_llm
from aiv_dse.llm.models import SpecPlan
from aiv_dse.tracing import observe

SPEC_SYSTEM_PROMPT = """\
You are an HLS (High-Level Synthesis) architect. Read the IP specification
below and propose:

1. Appropriate design constraints (latency, area, power thresholds) with
   severity levels (CRITICAL or WARNING) and violation actions (VETO or ESCALATE).
2. Initial synthesis parameters (unroll_factor, pipeline_depth, clock_period_ns,
   array_partition_factor) as a starting point for exploration.
3. Cite specific lines or sections from the spec to justify each choice.
4. Flag any architectural limitations (e.g., "BRAM bottleneck at unroll > 8",
   "pipeline depth > 4 gives diminishing returns").

Be conservative with initial parameters -- the loop will explore from there.
"""


def load_spec(path: str) -> str:
    """Load IP spec from .txt, .md, or .pdf file. Returns plain text."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDF spec files. "
                "Install with: pip install pdfplumber"
            )
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)

    # Plain text / markdown
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Path 1: LangChain
# ---------------------------------------------------------------------------
def _plan_via_langchain(
    spec_text: str,
    settings: LLMSettings,
) -> SpecPlan:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(SpecPlan)

    messages = [
        SystemMessage(content=SPEC_SYSTEM_PROMPT),
        HumanMessage(content=f"## IP Specification\n\n{spec_text}"),
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
        f"LangChain spec_planner failed after {settings.max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Path 2: Direct Anthropic SDK
# ---------------------------------------------------------------------------
_SPEC_TOOL_SCHEMA = {
    "name": "propose_spec_plan",
    "description": "Propose constraints and initial params from IP specification.",
    "input_schema": {
        "type": "object",
        "properties": {
            "constraints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "field": {"type": "string"},
                        "max": {"type": "number"},
                        "severity": {"type": "string"},
                        "on_violation": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["id", "field", "max", "severity", "on_violation", "reasoning"],
                },
            },
            "initial_params": {
                "type": "object",
                "properties": {
                    "unroll_factor": {"type": "integer"},
                    "pipeline_depth": {"type": "integer"},
                    "clock_period_ns": {"type": "number"},
                    "array_partition_factor": {"type": "integer"},
                },
                "required": ["unroll_factor", "pipeline_depth", "clock_period_ns", "array_partition_factor"],
            },
            "reasoning": {"type": "string"},
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["constraints", "initial_params", "reasoning", "warnings"],
    },
}


def _plan_via_anthropic(
    spec_text: str,
    settings: LLMSettings,
) -> SpecPlan:
    import anthropic

    client = anthropic.Anthropic()

    last_error = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            response = client.messages.create(
                model=settings.model_name,
                max_tokens=2048,
                system=SPEC_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"## IP Specification\n\n{spec_text}"},
                ],
                tools=[_SPEC_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "propose_spec_plan"},
            )

            tool_block = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_block = block
                    break

            if tool_block is None:
                raise ValueError("No tool_use block in Anthropic response")

            return SpecPlan.model_validate(tool_block.input)

        except Exception as e:
            last_error = e
            if attempt < settings.max_retries:
                continue

    raise RuntimeError(
        f"Anthropic spec_planner failed after {settings.max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@observe(name="spec_planner")
def plan_from_spec(
    spec_text: str,
    settings: LLMSettings,
) -> SpecPlan:
    """LLM reads the IP spec and proposes constraints + initial params.

    Goes through judge + HITL review before the loop starts (handled
    by run_loop.py, not here).
    """
    if settings.sdk_mode == "anthropic":
        plan = _plan_via_anthropic(spec_text, settings)
    else:
        plan = _plan_via_langchain(spec_text, settings)

    if settings.log_llm_io:
        os.makedirs(settings.log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(settings.log_dir, f"spec_plan_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": ts,
                "plan": plan.model_dump(),
            }, f, indent=2)

    return plan
