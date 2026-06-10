"""Pydantic models for LLM structured output.

ConstraintAdjustment and LLMProposal define the schema the LLM must
return. extra="forbid" ensures no unexpected fields leak through.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class ConstraintAdjustment(BaseModel):
    """A single proposed change to a policy constraint threshold."""
    model_config = {"extra": "forbid"}

    constraint_id: str = Field(description="Constraint id from policy, e.g. 'latency'")
    current_max: float = Field(description="Current threshold value")
    proposed_max: float = Field(description="Proposed new threshold value")
    reasoning: str = Field(description="Justification citing run_id and metric values")


class LLMProposal(BaseModel):
    """Complete LLM proposal for constraint adjustments."""
    model_config = {"extra": "forbid"}

    adjustments: List[ConstraintAdjustment] = Field(
        description="List of proposed constraint changes"
    )
    overall_reasoning: str = Field(
        description="Summary reasoning citing run history"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in this proposal (0.0-1.0)"
    )
    cited_runs: List[str] = Field(
        description="Run IDs referenced in reasoning, e.g. ['RUN-001', 'RUN-002']"
    )

    @field_validator("cited_runs")
    @classmethod
    def cited_runs_must_be_nonempty(cls, v):
        if len(v) == 0:
            raise ValueError("cited_runs must contain at least one run_id")
        return v


# ---------------------------------------------------------------------------
# Stage 3: Synthesis parameter models (closed-loop)
# ---------------------------------------------------------------------------

class SynthesisParams(BaseModel):
    """Synthesis knobs that an HLS tool accepts.

    These are the design variables the loop iterates on.
    Policy thresholds (latency max, area max, power max) are the SPEC
    and stay fixed. These parameters are the KNOBS you turn.
    """
    model_config = {"extra": "forbid"}

    # --- Phase 3 knobs (original) ---
    unroll_factor: int = Field(default=4, ge=1, le=64,
        description="Loop unrolling factor (1=none, higher=more parallelism)")
    pipeline_depth: int = Field(default=1, ge=1, le=16,
        description="Pipeline initiation interval target")
    clock_period_ns: float = Field(default=10.0, gt=0.0, le=100.0,
        description="Target clock period in nanoseconds")
    array_partition_factor: int = Field(default=1, ge=1, le=32,
        description="Array partitioning factor (1=no partition)")

    # --- Phase 4 knobs (HLS directives) ---
    clock_slack_ns: float = Field(default=0.0, ge=-5.0, le=50.0,
        description="Clock slack in ns (negative=tighter timing, positive=relaxed)")
    dpo_mode: str = Field(default="none",
        description="Datapath optimization: none, DPO_AUTO_ALL, DPO_AUTO_OPT, DPO_AUTO_EXPR")
    flatten: bool = Field(default=False,
        description="Flatten hierarchy for optimization")
    inline: bool = Field(default=False,
        description="Inline function calls")
    loop_merge: bool = Field(default=False,
        description="Merge adjacent loops")
    bitwidth_reduce: bool = Field(default=False,
        description="Automatic bitwidth reduction")
    resource_sharing: bool = Field(default=False,
        description="Enable resource sharing across operations")

    @field_validator("dpo_mode")
    @classmethod
    def dpo_mode_must_be_valid(cls, v):
        valid = {"none", "DPO_AUTO_ALL", "DPO_AUTO_OPT", "DPO_AUTO_EXPR"}
        if v not in valid:
            raise ValueError(f"dpo_mode must be one of {valid}, got '{v}'")
        return v


class SynthParamAdjustment(BaseModel):
    """A single proposed change to a synthesis parameter."""
    model_config = {"extra": "forbid"}

    param_name: str = Field(
        description="Parameter name, e.g. 'unroll_factor'")
    current_value: float = Field(
        description="Current value of the parameter")
    proposed_value: float = Field(
        description="Proposed new value")
    reasoning: str = Field(
        description="Justification citing run_id and metric values")


class SynthParamProposal(BaseModel):
    """Complete LLM proposal for synthesis parameter changes."""
    model_config = {"extra": "forbid"}

    adjustments: List[SynthParamAdjustment] = Field(
        description="List of proposed synthesis parameter changes")
    overall_reasoning: str = Field(
        description="Summary reasoning citing run history and tradeoffs")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in this proposal (0.0-1.0)")
    cited_runs: List[str] = Field(
        description="Run IDs referenced in reasoning")

    @field_validator("cited_runs")
    @classmethod
    def cited_runs_must_be_nonempty(cls, v):
        if len(v) == 0:
            raise ValueError("cited_runs must contain at least one run_id")
        return v


class JudgeVerdict(BaseModel):
    """LLM-as-judge cross-check verdict on a proposal."""
    model_config = {"extra": "forbid"}

    agree: bool = Field(
        description="Does the judge agree with the proposal?")
    disagreements: List[str] = Field(
        default_factory=list,
        description="Specific objections (empty if agree=True)")
    alternative_suggestion: str = Field(
        default="",
        description="What the judge would do instead (if disagree)")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Judge confidence in this verdict (0.0-1.0)")


class SpecConstraint(BaseModel):
    """A single constraint derived from an IP specification."""
    model_config = {"extra": "forbid"}

    id: str = Field(description="Constraint identifier, e.g. 'latency'")
    field: str = Field(description="Metric field name, e.g. 'latency_ns'")
    max: float = Field(description="Maximum allowed value")
    severity: str = Field(description="CRITICAL or WARNING")
    on_violation: str = Field(description="VETO or ESCALATE")
    reasoning: str = Field(description="Why this threshold, citing spec text")


class SpecPlan(BaseModel):
    """LLM-generated plan from reading an IP specification.

    The LLM reads the spec (txt or pdf), then proposes initial constraints
    and synthesis parameters tailored to that specific design. Goes through
    judge + HITL review before the loop starts.
    """
    model_config = {"extra": "forbid"}

    constraints: List[SpecConstraint] = Field(
        description="Proposed policy constraints derived from spec")
    initial_params: SynthesisParams = Field(
        description="Proposed starting synthesis parameters")
    reasoning: str = Field(
        description="Overall reasoning citing spec text")
    warnings: List[str] = Field(
        default_factory=list,
        description="Design-specific limitations from spec, "
        "e.g. 'BRAM bottleneck at unroll > 8'")


# ---------------------------------------------------------------------------
# Phase 5: Code analysis, knowledge retrieval, and code advisory models
# ---------------------------------------------------------------------------

class LoopInfo(BaseModel):
    """A loop extracted from SystemC source code."""
    model_config = {"extra": "forbid"}

    line_number: int = Field(description="Starting line number of the loop")
    loop_type: str = Field(description="'for' or 'while'")
    iteration_count: Optional[int] = Field(
        default=None,
        description="Static iteration count if determinable, else None")
    body_line_count: int = Field(default=0,
        description="Number of lines in loop body")
    has_pipeline_pragma: bool = Field(default=False)
    has_unroll_pragma: bool = Field(default=False)
    nesting_depth: int = Field(default=0,
        description="0 = top-level loop")


class ArrayInfo(BaseModel):
    """An array declaration extracted from SystemC source."""
    model_config = {"extra": "forbid"}

    line_number: int = Field(description="Line where array is declared")
    name: str = Field(description="Variable name")
    element_type: str = Field(description="Element type, e.g. 'sc_fixed<16,8>'")
    dimensions: List[int] = Field(
        description="Array dimensions, e.g. [256] or [16, 16]")
    has_partition_pragma: bool = Field(default=False)


class PragmaInfo(BaseModel):
    """An existing HLS pragma found in the source."""
    model_config = {"extra": "forbid"}

    line_number: int = Field(description="Line number of the pragma")
    directive: str = Field(
        description="Full pragma text, e.g. '#pragma HLS PIPELINE II=1'")
    category: str = Field(
        description="Category: 'pipeline', 'unroll', 'array_partition', "
        "'interface', 'inline', 'other'")


class FunctionInfo(BaseModel):
    """A function/method found in the source."""
    model_config = {"extra": "forbid"}

    line_number: int = Field(description="Line of function definition")
    name: str = Field(description="Function name")
    is_top_level: bool = Field(default=False,
        description="True if this is the HLS top function")
    calls: List[str] = Field(default_factory=list,
        description="Functions called from this one")


class CodeProfile(BaseModel):
    """Complete static analysis profile of a SystemC source file."""
    model_config = {"extra": "forbid"}

    file_path: str = Field(description="Path to the analyzed source file")
    total_lines: int = Field(description="Total lines in file")
    loops: List[LoopInfo] = Field(default_factory=list)
    arrays: List[ArrayInfo] = Field(default_factory=list)
    pragmas: List[PragmaInfo] = Field(default_factory=list)
    functions: List[FunctionInfo] = Field(default_factory=list)
    memory_access_pattern: str = Field(
        default="unknown",
        description="'sequential', 'strided', 'random', or 'unknown'")


class CodeSuggestion(BaseModel):
    """A single code-level suggestion from the LLM."""
    model_config = {"extra": "forbid"}

    category: str = Field(
        description="'pragma_insert', 'pragma_modify', 'coding_style', 'restructure'")
    target_line: int = Field(
        description="Line number the suggestion applies to (0 if file-level)")
    current_code: str = Field(
        default="",
        description="The existing code at that line (for context)")
    suggested_change: str = Field(
        description="What to do, e.g. 'Insert #pragma HLS PIPELINE II=2 before line 42'")
    reasoning: str = Field(
        description="Why this change helps, citing metrics if possible")
    expected_impact: str = Field(
        description="Expected effect, e.g. 'Reduce latency ~20%, area +5%'")
    priority: str = Field(
        default="medium",
        description="'high', 'medium', or 'low'")


class CodeAdvisoryReport(BaseModel):
    """Complete LLM advisory output for code-level improvements."""
    model_config = {"extra": "forbid"}

    suggestions: List[CodeSuggestion] = Field(
        description="Ordered list of code suggestions (highest priority first)")
    overall_assessment: str = Field(
        description="Summary of code quality and optimization opportunities")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in these suggestions")
    cited_metrics: List[str] = Field(
        default_factory=list,
        description="Metric names referenced, e.g. ['latency_ns', 'area_units']")


class KnowledgeChunk(BaseModel):
    """A retrieved chunk of domain knowledge."""
    model_config = {"extra": "forbid"}

    text: str = Field(description="The chunk content")
    source: str = Field(description="Source file or URL it came from")
    score: float = Field(default=0.0,
        description="Relevance score from retriever")
