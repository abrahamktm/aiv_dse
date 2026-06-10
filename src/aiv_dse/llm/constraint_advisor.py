"""LLM-powered constraint advisor.

Proposes constraint adjustments based on run history and violations.
Two implementations side by side:

  1. _propose_via_langchain()  -- uses LangChain's .with_structured_output()
     which handles JSON schema injection, parsing, and retry internally.

  2. _propose_via_anthropic()  -- uses the Anthropic Python SDK directly.
     You build the messages list, pass a JSON schema via the `tools`
     parameter, and parse the tool_use response block yourself.

Both return an LLMProposal (Pydantic model).
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

from aiv_dse.core.validator import ValidationResult
from aiv_dse.llm.config import LLMSettings, get_llm
from aiv_dse.llm.models import LLMProposal
from aiv_dse.llm.prompt_formatter import format_context
from aiv_dse.tracing import observe

SYSTEM_PROMPT = """\
You are a hardware design space exploration advisor. Your role is to analyze
synthesis run results and propose constraint adjustments to guide the design
toward feasibility.

Rules:
1. You MUST cite actual run_ids and metric values from the run history.
2. Propose only adjustments that are supported by the data.
3. If you don't have enough data to make a confident recommendation, set
   confidence to 0.0 and explain why in overall_reasoning.
4. Never propose relaxing a constraint beyond 2x its original value.
5. Prefer tightening constraints that have headroom over relaxing violated ones.
6. Your response must conform exactly to the requested JSON schema.
"""


def _log_io(settings: LLMSettings, prompt: str, response: str) -> None:
    """Write prompt/response pair to local debug log."""
    os.makedirs(settings.log_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(settings.log_dir, f"{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": ts, "prompt": prompt, "response": response},
            f, indent=2,
        )


# ---------------------------------------------------------------------------
# Path 1: LangChain  (.with_structured_output)
# ---------------------------------------------------------------------------
def _propose_via_langchain(
    context: str,
    settings: LLMSettings,
) -> LLMProposal:
    """Use LangChain's structured output to get an LLMProposal.

    How it works:
      - get_llm() returns a ChatOpenAI or ChatAnthropic instance.
      - .with_structured_output(LLMProposal) tells LangChain to:
          a) inject the Pydantic JSON schema into the request
          b) parse the LLM's JSON response into an LLMProposal
          c) raise if parsing fails
      - We wrap this in a retry loop for transient failures.

    This is the simplest path -- one line does schema + parsing.
    The tradeoff: you don't see what LangChain sends over the wire.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(LLMProposal)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    last_error = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            proposal = structured_llm.invoke(messages)
            return proposal
        except Exception as e:
            last_error = e
            if attempt < settings.max_retries:
                continue

    raise RuntimeError(
        f"LangChain structured output failed after "
        f"{settings.max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Path 2: Direct Anthropic SDK
# ---------------------------------------------------------------------------

# The JSON schema that tells Claude what shape to return.
# This is the same schema Pydantic would generate, but we write it
# explicitly so you can see exactly what goes to the API.
_PROPOSAL_TOOL_SCHEMA = {
    "name": "propose_adjustments",
    "description": (
        "Return a structured proposal for constraint adjustments "
        "based on the run history and violations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "adjustments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "constraint_id": {
                            "type": "string",
                            "description": "Constraint id from policy, e.g. 'latency'",
                        },
                        "current_max": {
                            "type": "number",
                            "description": "Current threshold value",
                        },
                        "proposed_max": {
                            "type": "number",
                            "description": "Proposed new threshold value",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Justification citing run_id + metric values",
                        },
                    },
                    "required": ["constraint_id", "current_max", "proposed_max", "reasoning"],
                },
                "description": "List of proposed constraint changes",
            },
            "overall_reasoning": {
                "type": "string",
                "description": "Summary reasoning citing run history",
            },
            "confidence": {
                "type": "number",
                "description": "Model confidence 0.0-1.0",
            },
            "cited_runs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Run IDs referenced in reasoning",
            },
        },
        "required": ["adjustments", "overall_reasoning", "confidence", "cited_runs"],
    },
}


def _propose_via_anthropic(
    context: str,
    settings: LLMSettings,
) -> LLMProposal:
    """Use the Anthropic Python SDK directly to get an LLMProposal.

    How it works:
      1. Create an anthropic.Anthropic() client (reads ANTHROPIC_API_KEY).
      2. Call client.messages.create() with:
         - model: the model name (e.g. "claude-sonnet-4-20250514")
         - system: the system prompt
         - messages: [{"role": "user", "content": context}]
         - tools: a list of tool schemas (we define one: propose_adjustments)
         - tool_choice: {"type": "tool", "name": "propose_adjustments"}
           ^ This forces Claude to call our tool (guaranteed structured output)
      3. The response contains content blocks.  We find the one with
         type="tool_use" and extract its "input" dict.
      4. We validate that dict through our Pydantic model.

    This path is more verbose but you see every API parameter.
    """
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    last_error = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            # --- THE API CALL ---
            response = client.messages.create(
                model=settings.model_name,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": context},
                ],
                # tools= tells Claude the shape of the output we want.
                # tool_choice= forces it to call this specific tool.
                tools=[_PROPOSAL_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "propose_adjustments"},
            )

            # --- PARSE THE RESPONSE ---
            # response.content is a list of content blocks.
            # When tool_choice forces a tool call, we get a ToolUseBlock.
            tool_block = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_block = block
                    break

            if tool_block is None:
                raise ValueError("No tool_use block in Anthropic response")

            # tool_block.input is a dict matching our schema.
            # Validate it through Pydantic for safety.
            proposal = LLMProposal.model_validate(tool_block.input)
            return proposal

        except Exception as e:
            last_error = e
            if attempt < settings.max_retries:
                continue

    raise RuntimeError(
        f"Anthropic SDK call failed after "
        f"{settings.max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public API -- dispatches based on sdk_mode
# ---------------------------------------------------------------------------
@observe(name="constraint_advisor")
def propose_adjustments(
    policy: Dict[str, Any],
    state: Dict[str, Any],
    result: ValidationResult,
    settings: LLMSettings,
) -> LLMProposal:
    """Propose constraint adjustments using the configured LLM.

    Dispatches to LangChain or direct Anthropic SDK based on
    settings.sdk_mode.

    Args:
        policy:   Parsed policy YAML dict.
        state:    Current state dict with run history.
        result:   Latest ValidationResult.
        settings: LLM configuration (provider, model, sdk_mode).

    Returns:
        LLMProposal with adjustments, reasoning, confidence, cited_runs.
    """
    context = format_context(policy, state, result)

    if settings.sdk_mode == "anthropic":
        proposal = _propose_via_anthropic(context, settings)
    else:
        proposal = _propose_via_langchain(context, settings)

    if settings.log_llm_io:
        _log_io(settings, context, proposal.model_dump_json(indent=2))

    return proposal
