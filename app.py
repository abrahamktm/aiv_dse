"""Gradio UI for AIV-DSE — deploy to Hugging Face Spaces.

Run locally:
    pip install gradio
    python app.py

Deploy to HF Spaces (private or public):
    1. huggingface-cli login
    2. huggingface-cli repo create aiv-dse --type space --space-sdk gradio --private
    3. git clone https://huggingface.co/spaces/<your-username>/aiv-dse
    4. Copy files and push
"""

import json
import sys
from typing import Any, Dict, List, Optional

# Add src to path for local development
sys.path.insert(0, "src")

import gradio as gr

from aiv_dse.adapters.dummy_hls import DummyHLSAdapter
from aiv_dse.core.validator import ValidationResult, validate
from aiv_dse.core.pareto import ParetoTracker
from aiv_dse.llm.models import SynthesisParams

# Default policy
DEFAULT_POLICY = {
    "constraints": [
        {"id": "latency", "field": "latency_ns", "max": 10000, "severity": "CRITICAL", "on_violation": "VETO"},
        {"id": "area", "field": "area_units", "max": 50000, "severity": "CRITICAL", "on_violation": "VETO"},
        {"id": "power", "field": "power_mw", "max": 500, "severity": "WARNING", "on_violation": "ESCALATE"},
    ]
}

# Global state for the session
_session_history: List[Dict[str, Any]] = []
_pareto_tracker = ParetoTracker()


def reset_session() -> str:
    """Reset the exploration session."""
    global _session_history, _pareto_tracker
    _session_history = []
    _pareto_tracker = ParetoTracker()
    return "Session reset. Ready for new exploration."


def validate_report(report_json: str) -> str:
    """Parse and validate a synthesis report against policy."""
    try:
        report = json.loads(report_json)
    except json.JSONDecodeError as e:
        return f"**Error:** Invalid JSON.\n\n```\n{e}\n```"

    # Check for required fields
    required = ["latency_ns", "area_units", "power_mw"]
    missing = [f for f in required if f not in report]
    if missing:
        return f"**Error:** Missing required fields: {', '.join(missing)}"

    # Validate
    result = validate(report, DEFAULT_POLICY)
    status = result.status
    emoji = {"APPROVED": "✅", "VETO": "❌", "ESCALATE": "⚠️", "HALT": "🛑"}.get(status, "❓")

    output = f"## {emoji} {status}\n\n"
    output += "### Metrics\n"
    output += f"| Metric | Value | Threshold | Status |\n"
    output += f"|--------|-------|-----------|--------|\n"

    for c in DEFAULT_POLICY["constraints"]:
        field = c["field"]
        threshold = c["max"]
        value = report.get(field, "N/A")
        if isinstance(value, (int, float)) and value > threshold:
            row_status = "❌ Over"
        else:
            row_status = "✅ OK"
        output += f"| {field} | {value} | {threshold} | {row_status} |\n"

    if result.violations:
        output += "\n### Violations\n"
        for v in result.violations:
            pct = ((v["observed"] - v["threshold"]) / v["threshold"]) * 100
            output += f"- **{v['constraint_id']}**: {v['observed']} vs max {v['threshold']} ({pct:.0f}% over)\n"

    return output


def run_synthesis(
    unroll: int,
    pipeline: int,
    clock: float,
    partition: int,
    dpo_mode: str,
    flatten: bool,
    inline: bool,
    loop_merge: bool,
    bitwidth_reduce: bool,
    resource_sharing: bool,
) -> tuple[str, str, str]:
    """Run a synthesis iteration with given parameters."""
    global _session_history, _pareto_tracker

    # Build params
    params = SynthesisParams(
        unroll_factor=unroll,
        pipeline_depth=pipeline,
        clock_period_ns=clock,
        array_partition_factor=partition,
        dpo_mode=dpo_mode if dpo_mode != "none" else "none",
        flatten=flatten,
        inline=inline,
        loop_merge=loop_merge,
        bitwidth_reduce=bitwidth_reduce,
        resource_sharing=resource_sharing,
    )

    # Run synthesis
    adapter = DummyHLSAdapter(noise_pct=5.0)
    run_id = f"RUN-{len(_session_history) + 1:03d}"
    report = adapter.run_synthesis(params, run_id)

    # Validate
    result = validate(report, DEFAULT_POLICY)

    # Update history
    entry = {
        "run_id": run_id,
        "params": params.model_dump(),
        "metrics": {
            "latency_ns": report["latency_ns"],
            "area_units": report["area_units"],
            "power_mw": report["power_mw"],
        },
        "status": result.status,
    }
    _session_history.append(entry)

    # Update Pareto tracker
    _pareto_tracker.add_point(
        run_id=run_id,
        metrics=entry["metrics"],
        synth_params=params.model_dump(),
        status=result.status,
    )

    # Format output
    status = result.status
    emoji = {"APPROVED": "✅", "VETO": "❌", "ESCALATE": "⚠️"}.get(status, "❓")

    metrics_output = f"## {emoji} {run_id}: {status}\n\n"
    metrics_output += f"| Metric | Value |\n|--------|-------|\n"
    metrics_output += f"| Latency | {report['latency_ns']} ns |\n"
    metrics_output += f"| Area | {report['area_units']} units |\n"
    metrics_output += f"| Power | {report['power_mw']} mW |\n"

    if result.violations:
        metrics_output += "\n### Violations\n"
        for v in result.violations:
            pct = ((v["observed"] - v["threshold"]) / v["threshold"]) * 100
            metrics_output += f"- {v['constraint_id']}: {pct:.0f}% over threshold\n"

    # History table
    history_output = get_history_table()

    # Pareto summary
    pareto_output = get_pareto_summary()

    return metrics_output, history_output, pareto_output


def get_history_table() -> str:
    """Get history as a markdown table."""
    if not _session_history:
        return "No runs yet."

    output = "| Run | Status | Latency | Area | Power |\n"
    output += "|-----|--------|---------|------|-------|\n"

    for entry in _session_history[-10:]:
        m = entry["metrics"]
        status_emoji = {"APPROVED": "✅", "VETO": "❌", "ESCALATE": "⚠️"}.get(entry["status"], "❓")
        output += f"| {entry['run_id']} | {status_emoji} {entry['status']} | {m['latency_ns']} | {m['area_units']} | {m['power_mw']} |\n"

    return output


def get_pareto_summary() -> str:
    """Get Pareto front summary."""
    if _pareto_tracker.front_size == 0:
        return "No Pareto-optimal points yet."

    output = f"**Pareto Front Size:** {_pareto_tracker.front_size} points\n\n"

    if _pareto_tracker.front:
        output += "| Run | Latency | Area | Power |\n"
        output += "|-----|---------|------|-------|\n"
        for point in _pareto_tracker.front[:5]:
            m = point["metrics"]
            output += f"| {point['run_id']} | {m['latency_ns']} | {m['area_units']} | {m['power_mw']} |\n"
        if len(_pareto_tracker.front) > 5:
            output += f"\n*...and {len(_pareto_tracker.front) - 5} more points*"

    return output


def suggest_params() -> str:
    """Suggest parameters based on history."""
    if not _session_history:
        return "**Suggestion:** Start with unroll=2, pipeline=2 (the sweet spot for the default policy)."

    # Analyze last run
    last = _session_history[-1]
    status = last["status"]
    metrics = last["metrics"]
    params = last["params"]

    suggestions = []

    if status == "APPROVED":
        suggestions.append("✅ Design passes all constraints!")
        suggestions.append("Try exploring nearby points to find better tradeoffs.")
    else:
        if metrics["area_units"] > 50000:
            suggestions.append(f"⚠️ Area over budget ({metrics['area_units']} > 50000)")
            suggestions.append("→ Try: Reduce unroll_factor, enable resource_sharing or bitwidth_reduce")
        if metrics["latency_ns"] > 10000:
            suggestions.append(f"⚠️ Latency over budget ({metrics['latency_ns']} > 10000)")
            suggestions.append("→ Try: Increase unroll_factor or pipeline_depth, enable flatten or inline")
        if metrics["power_mw"] > 500:
            suggestions.append(f"⚠️ Power over budget ({metrics['power_mw']} > 500)")
            suggestions.append("→ Try: Reduce unroll_factor, enable bitwidth_reduce, use DPO_AUTO_ALL")

    if not suggestions:
        suggestions.append("Keep exploring to find the optimal design point.")

    return "\n\n".join(suggestions)


# --- Gradio Interface ---

with gr.Blocks(
    title="AIV-DSE: HLS Design Space Explorer",
) as demo:
    gr.Markdown("""
    # AIV-DSE — Agentic HLS Design Space Exploration

    An AI-powered framework for exploring hardware design tradeoffs (latency, area, power)
    with multi-objective Pareto optimization and policy-based validation.

    **Default Policy:** latency ≤ 10,000 ns | area ≤ 50,000 units | power ≤ 500 mW
    """)

    with gr.Tab("Interactive Explorer"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Synthesis Parameters")

                unroll = gr.Slider(1, 64, value=4, step=1, label="Unroll Factor")
                pipeline = gr.Slider(1, 16, value=1, step=1, label="Pipeline Depth")
                clock = gr.Slider(1.0, 100.0, value=10.0, step=0.5, label="Clock Period (ns)")
                partition = gr.Slider(1, 32, value=1, step=1, label="Array Partition Factor")

                dpo_mode = gr.Dropdown(
                    choices=["none", "DPO_AUTO_ALL", "DPO_AUTO_OPT", "DPO_AUTO_EXPR"],
                    value="none",
                    label="Datapath Optimization"
                )

                with gr.Row():
                    flatten = gr.Checkbox(label="Flatten")
                    inline = gr.Checkbox(label="Inline")
                    loop_merge = gr.Checkbox(label="Loop Merge")

                with gr.Row():
                    bitwidth_reduce = gr.Checkbox(label="Bitwidth Reduce")
                    resource_sharing = gr.Checkbox(label="Resource Sharing")

                run_btn = gr.Button("Run Synthesis", variant="primary")
                reset_btn = gr.Button("Reset Session")

            with gr.Column(scale=2):
                metrics_output = gr.Markdown(label="Results")
                suggestion_output = gr.Markdown(label="Suggestions")

        with gr.Row():
            history_output = gr.Markdown(label="Run History")
            pareto_output = gr.Markdown(label="Pareto Front")

        # Wire up buttons
        run_btn.click(
            run_synthesis,
            inputs=[unroll, pipeline, clock, partition, dpo_mode,
                    flatten, inline, loop_merge, bitwidth_reduce, resource_sharing],
            outputs=[metrics_output, history_output, pareto_output],
        ).then(
            suggest_params,
            outputs=[suggestion_output],
        )

        reset_btn.click(
            reset_session,
            outputs=[metrics_output],
        ).then(
            lambda: ("", "", ""),
            outputs=[history_output, pareto_output, suggestion_output],
        )

    with gr.Tab("Validate Report"):
        gr.Markdown("### Validate a Synthesis Report")
        gr.Markdown("Paste a JSON report with `latency_ns`, `area_units`, and `power_mw` fields.")

        report_input = gr.Textbox(
            label="Synthesis Report (JSON)",
            placeholder='{"latency_ns": 8000, "area_units": 45000, "power_mw": 180}',
            lines=6,
        )
        validate_btn = gr.Button("Validate", variant="primary")
        validate_output = gr.Markdown(label="Validation Result")

        validate_btn.click(validate_report, inputs=[report_input], outputs=[validate_output])

        gr.Examples(
            examples=[
                ['{"latency_ns": 8000, "area_units": 38000, "power_mw": 200}'],
                ['{"latency_ns": 12000, "area_units": 55000, "power_mw": 350}'],
                ['{"latency_ns": 5000, "area_units": 120000, "power_mw": 450}'],
            ],
            inputs=[report_input],
            label="Example Reports",
        )

    with gr.Tab("About"):
        gr.Markdown("""
        ## About AIV-DSE

        **AIV-DSE** (Agentic Iterative Validation - Design Space Exploration) is a framework
        for autonomously exploring hardware design tradeoffs using:

        - **LangGraph state machine** for the optimization loop
        - **Multi-objective Pareto optimization** (NSGA-II)
        - **Policy-as-code validation** (deterministic, auditable)
        - **Three exploration strategies**: Shadow heuristic, Bayesian (Optuna), LLM-guided

        ### Key Features

        - **Governance-first**: LLM proposes, deterministic validator disposes
        - **Adversarial validation**: Two LLMs cross-check each other
        - **Human-in-the-loop**: Escalation on disagreement or critical decisions
        - **No vendor lock-in**: Works with any HLS tool via adapter pattern

        ### Sweet Spot

        For the default policy, the sweet spot is:
        - `unroll_factor=2, pipeline_depth=2`
        - Results in: ~8,165 ns latency, ~39,000 area, ~212 mW power
        - Status: **APPROVED**

        See the GitHub repository for documentation, source code, and tests.
        """)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
