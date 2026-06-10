from aiv_dse.core.validator import ValidationResult, validate, load_policy
from aiv_dse.core.state import append_result
from aiv_dse.llm.prompt_formatter import format_context


POLICY_PATH = "policy/default_policy.yaml"


def _build_state_and_result():
    """Build a state with 2 runs and a VETO result for testing."""
    policy = load_policy(POLICY_PATH)
    state = {"history": []}

    report1 = {"run_id": "RUN-001", "latency_ns": 8500, "area_units": 42000,
               "power_mw": 350, "unroll_factor": 4}
    result1 = validate(report1, policy)
    state = append_result(state, result1, report1)

    report2 = {"run_id": "RUN-002", "latency_ns": 15000, "area_units": 62000,
               "power_mw": 620, "unroll_factor": 32}
    result2 = validate(report2, policy)
    state = append_result(state, result2, report2)

    return policy, state, result2


def test_format_includes_history():
    """Formatted prompt should contain run_ids from history."""
    policy, state, result = _build_state_and_result()
    text = format_context(policy, state, result)
    assert "RUN-001" in text
    assert "RUN-002" in text


def test_format_includes_violations():
    """Formatted prompt should list constraint violations."""
    policy, state, result = _build_state_and_result()
    text = format_context(policy, state, result)
    assert "latency" in text.lower()
    assert "## Latest violations" in text


def test_format_includes_constraints():
    """Formatted prompt should show current constraint thresholds."""
    policy, state, result = _build_state_and_result()
    text = format_context(policy, state, result)
    assert "## Current constraints" in text
    assert "10000" in text  # latency max


def test_format_includes_relaxations():
    """Formatted prompt should show suggested relaxations."""
    policy, state, result = _build_state_and_result()
    text = format_context(policy, state, result)
    assert "## Suggested relaxations" in text


def test_no_private_data():
    """Formatted prompt must not contain API keys or file paths."""
    policy, state, result = _build_state_and_result()
    text = format_context(policy, state, result)
    # Should not contain common private data patterns
    assert "API_KEY" not in text
    assert "OPENAI" not in text
    assert "ANTHROPIC" not in text
    assert "C:\\" not in text
    assert "/home/" not in text
    assert ".env" not in text
