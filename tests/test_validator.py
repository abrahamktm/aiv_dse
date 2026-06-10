from aiv_dse.adapters.report_parser import load_report
from aiv_dse.core.validator import load_policy, validate

POLICY_PATH = "policy/default_policy.yaml"


def test_passing_report():
    report = load_report("samples/report_pass.json")
    policy = load_policy(POLICY_PATH)
    result = validate(report, policy)
    assert result.status == "APPROVED"
    assert len(result.violations) == 0


def test_failing_report():
    report = load_report("samples/report_fail.json")
    policy = load_policy(POLICY_PATH)
    result = validate(report, policy)
    assert result.status == "VETO"
    assert len(result.violations) > 0

    violated_ids = [v["constraint_id"] for v in result.violations]
    assert "latency" in violated_ids
    assert "area" in violated_ids
    assert "power" in violated_ids
    assert "unroll_factor" in violated_ids


def test_poison_report():
    result = validate({}, {}, is_poison=True)
    assert result.status == "HALT"
    assert "Poison" in result.reasons[0]


def test_suggested_relaxations():
    report = load_report("samples/report_fail.json")
    policy = load_policy(POLICY_PATH)
    result = validate(report, policy)
    assert len(result.suggested_relaxations) > 0
    assert any("increase" in s for s in result.suggested_relaxations)
