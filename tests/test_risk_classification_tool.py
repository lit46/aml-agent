"""Tests for RiskClassificationTool."""

from app.tools.risk_classification import RiskClassificationTool


def _score(account_id: str, hybrid_score: float) -> dict:
    return {
        "account_id": account_id,
        "hybrid_score": hybrid_score,
        "ml_score": 0.0,
        "rule_score": hybrid_score,
        "triggered_rules": [],
    }


def test_default_thresholds_classify_correctly():
    tool = RiskClassificationTool()
    scores = [_score("A1", 0.9), _score("A3", 0.5), _score("A5", 0.1)]
    result = tool.run(anomaly_scores=scores)

    assert result.success
    by_id = {row["account_id"]: row for row in result.data["classifications"]}
    assert by_id["A1"]["risk_level"] == "HIGH"
    assert by_id["A1"]["recommended_action"] == "report"
    assert by_id["A3"]["risk_level"] == "MEDIUM"
    assert by_id["A3"]["recommended_action"] == "flag_for_review"
    assert by_id["A5"]["risk_level"] == "LOW"
    assert by_id["A5"]["recommended_action"] == "monitor"


def test_boundary_values_are_inclusive_on_the_high_side():
    tool = RiskClassificationTool(high_threshold=0.7, medium_threshold=0.4)
    result = tool.run(anomaly_scores=[_score("AT_HIGH", 0.7), _score("AT_MEDIUM", 0.4)])

    by_id = {row["account_id"]: row for row in result.data["classifications"]}
    assert by_id["AT_HIGH"]["risk_level"] == "HIGH"
    assert by_id["AT_MEDIUM"]["risk_level"] == "MEDIUM"


def test_invalid_thresholds_raise_at_construction():
    import pytest

    with pytest.raises(ValueError):
        RiskClassificationTool(high_threshold=0.3, medium_threshold=0.5)


def test_empty_input_returns_success_not_crash():
    tool = RiskClassificationTool()
    result = tool.run(anomaly_scores=[])

    assert result.success
    assert result.data["classifications"] == []
