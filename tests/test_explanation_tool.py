"""Tests for ExplanationTool."""

from app.tools.explanation import ExplanationTool


def test_explanation_cites_specific_triggered_rules():
    tool = ExplanationTool()
    scores = [
        {
            "account_id": "A1",
            "hybrid_score": 0.9,
            "triggered_rules": ["near_threshold_transactions", "multi_currency_activity"],
        }
    ]
    features = [
        {
            "account_id": "A1",
            "near_threshold_count": 4,
            "unique_currencies_used": 3,
            "distinct_counterparties_in": 0,
            "distinct_counterparties_out": 6,
            "transaction_count": 6,
            "avg_hours_between_txns": 5.0,
        }
    ]
    result = tool.run(anomaly_scores=scores, account_features=features)

    assert result.success
    text = result.data["explanations"][0]["explanation"]
    assert "structuring" in text
    assert "4 transaction" in text
    assert "3 different currencies" in text


def test_explanation_falls_back_gracefully_with_no_triggered_rules():
    tool = ExplanationTool()
    scores = [{"account_id": "A2", "hybrid_score": 0.75, "triggered_rules": []}]
    result = tool.run(anomaly_scores=scores)

    assert result.success
    text = result.data["explanations"][0]["explanation"]
    assert "A2" in text
    assert "0.75" in text
    assert "statistical deviation" in text


def test_explanation_includes_risk_level_when_provided():
    tool = ExplanationTool()
    scores = [{"account_id": "A3", "hybrid_score": 0.85, "triggered_rules": []}]
    classifications = [{"account_id": "A3", "risk_level": "HIGH"}]
    result = tool.run(anomaly_scores=scores, risk_classifications=classifications)

    text = result.data["explanations"][0]["explanation"]
    assert "HIGH risk" in text


def test_missing_feature_for_a_triggered_rule_does_not_crash():
    """If account_features weren't passed at all, templates needing them
    should be skipped, not raise — falls back to the generic explanation."""
    tool = ExplanationTool()
    scores = [
        {
            "account_id": "A4",
            "hybrid_score": 0.6,
            "triggered_rules": ["near_threshold_transactions"],
        }
    ]
    result = tool.run(anomaly_scores=scores)  # no account_features provided

    assert result.success
    text = result.data["explanations"][0]["explanation"]
    assert "A4" in text  # didn't crash, produced fallback text


def test_empty_input_returns_success_not_crash():
    tool = ExplanationTool()
    result = tool.run(anomaly_scores=[])

    assert result.success
    assert result.data["explanations"] == []
