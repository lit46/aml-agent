"""Tests for AnomalyDetectionTool.

Rule-based and ML-based assertions below were verified by running the tool
directly first (see milestone conversation), not guessed.
"""

from pathlib import Path

import pytest

from app.services.data_loader import DataStore
from app.tools.anomaly_detection import AnomalyDetectionTool

FIXTURES = Path(__file__).parent / "fixtures"

CLEAN_ACCOUNT = {
    "account_id": "CLEAN01",
    "transaction_count": 2,
    "total_amount_sent": 500.0,
    "total_amount_received": 0.0,
    "avg_amount_sent": 250.0,
    "amount_std_sent": 50.0,
    "max_amount": 300.0,
    "distinct_counterparties_out": 2,
    "distinct_counterparties_in": 0,
    "near_threshold_count": 0,
    "avg_hours_between_txns": 72.0,
    "unique_currencies_used": 1,
}

SUSPICIOUS_ACCOUNT = {
    "account_id": "SUSPICIOUS01",
    "transaction_count": 6,
    "total_amount_sent": 55000.0,
    "total_amount_received": 0.0,
    "avg_amount_sent": 9166.0,
    "amount_std_sent": 400.0,
    "max_amount": 9900.0,
    "distinct_counterparties_out": 6,
    "distinct_counterparties_in": 0,
    "near_threshold_count": 4,
    "avg_hours_between_txns": 5.0,
    "unique_currencies_used": 3,
}


def _make_tool(min_samples_for_ml: int = 10) -> AnomalyDetectionTool:
    store = DataStore(
        FIXTURES / "sample_transactions.csv", FIXTURES / "sample_accounts.csv"
    )
    return AnomalyDetectionTool(store, min_samples_for_ml=min_samples_for_ml)


def test_rule_score_separates_clean_from_suspicious():
    tool = _make_tool()
    result = tool.run(account_features=[CLEAN_ACCOUNT, SUSPICIOUS_ACCOUNT])

    assert result.success
    assert result.data["ml_model_used"] is False  # only 2 samples, below threshold

    scores = {row["account_id"]: row for row in result.data["anomaly_scores"]}
    assert scores["SUSPICIOUS01"]["rule_score"] > scores["CLEAN01"]["rule_score"]
    assert scores["CLEAN01"]["triggered_rules"] == []


def test_all_four_rules_trigger_on_suspicious_account():
    tool = _make_tool()
    result = tool.run(account_features=[CLEAN_ACCOUNT, SUSPICIOUS_ACCOUNT])

    triggered = next(
        row["triggered_rules"]
        for row in result.data["anomaly_scores"]
        if row["account_id"] == "SUSPICIOUS01"
    )
    assert set(triggered) == {
        "near_threshold_transactions",
        "high_fan_in_or_out",
        "rapid_succession_transfers",
        "multi_currency_activity",
    }


def test_ml_path_skipped_below_sample_threshold():
    tool = _make_tool(min_samples_for_ml=10)
    result = tool.run(account_features=[CLEAN_ACCOUNT, SUSPICIOUS_ACCOUNT])

    assert result.data["ml_model_used"] is False
    assert all(row["ml_score"] == 0.0 for row in result.data["anomaly_scores"])


def test_ml_path_identifies_planted_outlier():
    tool = _make_tool(min_samples_for_ml=10)

    normal_accounts = [
        {
            "account_id": f"NORMAL{i}",
            "transaction_count": 2,
            "total_amount_sent": 400.0,
            "total_amount_received": 0.0,
            "avg_amount_sent": 200.0,
            "amount_std_sent": 20.0,
            "max_amount": 300.0,
            "distinct_counterparties_out": 1,
            "distinct_counterparties_in": 0,
            "near_threshold_count": 0,
            "avg_hours_between_txns": 72.0,
            "unique_currencies_used": 1,
        }
        for i in range(11)
    ]
    outlier = {
        "account_id": "OUTLIER",
        "transaction_count": 20,
        "total_amount_sent": 500000.0,
        "total_amount_received": 0.0,
        "avg_amount_sent": 25000.0,
        "amount_std_sent": 5000.0,
        "max_amount": 99000.0,
        "distinct_counterparties_out": 15,
        "distinct_counterparties_in": 0,
        "near_threshold_count": 5,
        "avg_hours_between_txns": 1.0,
        "unique_currencies_used": 4,
    }

    result = tool.run(account_features=[*normal_accounts, outlier])

    assert result.data["ml_model_used"] is True
    scores = {row["account_id"]: row["ml_score"] for row in result.data["anomaly_scores"]}
    assert scores["OUTLIER"] > 0.9
    avg_normal = sum(v for k, v in scores.items() if k != "OUTLIER") / len(normal_accounts)
    assert scores["OUTLIER"] > avg_normal + 0.5


def test_empty_account_features_returns_success_not_crash():
    tool = _make_tool()
    result = tool.run(account_features=[])

    assert result.success
    assert result.data["account_count"] == 0
    assert result.data["anomaly_scores"] == []


def test_computes_features_internally_when_not_provided():
    """Integration path: tool should chain into feature_engineering_tool itself."""
    tool = _make_tool()
    result = tool.run()  # no account_features passed

    assert result.success
    assert result.data["account_count"] == 9  # matches fixture's 9 distinct accounts
