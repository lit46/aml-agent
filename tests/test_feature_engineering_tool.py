"""Tests for FeatureEngineeringTool, run against small fixture files.

Expected values below were computed by running the tool against the
fixture and verifying by hand, not guessed — see account 8062C56E0's
3 outgoing transactions in tests/fixtures/sample_transactions.csv.
"""

from datetime import date
from pathlib import Path

import pytest

from app.schemas import QueryFilters
from app.services.data_loader import DataStore
from app.tools.feature_engineering import FeatureEngineeringTool

FIXTURES = Path(__file__).parent / "fixtures"


def _make_tool() -> FeatureEngineeringTool:
    store = DataStore(
        FIXTURES / "sample_transactions.csv", FIXTURES / "sample_accounts.csv"
    )
    return FeatureEngineeringTool(store)


def _get_account(result_data: dict, account_id: str) -> dict:
    matches = [
        row for row in result_data["account_features"] if row["account_id"] == account_id
    ]
    assert matches, f"account {account_id} not found in features"
    return matches[0]


def test_returns_features_for_every_account_touched():
    tool = _make_tool()
    result = tool.run()

    assert result.success
    # 9 distinct accounts appear across from_account/to_account in the fixture
    assert result.data["account_count"] == 9


def test_outbound_stats_correct_for_known_account():
    tool = _make_tool()
    result = tool.run()
    account = _get_account(result.data, "8062C56E0")

    assert account["transaction_count"] == 3
    assert account["total_amount_sent"] == pytest.approx(12802.59)
    assert account["avg_amount_sent"] == pytest.approx(4267.53, abs=0.01)
    assert account["max_amount"] == pytest.approx(5602.59)
    assert account["distinct_counterparties_out"] == 3


def test_account_with_no_inbound_activity_gets_zero_not_error():
    tool = _make_tool()
    result = tool.run()
    account = _get_account(result.data, "8062C56E0")

    assert account["total_amount_received"] == 0.0
    assert account["distinct_counterparties_in"] == 0


def test_velocity_computed_correctly():
    tool = _make_tool()
    result = tool.run()
    account = _get_account(result.data, "8062C56E0")

    # timestamps 09-03 13:09 -> 09-04 09:00 -> 09-05 10:00: gaps 19.85h, 25h
    assert account["avg_hours_between_txns"] == pytest.approx(22.425, abs=0.01)


def test_single_transaction_account_has_zero_velocity_not_nan():
    tool = _make_tool()
    result = tool.run()
    # 823D5EB90 has exactly one outgoing transaction in the fixture
    account = _get_account(result.data, "823D5EB90")

    assert account["avg_hours_between_txns"] == 0.0
    assert account["amount_std_sent"] == 0.0


def test_empty_filtered_result_returns_success_not_crash():
    tool = _make_tool()
    filters = QueryFilters(start_date=date(2099, 1, 1))
    result = tool.run(filters=filters)

    assert result.success
    assert result.data["account_count"] == 0
    assert result.data["account_features"] == []
