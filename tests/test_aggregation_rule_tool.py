"""Tests for AggregationRuleTool, run against small fixture files."""

from pathlib import Path

from app.schemas import QueryFilters
from app.services.data_loader import DataStore
from app.tools.aggregation_rule import AggregationRuleTool

FIXTURES = Path(__file__).parent / "fixtures"


def _make_tool() -> AggregationRuleTool:
    store = DataStore(
        FIXTURES / "sample_transactions.csv", FIXTURES / "sample_accounts.csv"
    )
    return AggregationRuleTool(store)


def test_finds_account_over_transaction_count_threshold():
    tool = _make_tool()
    result = tool.run(min_transaction_count=3)

    assert result.success
    assert result.data["match_count"] == 1
    assert result.data["matching_accounts"][0]["from_account"] == "8062C56E0"
    assert result.data["matching_accounts"][0]["transaction_count"] == 3


def test_threshold_of_one_matches_everyone():
    tool = _make_tool()
    result = tool.run(min_transaction_count=1)

    assert result.success
    # 4 distinct from_account values in the fixture
    assert result.data["match_count"] == 4


def test_amount_filter_narrows_results():
    tool = _make_tool()
    filters = QueryFilters(max_amount=10000)
    result = tool.run(filters=filters, min_transaction_count=1)

    assert result.success
    # the 16898.29 transaction should be excluded, its account shouldn't appear
    matched_accounts = {
        row["from_account"] for row in result.data["matching_accounts"]
    }
    assert "81363F410" not in matched_accounts


def test_tool_does_not_raise_on_bad_input():
    tool = _make_tool()
    # filters with an impossible date range should return an empty, still-successful result
    from datetime import date

    filters = QueryFilters(start_date=date(2099, 1, 1))
    result = tool.run(filters=filters)

    assert result.success
    assert result.data["match_count"] == 0
