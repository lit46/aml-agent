"""Tests for RuleBasedOrchestrator, run against real fixture data (no mocking)."""

from pathlib import Path

from app.agents.rule_based_orchestrator import RuleBasedOrchestrator
from app.schemas import ToolName
from app.services.data_loader import DataStore

FIXTURES = Path(__file__).parent / "fixtures"


def _make_orchestrator() -> RuleBasedOrchestrator:
    store = DataStore(
        FIXTURES / "sample_transactions.csv", FIXTURES / "sample_accounts.csv"
    )
    return RuleBasedOrchestrator(store)


def test_aggregation_query_routes_to_aggregation_tool_only():
    orchestrator = _make_orchestrator()
    result = orchestrator.handle_query(
        "Which accounts made 3+ transactions under $10,000?"
    )

    called_tools = [s.tool for s in result.plan.steps]
    assert called_tools == [ToolName.AGGREGATION_RULE]
    assert set(result.plan.skipped_tools) == {
        ToolName.FEATURE_ENGINEERING,
        ToolName.ANOMALY_DETECTION,
        ToolName.RISK_CLASSIFICATION,
        ToolName.EXPLANATION,
    }
    assert result.supporting_metrics["used_fallback"] is True
    assert result.supporting_metrics["match_count"] == 1


def test_plural_accounts_does_not_trigger_single_entity_lookup():
    """Regression test: 'accounts'/'customers' (plural) must not match the
    account-ID regex as a substring of 'account'/'customer'."""
    orchestrator = _make_orchestrator()
    result = orchestrator.handle_query(
        "Which customers made 10+ transactions under $10,000?"
    )
    called_tools = [s.tool for s in result.plan.steps]
    assert called_tools == [ToolName.AGGREGATION_RULE]


def test_single_entity_query_routes_to_targeted_pipeline():
    orchestrator = _make_orchestrator()
    result = orchestrator.handle_query("Is account 8062C56E0 suspicious?")

    called_tools = [s.tool for s in result.plan.steps]
    assert called_tools == [
        ToolName.FEATURE_ENGINEERING,
        ToolName.ANOMALY_DETECTION,
        ToolName.RISK_CLASSIFICATION,
        ToolName.EXPLANATION,
    ]
    assert ToolName.AGGREGATION_RULE in result.plan.skipped_tools
    assert "8062C56E0" in result.execution_summary


def test_customer_id_phrasing_also_triggers_single_entity_lookup():
    orchestrator = _make_orchestrator()
    result = orchestrator.handle_query("Is customer ID 4521 suspicious?")

    called_tools = [s.tool for s in result.plan.steps]
    assert called_tools == [
        ToolName.FEATURE_ENGINEERING,
        ToolName.ANOMALY_DETECTION,
        ToolName.RISK_CLASSIFICATION,
        ToolName.EXPLANATION,
    ]


def test_generic_query_runs_full_pipeline():
    orchestrator = _make_orchestrator()
    result = orchestrator.handle_query("Find structuring patterns in the last 30 days")

    called_tools = [s.tool for s in result.plan.steps]
    assert called_tools == [
        ToolName.FEATURE_ENGINEERING,
        ToolName.ANOMALY_DETECTION,
        ToolName.RISK_CLASSIFICATION,
        ToolName.EXPLANATION,
    ]
    assert result.plan.skipped_tools == []


def test_response_is_always_marked_as_fallback():
    orchestrator = _make_orchestrator()
    result = orchestrator.handle_query("anything at all")
    assert result.supporting_metrics["used_fallback"] is True
