"""Tests for LLMOrchestrator.

These test the orchestrator's control flow (parsing tool_use blocks,
threading pipeline data between real tool calls, terminating on final
text) using a scripted fake Anthropic client. They do NOT validate real
Claude API behavior — that must be checked manually with a real key
(see README). What they do prove: the tool calls that get made actually
run against real fixture data and produce correct results.
"""

from pathlib import Path

from app.agents.llm_orchestrator import LLMOrchestrator
from app.schemas import ToolName
from app.services.data_loader import DataStore
from tests.fakes.anthropic_client import ScriptedClient, fake_response, text_block, tool_use_block

FIXTURES = Path(__file__).parent / "fixtures"


def _make_orchestrator(client) -> LLMOrchestrator:
    store = DataStore(
        FIXTURES / "sample_transactions.csv", FIXTURES / "sample_accounts.csv"
    )
    return LLMOrchestrator(store, client=client)


def test_single_tool_call_then_final_answer():
    client = ScriptedClient(
        [
            fake_response(
                [
                    tool_use_block(
                        "aggregation_rule_tool",
                        {"reason": "threshold query", "filters": {"min_transaction_count": 3}},
                    )
                ]
            ),
            fake_response([text_block("Found 1 account exceeding the threshold.")]),
        ]
    )
    orchestrator = _make_orchestrator(client)
    result = orchestrator.handle_query("Which accounts have 3+ transactions?")

    assert result.execution_summary == "Found 1 account exceeding the threshold."
    assert len(result.plan.steps) == 1
    assert result.plan.steps[0].tool == ToolName.AGGREGATION_RULE
    assert result.supporting_metrics["used_fallback"] is False
    assert result.supporting_metrics["match_count"] == 1
    assert client.call_count == 2


def test_multi_step_pipeline_threads_data_between_real_tool_calls():
    """Feature engineering -> anomaly detection -> risk classification ->
    explanation, scripted turn by turn, verifying real data flows through."""
    client = ScriptedClient(
        [
            fake_response(
                [tool_use_block("feature_engineering_tool", {"reason": "broad query"}, "t1")]
            ),
            fake_response(
                [tool_use_block("anomaly_detection_tool", {"reason": "score accounts"}, "t2")]
            ),
            fake_response(
                [tool_use_block("risk_classification_tool", {"reason": "classify"}, "t3")]
            ),
            fake_response(
                [tool_use_block("explanation_tool", {"reason": "explain"}, "t4")]
            ),
            fake_response([text_block("Analysis complete across all accounts.")]),
        ]
    )
    orchestrator = _make_orchestrator(client)
    result = orchestrator.handle_query("Find suspicious activity")

    assert result.execution_summary == "Analysis complete across all accounts."
    called_tools = [s.tool for s in result.plan.steps]
    assert called_tools == [
        ToolName.FEATURE_ENGINEERING,
        ToolName.ANOMALY_DETECTION,
        ToolName.RISK_CLASSIFICATION,
        ToolName.EXPLANATION,
    ]
    # 9 accounts in the fixture -> feature engineering should have run for real
    assert orchestrator._pipeline_state["account_features"]
    assert len(orchestrator._pipeline_state["account_features"]) == 9


def test_stops_after_max_turns_if_no_final_answer():
    client = ScriptedClient(
        [
            fake_response([tool_use_block("feature_engineering_tool", {"reason": "r1"}, "a")]),
            fake_response([tool_use_block("feature_engineering_tool", {"reason": "r2"}, "b")]),
        ]
    )
    store = DataStore(
        FIXTURES / "sample_transactions.csv", FIXTURES / "sample_accounts.csv"
    )
    orchestrator = LLMOrchestrator(store, client=client, max_turns=2)
    result = orchestrator.handle_query("some query")

    assert "maximum" in result.execution_summary.lower()
    assert client.call_count == 2
