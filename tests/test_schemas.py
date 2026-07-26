"""Smoke tests for the schema layer.

These don't test business logic (there isn't any yet) — they confirm the
schema module imports cleanly and models can be constructed and validated,
so later milestones aren't debugging import errors from day one.
"""

from app.schemas import (
    AMLPattern,
    AgentResponse,
    ExecutionPlan,
    ExecutionStep,
    FlaggedItem,
    ParsedIntent,
    QueryFilters,
    RecommendedAction,
    RiskLevel,
    ToolName,
)


def test_parsed_intent_defaults():
    intent = ParsedIntent(raw_query="Is customer ID 4521 suspicious?")
    assert intent.wants_full_eda is False
    assert intent.target_pattern == AMLPattern.ANY
    assert isinstance(intent.filters, QueryFilters)


def test_execution_plan_construction():
    plan = ExecutionPlan(
        steps=[
            ExecutionStep(
                tool=ToolName.AGGREGATION_RULE,
                reason="Simple threshold query, no ML needed.",
            )
        ],
        skipped_tools=[ToolName.EDA, ToolName.ANOMALY_DETECTION],
    )
    assert len(plan.steps) == 1
    assert ToolName.EDA in plan.skipped_tools


def test_flagged_item_and_agent_response():
    item = FlaggedItem(
        entity_id="TXN-001",
        entity_type="transaction",
        risk_score=0.87,
        risk_level=RiskLevel.HIGH,
        pattern_type=AMLPattern.STACK,
        explanation="Two rapid transfers just under reporting threshold.",
        recommended_action=RecommendedAction.FLAG_FOR_REVIEW,
    )
    response = AgentResponse(
        query="Find structuring in the last 30 days",
        execution_summary="Applied time filter, ran feature engineering and anomaly detection.",
        plan=ExecutionPlan(),
        flagged_items=[item],
    )
    assert response.flagged_items[0].risk_level == RiskLevel.HIGH
