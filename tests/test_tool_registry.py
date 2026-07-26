"""Tests for the tool registry."""

from pathlib import Path

from app.agents.tool_registry import build_tool_instances, get_anthropic_tool_schemas
from app.schemas import ToolName
from app.services.data_loader import DataStore

FIXTURES = Path(__file__).parent / "fixtures"


def test_builds_all_five_tools():
    store = DataStore(FIXTURES / "sample_transactions.csv", FIXTURES / "sample_accounts.csv")
    tools = build_tool_instances(store)

    assert set(tools.keys()) == {
        ToolName.AGGREGATION_RULE,
        ToolName.FEATURE_ENGINEERING,
        ToolName.ANOMALY_DETECTION,
        ToolName.RISK_CLASSIFICATION,
        ToolName.EXPLANATION,
    }


def test_schemas_have_required_reason_field():
    schemas = get_anthropic_tool_schemas()
    assert len(schemas) == 5
    for schema in schemas:
        assert "reason" in schema["input_schema"]["required"]
        assert "reason" in schema["input_schema"]["properties"]


def test_schema_names_match_tool_name_enum_values():
    schemas = get_anthropic_tool_schemas()
    schema_names = {s["name"] for s in schemas}
    # eda_tool is intentionally not yet wired into the LLM-facing schemas
    expected = {t.value for t in ToolName if t != ToolName.EDA}
    assert schema_names == expected
