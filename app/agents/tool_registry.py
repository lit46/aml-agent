"""Tool registry: instantiates tools and exposes their Claude-compatible schemas.

Only parameters the LLM should actually decide (reason, filters, thresholds)
are exposed in the schemas below. Parameters that thread data between tool
calls (account_features, anomaly_scores, risk_classifications) are injected
by the orchestrator itself in app/agents/base_orchestrator.py and are never
exposed to the model — the LLM should not be responsible for copying large
intermediate data structures between tool calls.
"""

from __future__ import annotations

from app.schemas import ToolName
from app.services.data_loader import DataStore
from app.tools.aggregation_rule import AggregationRuleTool
from app.tools.anomaly_detection import AnomalyDetectionTool
from app.tools.base import Tool
from app.tools.explanation import ExplanationTool
from app.tools.feature_engineering import FeatureEngineeringTool
from app.tools.risk_classification import RiskClassificationTool

FILTERS_SCHEMA = {
    "type": "object",
    "description": "Optional filters to scope the analysis to a subset of transactions.",
    "properties": {
        "start_date": {
            "type": "string",
            "description": "ISO date (YYYY-MM-DD), earliest transaction to include.",
        },
        "end_date": {
            "type": "string",
            "description": "ISO date (YYYY-MM-DD), latest transaction to include.",
        },
        "min_amount": {"type": "number", "description": "Minimum transaction amount."},
        "max_amount": {"type": "number", "description": "Maximum transaction amount."},
        "account_id": {
            "type": "string",
            "description": "Restrict analysis to a single account number.",
        },
        "bank_id": {"type": "string", "description": "Restrict to a single bank ID."},
        "payment_format": {
            "type": "string",
            "description": "Restrict to a payment rail, e.g. 'ACH', 'Cheque'.",
        },
        "min_transaction_count": {
            "type": "integer",
            "description": "Minimum transaction count, for aggregation-style queries.",
        },
    },
}

REASON_PROPERTY = {
    "type": "string",
    "description": "One sentence explaining why this tool is needed to answer the query.",
}


def build_tool_instances(data_store: DataStore) -> dict[ToolName, Tool]:
    """Instantiate every tool the orchestrator can invoke."""
    return {
        ToolName.AGGREGATION_RULE: AggregationRuleTool(data_store),
        ToolName.FEATURE_ENGINEERING: FeatureEngineeringTool(data_store),
        ToolName.ANOMALY_DETECTION: AnomalyDetectionTool(data_store),
        ToolName.RISK_CLASSIFICATION: RiskClassificationTool(),
        ToolName.EXPLANATION: ExplanationTool(),
    }


def get_anthropic_tool_schemas() -> list[dict]:
    """Claude tool-use schemas for every tool the orchestrator can invoke.

    eda_tool is intentionally excluded for now, pending a later milestone.
    """
    return [
        {
            "name": ToolName.AGGREGATION_RULE.value,
            "description": (
                "Runs direct pandas aggregation/threshold rules over "
                "transactions (e.g. counting transactions per account "
                "under a given amount). Use for direct counting/threshold "
                "queries; skip when the query needs learned anomaly "
                "patterns instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"reason": REASON_PROPERTY, "filters": FILTERS_SCHEMA},
                "required": ["reason"],
            },
        },
        {
            "name": ToolName.FEATURE_ENGINEERING.value,
            "description": (
                "Computes per-account behavioral features (transaction "
                "velocity, amount statistics, fan-in/fan-out degree, "
                "near-threshold transaction counts, currency diversity). "
                "Required before anomaly detection for most queries; skip "
                "for simple aggregation/threshold queries."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"reason": REASON_PROPERTY, "filters": FILTERS_SCHEMA},
                "required": ["reason"],
            },
        },
        {
            "name": ToolName.ANOMALY_DETECTION.value,
            "description": (
                "Scores accounts for suspicious activity using a hybrid of "
                "an Isolation Forest anomaly model and rule-based AML "
                "signals. Automatically uses features from a prior "
                "feature_engineering_tool call in this conversation, or "
                "computes them itself if not already available."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"reason": REASON_PROPERTY, "filters": FILTERS_SCHEMA},
                "required": ["reason"],
            },
        },
        {
            "name": ToolName.RISK_CLASSIFICATION.value,
            "description": (
                "Maps hybrid anomaly scores to LOW/MEDIUM/HIGH risk tiers "
                "and a recommended escalation action. Must run after "
                "anomaly_detection_tool."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"reason": REASON_PROPERTY},
                "required": ["reason"],
            },
        },
        {
            "name": ToolName.EXPLANATION.value,
            "description": (
                "Generates a concise natural-language explanation for why "
                "each account was flagged. Should typically be the last "
                "tool called, after risk_classification_tool."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"reason": REASON_PROPERTY},
                "required": ["reason"],
            },
        },
    ]
