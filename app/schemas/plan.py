"""Schemas describing the agent's dynamic execution plan.

The orchestrator does not follow a fixed pipeline. Instead it builds an
ExecutionPlan at runtime, listing only the tools required to answer the
specific query, in the order they need to run.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    """Identifiers for each tool the orchestrator can invoke."""

    EDA = "eda_tool"
    FEATURE_ENGINEERING = "feature_engineering_tool"
    AGGREGATION_RULE = "aggregation_rule_tool"
    ANOMALY_DETECTION = "anomaly_detection_tool"
    RISK_CLASSIFICATION = "risk_classification_tool"
    EXPLANATION = "explanation_tool"


class ExecutionStep(BaseModel):
    """A single tool invocation within the plan."""

    tool: ToolName
    reason: str = Field(
        description="Short justification for why this tool was selected, "
        "used for the query-aware execution summary shown to the user."
    )


class ExecutionPlan(BaseModel):
    """The full sequence of tool invocations the agent decided on for a query."""

    steps: list[ExecutionStep] = Field(default_factory=list)
    skipped_tools: list[ToolName] = Field(
        default_factory=list,
        description="Tools deliberately not invoked, for transparency in the summary.",
    )
