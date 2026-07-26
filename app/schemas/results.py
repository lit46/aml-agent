"""Schemas for tool outputs and the final composed agent response."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.plan import ExecutionPlan
from app.schemas.query import AMLPattern


class RiskLevel(str, Enum):
    """Business-facing risk tiers, mapped from raw model/rule scores."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RecommendedAction(str, Enum):
    """Escalation actions the agent can suggest for a flagged item."""

    MONITOR = "monitor"
    FLAG_FOR_REVIEW = "flag_for_review"
    REPORT = "report"


class ToolResult(BaseModel):
    """Generic wrapper returned by every tool.

    Keeping this uniform lets the orchestrator log and chain tool calls
    without needing to know each tool's internal output shape in advance.
    """

    tool: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class FlaggedItem(BaseModel):
    """A single transaction or account the agent is flagging as suspicious."""

    entity_id: str = Field(description="Transaction ID or account/customer ID.")
    entity_type: str = Field(description="'transaction' or 'account'.")
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    pattern_type: AMLPattern
    explanation: str = Field(
        description="Human-readable reason this item was flagged, tied to the "
        "specific features that triggered it."
    )
    recommended_action: RecommendedAction


class AgentResponse(BaseModel):
    """The final, complete response returned to the user for a query."""

    query: str
    execution_summary: str = Field(
        description="Plain-language summary of what the agent detected in the "
        "query and which tools it decided to invoke."
    )
    plan: ExecutionPlan
    flagged_items: list[FlaggedItem] = Field(default_factory=list)
    supporting_metrics: dict[str, Any] = Field(default_factory=dict)
