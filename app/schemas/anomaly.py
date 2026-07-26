"""Schema for anomaly detection tool output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnomalyScore(BaseModel):
    """Hybrid suspicion score for a single account.

    ml_score and rule_score are both normalized to [0, 1] so they can be
    combined; hybrid_score is their weighted average (or rule_score alone
    when there isn't enough data to fit the ML model meaningfully).
    """

    account_id: str
    ml_score: float = Field(
        ge=0.0, le=1.0, description="Isolation Forest anomaly score, normalized."
    )
    rule_score: float = Field(
        ge=0.0, le=1.0, description="Rule-based AML signal score."
    )
    hybrid_score: float = Field(ge=0.0, le=1.0)
    triggered_rules: list[str] = Field(
        default_factory=list,
        description="Names of rule-based signals that fired for this account, "
        "used by the explanation tool to describe why it was flagged.",
    )
