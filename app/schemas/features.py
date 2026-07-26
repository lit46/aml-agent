"""Schema for per-account behavioral features."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AccountFeatures(BaseModel):
    """Descriptive statistics computed for a single account.

    This model makes no suspicious/not-suspicious judgment — it purely
    describes behavior. The anomaly detection tool consumes these features
    to make that judgment.
    """

    account_id: str
    transaction_count: int = Field(description="Total transactions sent + received.")
    total_amount_sent: float
    total_amount_received: float
    avg_amount_sent: float
    amount_std_sent: float = Field(
        description="Standard deviation of sent amounts; 0 if fewer than 2 transactions."
    )
    max_amount: float = Field(description="Largest single amount sent.")
    distinct_counterparties_out: int
    distinct_counterparties_in: int
    near_threshold_count: int = Field(
        description="Sent transactions between $9,000-$9,999.99 — a structuring signal."
    )
    avg_hours_between_txns: float = Field(
        description="Average gap between consecutive outgoing transactions, in hours."
    )
    unique_currencies_used: int
