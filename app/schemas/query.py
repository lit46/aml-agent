"""Schemas for representing a parsed user query.

These models describe the output of intent extraction: what the user is
asking for, in a structured form the planner can reason about.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class AMLPattern(str, Enum):
    """Laundering typologies the agent knows how to reason about.

    Mirrors the pattern labels present in the sampled dataset
    (see data/DATA_CARD.md) plus a catch-all for unspecified/general queries.
    """

    STACK = "STACK"
    CYCLE = "CYCLE"
    FAN_IN = "FAN-IN"
    FAN_OUT = "FAN-OUT"
    BIPARTITE = "BIPARTITE"
    GATHER_SCATTER = "GATHER-SCATTER"
    SCATTER_GATHER = "SCATTER-GATHER"
    RANDOM = "RANDOM"
    UNSPECIFIED = "UNSPECIFIED"
    ANY = "ANY"  # user did not name a specific pattern


class QueryFilters(BaseModel):
    """Structured filters extracted from a natural language query.

    All fields are optional — an empty QueryFilters means "no filtering,
    consider the full dataset".
    """

    start_date: date | None = Field(
        default=None, description="Earliest transaction date to include."
    )
    end_date: date | None = Field(
        default=None, description="Latest transaction date to include."
    )
    min_amount: float | None = Field(
        default=None, description="Minimum transaction amount, in USD equivalent."
    )
    max_amount: float | None = Field(
        default=None, description="Maximum transaction amount, in USD equivalent."
    )
    account_id: str | None = Field(
        default=None, description="Restrict analysis to a single account number."
    )
    bank_id: str | None = Field(
        default=None, description="Restrict analysis to a single bank ID."
    )
    payment_format: str | None = Field(
        default=None, description="Restrict to a payment rail, e.g. 'ACH', 'Cheque'."
    )
    min_transaction_count: int | None = Field(
        default=None,
        description="For aggregation-style queries, e.g. 'accounts with 10+ transactions'.",
    )


class ParsedIntent(BaseModel):
    """The structured result of parsing a user's natural language query."""

    raw_query: str = Field(description="The original, unmodified user query.")
    wants_full_eda: bool = Field(
        default=False,
        description="True only for broad exploratory requests, e.g. 'analyse this dataset'.",
    )
    wants_single_entity_lookup: bool = Field(
        default=False,
        description="True for queries targeting one specific account/customer.",
    )
    requires_ml_anomaly_detection: bool = Field(
        default=True,
        description=(
            "False for simple aggregation/threshold queries that can be answered "
            "with a direct rule instead of running the ML model."
        ),
    )
    target_pattern: AMLPattern = Field(
        default=AMLPattern.ANY,
        description="The laundering typology the user is asking about, if any.",
    )
    filters: QueryFilters = Field(default_factory=QueryFilters)
