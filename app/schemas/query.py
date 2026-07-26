"""Schemas for representing a parsed user query.

These models describe the output of intent extraction: what the user is
asking for, in a structured form the planner can reason about.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from enum import Enum

from dateutil import parser as dateutil_parser
from pydantic import BaseModel, Field, field_validator

_RELATIVE_UNIT_DAYS = {
    "day": 1,
    "days": 1,
    "week": 7,
    "weeks": 7,
    "month": 30,
    "months": 30,
    "year": 365,
    "years": 365,
}

_RELATIVE_PATTERN = re.compile(
    r"^\s*(\d+)\s*(day|days|week|weeks|month|months|year|years)\s*ago\s*$",
    re.IGNORECASE,
)


def _coerce_to_date(value: object) -> object:
    """Best-effort conversion of LLM-provided date strings to a date.

    LLMs asked for an ISO date sometimes still hand back natural language
    like '30 days ago' or 'today' instead of computing the exact date
    themselves. Rather than hard-failing validation (and silently dropping
    to the rule-based fallback), normalize the common relative phrases and
    fall back to a fuzzy absolute-date parse for everything else.
    """
    if value is None or isinstance(value, date):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip().lower()
    if not text:
        return None

    if text in {"today", "now"}:
        return date.today()
    if text == "yesterday":
        return date.today() - timedelta(days=1)
    if text == "tomorrow":
        return date.today() + timedelta(days=1)

    match = _RELATIVE_PATTERN.match(text)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        return date.today() - timedelta(days=amount * _RELATIVE_UNIT_DAYS[unit])

    try:
        return dateutil_parser.parse(value, default=datetime(1900, 1, 1)).date()
    except (ValueError, OverflowError):
        # Give up gracefully rather than raising - an unparseable date
        # filter becomes "no filter" instead of crashing the whole query.
        return None


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

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _normalize_date(cls, value: object) -> object:
        return _coerce_to_date(value)


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
