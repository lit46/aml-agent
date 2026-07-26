"""Aggregation / threshold rule tool.

Answers direct counting or threshold queries — e.g. "which accounts made
10+ transactions under $10,000?" — using pandas grouping only. No ML model
is invoked. The planner should select this tool when a query maps to a
simple, explicit rule rather than a learned anomaly pattern.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.schemas import QueryFilters, ToolResult
from app.services.data_loader import DataStore
from app.tools.base import Tool


class AggregationRuleTool(Tool):
    """Direct pandas aggregation over transactions, no ML involved."""

    name = "aggregation_rule_tool"
    description = (
        "Runs direct pandas aggregation/threshold rules over transactions, "
        "e.g. counting transactions per account under a given amount within "
        "a date range. Use for direct counting/threshold queries; skip when "
        "the query requires learned anomaly patterns instead."
    )

    def __init__(self, data_store: DataStore) -> None:
        self._data_store = data_store

    def run(
        self,
        filters: QueryFilters | None = None,
        min_transaction_count: int | None = None,
        **_: Any,
    ) -> ToolResult:
        """Group transactions by sender account and apply a count threshold.

        `min_transaction_count` can be passed explicitly, or read from
        `filters.min_transaction_count` — the explicit argument wins if both
        are given.
        """
        try:
            df = self._apply_filters(self._data_store.transactions, filters)

            threshold = (
                min_transaction_count
                if min_transaction_count is not None
                else (filters.min_transaction_count if filters else None)
            ) or 1

            grouped = (
                df.groupby("from_account")
                .agg(
                    transaction_count=("from_account", "count"),
                    total_amount=("amount_paid", "sum"),
                )
                .reset_index()
            )
            matches = grouped[grouped["transaction_count"] >= threshold]
            matches = matches.sort_values("transaction_count", ascending=False)

            return ToolResult(
                tool=self.name,
                success=True,
                data={
                    "matching_accounts": matches.to_dict(orient="records"),
                    "match_count": int(len(matches)),
                    "threshold_used": threshold,
                },
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary must not raise
            return ToolResult(tool=self.name, success=False, error=str(exc))

    @staticmethod
    def _apply_filters(
        df: pd.DataFrame, filters: QueryFilters | None
    ) -> pd.DataFrame:
        """Apply the subset of QueryFilters relevant to a flat transaction table."""
        if filters is None:
            return df

        result = df
        if filters.start_date:
            result = result[result["timestamp"].dt.date >= filters.start_date]
        if filters.end_date:
            result = result[result["timestamp"].dt.date <= filters.end_date]
        if filters.min_amount is not None:
            result = result[result["amount_paid"] >= filters.min_amount]
        if filters.max_amount is not None:
            result = result[result["amount_paid"] <= filters.max_amount]
        if filters.account_id:
            result = result[
                (result["from_account"] == filters.account_id)
                | (result["to_account"] == filters.account_id)
            ]
        if filters.bank_id:
            result = result[
                (result["from_bank"] == filters.bank_id)
                | (result["to_bank"] == filters.bank_id)
            ]
        if filters.payment_format:
            result = result[result["payment_format"] == filters.payment_format]
        return result
