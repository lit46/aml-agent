"""Feature engineering tool.

Transforms filtered transaction data into per-account behavioral features,
consumed by the anomaly detection tool. This tool makes no suspicious /
not-suspicious judgment of its own — it only describes behavior
numerically. Judgment happens downstream.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.schemas import AccountFeatures, QueryFilters, ToolResult
from app.services.data_loader import DataStore
from app.tools.aggregation_rule import AggregationRuleTool
from app.tools.base import Tool

# Transactions in this range are a classic structuring signal — just under
# the $10,000 reporting threshold used by FinCEN and similar regulators.
NEAR_THRESHOLD_LOW = 9000.0
NEAR_THRESHOLD_HIGH = 9999.99


class FeatureEngineeringTool(Tool):
    """Computes per-account behavioral features from filtered transactions."""

    name = "feature_engineering_tool"
    description = (
        "Computes per-account behavioral features — transaction velocity, "
        "amount statistics, counterparty fan-in/fan-out degree, near-"
        "threshold transaction counts, currency diversity — from filtered "
        "transaction data. Required before running anomaly detection; not "
        "needed for simple aggregation/threshold queries."
    )

    def __init__(self, data_store: DataStore) -> None:
        self._data_store = data_store

    def run(self, filters: QueryFilters | None = None, **_: Any) -> ToolResult:
        try:
            df = AggregationRuleTool._apply_filters(
                self._data_store.transactions, filters
            )
            if df.empty:
                return ToolResult(
                    tool=self.name,
                    success=True,
                    data={"account_features": [], "account_count": 0},
                )

            merged = (
                self._outbound_stats(df)
                .join(self._inbound_stats(df), how="outer")
                .join(self._velocity_stats(df), how="outer")
                .join(self._currency_diversity(df), how="outer")
            )
            merged = merged.fillna(0)
            merged.index.name = "account_id"
            merged = merged.reset_index()

            features = [
                AccountFeatures(**row).model_dump()
                for row in merged.to_dict(orient="records")
            ]

            return ToolResult(
                tool=self.name,
                success=True,
                data={"account_features": features, "account_count": len(features)},
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary must not raise
            return ToolResult(tool=self.name, success=False, error=str(exc))

    @staticmethod
    def _outbound_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Stats derived from transactions where the account is the sender."""
        near_threshold = df[
            (df["amount_paid"] >= NEAR_THRESHOLD_LOW)
            & (df["amount_paid"] <= NEAR_THRESHOLD_HIGH)
        ]
        near_threshold_counts = (
            near_threshold.groupby("from_account")
            .size()
            .rename("near_threshold_count")
        )

        stats = df.groupby("from_account").agg(
            transaction_count=("from_account", "count"),
            total_amount_sent=("amount_paid", "sum"),
            avg_amount_sent=("amount_paid", "mean"),
            amount_std_sent=("amount_paid", "std"),
            max_amount=("amount_paid", "max"),
            distinct_counterparties_out=("to_account", "nunique"),
        )
        stats = stats.join(near_threshold_counts, how="left")
        stats.index.name = "account_id"
        return stats

    @staticmethod
    def _inbound_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Stats derived from transactions where the account is the receiver."""
        stats = df.groupby("to_account").agg(
            total_amount_received=("amount_received", "sum"),
            distinct_counterparties_in=("from_account", "nunique"),
        )
        stats.index.name = "account_id"
        return stats

    @staticmethod
    def _velocity_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Average time gap between an account's consecutive outgoing transactions."""

        def avg_gap_hours(group: pd.DataFrame) -> float:
            if len(group) < 2:
                return 0.0
            sorted_ts = group["timestamp"].sort_values()
            gaps_hours = sorted_ts.diff().dropna().dt.total_seconds() / 3600
            return float(gaps_hours.mean()) if not gaps_hours.empty else 0.0

        result = df.groupby("from_account").apply(avg_gap_hours, include_groups=False)
        result = result.rename("avg_hours_between_txns")
        result.index.name = "account_id"
        return result.to_frame()

    @staticmethod
    def _currency_diversity(df: pd.DataFrame) -> pd.DataFrame:
        """Count of distinct currencies an account has sent or received in."""
        out_currencies = df.groupby("from_account")["payment_currency"].nunique()
        in_currencies = df.groupby("to_account")["receiving_currency"].nunique()
        combined = pd.concat(
            [out_currencies.rename("out"), in_currencies.rename("in")], axis=1
        )
        combined["unique_currencies_used"] = combined.max(axis=1)
        combined.index.name = "account_id"
        return combined[["unique_currencies_used"]]
