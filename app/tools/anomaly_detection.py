"""Anomaly detection tool.

Combines an unsupervised Isolation Forest model with rule-based AML
signals into a single hybrid suspicion score per account. Consumes the
output of feature_engineering_tool (or computes it internally if not
already provided by the orchestrator).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.schemas import AnomalyScore, QueryFilters, ToolResult
from app.services.data_loader import DataStore
from app.tools.base import Tool
from app.tools.feature_engineering import FeatureEngineeringTool

FEATURE_COLUMNS = [
    "transaction_count",
    "total_amount_sent",
    "total_amount_received",
    "avg_amount_sent",
    "amount_std_sent",
    "max_amount",
    "distinct_counterparties_out",
    "distinct_counterparties_in",
    "near_threshold_count",
    "avg_hours_between_txns",
    "unique_currencies_used",
]

# Below this many accounts, Isolation Forest has too little data to be
# meaningful — fall back to the rule score alone rather than return noise.
DEFAULT_MIN_SAMPLES_FOR_ML = 10


class AnomalyDetectionTool(Tool):
    """Hybrid (ML + rule-based) suspicion scoring for accounts."""

    name = "anomaly_detection_tool"
    description = (
        "Scores accounts for suspicious activity using a hybrid of an "
        "Isolation Forest anomaly model and rule-based AML signals "
        "(structuring, fan-in/fan-out, rapid succession transfers, "
        "currency hopping). Requires account features; computes them "
        "internally via feature_engineering_tool if not already provided."
    )

    def __init__(
        self,
        data_store: DataStore,
        contamination: float = 0.1,
        random_state: int = 42,
        min_samples_for_ml: int = DEFAULT_MIN_SAMPLES_FOR_ML,
    ) -> None:
        self._data_store = data_store
        self._feature_tool = FeatureEngineeringTool(data_store)
        self._contamination = contamination
        self._random_state = random_state
        self._min_samples_for_ml = min_samples_for_ml

    def run(
        self,
        account_features: list[dict[str, Any]] | None = None,
        filters: QueryFilters | None = None,
        **_: Any,
    ) -> ToolResult:
        try:
            if account_features is None:
                feature_result = self._feature_tool.run(filters=filters)
                if not feature_result.success:
                    return ToolResult(
                        tool=self.name,
                        success=False,
                        error=f"upstream feature engineering failed: {feature_result.error}",
                    )
                account_features = feature_result.data["account_features"]

            if not account_features:
                return ToolResult(
                    tool=self.name,
                    success=True,
                    data={
                        "anomaly_scores": [],
                        "account_count": 0,
                        "ml_model_used": False,
                    },
                )

            df = pd.DataFrame(account_features)
            rule_scores, triggered_rules = self._compute_rule_scores(df)
            ml_scores, ml_used = self._compute_ml_scores(df)

            hybrid_scores = (
                0.5 * ml_scores + 0.5 * rule_scores if ml_used else rule_scores
            )

            scores = [
                AnomalyScore(
                    account_id=str(df.iloc[i]["account_id"]),
                    ml_score=float(ml_scores[i]) if ml_used else 0.0,
                    rule_score=float(rule_scores[i]),
                    hybrid_score=float(hybrid_scores[i]),
                    triggered_rules=triggered_rules[i],
                ).model_dump()
                for i in range(len(df))
            ]

            return ToolResult(
                tool=self.name,
                success=True,
                data={
                    "anomaly_scores": scores,
                    "account_count": len(scores),
                    "ml_model_used": ml_used,
                },
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary must not raise
            return ToolResult(tool=self.name, success=False, error=str(exc))

    @staticmethod
    def _compute_rule_scores(
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, list[list[str]]]:
        """Score each account against explicit AML heuristics.

        Each component is normalized to [0, 1]; the final rule_score is
        their unweighted average. Thresholds are deliberately simple and
        documented here so they're easy to tune later.
        """
        structuring = (df["near_threshold_count"] / 3).clip(upper=1.0)
        fan_degree = df[["distinct_counterparties_in", "distinct_counterparties_out"]].max(
            axis=1
        )
        fan = (fan_degree / 8).clip(upper=1.0)
        rapid = (
            (df["avg_hours_between_txns"] > 0)
            & (df["avg_hours_between_txns"] < 24)
            & (df["transaction_count"] >= 3)
        ).astype(float)
        currency = (df["unique_currencies_used"] >= 2).astype(float)

        rule_score = (structuring + fan + rapid + currency) / 4.0

        triggered: list[list[str]] = []
        for i in range(len(df)):
            reasons = []
            if structuring.iloc[i] > 0:
                reasons.append("near_threshold_transactions")
            if fan.iloc[i] > 0.5:
                reasons.append("high_fan_in_or_out")
            if rapid.iloc[i] == 1.0:
                reasons.append("rapid_succession_transfers")
            if currency.iloc[i] == 1.0:
                reasons.append("multi_currency_activity")
            triggered.append(reasons)

        return rule_score.to_numpy(), triggered

    def _compute_ml_scores(self, df: pd.DataFrame) -> tuple[np.ndarray, bool]:
        """Fit Isolation Forest and return normalized anomaly scores.

        Returns (zeros, False) if there isn't enough data to fit
        meaningfully, so callers can fall back to rule_score alone.
        """
        if len(df) < self._min_samples_for_ml:
            return np.zeros(len(df)), False

        X = df[FEATURE_COLUMNS].to_numpy()
        model = IsolationForest(
            contamination=self._contamination, random_state=self._random_state
        )
        model.fit(X)
        raw_scores = -model.score_samples(X)  # higher = more anomalous

        min_score, max_score = raw_scores.min(), raw_scores.max()
        if max_score - min_score < 1e-9:
            normalized = np.zeros_like(raw_scores)
        else:
            normalized = (raw_scores - min_score) / (max_score - min_score)

        return normalized, True
