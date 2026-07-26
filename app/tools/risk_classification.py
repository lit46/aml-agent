"""Risk classification tool.

Maps hybrid anomaly scores (from anomaly_detection_tool) into business-facing
risk tiers and a recommended escalation action, using configurable
thresholds. Pure business logic, no ML or data access — deliberately
simple and easy to tune.
"""

from __future__ import annotations

from typing import Any

from app.schemas import RecommendedAction, RiskLevel, ToolResult
from app.tools.base import Tool


class RiskClassificationTool(Tool):
    """Converts a hybrid anomaly score into a risk tier and recommended action."""

    name = "risk_classification_tool"
    description = (
        "Maps hybrid anomaly scores to LOW/MEDIUM/HIGH risk tiers and a "
        "recommended escalation action (monitor / flag for review / report), "
        "using configurable thresholds. Runs after anomaly_detection_tool."
    )

    def __init__(self, high_threshold: float = 0.7, medium_threshold: float = 0.4) -> None:
        if not 0.0 <= medium_threshold < high_threshold <= 1.0:
            raise ValueError(
                "Thresholds must satisfy 0 <= medium_threshold < high_threshold <= 1"
            )
        self._high_threshold = high_threshold
        self._medium_threshold = medium_threshold

    def run(
        self, anomaly_scores: list[dict[str, Any]] | None = None, **_: Any
    ) -> ToolResult:
        try:
            anomaly_scores = anomaly_scores or []
            if not anomaly_scores:
                return ToolResult(
                    tool=self.name,
                    success=True,
                    data={"classifications": [], "account_count": 0},
                )

            classifications = []
            for row in anomaly_scores:
                score = row["hybrid_score"]
                level, action = self._classify(score)
                classifications.append(
                    {
                        "account_id": row["account_id"],
                        "hybrid_score": score,
                        "risk_level": level.value,
                        "recommended_action": action.value,
                    }
                )

            return ToolResult(
                tool=self.name,
                success=True,
                data={
                    "classifications": classifications,
                    "account_count": len(classifications),
                },
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary must not raise
            return ToolResult(tool=self.name, success=False, error=str(exc))

    def _classify(self, score: float) -> tuple[RiskLevel, RecommendedAction]:
        if score >= self._high_threshold:
            return RiskLevel.HIGH, RecommendedAction.REPORT
        if score >= self._medium_threshold:
            return RiskLevel.MEDIUM, RecommendedAction.FLAG_FOR_REVIEW
        return RiskLevel.LOW, RecommendedAction.MONITOR
