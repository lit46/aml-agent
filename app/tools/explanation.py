"""Explanation tool.

Converts triggered rule-based signals and feature values into a concise,
human-readable explanation for why an account was flagged. Template-based
(not an LLM call) so it is deterministic, fast, and testable without an
API key. The orchestrator may optionally ask Claude to rephrase this text
more conversationally, but the underlying reasoning is generated here.
"""

from __future__ import annotations

from typing import Any

from app.schemas import ToolResult
from app.tools.base import Tool

RULE_TEMPLATES: dict[str, str] = {
    "near_threshold_transactions": (
        "made {near_threshold_count:.0f} transaction(s) between $9,000 and "
        "$9,999.99 — just under the $10,000 reporting threshold, "
        "consistent with structuring"
    ),
    "high_fan_in_or_out": (
        "transacted with {fan_degree:.0f} distinct counterparties, "
        "indicating possible fan-in/fan-out (smurfing) behavior"
    ),
    "rapid_succession_transfers": (
        "sent {transaction_count:.0f} transfers averaging only "
        "{avg_hours_between_txns:.1f} hours apart, suggesting rapid "
        "layering activity"
    ),
    "multi_currency_activity": (
        "used {unique_currencies_used:.0f} different currencies across its "
        "transactions, a common technique to obscure fund origin"
    ),
}


class ExplanationTool(Tool):
    """Generates human-readable explanations from triggered rule signals."""

    name = "explanation_tool"
    description = (
        "Generates a concise natural-language explanation for why an "
        "account was flagged, tied to its specific triggered rule signals "
        "and feature values. Typically runs last, after "
        "risk_classification_tool."
    )

    def run(
        self,
        anomaly_scores: list[dict[str, Any]] | None = None,
        account_features: list[dict[str, Any]] | None = None,
        risk_classifications: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> ToolResult:
        try:
            anomaly_scores = anomaly_scores or []
            if not anomaly_scores:
                return ToolResult(tool=self.name, success=True, data={"explanations": []})

            features_by_account = {
                row["account_id"]: row for row in (account_features or [])
            }
            risk_by_account = {
                row["account_id"]: row for row in (risk_classifications or [])
            }

            explanations = []
            for score_row in anomaly_scores:
                account_id = score_row["account_id"]
                text = self._build_explanation(
                    account_id=account_id,
                    hybrid_score=score_row["hybrid_score"],
                    triggered_rules=score_row.get("triggered_rules", []),
                    features=features_by_account.get(account_id, {}),
                    risk_level=risk_by_account.get(account_id, {}).get("risk_level"),
                )
                explanations.append({"account_id": account_id, "explanation": text})

            return ToolResult(tool=self.name, success=True, data={"explanations": explanations})
        except Exception as exc:  # noqa: BLE001 - tool boundary must not raise
            return ToolResult(tool=self.name, success=False, error=str(exc))

    @staticmethod
    def _build_explanation(
        account_id: str,
        hybrid_score: float,
        triggered_rules: list[str],
        features: dict[str, Any],
        risk_level: str | None,
    ) -> str:
        fan_degree = max(
            features.get("distinct_counterparties_in", 0),
            features.get("distinct_counterparties_out", 0),
        )
        context = {**features, "fan_degree": fan_degree}

        phrases = []
        for rule in triggered_rules:
            template = RULE_TEMPLATES.get(rule)
            if template is None:
                continue
            try:
                phrases.append(template.format(**context))
            except (KeyError, ValueError):
                # Missing/malformed feature value for this rule — skip rather
                # than crash the whole explanation.
                continue

        risk_prefix = f"{risk_level} risk" if risk_level else "flagged"

        if not phrases:
            return (
                f"Account {account_id} was {risk_prefix} (score {hybrid_score:.2f}) "
                "based on statistical deviation from typical account behavior "
                "detected by the anomaly model, without a specific rule-based "
                "signal firing."
            )

        joined = "; ".join(phrases)
        return (
            f"Account {account_id} was {risk_prefix} (score {hybrid_score:.2f}) "
            f"because it {joined}."
        )
