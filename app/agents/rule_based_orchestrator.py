"""Deterministic fallback orchestrator.

Used when the LLM orchestrator is unavailable — missing API key, network
error, rate limit — so the agent never simply crashes during a demo or
judging session. Uses regex-based intent parsing instead of an LLM, but
still follows the core adaptive-tool-selection principle: different query
shapes invoke different tool subsets, not always the same fixed pipeline.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from app.agents.base_orchestrator import BaseOrchestrator
from app.schemas import AgentResponse, ToolName

THRESHOLD_COUNT_RE = re.compile(r"(\d+)\s*\+?\s*transactions?", re.IGNORECASE)
UNDER_AMOUNT_RE = re.compile(r"under\s*\$?\s*([\d,]+)", re.IGNORECASE)
LAST_N_DAYS_RE = re.compile(r"last\s+(\d+)\s+days?", re.IGNORECASE)
ACCOUNT_ID_RE = re.compile(
    r"\b(?:customer|account)\b\s*(?:id)?\s*#?\s*([A-Za-z0-9]+)", re.IGNORECASE
)


class RuleBasedOrchestrator(BaseOrchestrator):
    """Regex-based deterministic query router, no LLM involved."""

    def handle_query(self, query: str) -> AgentResponse:
        self._reset_state()
        filters_dict: dict[str, Any] = {}

        days_match = LAST_N_DAYS_RE.search(query)
        if days_match:
            n_days = int(days_match.group(1))
            filters_dict["start_date"] = (
                date.today() - timedelta(days=n_days)
            ).isoformat()

        account_match = ACCOUNT_ID_RE.search(query)
        threshold_match = THRESHOLD_COUNT_RE.search(query)
        amount_match = UNDER_AMOUNT_RE.search(query)

        if account_match:
            summary = self._run_single_entity_lookup(
                filters_dict, account_match.group(1)
            )
        elif threshold_match and amount_match:
            summary = self._run_aggregation_only(
                filters_dict, threshold_match, amount_match
            )
        else:
            summary = self._run_full_pipeline(filters_dict)

        return self._compose_response(query, summary, used_fallback=True)

    def _run_single_entity_lookup(
        self, filters_dict: dict[str, Any], account_id: str
    ) -> str:
        filters_dict["account_id"] = account_id
        self._execute_tool(
            ToolName.FEATURE_ENGINEERING,
            {"filters": filters_dict},
            reason="Single-entity lookup: computing features scoped to the named account.",
        )
        self._execute_tool(
            ToolName.ANOMALY_DETECTION,
            {"filters": filters_dict},
            reason="Scoring the named account for suspicious activity.",
        )
        self._execute_tool(
            ToolName.RISK_CLASSIFICATION,
            {},
            reason="Classifying the account's risk tier.",
        )
        self._execute_tool(
            ToolName.EXPLANATION,
            {},
            reason="Explaining why this account was or wasn't flagged.",
        )
        self._skipped_tools = [ToolName.AGGREGATION_RULE]
        return (
            f"Detected a single-entity lookup query. Ran targeted scoring for "
            f"account '{account_id}', skipping full-dataset tools."
        )

    def _run_aggregation_only(
        self,
        filters_dict: dict[str, Any],
        threshold_match: re.Match,
        amount_match: re.Match,
    ) -> str:
        min_count = int(threshold_match.group(1))
        max_amount = float(amount_match.group(1).replace(",", ""))
        filters_dict["max_amount"] = max_amount

        self._execute_tool(
            ToolName.AGGREGATION_RULE,
            {"filters": filters_dict, "min_transaction_count": min_count},
            reason="Direct threshold/count query — no learned pattern needed.",
        )
        self._skipped_tools = [
            ToolName.FEATURE_ENGINEERING,
            ToolName.ANOMALY_DETECTION,
            ToolName.RISK_CLASSIFICATION,
            ToolName.EXPLANATION,
        ]
        return (
            f"Detected a direct aggregation query ({min_count}+ transactions under "
            f"${max_amount:,.0f}). Ran aggregation_rule_tool only, skipping ML."
        )

    def _run_full_pipeline(self, filters_dict: dict[str, Any]) -> str:
        self._execute_tool(
            ToolName.FEATURE_ENGINEERING,
            {"filters": filters_dict},
            reason="Broad query — computing features across the filtered dataset.",
        )
        self._execute_tool(
            ToolName.ANOMALY_DETECTION,
            {"filters": filters_dict},
            reason="Running hybrid anomaly detection across all accounts in scope.",
        )
        self._execute_tool(
            ToolName.RISK_CLASSIFICATION,
            {},
            reason="Classifying all scored accounts into risk tiers.",
        )
        self._execute_tool(
            ToolName.EXPLANATION,
            {},
            reason="Explaining why each account was flagged.",
        )
        return "No specific pattern detected in the query; ran the full detection pipeline."
