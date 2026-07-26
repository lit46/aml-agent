"""Shared pipeline execution logic for all orchestrator strategies.

Both the LLM-driven orchestrator and the rule-based fallback subclass this,
so tool execution, data threading between tools, and response composition
are implemented exactly once rather than duplicated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.agents.tool_registry import build_tool_instances
from app.schemas import (
    AgentResponse,
    AMLPattern,
    ExecutionPlan,
    ExecutionStep,
    FlaggedItem,
    QueryFilters,
    RecommendedAction,
    RiskLevel,
    ToolName,
)
from app.services.data_loader import DataStore


class BaseOrchestrator(ABC):
    """Shared tool execution and response composition for orchestrators."""

    def __init__(self, data_store: DataStore) -> None:
        self._data_store = data_store
        self._tools = build_tool_instances(data_store)
        self._pipeline_state: dict[str, Any] = {}
        self._plan_steps: list[ExecutionStep] = []
        self._skipped_tools: list[ToolName] = []

    @abstractmethod
    def handle_query(self, query: str) -> AgentResponse:
        """Process a natural language query and return a full agent response."""
        raise NotImplementedError

    def _reset_state(self) -> None:
        self._pipeline_state = {}
        self._plan_steps = []
        self._skipped_tools = []

    def _execute_tool(
        self, tool_name: ToolName, args: dict[str, Any], reason: str = ""
    ) -> Any:
        """Run one tool, automatically threading pipeline state in and out.

        `args` holds only the query-derived parameters (filters,
        min_transaction_count) — never intermediate results. Those are
        injected here from self._pipeline_state based on which tool is
        being called.
        """
        self._plan_steps.append(
            ExecutionStep(tool=tool_name, reason=reason or "No reason given.")
        )

        kwargs: dict[str, Any] = {}
        if args.get("filters"):
            kwargs["filters"] = self._parse_filters(args["filters"])
        if args.get("min_transaction_count") is not None:
            kwargs["min_transaction_count"] = args["min_transaction_count"]

        if (
            tool_name == ToolName.ANOMALY_DETECTION
            and self._pipeline_state.get("account_features") is not None
        ):
            kwargs["account_features"] = self._pipeline_state["account_features"]
        if tool_name == ToolName.RISK_CLASSIFICATION:
            kwargs["anomaly_scores"] = self._pipeline_state.get("anomaly_scores", [])
        if tool_name == ToolName.EXPLANATION:
            kwargs["anomaly_scores"] = self._pipeline_state.get("anomaly_scores", [])
            kwargs["account_features"] = self._pipeline_state.get(
                "account_features", []
            )
            kwargs["risk_classifications"] = self._pipeline_state.get(
                "risk_classifications", []
            )

        result = self._tools[tool_name].run(**kwargs)

        if result.success:
            data = result.data
            if "account_features" in data:
                self._pipeline_state["account_features"] = data["account_features"]
            if "anomaly_scores" in data:
                self._pipeline_state["anomaly_scores"] = data["anomaly_scores"]
            if "classifications" in data:
                self._pipeline_state["risk_classifications"] = data["classifications"]
            if "explanations" in data:
                self._pipeline_state["explanations"] = data["explanations"]
            if "matching_accounts" in data:
                self._pipeline_state["matching_accounts"] = data["matching_accounts"]

        return result

    @staticmethod
    def _parse_filters(raw: dict[str, Any] | QueryFilters | None) -> QueryFilters:
        if raw is None:
            return QueryFilters()
        if isinstance(raw, QueryFilters):
            return raw
        return QueryFilters.model_validate(raw)

    def _compose_response(
        self, query: str, execution_summary: str, used_fallback: bool
    ) -> AgentResponse:
        plan = ExecutionPlan(steps=self._plan_steps, skipped_tools=self._skipped_tools)
        flagged_items = self._build_flagged_items()

        metrics: dict[str, Any] = {"used_fallback": used_fallback}
        if "matching_accounts" in self._pipeline_state:
            metrics["matching_accounts"] = self._pipeline_state["matching_accounts"]
            metrics["match_count"] = len(self._pipeline_state["matching_accounts"])

        return AgentResponse(
            query=query,
            execution_summary=execution_summary,
            plan=plan,
            flagged_items=flagged_items,
            supporting_metrics=metrics,
        )

    def _build_flagged_items(self) -> list[FlaggedItem]:
        classifications = {
            row["account_id"]: row
            for row in self._pipeline_state.get("risk_classifications", [])
        }
        scores = {
            row["account_id"]: row
            for row in self._pipeline_state.get("anomaly_scores", [])
        }
        explanations = {
            row["account_id"]: row["explanation"]
            for row in self._pipeline_state.get("explanations", [])
        }

        items: list[FlaggedItem] = []
        for account_id, classification in classifications.items():
            if classification["risk_level"] == "LOW":
                continue
            score_row = scores.get(account_id, {})
            items.append(
                FlaggedItem(
                    entity_id=account_id,
                    entity_type="account",
                    risk_score=score_row.get("hybrid_score", 0.0),
                    risk_level=RiskLevel(classification["risk_level"]),
                    pattern_type=AMLPattern.UNSPECIFIED,
                    explanation=explanations.get(account_id, "No explanation generated."),
                    recommended_action=RecommendedAction(
                        classification["recommended_action"]
                    ),
                )
            )
        items.sort(key=lambda item: item.risk_score, reverse=True)
        return items
