"""LLM-driven orchestrator using Claude's native tool-use.

Claude decides which tools to call, in what order, based on the user's
natural language query — this is what makes tool selection dynamic rather
than a fixed pipeline. Data threading between tool calls (e.g. passing
computed account features into the anomaly detector) is handled by
BaseOrchestrator, not by the model — asking an LLM to copy large
intermediate data structures between calls would be both wasteful and
unreliable.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from app.agents.base_orchestrator import BaseOrchestrator
from app.agents.tool_registry import get_anthropic_tool_schemas
from app.config import settings
from app.schemas import AgentResponse, ToolName
from app.services.data_loader import DataStore

# Tool results can contain one record per account (e.g. account_features,
# anomaly_scores) — on an unfiltered query over the full ~95k-account
# dataset that's tens of thousands of JSON objects. The LLM never needs to
# see that raw data (BaseOrchestrator already threads it between tool
# calls internally via pipeline_state), it only needs enough to decide the
# next step and write a final summary. Sending the full list back into the
# conversation on every turn made the payload grow each turn until
# providers with tighter request-size limits (e.g. Groq) started rejecting
# it with a 413.
_MAX_ITEMS_PER_LIST = 5


def _summarize_for_llm(result: Any) -> dict[str, Any]:
    """Cap any large list fields in a tool result before it goes back to the LLM."""
    dumped = result.model_dump()
    data = dumped.get("data")
    if isinstance(data, dict):
        summarized_data = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) > _MAX_ITEMS_PER_LIST:
                summarized_data[key] = {
                    "total_count": len(value),
                    "sample": value[:_MAX_ITEMS_PER_LIST],
                    "note": (
                        f"{len(value)} items total; showing first "
                        f"{_MAX_ITEMS_PER_LIST} only. Full results are "
                        "already available to the final response."
                    ),
                }
            else:
                summarized_data[key] = value
        dumped["data"] = summarized_data
    return dumped


SYSTEM_PROMPT_TEMPLATE = """You are an AML (Anti-Money Laundering) investigation agent.

Today's date is {today}. When a query mentions a relative time window
(e.g. "last 30 days", "this month", "since last week"), compute the exact
start_date/end_date yourself and pass them as ISO dates (YYYY-MM-DD) —
never pass phrases like "30 days ago" or "today" literally into a filter.

You have access to a set of tools for analyzing financial transactions. You
must NOT run every tool for every query — decide which tools are actually
needed based on the user's request, and invoke only those, in a sensible
order.

Guidance:
- Simple counting/threshold questions (e.g. "which accounts made 10+
  transactions under $10,000?") need ONLY aggregation_rule_tool. Do not run
  feature engineering or anomaly detection for these.
- Single-entity questions (e.g. "is account X suspicious?") should scope
  filters to that entity and run feature_engineering_tool ->
  anomaly_detection_tool -> risk_classification_tool -> explanation_tool.
- Broad exploratory questions (e.g. "find suspicious activity in the last
  30 days") should run feature_engineering_tool -> anomaly_detection_tool
  -> risk_classification_tool -> explanation_tool, scoped by any date or
  amount filters mentioned in the query.
- risk_classification_tool must run after anomaly_detection_tool.
- explanation_tool should typically run last.

Every tool call must include a brief "reason" explaining why you're calling
it. When you have enough information to answer the user, respond with a
final plain-text summary (no more tool calls) describing what you found and
which tools you used."""


class LLMOrchestrator(BaseOrchestrator):
    """Claude tool-use loop tying all AML tools together dynamically."""

    def __init__(
        self,
        data_store: DataStore,
        client: Any,
        model: str | None = None,
        max_turns: int = 6,
    ) -> None:
        super().__init__(data_store)
        self._client = client
        self._model = model or settings.claude_model
        self._max_turns = max_turns

    def handle_query(self, query: str) -> AgentResponse:
        self._reset_state()
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]

        for _ in range(self._max_turns):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1500,
                system=SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat()),
                tools=get_anthropic_tool_schemas(),
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                final_text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                return self._compose_response(query, final_text, used_fallback=False)

            tool_result_content = []
            for block in tool_use_blocks:
                tool_name = ToolName(block.name)
                reason = block.input.get("reason", "")
                result = self._execute_tool(tool_name, block.input, reason=reason)
                tool_result_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(_summarize_for_llm(result)),
                    }
                )
            messages.append({"role": "user", "content": tool_result_content})

        return self._compose_response(
            query,
            "Reached the maximum number of reasoning turns without a final answer.",
            used_fallback=False,
        )
