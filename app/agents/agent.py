"""Top-level agent facade.

Picks the LLM orchestrator when a usable client is available, and
transparently falls back to the deterministic rule-based orchestrator if
the LLM call fails for any reason (missing key, network error, rate limit)
— so the agent never simply crashes during a demo or judging session.
"""

from __future__ import annotations

from typing import Any

from app.agents.llm_orchestrator import LLMOrchestrator
from app.agents.rule_based_orchestrator import RuleBasedOrchestrator
from app.config import settings
from app.schemas import AgentResponse
from app.services.data_loader import DataStore


class AMLAgent:
    """Facade choosing between LLM-driven and rule-based orchestration."""

    def __init__(
        self,
        data_store: DataStore,
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Args:
            data_store: shared data access layer.
            client: an already-constructed Anthropic-compatible client
                (mainly for tests). Takes priority over `api_key` and
                `.env` if provided.
            api_key: an explicit API key to use for this session — lets a
                UI (e.g. Streamlit) pass in a key a user typed at runtime,
                without requiring it to be written to .env. Falls back to
                settings.anthropic_api_key (from .env) if not given.
        """
        self._data_store = data_store
        self._rule_based = RuleBasedOrchestrator(data_store)
        self._llm: LLMOrchestrator | None = None

        resolved_key = api_key or settings.anthropic_api_key

        if client is not None:
            self._llm = LLMOrchestrator(data_store, client=client)
        elif resolved_key:
            try:
                from anthropic import Anthropic

                self._llm = LLMOrchestrator(
                    data_store, client=Anthropic(api_key=resolved_key)
                )
            except Exception:
                self._llm = None

    def handle_query(self, query: str) -> AgentResponse:
        if self._llm is not None:
            try:
                return self._llm.handle_query(query)
            except Exception:
                # Any API/network failure at query time — fall back rather
                # than crash. This is the resilience path for judging.
                pass
        return self._rule_based.handle_query(query)
