"""Tests for AMLAgent — the facade choosing LLM vs rule-based orchestration.

These verify the resilience story explicitly: a failing/unavailable LLM
client must never crash a query, it must fall back to the rule-based
orchestrator instead.
"""

from pathlib import Path

from app.agents.agent import AMLAgent
from app.config import settings
from app.services.data_loader import DataStore
from tests.fakes.anthropic_client import RaisingClient, ScriptedClient, fake_response, text_block

FIXTURES = Path(__file__).parent / "fixtures"


def _make_store() -> DataStore:
    return DataStore(
        FIXTURES / "sample_transactions.csv", FIXTURES / "sample_accounts.csv"
    )


def test_uses_llm_path_when_client_provided_and_succeeds():
    client = ScriptedClient([fake_response([text_block("No tools needed for this.")])])
    agent = AMLAgent(_make_store(), client=client)

    result = agent.handle_query("simple question")

    assert result.execution_summary == "No tools needed for this."
    assert result.supporting_metrics["used_fallback"] is False


def test_falls_back_when_client_raises():
    agent = AMLAgent(_make_store(), client=RaisingClient())

    result = agent.handle_query("Which accounts made 3+ transactions under $10,000?")

    # Should not raise, and should have used the rule-based path instead
    assert result.supporting_metrics["used_fallback"] is True
    assert "simulated API failure" in result.supporting_metrics["fallback_reason"]
    assert len(result.plan.steps) > 0


def test_uses_rule_based_directly_when_no_client_and_no_api_key(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    agent = AMLAgent(_make_store())  # no client injected

    result = agent.handle_query("Find suspicious activity in the last 30 days")

    assert result.supporting_metrics["used_fallback"] is True
    assert "No API key" in result.supporting_metrics["fallback_reason"]
    assert agent._llm is None


def test_explicit_api_key_override_constructs_llm_orchestrator(monkeypatch):
    """A UI-supplied key (e.g. pasted into Streamlit) should work even
    when settings.anthropic_api_key (.env) is not set at all."""
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    agent = AMLAgent(_make_store(), api_key="sk-ant-fake-key-for-construction-test")

    assert agent._llm is not None
