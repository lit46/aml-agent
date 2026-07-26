"""Streamlit UI for Sentinel AML.

Lets a user (or judge) enter their own Anthropic API key at runtime — it
is used only for this session and is never written to disk. If no key is
given, the agent automatically falls back to the deterministic rule-based
detection engine, so the app is always usable with zero setup.
"""

from __future__ import annotations

import streamlit as st

from app.agents.agent import AMLAgent
from app.agents.providers.openai_compatible_client import KNOWN_PROVIDERS, OpenAICompatibleClient
from app.schemas import AgentResponse, ExecutionPlan, FlaggedItem
from app.services.data_loader import DataStore

st.set_page_config(page_title="Sentinel AML", page_icon="🕵️", layout="wide")

PROVIDER_OPTIONS = ["Anthropic (Claude)", *KNOWN_PROVIDERS.keys(), "Custom OpenAI-compatible"]


@st.cache_resource
def get_data_store() -> DataStore:
    return DataStore()


def render_plan(plan: ExecutionPlan) -> None:
    st.subheader("Execution plan")
    for step in plan.steps:
        st.markdown(f"- **{step.tool.value}** — {step.reason}")
    if plan.skipped_tools:
        skipped = ", ".join(t.value for t in plan.skipped_tools)
        st.caption(f"Skipped: {skipped}")


def render_flagged_items(flagged_items: list[FlaggedItem]) -> None:
    if not flagged_items:
        st.info("No accounts were flagged as MEDIUM or HIGH risk for this query.")
        return
    st.subheader(f"Flagged items ({len(flagged_items)})")
    rows = [
        {
            "Account": item.entity_id,
            "Risk": item.risk_level.value,
            "Score": round(item.risk_score, 2),
            "Action": item.recommended_action.value,
            "Explanation": item.explanation,
        }
        for item in flagged_items
    ]
    st.dataframe(rows, width='stretch')


def render_result(result: AgentResponse) -> None:
    if result.supporting_metrics.get("used_fallback"):
        reason = result.supporting_metrics.get("fallback_reason")
        if reason:
            st.warning(f"Ran in rule-based fallback mode. Reason: {reason}")
        else:
            st.warning("Ran in rule-based fallback mode (no LLM call was used).")

    st.subheader("Summary")
    st.write(result.execution_summary)

    render_plan(result.plan)
    render_flagged_items(result.flagged_items)

    if "matching_accounts" in result.supporting_metrics:
        st.subheader(f"Matching accounts ({result.supporting_metrics['match_count']})")
        st.dataframe(
            result.supporting_metrics["matching_accounts"], width='stretch'
        )


def main() -> None:
    st.title("🕵️ Sentinel AML")
    st.caption(
        "AI-Powered Suspicious Activity Detection — dynamically orchestrates "
        "feature engineering and anomaly detection tools based on natural "
        "language queries."
    )

    with st.sidebar:
        st.header("Configuration")
        provider = st.selectbox(
            "LLM Provider",
            PROVIDER_OPTIONS,
            help=(
                "No Anthropic balance? Groq offers a genuinely free tier "
                "(no credit card) with tool-calling support — get a key at "
                "console.groq.com."
            ),
        )

        api_key_input = st.text_input(
            f"{provider} API Key",
            type="password",
            help="Used only for this session, never written to disk.",
        )

        base_url = None
        default_model = ""
        if provider == "Anthropic (Claude)":
            default_model = "claude-sonnet-5"
        elif provider == "Custom OpenAI-compatible":
            base_url = st.text_input(
                "Base URL", placeholder="https://api.example.com/v1"
            )
        else:
            base_url = KNOWN_PROVIDERS[provider]["base_url"]
            default_model = KNOWN_PROVIDERS[provider]["default_model"]

        model_input = st.text_input("Model name", value=default_model)

        if api_key_input:
            st.success(f"Using your {provider} key for this session.")
        else:
            st.info("No API key entered — running in rule-based fallback mode.")

    try:
        data_store = get_data_store()
    except Exception as exc:  # noqa: BLE001 - surface to the user, don't crash the app
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    query = st.text_input(
        "Ask a question about the transaction data",
        placeholder="e.g. Which accounts made 10+ transactions under $10,000?",
    )
    run_clicked = st.button("Analyze", type="primary")

    if run_clicked and query:
        agent = _build_agent(data_store, provider, api_key_input, base_url, model_input)
        with st.spinner("Running the agent..."):
            result = agent.handle_query(query)
        render_result(result)
    elif run_clicked and not query:
        st.warning("Please enter a query first.")


def _build_agent(
    data_store: DataStore,
    provider: str,
    api_key_input: str,
    base_url: str | None,
    model_input: str,
) -> AMLAgent:
    """Construct an AMLAgent wired to whichever provider was selected."""
    if not api_key_input:
        return AMLAgent(data_store)  # no key -> rule-based fallback

    if provider == "Anthropic (Claude)":
        return AMLAgent(data_store, api_key=api_key_input, model=model_input or None)

    if not base_url:
        st.error("A base URL is required for a custom OpenAI-compatible provider.")
        st.stop()

    client = OpenAICompatibleClient(base_url=base_url, api_key=api_key_input)
    return AMLAgent(data_store, client=client, model=model_input or None)


if __name__ == "__main__":
    main()
