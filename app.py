"""Streamlit UI for Sentinel AML.

Lets a user (or judge) enter their own Anthropic API key at runtime — it
is used only for this session and is never written to disk. If no key is
given, the agent automatically falls back to the deterministic rule-based
detection engine, so the app is always usable with zero setup.
"""

from __future__ import annotations

import streamlit as st

from app.agents.agent import AMLAgent
from app.schemas import AgentResponse, ExecutionPlan, FlaggedItem
from app.services.data_loader import DataStore

st.set_page_config(page_title="Sentinel AML", page_icon="🕵️", layout="wide")


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
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            help=(
                "Used only for this session, never written to disk. Leave "
                "blank to run in rule-based fallback mode (no key required)."
            ),
        )
        if api_key_input:
            st.success("Using your API key for this session.")
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
        agent = AMLAgent(data_store, api_key=api_key_input or None)
        with st.spinner("Running the agent..."):
            result = agent.handle_query(query)
        render_result(result)
    elif run_clicked and not query:
        st.warning("Please enter a query first.")


if __name__ == "__main__":
    main()
