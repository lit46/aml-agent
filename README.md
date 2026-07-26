# 🕵️ Sentinel AML

AI-powered Anti-Money Laundering investigation agent. Ask a question in
plain English — an LLM plans which detection tools to run and in what
order, executes them against real transaction data, and returns
explainable, risk-tiered results. No LLM key? It falls back to a fully
deterministic rule-based engine automatically, so the app is always usable.

## What it does

Sentinel AML doesn't run a fixed pipeline on every query. Claude (or any
OpenAI-compatible model) reads your question, decides which of five tools
are actually needed, and calls them in a sensible order:

| Tool | Purpose |
|---|---|
| `aggregation_rule_tool` | Fast threshold/counting queries (e.g. "accounts with 10+ transactions") — no ML needed. |
| `feature_engineering_tool` | Computes per-account behavioral features (transaction velocity, counterparty spread, near-threshold amounts, currency hopping, etc.). |
| `anomaly_detection_tool` | Hybrid suspicion score — an Isolation Forest model combined with rule-based AML signals (structuring, fan-in/fan-out, rapid succession transfers). |
| `risk_classification_tool` | Buckets scored accounts into LOW / MEDIUM / HIGH risk tiers with a recommended action. |
| `explanation_tool` | Generates a human-readable explanation for *why* each account was flagged. |

A simple query like *"which accounts made 10+ transactions under
$10,000?"* only triggers `aggregation_rule_tool`. A broad query like
*"find suspicious activity in the last 30 days"* runs the full pipeline,
scoped by whatever date/amount/account filters it can pull out of your
question.

## Architecture

```
app.py                          Streamlit UI
app/
  agents/
    agent.py                    Top-level facade — picks LLM or rule-based mode
    llm_orchestrator.py         Claude/OpenAI-compatible tool-use loop
    rule_based_orchestrator.py  Deterministic fallback (no LLM required)
    base_orchestrator.py        Shared tool execution + data threading
    tool_registry.py            Tool instantiation + JSON schemas for the LLM
    providers/
      openai_compatible_client.py   Adapter: Groq / Gemini / OpenRouter / any
                                     OpenAI-compatible endpoint speaks the
                                     Anthropic Messages API shape
  tools/                        The 5 tools above, each a standalone, testable unit
  schemas/                      Pydantic models for queries, results, and responses
  services/
    data_loader.py               Loads and filters the transaction/account CSVs
  config.py                     Environment-driven settings (see below)
data/
  DATA_CARD.md                  Dataset provenance, sampling methodology, known bias
  accounts_sample.csv / transactions_sample.csv
scripts/                        Reproducible dataset sampling scripts
tests/                          58 tests across tools, orchestrators, schemas, adapter
```

**Why an orchestrator facade?** `AMLAgent` never crashes on an LLM
failure (bad key, no key, rate limit, network error) — it transparently
falls back to `RuleBasedOrchestrator` and reports *why* in
`supporting_metrics.fallback_reason`, so a failure is visible in the UI
instead of silently changing behavior.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add a key if you want the LLM-driven
orchestrator:

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

**No key? No problem.** Leave `.env` unset (or don't create it at all)
and the app runs entirely in deterministic rule-based mode — this is a
first-class supported mode, not a degraded one.

You can also skip `.env` entirely and paste any provider's key directly
into the sidebar at runtime (Anthropic, Groq, Google Gemini, OpenRouter,
or any custom OpenAI-compatible endpoint) — it's used only for that
session and never written to disk.

## Run it

```bash
streamlit run app.py
```

## Test it

```bash
pytest tests/ -q
```

58 tests cover every tool in isolation, both orchestrators, the
schema/validation layer, and the OpenAI-compatible provider adapter
(scripted against fake clients — no network calls, no API key required
to run the suite).

## Example queries

```
Which accounts made 10+ transactions under $10,000?
Find suspicious activity in the last 30 days
Is account <account_id> suspicious?
Show anomalous transactions between $5,000 and $15,000
Are there any structuring patterns in the data?
```

## Dataset

Sampled from IBM's synthetic AML transaction dataset (Kaggle,
`HI-Medium` variant) — 66,722 transactions across 95,057 accounts, with
every illicit transaction retained and cross-referenced against its
laundering typology. Full provenance, sampling methodology, and an
explicit disclosure of the dataset's deliberate illicit-rate oversampling
are documented in [`data/DATA_CARD.md`](data/DATA_CARD.md).

## Known limitations

- The Isolation Forest model requires at least 10 accounts in scope to
  produce meaningful scores; below that it falls back to the rule-based
  score alone.
- The illicit-to-normal transaction ratio in the sample (~53:47) is
  vastly higher than real-world prevalence (<1%) — a deliberate dataset
  construction choice for demo purposes, not a claim about real-world AML
  base rates. See `data/DATA_CARD.md`.
