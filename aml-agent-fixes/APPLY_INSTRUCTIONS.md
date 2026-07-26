# How to apply these fixes to your repo

Copy each file below into your local `aml-agent` repo at the **exact same
path** (overwriting the existing file), then commit and push.

| File in this zip | Destination in your repo |
|---|---|
| `app.py` | `app.py` (overwrite) |
| `app/schemas/query.py` | `app/schemas/query.py` (overwrite) |
| `app/agents/llm_orchestrator.py` | `app/agents/llm_orchestrator.py` (overwrite) |
| `requirements.txt` | `requirements.txt` (overwrite) |
| `tests/test_llm_orchestrator.py` | `tests/test_llm_orchestrator.py` (overwrite) |
| `.streamlit/config.toml` | `.streamlit/config.toml` (**new file** — this folder probably doesn't exist yet, create it) |

## Fastest way to copy them in (terminal)

If your repo is cloned locally, and this zip is extracted to e.g.
`~/Downloads/aml-agent-fixes/`, run from the root of your repo:

```bash
cp ~/Downloads/aml-agent-fixes/app.py .
cp ~/Downloads/aml-agent-fixes/app/schemas/query.py app/schemas/query.py
cp ~/Downloads/aml-agent-fixes/app/agents/llm_orchestrator.py app/agents/llm_orchestrator.py
cp ~/Downloads/aml-agent-fixes/requirements.txt .
cp ~/Downloads/aml-agent-fixes/tests/test_llm_orchestrator.py tests/test_llm_orchestrator.py
mkdir -p .streamlit
cp ~/Downloads/aml-agent-fixes/.streamlit/config.toml .streamlit/config.toml
```

## Then install the new dependency, test, and push

```bash
pip install -r requirements.txt
pytest tests/ -q          # should show 56 passed
streamlit run app.py      # eyeball the new theme before you submit
git add -A
git commit -m "Fix relative-date ValidationError crash; add investigation-terminal theme"
git push
```

## What changed and why (quick reference)

- **The crash from your screenshot**: `QueryFilters` in `app/schemas/query.py`
  now accepts natural-language dates like "30 days ago" / "today" and
  converts them to real dates instead of raising a `ValidationError` and
  silently dropping into rule-based fallback mode.
- **413 Payload Too Large** (found after the date fix, on real full-scale
  data): `app/agents/llm_orchestrator.py` was echoing the *full* result of
  every tool call (one record per account) back into the conversation on
  every turn. On the real ~95k-account dataset, a single unfiltered
  `feature_engineering_tool` call serialized to **32 MB** — a size no LLM
  provider's request limit will accept. Added `_summarize_for_llm()`,
  which caps large list fields to a 5-item sample + total count before
  they're sent to the LLM. The full data still flows correctly into the
  final response via `pipeline_state`, which was never touched by this bug.
- `app/agents/llm_orchestrator.py`: the system prompt now tells the LLM
  today's actual date, so it computes correct ISO dates itself instead of
  guessing — this is the root-cause fix, the schema change above is the
  safety net.
- `requirements.txt`: added `python-dateutil` explicitly (needed by the fix
  above; it was already present transitively via pandas, but relying on
  that is fragile).
- `tests/test_llm_orchestrator.py`: added a regression test that reproduces
  your exact screenshot scenario end-to-end.
- `app.py` + `.streamlit/config.toml`: UI theme — dark background, amber
  accent, monospace section headers, color-coded risk indicators
  (🔴 HIGH / 🟠 MEDIUM / 🟢 LOW) in the results table.
