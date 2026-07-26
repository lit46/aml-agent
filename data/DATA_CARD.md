# Data Card — AML Transaction Dataset

## Source

- **Original dataset**: [IBM Transactions for Anti-Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) — Kaggle
- **Variant used**: `HI-Medium` (Higher Illicit ratio, Medium size)
- **License**: Data released under CDLA-Sharing-1.0 (Community Data License Agreement); accompanying code under Apache-2.0. See the Kaggle page for full terms.
- **Generation method**: Synthetic multi-agent transaction simulation (AMLworld / AMLSim-based). Real, labeled money-laundering transaction data does not exist in public form for privacy and regulatory reasons, so a well-documented synthetic simulator is the standard approach in this research area.

## Why this dataset

- Provides a binary `Is Laundering` flag on every transaction **and** a separate patterns file naming the specific laundering typology (structuring/layering-style behaviors) for most illicit transactions — this lets the agent's explanation tool describe *why* a transaction was flagged in specific, pattern-aware language rather than a generic anomaly score.
- Includes linked account and bank identifiers, enabling account-level feature engineering (transaction velocity, rolling sums, fan-in/fan-out behavior).

## Sampling methodology

The full `HI-Medium` release contains:
- `HI-Medium_Trans.csv` — 31,898,238 transactions (~2.9 GB)
- `HI-Medium_accounts.csv` — 2,087,786 accounts (~139 MB)
- `HI-Medium_Patterns.txt` — 22,743 transactions labeled across 8 named laundering typologies, grouped into individual "attempts"

Random row sampling was **not** used, because it can shatter a laundering pattern (e.g. a CYCLE attempt is only meaningful as a complete multi-hop chain — sampling 2 of 6 hops destroys the pattern). Instead:

1. **All transactions flagged `Is Laundering = 1`** in the full transaction file were kept (35,230 rows), each cross-referenced against the patterns file and tagged with its specific typology (`Pattern Type`) and a synthetic `Attempt ID` grouping transactions belonging to the same laundering attempt. Illicit transactions not matched to a named typology are tagged `UNSPECIFIED`.
2. **A random 0.1% sample of normal (`Is Laundering = 0`) transactions** was drawn (31,492 rows) to provide baseline/legitimate behavior for contrast.
3. **Accounts were filtered** to only those referenced by at least one sampled transaction (95,057 of ~2.09M accounts).

Reproduction scripts: `scripts/sample_dataset.py` (transaction sampling) and `scripts/fix_accounts.py` (account filtering).

### ⚠️ Known deliberate bias — do not treat as real-world prevalence

The resulting sample has an illicit-to-normal ratio of roughly **53% : 47%** — vastly higher than real-world money laundering prevalence (typically well under 1% of transactions). This is intentional oversampling to ensure the anomaly detection and explanation tools have enough positive examples to demonstrate meaningfully within a hackathon dataset. This should be stated explicitly in any demo or write-up — it is a dataset construction choice, not a claim about real-world AML base rates.

## Final sample stats

| File | Rows (excl. header) | Size |
|---|---|---|
| `transactions_sample.csv` | 66,722 | 6.8 MB |
| `accounts_sample.csv` | 95,057 | 6.4 MB |

**Pattern type distribution** (illicit transactions only, 35,230 total):

| Pattern Type | Count |
|---|---|
| UNSPECIFIED | 12,487 |
| GATHER-SCATTER | 4,289 |
| SCATTER-GATHER | 3,988 |
| STACK | 3,986 |
| FAN-IN | 2,315 |
| CYCLE | 2,235 |
| BIPARTITE | 2,135 |
| FAN-OUT | 2,128 |
| RANDOM | 1,667 |

## Column dictionary

### `transactions_sample.csv`

| Column | Description |
|---|---|
| `Timestamp` | Transaction date/time (`YYYY/MM/DD HH:MM`) |
| `From Bank` | Sender's bank ID |
| `Account` | Sender's account number |
| `To Bank` | Receiver's bank ID |
| `Account.1` | Receiver's account number |
| `Amount Received` | Amount received, in receiving currency |
| `Receiving Currency` | Currency of amount received |
| `Amount Paid` | Amount paid, in payment currency |
| `Payment Currency` | Currency of amount paid |
| `Payment Format` | Payment rail/method (e.g. ACH, Cheque, Reinvestment) |
| `Is Laundering` | Binary flag, 1 = confirmed illicit transaction |
| `Pattern Type` | Laundering typology (STACK / CYCLE / FAN-IN / FAN-OUT / BIPARTITE / GATHER-SCATTER / SCATTER-GATHER / RANDOM / UNSPECIFIED / NONE) |
| `Attempt ID` | Groups transactions belonging to the same laundering attempt; -1 for non-laundering transactions |

### `accounts_sample.csv`

| Column | Description |
|---|---|
| `Bank Name` | Name of the bank |
| `Bank ID` | Numeric bank identifier (matches `From Bank`/`To Bank`, note: not zero-padded here — see gotcha below) |
| `Account Number` | Account identifier (matches `Account`/`Account.1`) |
| `Entity ID` | Internal entity identifier |
| `Entity Name` | Entity type and ID (e.g. "Corporation #183669") — no real names, synthetic |

## Known gotcha for anyone joining these tables

`Bank ID` in `accounts_sample.csv` is a plain integer (e.g. `11304`), while `From Bank`/`To Bank` in `transactions_sample.csv` may be zero-padded strings (e.g. `011304`). **Normalize both to integers before joining** — matching as raw strings will silently fail (this caused a 0-match bug during dataset preparation).
