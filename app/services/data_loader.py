"""Data loading service.

This is the single place in the codebase that reads the raw CSV files and
normalizes them. Every tool depends on this rather than reading CSVs
directly, so schema fixes (like bank ID zero-padding) only need to happen
in one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import settings

TRANSACTION_COLUMN_MAP = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
    "Pattern Type": "pattern_type",
    "Attempt ID": "attempt_id",
}

ACCOUNT_COLUMN_MAP = {
    "Bank Name": "bank_name",
    "Bank ID": "bank_id",
    "Account Number": "account_number",
    "Entity ID": "entity_id",
    "Entity Name": "entity_name",
}


def _normalize_bank_id(series: pd.Series) -> pd.Series:
    """Normalize bank IDs to plain integer strings.

    The raw files are inconsistent about zero-padding (e.g. '011304' in
    transactions vs '11304' in accounts) — this was the cause of a 0-match
    bug during dataset preparation. Always route bank ID comparisons
    through this function.
    """
    return series.astype(str).astype(int).astype(str)


def load_transactions(path: Path | str | None = None) -> pd.DataFrame:
    """Load and normalize the transactions CSV.

    Renames columns to snake_case, parses timestamps, normalizes bank IDs,
    and casts the laundering flag to bool.
    """
    resolved_path = Path(path) if path else settings.transactions_path
    df = pd.read_csv(resolved_path)
    df = df.rename(columns=TRANSACTION_COLUMN_MAP)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    df["from_bank"] = _normalize_bank_id(df["from_bank"])
    df["to_bank"] = _normalize_bank_id(df["to_bank"])
    df["from_account"] = df["from_account"].astype(str)
    df["to_account"] = df["to_account"].astype(str)
    df["is_laundering"] = df["is_laundering"].astype(bool)

    return df


def load_accounts(path: Path | str | None = None) -> pd.DataFrame:
    """Load and normalize the accounts CSV."""
    resolved_path = Path(path) if path else settings.accounts_path
    df = pd.read_csv(resolved_path)
    df = df.rename(columns=ACCOUNT_COLUMN_MAP)

    df["bank_id"] = _normalize_bank_id(df["bank_id"])
    df["account_number"] = df["account_number"].astype(str)

    return df


class DataStore:
    """Lazily loads and caches the transaction and account datasets.

    Passed into tools via dependency injection so the orchestrator controls
    a single shared instance, and tests can point it at fixture files
    instead of the full dataset.
    """

    def __init__(
        self,
        transactions_path: Path | str | None = None,
        accounts_path: Path | str | None = None,
    ) -> None:
        self._transactions_path = transactions_path
        self._accounts_path = accounts_path
        self._transactions: pd.DataFrame | None = None
        self._accounts: pd.DataFrame | None = None

    @property
    def transactions(self) -> pd.DataFrame:
        if self._transactions is None:
            self._transactions = load_transactions(self._transactions_path)
        return self._transactions

    @property
    def accounts(self) -> pd.DataFrame:
        if self._accounts is None:
            self._accounts = load_accounts(self._accounts_path)
        return self._accounts
