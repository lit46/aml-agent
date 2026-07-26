"""Tests for the data loading service, run against small fixture files."""

from pathlib import Path

from app.services.data_loader import DataStore, load_accounts, load_transactions

FIXTURES = Path(__file__).parent / "fixtures"
TRANSACTIONS_FIXTURE = FIXTURES / "sample_transactions.csv"
ACCOUNTS_FIXTURE = FIXTURES / "sample_accounts.csv"


def test_load_transactions_renames_columns_to_snake_case():
    df = load_transactions(TRANSACTIONS_FIXTURE)
    assert "from_bank" in df.columns
    assert "to_account" in df.columns
    assert "From Bank" not in df.columns


def test_load_transactions_parses_timestamp():
    df = load_transactions(TRANSACTIONS_FIXTURE)
    assert df["timestamp"].dtype.kind == "M"  # numpy datetime64


def test_load_transactions_casts_laundering_flag_to_bool():
    df = load_transactions(TRANSACTIONS_FIXTURE)
    assert df["is_laundering"].dtype == bool
    assert df["is_laundering"].sum() == 3  # 3 rows flagged 1 in the fixture


def test_bank_id_zero_padding_is_normalized():
    """Regression test for the leading-zero join bug found during dataset prep."""
    df = load_transactions(TRANSACTIONS_FIXTURE)
    # fixture has From Bank '0111632' -> should normalize to '111632'
    assert (df["from_bank"] == "111632").any()
    assert not (df["from_bank"] == "0111632").any()


def test_load_accounts_renames_columns():
    df = load_accounts(ACCOUNTS_FIXTURE)
    assert "account_number" in df.columns
    assert "bank_id" in df.columns


def test_data_store_lazily_caches_transactions():
    store = DataStore(TRANSACTIONS_FIXTURE, ACCOUNTS_FIXTURE)
    first = store.transactions
    second = store.transactions
    assert first is second  # same object, not reloaded from disk
