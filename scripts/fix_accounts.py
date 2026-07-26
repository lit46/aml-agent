import csv

TRANS_SAMPLE = "sampled/transactions_sample.csv"
ACCOUNTS_FILE = "HI-Medium_accounts.csv"
OUT_PATH = "sampled/accounts_sample.csv"


def normalize_bank_id(bank_id: str) -> int:
    """Strip leading zeros / normalize to int so '011304' == '11304'."""
    return int(bank_id)


def main():
    used_accounts = set()

    with open(TRANS_SAMPLE, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # From Bank, Account, To Bank, Account.1
            used_accounts.add((normalize_bank_id(row[1]), row[2]))
            used_accounts.add((normalize_bank_id(row[3]), row[4]))

    print(f"Unique accounts referenced in sample: {len(used_accounts):,}")

    kept = 0
    with open(ACCOUNTS_FILE, "r", encoding="utf-8", newline="") as fin, \
         open(OUT_PATH, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        header = next(reader)
        writer = csv.writer(fout)
        writer.writerow(header)
        for row in reader:
            # Bank ID, Account Number
            key = (normalize_bank_id(row[1]), row[2])
            if key in used_accounts:
                writer.writerow(row)
                kept += 1

    print(f"Accounts matched and written: {kept:,}")


if __name__ == "__main__":
    main()
