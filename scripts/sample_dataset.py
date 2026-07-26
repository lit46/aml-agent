import csv
import random
import re
import os

PATTERNS_FILE = "HI-Medium_Patterns.txt"
TRANS_FILE = "HI-Medium_Trans.csv"
ACCOUNTS_FILE = "HI-Medium_accounts.csv"

OUT_DIR = "sampled"
os.makedirs(OUT_DIR, exist_ok=True)

NORMAL_SAMPLE_RATE = 0.001  # ~0.1% of normal txns — tune after first run
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def parse_patterns(path):
    """Parse the block-structured patterns file into a lookup:
    exact raw transaction row (tuple) -> (pattern_type, attempt_id)
    """
    lookup = {}
    current_type = None
    attempt_id = -1
    begin_re = re.compile(r"BEGIN LAUNDERING ATTEMPT - (.+)")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                match = begin_re.match(line)
                current_type = match.group(1).split(":")[0].strip()
                attempt_id += 1
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                current_type = None
                continue
            fields = tuple(line.split(","))
            lookup[fields] = (current_type, attempt_id)
    return lookup


def main():
    print("Parsing patterns file...")
    pattern_lookup = parse_patterns(PATTERNS_FILE)
    print(f"Loaded {len(pattern_lookup):,} labeled laundering transactions.")

    used_accounts = set()
    illicit_count = 0
    normal_kept = 0
    total_rows = 0

    trans_out_path = os.path.join(OUT_DIR, "transactions_sample.csv")

    with open(TRANS_FILE, "r", encoding="utf-8", newline="") as fin, \
         open(trans_out_path, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        header = next(reader)
        writer = csv.writer(fout)
        writer.writerow(header + ["Pattern Type", "Attempt ID"])

        for row in reader:
            total_rows += 1
            if total_rows % 2_000_000 == 0:
                print(f"...processed {total_rows:,} rows "
                      f"(illicit kept: {illicit_count:,}, "
                      f"normal kept: {normal_kept:,})")

            is_laundering = row[-1] == "1"

            if is_laundering:
                pattern_type, attempt_id = pattern_lookup.get(
                    tuple(row), ("UNSPECIFIED", -1)
                )
                writer.writerow(row + [pattern_type, attempt_id])
                illicit_count += 1
                used_accounts.add((row[1], row[2]))
                used_accounts.add((row[3], row[4]))
            else:
                if random.random() < NORMAL_SAMPLE_RATE:
                    writer.writerow(row + ["NONE", -1])
                    normal_kept += 1
                    used_accounts.add((row[1], row[2]))
                    used_accounts.add((row[3], row[4]))

    print(f"\nDone. Total rows scanned: {total_rows:,}")
    print(f"Illicit transactions kept: {illicit_count:,}")
    print(f"Normal transactions kept: {normal_kept:,}")
    print(f"Unique accounts referenced: {len(used_accounts):,}")

    accounts_out_path = os.path.join(OUT_DIR, "accounts_sample.csv")
    kept_accounts = 0
    with open(ACCOUNTS_FILE, "r", encoding="utf-8", newline="") as fin, \
         open(accounts_out_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        header = next(reader)
        writer = csv.writer(fout)
        writer.writerow(header)
        for row in reader:
            if (row[1], row[2]) in used_accounts:
                writer.writerow(row)
                kept_accounts += 1

    print(f"Accounts kept: {kept_accounts:,}")


if __name__ == "__main__":
    main()
