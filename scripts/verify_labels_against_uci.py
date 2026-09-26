"""Check label mappings using uniquely matched Cleveland records."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
# Match using attributes whose coding is unchanged between the files.
MATCH_COLUMNS = [
    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang",
    "oldpeak",
]
CP_MAPPING = {1: 3, 2: 1, 3: 2, 4: 0}


def main():
    original = pd.read_csv(UCI_URL, names=COLUMNS, na_values="?")
    ours = pd.read_csv(ROOT / "heart.csv", encoding="utf-8-sig")
    ours = ours.drop_duplicates().copy()

    # Normalize numeric representations before matching.
    for frame in (original, ours):
        for column in MATCH_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="raise").round(6)

    # Do not guess when multiple records share a matching signature.
    original_ambiguous = original.duplicated(MATCH_COLUMNS, keep=False)
    ours_ambiguous = ours.duplicated(MATCH_COLUMNS, keep=False)
    original_unique = original.loc[~original_ambiguous]
    ours_unique = ours.loc[~ours_ambiguous]

    matched = ours_unique.merge(
        original_unique,
        on=MATCH_COLUMNS,
        how="inner",
        suffixes=("_ours", "_uci"),
        validate="one_to_one",
    )

    print(f"Local rows after deduplication: {len(ours)}")
    print(f"Uniquely matched rows: {len(matched)}")
    print(f"Local ambiguous rows excluded: {int(ours_ambiguous.sum())}")
    print(
        "Local unambiguous rows without a unique UCI match: "
        f"{len(ours_unique) - len(matched)}"
    )

    if matched.empty:
        raise SystemExit("FAIL: no unique matches; mappings are unverified.")

    expected_target = (matched["num"] == 0).astype(int)
    expected_cp = matched["cp_uci"].map(CP_MAPPING)

    target_mismatches = int((matched["target"] != expected_target).sum())
    cp_mismatches = int((matched["cp_ours"] != expected_cp).sum())

    print(f"Target mapping mismatches: {target_mismatches}")
    print(f"Chest-pain mapping mismatches: {cp_mismatches}")

    # Report evidence about sentinel codes without assuming equivalence.
    for column, sentinel in [("ca", 4), ("thal", 0)]:
        local_sentinel = matched[f"{column}_ours"] == sentinel
        uci_missing = matched[f"{column}_uci"].isna()
        print(
            f"{column}: local sentinel rows={int(local_sentinel.sum())}; "
            "also missing in matched UCI records="
            f"{int((local_sentinel & uci_missing).sum())}"
        )

    if target_mismatches or cp_mismatches:
        raise SystemExit(
            "FAIL: mapping discrepancies found. Investigate before "
            "changing labels or reporting findings."
        )

    print(
        "PASS: target and chest-pain mappings agree on uniquely "
        "matched records only."
    )
    if len(matched) != len(ours):
        print(
            "LIMITATION: unmatched or ambiguous records remain; "
            "this is not complete row-level verification."
        )


if __name__ == "__main__":
    main()
