"""
SQLForge Phase 3 — targeted DPO augmentation for the categorical-literal bug.

On-policy mining over TRAINING questions misses the churn_risk='High' bug, because
the SFT model already uses the correct literal on training phrasings — the mistake
only appears on novel EVAL phrasings. So we construct the preference signal directly:
for every training gold SQL that uses a correct categorical literal, emit a pair whose
`rejected` corrupts that literal to the buggy form the model reverts to. DPO then
learns 'HIGH_RISK' > 'High' and generalizes it.

Verified: the corrupted `rejected` is executed and only kept if it actually returns a
DIFFERENT result from gold (i.e. the corruption genuinely breaks the query).

CPU-only, no GPU. Appends to dpo_pairs.jsonl (run AFTER mine_dpo_pairs.py).

Run:
    python sqlforge/data/augment_literal_pairs.py
"""

import os
import re
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase0"))

from eval_compare import execute_sql, compare_results  # noqa: E402
from schema_context import DEFAULT_DB_PATH             # noqa: E402

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
TRAIN_PATH = os.path.join(DATASETS_DIR, "train.jsonl")
OUT_PATH = os.path.join(DATASETS_DIR, "dpo_pairs.jsonl")

# correct literal -> the buggy form the base/SFT model reverts to on eval
CORRUPTIONS = {
    "HIGH_RISK": "High",
    "MEDIUM_RISK": "Medium",
    "LOW_RISK": "Low",
}


def corrupt(sql: str):
    """Return a copy of sql with churn literals swapped to the buggy form, or None
    if it contains no correct churn literal to corrupt."""
    out, changed = sql, False
    for good, bad in CORRUPTIONS.items():
        pat = re.compile(f"'{good}'")
        if pat.search(out):
            out = pat.sub(f"'{bad}'", out)
            changed = True
    return out if changed else None


def main():
    records = []
    with open(TRAIN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    made = 0
    skipped_same = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for rec in records:
            gold = rec["messages"][2]["content"]
            rejected = corrupt(gold)
            if rejected is None:
                continue

            gold_df, gerr = execute_sql(gold, DEFAULT_DB_PATH)
            if gerr or gold_df is None or gold_df.empty:
                continue
            rej_df, rerr = execute_sql(rejected, DEFAULT_DB_PATH)
            # keep only if the corruption actually changes the result (it should:
            # the bad literal matches no rows -> empty / different)
            if not rerr and rej_df is not None and compare_results(gold_df, rej_df)["match"]:
                skipped_same += 1
                continue

            out.write(json.dumps({
                "prompt": [rec["messages"][0], rec["messages"][1]],
                "chosen": gold,
                "rejected": rejected,
                "meta": {"tier": rec.get("meta", {}).get("tier", "?"), "kind": "literal_aug"},
            }) + "\n")
            made += 1

    print(f"Appended {made} literal-augmentation pairs -> {OUT_PATH}")
    print(f"  (skipped {skipped_same} where corruption didn't change the result)")


if __name__ == "__main__":
    main()
