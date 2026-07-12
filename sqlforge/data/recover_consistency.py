"""
Recover consistency-rejected SFT pairs — a $0 rescue.

The Haiku self-consistency check in generate_sft.py proved unreliable (it dropped
~1000 pairs that were actually correct and house-style compliant). Those rejects
had ALREADY passed execution + house-style validation (consistency runs last), so
they are recoverable with no additional API cost.

This reads rejects.jsonl, takes every `consistency_no` entry, re-validates its SQL
(execution + house-style), and appends the survivors to sft_pairs.jsonl with a
simulation-aware system prompt and a `recovered_consistency` source tag.

Usage:
    python sqlforge/data/recover_consistency.py
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from schema_context import get_schema, is_simulation, DEFAULT_DB_PATH
from validate import validate_pair
import generate_sft as g

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
PAIRS_PATH = os.path.join(DATASETS_DIR, "sft_pairs.jsonl")
REJECTS_PATH = os.path.join(DATASETS_DIR, "rejects.jsonl")


def main():
    schema = get_schema(DEFAULT_DB_PATH)
    bench = g.load_benchmark_questions()

    have = set()
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                have.add(g.norm_q(json.loads(line)["messages"][1]["content"]))
            except Exception:
                continue

    recovered = {}
    with open(PAIRS_PATH, "a", encoding="utf-8") as out, \
         open(REJECTS_PATH, encoding="utf-8") as rej:
        for line in rej:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("reason") != "consistency_no" or not r.get("sql"):
                continue
            q, sql, tier = r["question"], r["sql"], r.get("tier", "unknown")
            qn = g.norm_q(q)
            if qn in have or qn in bench or g.is_contaminated(qn, bench):
                continue
            v = validate_pair(sql, DEFAULT_DB_PATH, enforce_house_style=True)
            if not v["ok"]:
                continue
            have.add(qn)
            rec = {
                "messages": [
                    {"role": "system", "content": g.build_system_prompt(schema, is_simulation(q))},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": sql},
                ],
                "meta": {"tier": tier, "source": "recovered_consistency",
                         "rows": v["rows"], "cols": v["cols"]},
            }
            out.write(json.dumps(rec) + "\n")
            recovered[tier] = recovered.get(tier, 0) + 1

    total = sum(recovered.values())
    print(f"Recovered {total} pairs -> {PAIRS_PATH}")
    for t, c in sorted(recovered.items(), key=lambda x: -x[1]):
        print(f"  {t:18s} +{c}")


if __name__ == "__main__":
    main()
