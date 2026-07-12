"""
Split sft_pairs.jsonl into train / val (default 95/5), stratified by tier so the
validation set keeps every tier represented.

Usage:
    python sqlforge/data/split.py                 # 95/5
    python sqlforge/data/split.py --val-frac 0.1  # 90/10
"""

import os
import json
import random
import argparse
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(DATASETS_DIR, "sft_pairs.jsonl"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    by_tier = defaultdict(list)
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_tier[rec.get("meta", {}).get("tier", "unknown")].append(rec)

    rng = random.Random(args.seed)
    train, val = [], []
    for tier, recs in by_tier.items():
        rng.shuffle(recs)
        k = max(1, round(len(recs) * args.val_frac))  # >=1 val per tier
        val.extend(recs[:k])
        train.extend(recs[k:])

    rng.shuffle(train)
    rng.shuffle(val)

    train_path = os.path.join(DATASETS_DIR, "train.jsonl")
    val_path = os.path.join(DATASETS_DIR, "val.jsonl")
    for path, rows in ((train_path, train), (val_path, val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    total = len(train) + len(val)
    print(f"Split {total} pairs -> {len(train)} train / {len(val)} val "
          f"({args.val_frac:.0%} val, stratified by tier)")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")
    print("\n  Per-tier (train/val):")
    for tier in sorted(by_tier):
        tv = sum(1 for r in val if r["meta"]["tier"] == tier)
        tt = sum(1 for r in train if r["meta"]["tier"] == tier)
        print(f"    {tier:18s} {tt:4d} / {tv}")


if __name__ == "__main__":
    main()
