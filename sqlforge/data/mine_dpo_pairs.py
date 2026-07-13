"""
SQLForge Phase 3 — on-policy DPO preference-pair mining.

Samples the SFT model (base + SFT LoRA adapter) over the TRAINING questions, executes
every sample against DuckDB, and grades it against the training gold result set. Then
forms preference pairs:

  - on_policy : the model produced BOTH a correct and an incorrect sample for a question
                -> (chosen = a correct sample, rejected = an incorrect sample).
  - gold_chosen : the model got the question wrong in ALL k samples (e.g. the systematic
                  churn_risk='High' bug) -> (chosen = the gold SQL, rejected = a wrong
                  sample). This is what teaches the fixes the model can't yet sample.

Zero API cost — everything runs on the local SFT model via vLLM. Reuses the exact
comparison logic behind the 74.5% eval, so "correct" means the same thing here.

Run (pod, in the vllm serve venv):
    python sqlforge/data/mine_dpo_pairs.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --adapter /workspace/sqlforge-sft-7b --max-lora-rank 32
"""

import os
import re
import sys
import json
import argparse
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)                                   # validate, schema_context
sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase0"))      # eval_compare

from eval_compare import extract_sql, execute_sql, compare_results  # noqa: E402
from validate import execute_and_validate                          # noqa: E402
from schema_context import DEFAULT_DB_PATH                          # noqa: E402

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
TRAIN_PATH = os.path.join(DATASETS_DIR, "train.jsonl")

_WS = re.compile(r"\s+")


def _norm(sql: str) -> str:
    return _WS.sub(" ", sql.strip().lower())


def sample_all(conversations, model, adapter, k, temperature, max_tokens,
               max_model_len, gpu_mem, max_lora_rank):
    """Batched vLLM sampling: k completions per conversation from base+adapter."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"Loading SFT model: {model} + adapter {adapter}")
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem, trust_remote_code=True,
              enable_lora=True, max_lora_rank=max_lora_rank)
    sp = SamplingParams(n=k, temperature=temperature, top_p=0.95,
                        max_tokens=max_tokens, seed=0)
    lora = LoRARequest("sqlforge-sft", 1, adapter)
    outputs = llm.chat(conversations, sp, lora_request=lora)
    return [[o.text for o in out.outputs] for out in outputs]


def grade_samples(samples, gold_df, db_path):
    """Split k raw completions into deduped correct / incorrect SQL lists."""
    correct, wrong, seen = [], [], set()
    for raw in samples:
        sql = extract_sql(raw)
        if not sql:
            continue
        key = _norm(sql)
        if key in seen:
            continue
        seen.add(key)
        ok, df, _reason = execute_and_validate(sql, db_path)
        if not ok:                                   # didn't run / empty / cartesian
            wrong.append(sql)
            continue
        if compare_results(gold_df, df)["match"]:
            correct.append(sql)
        else:
            wrong.append(sql)
    return correct, wrong


def main():
    ap = argparse.ArgumentParser(description="Mine on-policy DPO pairs from the SFT model.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter", required=True, help="Path to the SFT LoRA adapter dir.")
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--k", type=int, default=8, help="Samples per question.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-pairs-per-q", type=int, default=1)
    ap.add_argument("--no-gold-fallback", dest="gold_fallback", action="store_false",
                    help="Don't use gold-as-chosen for questions with 0 correct samples.")
    ap.add_argument("--limit", type=int, default=0, help="Only first N questions (smoke test).")
    ap.add_argument("--train", default=TRAIN_PATH)
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--out", default=os.path.join(DATASETS_DIR, "dpo_pairs.jsonl"))
    args = ap.parse_args()

    # --- load training prompts + gold ---
    records = []
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]

    conversations = [[r["messages"][0], r["messages"][1]] for r in records]
    golds = [r["messages"][2]["content"] for r in records]

    print("=" * 70)
    print("SQLForge Phase 3 — DPO pair mining")
    print(f"  Questions: {len(records)}   k={args.k}   temp={args.temperature}")
    print(f"  Model: {args.model} + {args.adapter}")
    print("=" * 70)

    all_samples = sample_all(conversations, args.model, args.adapter, args.k,
                             args.temperature, args.max_tokens, args.max_model_len,
                             args.gpu_mem, args.max_lora_rank)

    pairs = []
    kinds = Counter()
    tier_pairs = Counter()
    stats = {"gold_error": 0, "no_signal": 0, "all_correct": 0}

    for rec, samples, gold_sql in zip(records, all_samples, golds):
        tier = rec.get("meta", {}).get("tier", "?")
        gold_df, gold_err = execute_sql(gold_sql, args.db)
        if gold_err or gold_df is None or gold_df.empty:
            stats["gold_error"] += 1
            continue

        correct, wrong = grade_samples(samples, gold_df, args.db)
        prompt_msgs = [rec["messages"][0], rec["messages"][1]]

        if correct and wrong:
            chosen = correct[0]
            for w in wrong[: args.max_pairs_per_q]:
                pairs.append({"prompt": prompt_msgs, "chosen": chosen, "rejected": w,
                              "meta": {"tier": tier, "kind": "on_policy"}})
                kinds["on_policy"] += 1
                tier_pairs[tier] += 1
        elif wrong and not correct and args.gold_fallback:
            for w in wrong[: args.max_pairs_per_q]:
                pairs.append({"prompt": prompt_msgs, "chosen": gold_sql, "rejected": w,
                              "meta": {"tier": tier, "kind": "gold_chosen"}})
                kinds["gold_chosen"] += 1
                tier_pairs[tier] += 1
        elif not wrong:
            stats["all_correct"] += 1
        else:
            stats["no_signal"] += 1

    os.makedirs(DATASETS_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print("\n" + "=" * 70)
    print(f"MINED {len(pairs)} DPO pairs -> {args.out}")
    print(f"  on_policy (correct vs wrong sample):   {kinds['on_policy']}")
    print(f"  gold_chosen (gold vs wrong, 0 correct): {kinds['gold_chosen']}")
    print(f"  questions all-correct (skipped):        {stats['all_correct']}")
    print(f"  questions no usable signal (skipped):   {stats['no_signal']}")
    print(f"  gold SQL failed to execute (skipped):   {stats['gold_error']}")
    print("\n  Pairs per tier:")
    for t in ["simple_select", "single_join", "aggregation", "multi_hop",
              "window_function", "simulation"]:
        if tier_pairs.get(t):
            print(f"    {t:18s}: {tier_pairs[t]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
