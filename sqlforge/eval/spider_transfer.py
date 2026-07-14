"""
SQLForge Phase 5 — zero-shot transfer to Spider (the generalization test).

THE QUESTION: did SFT teach the model *general* Text-to-SQL, or only THIS schema's
house style? The 55-query suite can't tell us — it rewards exactly the conventions we
trained on. Spider is a different dialect (SQLite), different schemas, and no house
style, so it isolates general capability from domain specialization.

FAIRNESS: base and SFT get the IDENTICAL prompt — the target DB's schema + the question,
with NO TPC-H house-style rules (those are meaningless here). Any delta is therefore
attributable to the weights, not the prompt.

Grading is execution-based and reuses the SAME comparison logic as the main eval
(phase0/eval_compare.compare_results), so "correct" means the same thing it does
everywhere else in this project.

Expected outcome is genuinely open, and both directions are informative:
  - SFT >= base  -> fine-tuning improved general SQL ability.
  - SFT <  base  -> we traded generality for domain accuracy (specialization cost).

Run (pod, in the vllm serve venv):
    # base
    python sqlforge/eval/spider_transfer.py \
        --dev-json /workspace/spider/dev.json --db-dir /workspace/spider/database \
        --model Qwen/Qwen2.5-Coder-7B-Instruct --limit 300 \
        --out /workspace/spider_base.json
    # SFT
    python sqlforge/eval/spider_transfer.py \
        --dev-json /workspace/spider/dev.json --db-dir /workspace/spider/database \
        --model Qwen/Qwen2.5-Coder-7B-Instruct --adapter /workspace/sqlforge-sft-7b \
        --max-lora-rank 32 --limit 300 --out /workspace/spider_sft.json
"""

import os
import sys
import json
import time
import argparse
import sqlite3
from collections import Counter

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase0"))   # eval_compare

from eval_compare import extract_sql, compare_results  # noqa: E402

SYSTEM_TMPL = """You are an expert SQL analyst. Given a database schema, write a single \
SQLite query that answers the user's question.

DATABASE SCHEMA:
{schema}

Rules:
- Output ONLY the SQL query inside a ```sql code block.
- Use only tables and columns that exist in the schema above.
- Return exactly the columns the question asks for."""


# ---------------------------------------------------------------------------
# Spider DB helpers
# ---------------------------------------------------------------------------
def db_path(db_dir: str, db_id: str) -> str:
    return os.path.join(db_dir, db_id, f"{db_id}.sqlite")


def load_schema(path: str) -> str:
    """Schema as the DB's own CREATE TABLE statements (no tables.json needed)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    return "\n\n".join(r[0].strip() for r in rows if r[0])


def execute_sqlite(sql: str, path: str, timeout: float = 5.0):
    """Execute read-only with a wall-clock guard. Returns (DataFrame|None, err|None)."""
    clean = sql.replace("```sql", "").replace("```", "").strip()
    if not clean:
        return None, "empty SQL"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    deadline = time.time() + timeout
    # sqlite has no statement timeout; abort from the progress handler instead.
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        df = pd.read_sql_query(clean, con)
        return df, None
    except Exception as e:  # noqa: BLE001 — we want the message text
        return None, str(e)[:200]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(conversations, model, adapter, max_lora_rank, max_model_len, gpu_mem, max_tokens):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"Loading vLLM: {model}" + (f"  + LoRA {adapter}" if adapter else ""))
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem, trust_remote_code=True,
              enable_lora=bool(adapter), max_lora_rank=max_lora_rank if adapter else 16,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    lora = LoRARequest("sqlforge-sft", 1, adapter) if adapter else None
    outs = llm.chat(conversations, sp, lora_request=lora)
    return [o.outputs[0].text for o in outs]


def main():
    ap = argparse.ArgumentParser(description="Zero-shot Spider transfer eval.")
    ap.add_argument("--dev-json", required=True, help="Spider dev.json")
    ap.add_argument("--db-dir", required=True, help="Spider database/ dir")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--limit", type=int, default=300, help="First N dev examples (0 = all).")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.dev_json, encoding="utf-8") as f:
        dev = json.load(f)
    if args.limit:
        dev = dev[: args.limit]

    # --- build prompts (schema cached per db_id) ---
    schema_cache, records = {}, []
    for ex in dev:
        db_id = ex["db_id"]
        p = db_path(args.db_dir, db_id)
        if not os.path.exists(p):
            continue
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(p)
        records.append({
            "db_id": db_id, "path": p,
            "question": ex["question"],
            "gold": ex.get("query") or ex.get("SQL"),
        })

    conversations = [
        [{"role": "system", "content": SYSTEM_TMPL.format(schema=schema_cache[r["db_id"]])},
         {"role": "user", "content": r["question"]}]
        for r in records
    ]

    print("=" * 70)
    print("SPIDER ZERO-SHOT TRANSFER")
    print(f"  Model:    {args.model}" + (f"  + adapter {args.adapter}" if args.adapter else "  (base)"))
    print(f"  Examples: {len(records)}   DBs: {len(schema_cache)}")
    print("=" * 70)

    t0 = time.time()
    raw = generate(conversations, args.model, args.adapter, args.max_lora_rank,
                   args.max_model_len, args.gpu_mem, args.max_tokens)
    elapsed = time.time() - t0

    # --- grade: execute prediction AND gold, compare result sets ---
    correct = valid = gold_ok = 0
    fails = Counter()
    results = []
    for r, text in zip(records, raw):
        sql = extract_sql(text)
        pred_df, perr = execute_sqlite(sql, r["path"])
        gold_df, gerr = execute_sqlite(r["gold"], r["path"])
        if gerr or gold_df is None:
            gold_ok += 0
            fails["gold_unrunnable"] += 1          # skip: can't grade fairly
            results.append({**{k: r[k] for k in ("db_id", "question", "gold")},
                            "pred_sql": sql, "status": "gold_unrunnable"})
            continue
        gold_ok += 1
        if perr or pred_df is None:
            fails["invalid_sql"] += 1
            results.append({**{k: r[k] for k in ("db_id", "question", "gold")},
                            "pred_sql": sql, "status": "invalid_sql", "error": perr})
            continue
        valid += 1
        cmp = compare_results(gold_df, pred_df)
        if cmp["match"]:
            correct += 1
            results.append({**{k: r[k] for k in ("db_id", "question", "gold")},
                            "pred_sql": sql, "status": "correct",
                            "match_type": cmp.get("match_type")})
        else:
            fails["wrong_result"] += 1
            results.append({**{k: r[k] for k in ("db_id", "question", "gold")},
                            "pred_sql": sql, "status": "wrong_result"})

    n = max(gold_ok, 1)
    summary = {
        "model": args.model, "adapter": args.adapter,
        "graded": gold_ok, "correct": correct, "valid_sql": valid,
        "execution_accuracy_pct": round(100 * correct / n, 1),
        "valid_sql_pct": round(100 * valid / n, 1),
        "elapsed_s": round(elapsed, 1),
        "failures": dict(fails),
    }

    print("\n" + "=" * 70)
    print("SPIDER TRANSFER SUMMARY")
    print("=" * 70)
    print(f"  Model:              {args.model}" + ("  + adapter" if args.adapter else "  (base)"))
    print(f"  Gradeable examples: {gold_ok}  (skipped {fails['gold_unrunnable']} unrunnable gold)")
    print(f"  Execution Accuracy: {summary['execution_accuracy_pct']}%   ({correct}/{gold_ok})")
    print(f"  Valid SQL:          {summary['valid_sql_pct']}%   ({valid}/{gold_ok})")
    print(f"  Generation time:    {elapsed:.1f}s")
    print("\n  Failure categories:")
    for k, v in fails.most_common():
        print(f"    {k:18s}: {v}")
    print("=" * 70)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"\nFull report saved to: {args.out}")


if __name__ == "__main__":
    main()
