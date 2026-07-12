"""
PHASE 0 — Base-model baseline on the 55-query suite
===================================================

Answers the single go/no-go question of the whole SQLForge project:

    Does base Qwen2.5-Coder already crush TPC-H (>= 85%)?

If yes on the 7B, the 7B fine-tune has no headroom and the story pivots to the
1.5B model + harder synthetic data (see spec section 9). You need this number
before generating any data.

What it does
------------
1. Loads the 55 non-safety queries from eval/benchmark.json.
2. Builds the SAME prompt your production agent uses (schema DDL + rules +
   simulation-mode block for simulation queries). Few-shot is opt-in.
3. Generates SQL with a base model (in-process vLLM by default) at temp 0,
   single-shot (no self-healing retry loop — that's scaffolding).
4. Executes generated + gold SQL on the DuckDB demo DB and compares result sets
   using the exact logic from eval/accuracy_eval.py (comparable to your 91.7%).
5. Reports execution accuracy overall + per tier, valid-SQL %, and throughput,
   and writes a full JSON report to phase0/results/.

Usage (on the pod)
------------------
    # 7B baseline (the decisive number)
    python phase0/run_baseline.py --model Qwen/Qwen2.5-Coder-7B-Instruct

    # 1.5B baseline (guaranteed-headroom fallback model)
    python phase0/run_baseline.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct

    # add few-shot (3 examples, keyword-selected from data/sql_examples.json)
    python phase0/run_baseline.py --model ... --fewshot 3

Usage (locally, no GPU — validate the harness first)
----------------------------------------------------
    python phase0/run_baseline.py --self-test
    # feeds GOLD sql through the pipeline; should report ~100% and prove plumbing.
"""

import os
import re
import sys
import json
import time
import argparse

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, BASE_DIR)

from eval_compare import extract_sql, execute_sql, compare_results, normalize_df  # noqa: E402

DEMO_DB_PATH = os.path.join(PROJECT_ROOT, "data", "sql_agent_demo.db")
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "eval", "benchmark.json")
EXAMPLES_PATH = os.path.join(PROJECT_ROOT, "data", "sql_examples.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TIER_ORDER = [
    "simple_select", "single_join", "aggregation",
    "multi_hop", "window_function", "simulation",
]


# ---------------------------------------------------------------------------
# Prompt construction  (mirrors src/agent_graph.py so the baseline is fair)
# ---------------------------------------------------------------------------
def get_schema(db_path: str) -> str:
    """Return the schema string exactly as the production agent builds it."""
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    schema_str = "Database Schema (DuckDB):\n"
    target_tables = ["customer", "lineitem", "orders", "supplier",
                     "nation", "part", "region", "partsupp"]
    for table in target_tables:
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        col_list = [f"{col[0]} {col[1]}" for col in cols]
        schema_str += f"- {table}: {', '.join(col_list)}\n"
    con.close()
    return schema_str


def is_simulation(question: str) -> bool:
    q = question.lower()
    return ("what if" in q or "simulate" in q or "sensitivity" in q or "scenario" in q)


SIMULATION_BLOCK = """

        SIMULATION MODE ACTIVE — FOLLOW THESE RULES STRICTLY:

        1. MULTI-VARIABLE SUPPORT:
           - The user may change MULTIPLE variables at once (e.g., "increase price by 5% AND reduce discount by 3%").
           - Apply ALL modifications in the same CTE.
           - If the user specifies a SCOPE (e.g., "for EUROPE only" or "for AUTOMOBILE segment"),
             apply the modification ONLY to matching rows using CASE WHEN. Keep other rows unchanged.

        2. MANDATORY OUTPUT FORMAT — SIDE-BY-SIDE COMPARISON:
           - You MUST always return BOTH original AND simulated values in the same result set.
           - Required output columns: group column (if grouped), original_value, simulated_value, difference, pct_change.
           - If grouped (e.g., by region), show original vs simulated PER GROUP.
           - Always include: ROUND(((simulated - original) / original) * 100, 2) as pct_change

        3. SENSITIVITY ANALYSIS:
           - If the user asks "how sensitive is revenue to discount" or similar,
             generate a query that tests MULTIPLE levels (e.g., +5%, +10%, +15%, +20%).
           - Use UNION ALL to combine results from each level.
           - Output columns: scenario_label, original_value, simulated_value, difference, pct_change.

        4. ALWAYS use revenue formula: SUM(total_value * (1 - promo_reduction))
        """


def build_system_prompt(schema: str, examples: str, simulation: bool) -> str:
    """Replicates the agent's system prompt (agent_graph.py) minus retriever wiring."""
    system_prompt = f"""
    You are an expert SQL Data Analyst.
    You are querying a TPC-H database with CUSTOM column names.
    You have access to the following tables:
    {schema}

    RULES:
    1. Use ONLY DuckDB syntax.
    2. IMPORTANT: The schema has been modified.
       - Revenue is calculated as: sum(total_value * (1 - promo_reduction))
       - Do NOT use 'l_extendedprice' or 'l_discount'. Use 'total_value' and 'promo_reduction'.
    3. Return ONLY the SQL query. No markdown formatting. No explanation.
    4. If the user asks to delete or change data, politely refuse.

    5. COMPLEX QUERY HANDLING (CHAIN OF THOUGHT):
       - If a user asks a complex question (e.g. "Find the region with lowest revenue"),
         you MUST use a CTE.
       - Example Logic: WITH regional_revenue AS (...) SELECT ...

    6. VISUALIZATION RULES (CRITICAL):
       - When grouping by a category, YOU MUST SELECT THE NAME, NOT THE ID.
       - CORRECT: SELECT r_name, sum(revenue)...

    7. COLUMN SECURITY:
       - If the user asks for a missing column (e.g. Profit), REFUSE and explain why.
       - Return text starting with "MISSING DATA:"

    Here are some examples:
    {examples}
    """
    if simulation:
        system_prompt += SIMULATION_BLOCK
    return system_prompt


# ---------------------------------------------------------------------------
# Lightweight, dependency-free few-shot selector (opt-in)
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set:
    return set(_WORD.findall(s.lower()))


def select_fewshot(question: str, examples: list, k: int, benchmark_questions: set) -> str:
    """Keyword-overlap top-k selection from data/sql_examples.json.

    Skips any example whose question exactly matches a benchmark question
    (contamination guard for the baseline).
    """
    if k <= 0 or not examples:
        return ""
    q_tokens = _tokens(question)
    scored = []
    for ex in examples:
        if ex["question"].strip().lower() in benchmark_questions:
            continue  # contamination guard
        overlap = len(q_tokens & _tokens(ex["question"]))
        scored.append((overlap, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [ex for _, ex in scored[:k]]

    out = ""
    for i, ex in enumerate(chosen):
        out += f"Example {i+1}:\nUser Q: {ex['question']}\nSQL: {ex['sql']}\n\n"
    return out


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------
def generate_vllm(conversations, model_id, temperature, max_tokens, max_model_len, gpu_mem,
                  adapter=None, max_lora_rank=32):
    """In-process, batched vLLM generation. Returns (completions, meta).

    If `adapter` is a path to a trained LoRA adapter dir, it is applied on top of the
    base `model_id` — this is how we evaluate the SFT model on the 55-query suite.
    """
    from vllm import LLM, SamplingParams

    print(f"Loading vLLM model: {model_id}" + (f"  + LoRA adapter: {adapter}" if adapter else " ..."))
    llm_kwargs = dict(
        model=model_id,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
    )
    if adapter:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank  # must be >= the adapter's LoRA r
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=0)

    lora_request = None
    if adapter:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("sqlforge-sft", 1, adapter)

    t0 = time.time()
    outputs = llm.chat(conversations, sp, lora_request=lora_request)
    elapsed = time.time() - t0

    completions = [o.outputs[0].text for o in outputs]
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    meta = {
        "wall_s": round(elapsed, 2),
        "gen_tokens": gen_tokens,
        "tokens_per_s": round(gen_tokens / elapsed, 1) if elapsed else 0,
    }
    return completions, meta


def generate_openai(conversations, model_id, temperature, max_tokens, base_url, api_key, sleep):
    """OpenAI-compatible HTTP backend (e.g. a running vLLM server, or a hosted API)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    completions = []
    t0 = time.time()
    for i, conv in enumerate(conversations):
        resp = client.chat.completions.create(
            model=model_id, messages=conv,
            temperature=temperature, max_tokens=max_tokens,
        )
        completions.append(resp.choices[0].message.content or "")
        if sleep and i < len(conversations) - 1:
            time.sleep(sleep)
    meta = {"wall_s": round(time.time() - t0, 2)}
    return completions, meta


def generate_selftest(items):
    """Feed GOLD sql through the pipeline to validate comparison plumbing locally."""
    return [t["gold_sql"] for t in items], {"wall_s": 0.0, "note": "self-test (gold SQL)"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Phase 0 base-model baseline on the 55-query suite.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="HF model id (vllm) or served model name (openai backend).")
    ap.add_argument("--backend", choices=["vllm", "openai", "selftest"], default="vllm")
    ap.add_argument("--fewshot", type=int, default=0, help="Number of few-shot examples (0 = zero-shot).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096, help="vLLM context length.")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="vLLM gpu_memory_utilization.")
    ap.add_argument("--include-safety", action="store_true",
                    help="Include the 5 safety/refusal queries (default: excluded → 55-query suite).")
    ap.add_argument("--limit", type=int, default=0, help="Only run first N queries (smoke test).")
    ap.add_argument("--db", default=DEMO_DB_PATH)
    ap.add_argument("--benchmark", default=BENCHMARK_PATH)
    # LoRA adapter (evaluate the SFT model): base --model + trained adapter dir
    ap.add_argument("--adapter", default=None,
                    help="Path to a trained LoRA adapter dir to apply on top of --model (SFT eval).")
    ap.add_argument("--max-lora-rank", type=int, default=32,
                    help="Must be >= the adapter's LoRA r (sft.yaml lora_r, default 32).")
    # openai backend options
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--sleep", type=float, default=0.0, help="Delay between calls (openai backend).")
    # convenience alias
    ap.add_argument("--self-test", dest="selftest", action="store_true",
                    help="Shortcut for --backend selftest (validate harness locally).")
    ap.add_argument("--out", default=None, help="Output JSON path (default: auto under phase0/results/).")
    args = ap.parse_args()

    if args.selftest:
        args.backend = "selftest"

    # --- Preconditions ---
    if not os.path.exists(args.db):
        sys.exit(f"[FATAL] DB not found: {args.db}\n"
                 f"        Generate it first:  python scripts/demo_db.py")
    if not os.path.exists(args.benchmark):
        sys.exit(f"[FATAL] Benchmark not found: {args.benchmark}")

    with open(args.benchmark, "r") as f:
        benchmark = json.load(f)

    examples = []
    if os.path.exists(EXAMPLES_PATH):
        with open(EXAMPLES_PATH, "r") as f:
            examples = json.load(f)

    # --- Filter to the 55-query suite (drop safety/refusal by default) ---
    items = [t for t in benchmark if args.include_safety or not t.get("expect_refusal", False)]
    if args.limit:
        items = items[: args.limit]

    benchmark_questions = {t["question"].strip().lower() for t in benchmark}

    # --- Build prompts ---
    schema = get_schema(args.db)
    conversations = []
    for t in items:
        sim = is_simulation(t["question"])
        few = select_fewshot(t["question"], examples, args.fewshot, benchmark_questions)
        sys_prompt = build_system_prompt(schema, few, sim)
        conversations.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": t["question"]},
        ])

    print("=" * 72)
    print("SFT-MODEL EVAL" if args.adapter else "PHASE 0 — BASE-MODEL BASELINE")
    print(f"  Model:      {args.model}")
    if args.adapter:
        print(f"  Adapter:    {args.adapter}")
    print(f"  Backend:    {args.backend}")
    print(f"  Queries:    {len(items)}  ({'incl. safety' if args.include_safety else '55-query suite, safety excluded'})")
    print(f"  Few-shot:   {args.fewshot}")
    print(f"  Temp:       {args.temperature}   Max tokens: {args.max_tokens}")
    print(f"  DB:         {args.db}")
    print("=" * 72)

    # --- Generate ---
    if args.backend == "vllm":
        completions, meta = generate_vllm(
            conversations, args.model, args.temperature,
            args.max_tokens, args.max_model_len, args.gpu_mem,
            adapter=args.adapter, max_lora_rank=args.max_lora_rank,
        )
    elif args.backend == "openai":
        completions, meta = generate_openai(
            conversations, args.model, args.temperature,
            args.max_tokens, args.base_url, args.api_key, args.sleep,
        )
    else:  # selftest
        completions, meta = generate_selftest(items)

    # --- Evaluate ---
    results = []
    n_valid_sql = 0   # parses + executes without error
    n_correct = 0     # result set matches gold
    match_breakdown = {}
    tier_stats = {}
    fail_categories = {}

    for t, raw in zip(items, completions):
        tier = t["difficulty"]
        tier_stats.setdefault(tier, {"total": 0, "correct": 0})
        tier_stats[tier]["total"] += 1

        raw = raw or ""
        sql = extract_sql(raw)
        refused = raw.strip().upper().startswith("MISSING DATA") or "I CANNOT" in raw.strip().upper()[:20]

        agent_df, agent_err = (None, "refusal") if refused else execute_sql(sql, args.db)

        # Gold may be absent (e.g. sim_05): fall back to an execution-only check
        # on expected_columns / expected_min_rows, exactly like eval/accuracy_eval.py.
        gold_sql = t.get("gold_sql")
        gold_df, gold_err = (None, None)
        if gold_sql:
            gold_df, gold_err = execute_sql(gold_sql, args.db)

        entry = {
            "id": t["id"],
            "difficulty": tier,
            "question": t["question"],
            "generated_sql": sql,
            "gold_sql": gold_sql,
        }

        if gold_err:
            entry["status"] = "GOLD_ERROR"
            entry["error"] = gold_err
            fail_categories["gold_error"] = fail_categories.get("gold_error", 0) + 1
            results.append(entry)
            print(f"  [{t['id']:<14}] GOLD_ERROR: {gold_err[:60]}")
            continue

        valid = isinstance(agent_df, pd.DataFrame) and agent_err is None
        if valid:
            n_valid_sql += 1

        if not isinstance(agent_df, pd.DataFrame):
            entry["status"] = "FAIL"
            cat = "refusal" if refused else _err_category(agent_err)
            entry["failure_category"] = cat
            entry["error"] = agent_err
            fail_categories[cat] = fail_categories.get(cat, 0) + 1
            results.append(entry)
            print(f"  [{t['id']:<14}] FAIL ({cat})")
            continue

        entry["rows_returned"] = len(agent_df)

        # --- No gold SQL: execution-only pass/fail on expected shape ---
        if gold_df is None:
            col_ok = all(c in agent_df.columns for c in t.get("expected_columns", []))
            row_ok = len(agent_df) >= t.get("expected_min_rows", 1)
            if col_ok and row_ok:
                entry["status"] = "PASS"
                entry["match_type"] = "execution_only"
                n_correct += 1
                tier_stats[tier]["correct"] += 1
                match_breakdown["execution_only"] = match_breakdown.get("execution_only", 0) + 1
                print(f"  [{t['id']:<14}] PASS (execution_only, no gold)")
            else:
                entry["status"] = "FAIL"
                entry["failure_category"] = "wrong_result"
                fail_categories["wrong_result"] = fail_categories.get("wrong_result", 0) + 1
                print(f"  [{t['id']:<14}] FAIL (execution_only check: cols={col_ok}, rows={row_ok})")
            results.append(entry)
            continue

        cmp = compare_results(gold_df, agent_df)
        entry["match_type"] = cmp["match_type"]
        entry["match_details"] = cmp["details"]

        if cmp["match"]:
            entry["status"] = "PASS"
            n_correct += 1
            tier_stats[tier]["correct"] += 1
            match_breakdown[cmp["match_type"]] = match_breakdown.get(cmp["match_type"], 0) + 1
            print(f"  [{t['id']:<14}] PASS ({cmp['match_type']})")
        else:
            entry["status"] = "FAIL"
            entry["failure_category"] = "wrong_result"
            fail_categories["wrong_result"] = fail_categories.get("wrong_result", 0) + 1
            print(f"  [{t['id']:<14}] FAIL (wrong_result): {cmp['details'][:70]}")

        results.append(entry)

    # --- Summary ---
    n = len(items)
    exec_acc = round(n_correct / n * 100, 1) if n else 0.0
    valid_pct = round(n_valid_sql / n * 100, 1) if n else 0.0

    tier_summary = {
        tier: {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy_pct": round(s["correct"] / s["total"] * 100, 1) if s["total"] else 0.0,
        }
        for tier, s in tier_stats.items()
    }

    report = {
        "config": {
            "model": args.model,
            "backend": args.backend,
            "fewshot": args.fewshot,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "num_queries": n,
            "include_safety": args.include_safety,
        },
        "summary": {
            "execution_accuracy_pct": exec_acc,
            "correct": n_correct,
            "valid_sql_pct": valid_pct,
            "valid_sql": n_valid_sql,
            "match_breakdown": match_breakdown,
            "failure_categories": fail_categories,
            "generation_meta": meta,
        },
        "tier_results": tier_summary,
        "results": results,
    }

    print("\n" + "=" * 72)
    print("SFT-MODEL EVAL SUMMARY" if args.adapter else "PHASE 0 SUMMARY")
    print("=" * 72)
    print(f"  Model:                 {args.model}" + (f"  + adapter" if args.adapter else ""))
    print(f"  Execution Accuracy:    {exec_acc}%   ({n_correct}/{n})")
    print(f"  Valid SQL:             {valid_pct}%   ({n_valid_sql}/{n})")
    if "tokens_per_s" in meta:
        print(f"  Throughput:            {meta['tokens_per_s']} tok/s  ({meta['wall_s']}s total)")
    print("\n  Per-Tier Accuracy:")
    for tier in TIER_ORDER:
        if tier in tier_summary:
            s = tier_summary[tier]
            print(f"    {tier:18s}: {s['correct']}/{s['total']}  ({s['accuracy_pct']}%)")
    if match_breakdown:
        print("\n  Match Breakdown:")
        for k, v in sorted(match_breakdown.items(), key=lambda x: -x[1]):
            print(f"    {k:18s}: {v}")
    if fail_categories:
        print("\n  Failure Categories:")
        for k, v in sorted(fail_categories.items(), key=lambda x: -x[1]):
            print(f"    {k:18s}: {v}")

    # --- Verdict ---
    print("\n  " + "-" * 68)
    if args.backend == "selftest":
        pass
    elif args.adapter:
        print(f"  SFT MODEL: {exec_acc}% on the 55-query suite.")
        print("           Reference: base 7B 54.5% (zero-shot) / 61.8% (few-shot); Claude 91.7%.")
        print(f"           Delta vs base zero-shot: {exec_acc - 54.5:+.1f} pp.")
    elif exec_acc >= 85.0:
        print(f"  VERDICT: {exec_acc}% >= 85% → LOW HEADROOM on this model.")
        print("           If this is the 7B: pivot the headline gains to the 1.5B")
        print("           model and make synthetic training data harder (spec §9).")
    else:
        print(f"  VERDICT: {exec_acc}% < 85% → HEADROOM EXISTS. Fine-tuning story holds.")
    print("  " + "-" * 68)
    print("=" * 72)

    # --- Save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.out:
        out_path = args.out
    else:
        safe_model = args.model.replace("/", "_")
        tag = f"fs{args.fewshot}" if args.fewshot else "zeroshot"
        prefix = "sft" if args.adapter else "baseline"
        out_path = os.path.join(RESULTS_DIR, f"{prefix}_{safe_model}_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {out_path}")

    return report


def _err_category(err: str) -> str:
    if not err:
        return "unknown_error"
    e = err.lower()
    if "syntax" in e:
        return "syntax_error"
    if "does not exist" in e or "referenced table" in e:
        return "wrong_tables"
    if "column" in e or "not found" in e:
        return "wrong_columns"
    if "empty sql" in e:
        return "no_sql"
    return "execution_error"


if __name__ == "__main__":
    main()
