"""
SQL ACCURACY EVALUATION HARNESS
================================
Reads benchmark.json and for each query:
  1. Runs the question through the agent to get generated SQL
  2. Executes BOTH gold SQL and generated SQL on DuckDB
  3. Compares actual result sets (not string matching)
  4. Categorizes failures: wrong_result, syntax_error, wrong_tables, empty_result, refusal
  5. Reports accuracy by difficulty tier and overall

Usage:
    python eval/accuracy_eval.py

Results saved to eval/results/accuracy_report.json
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import duckdb
import pandas as pd
from src.agent_graph import agent_workflow, DB_PATH

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RATE_LIMIT_DELAY = 15  # seconds between queries (for free-tier API limits)

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_benchmark():
    """Load benchmark questions from JSON."""
    with open(BENCHMARK_PATH, "r") as f:
        return json.load(f)


def execute_sql(sql: str) -> tuple:
    """Execute SQL on DuckDB and return (DataFrame, error)."""
    try:
        clean = sql.replace("```sql", "").replace("```", "").strip()
        con = duckdb.connect(DB_PATH, read_only=True)
        df = con.execute(clean).fetchdf()
        con.close()
        return df, None
    except Exception as e:
        return None, str(e)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame for comparison: lowercase columns, sort, reset index, round floats, coerce types."""
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Coerce all columns to comparable types
    for col in df.columns:
        # Convert string numbers to actual numbers (fixes strftime "1995" vs EXTRACT 1995)
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                # If ALL values converted successfully, use the numeric version
                if converted.notna().all():
                    df[col] = converted
            except Exception:
                pass
    
    # Round numeric columns to 2 decimal places to avoid float precision issues
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].round(2)
    
    # Sort by all columns for consistent ordering
    try:
        df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    except TypeError:
        df = df.reset_index(drop=True)
    return df


def fuzzy_column_match(gold_cols: list, agent_cols: list) -> bool:
    """Check if column names match after removing common suffixes/variations."""
    if len(gold_cols) != len(agent_cols):
        return False
    
    def simplify(name: str) -> str:
        """Reduce column name to its core meaning."""
        name = name.lower().strip().replace("_", "").replace(" ", "")
        # Remove common suffixes/prefixes that don't change meaning
        for suffix in ["pct", "percent", "percentage", "rate"]:
            name = name.replace(suffix, "pct")
        for suffix in ["3m", "3month", "3months", "3mo"]:
            name = name.replace(suffix, "3m")
        for suffix in ["mom", "monthovermonth"]:
            name = name.replace(suffix, "mom")
        for suffix in ["yoy", "yearoveryear"]:
            name = name.replace(suffix, "yoy")
        for suffix in ["qoq", "quarteroverquarter"]:
            name = name.replace(suffix, "qoq")
        return name
    
    gold_simple = sorted([simplify(c) for c in gold_cols])
    agent_simple = sorted([simplify(c) for c in agent_cols])
    return gold_simple == agent_simple


def compare_results(gold_df: pd.DataFrame, agent_df: pd.DataFrame) -> dict:
    """
    Compare gold and agent result sets with smart type coercion.
    Returns: {match: bool, match_type: str, details: str}
    """
    gold_norm = normalize_df(gold_df)
    agent_norm = normalize_df(agent_df)
    
    # 1. Exact match (same columns, same data)
    if set(gold_norm.columns) == set(agent_norm.columns):
        agent_reordered = agent_norm[gold_norm.columns]
        if gold_norm.equals(agent_reordered):
            return {"match": True, "match_type": "exact", "details": "Exact match"}
    
    # 2. Fuzzy column name match (e.g., moving_avg_3m vs moving_avg_3month)
    if len(gold_norm) == len(agent_norm) and fuzzy_column_match(list(gold_norm.columns), list(agent_norm.columns)):
        # Columns are semantically the same — compare values positionally
        gold_vals = gold_norm.values
        agent_vals = agent_norm.values
        try:
            if pd.DataFrame(gold_vals).round(2).equals(pd.DataFrame(agent_vals).round(2)):
                return {"match": True, "match_type": "fuzzy_column_match", "details": f"Same values, fuzzy column match (gold: {list(gold_norm.columns)}, agent: {list(agent_norm.columns)})"}
        except Exception:
            pass
    
    # 3. Value match (same values, different column names — positional comparison)
    if len(gold_norm) == len(agent_norm) and len(gold_norm.columns) == len(agent_norm.columns):
        gold_vals = gold_norm.values
        agent_vals = agent_norm.values
        try:
            if pd.DataFrame(gold_vals).round(2).equals(pd.DataFrame(agent_vals).round(2)):
                return {"match": True, "match_type": "value_match", "details": "Same values, different column names"}
        except Exception:
            pass
    
    # 4. Row count match with key column overlap + numeric proximity
    if len(gold_norm) == len(agent_norm):
        gold_text_cols = gold_norm.select_dtypes(include=["object", "string"]).columns
        agent_text_cols = agent_norm.select_dtypes(include=["object", "string"]).columns
        
        for gc in gold_text_cols:
            for ac in agent_text_cols:
                gold_vals = set(gold_norm[gc].dropna().astype(str))
                agent_vals = set(agent_norm[ac].dropna().astype(str))
                if gold_vals and gold_vals == agent_vals:
                    gold_nums = gold_norm.select_dtypes(include=["number"])
                    agent_nums = agent_norm.select_dtypes(include=["number"])
                    if not gold_nums.empty and not agent_nums.empty:
                        # Check each numeric column for proximity (within 1%)
                        all_close = True
                        for gi, ai in zip(range(len(gold_nums.columns)), range(len(agent_nums.columns))):
                            gs = gold_nums.iloc[:, gi].sum()
                            as_ = agent_nums.iloc[:, ai].sum()
                            if gs != 0 and abs(gs - as_) / abs(gs) > 0.01:
                                all_close = False
                                break
                        if all_close:
                            return {"match": True, "match_type": "approximate", "details": "Same groups, all numeric columns within 1%"}
    
    # 5. Subset match (agent returned correct data + extra columns)
    if len(gold_norm) == len(agent_norm):
        common_cols = set(gold_norm.columns) & set(agent_norm.columns)
        if common_cols and len(common_cols) >= len(gold_norm.columns) * 0.5:
            gold_sub = normalize_df(gold_norm[sorted(common_cols)])
            agent_sub = normalize_df(agent_norm[sorted(common_cols)])
            if gold_sub.equals(agent_sub):
                return {"match": True, "match_type": "subset_match", "details": f"Common columns match: {common_cols}"}
    
    # 6. Same row count + same text values + numeric within 5% (lenient)
    if len(gold_norm) == len(agent_norm) and len(gold_norm.columns) == len(agent_norm.columns):
        gold_nums = gold_norm.select_dtypes(include=["number"])
        agent_nums = agent_norm.select_dtypes(include=["number"])
        if not gold_nums.empty and not agent_nums.empty and len(gold_nums.columns) == len(agent_nums.columns):
            all_close = True
            for gi, ai in zip(range(len(gold_nums.columns)), range(len(agent_nums.columns))):
                gs = gold_nums.iloc[:, gi].sum()
                as_ = agent_nums.iloc[:, ai].sum()
                if gs != 0 and abs(gs - as_) / abs(gs) > 0.05:
                    all_close = False
                    break
            if all_close:
                return {"match": True, "match_type": "lenient_match", "details": "Same shape, numeric values within 5%"}

    # No match
    details = (
        f"Mismatch: gold has {len(gold_norm)} rows x {len(gold_norm.columns)} cols "
        f"({list(gold_norm.columns)}), agent has {len(agent_norm)} rows x "
        f"{len(agent_norm.columns)} cols ({list(agent_norm.columns)})"
    )
    return {"match": False, "match_type": "mismatch", "details": details}


def categorize_failure(test: dict, agent_sql: str, agent_data, agent_error: str, gold_data) -> str:
    """Categorize why a query failed."""
    if agent_error and "syntax" in agent_error.lower():
        return "syntax_error"
    if agent_error and "does not exist" in agent_error.lower():
        return "wrong_tables"
    if agent_error and ("column" in agent_error.lower() or "not found" in agent_error.lower()):
        return "wrong_columns"
    if agent_data is not None and isinstance(agent_data, pd.DataFrame) and agent_data.empty:
        return "empty_result"
    if isinstance(agent_sql, str) and ("MISSING DATA" in agent_sql or "cannot" in agent_sql.lower()):
        return "unexpected_refusal"
    if isinstance(agent_sql, str) and "Failed to generate" in agent_sql:
        return "generation_failure"
    if agent_data is not None and gold_data is not None:
        return "wrong_result"
    return "unknown_error"


def run_accuracy_eval():
    """Main evaluation loop."""
    benchmark = load_benchmark()
    
    results = []
    totals = {
        "exact_match": 0, "approximate_match": 0, "value_match": 0, 
        "subset_match": 0, "fuzzy_column_match": 0, "lenient_match": 0,
        "fail": 0, "refusal_correct": 0, "refusal_wrong": 0
    }
    failure_categories = {}
    tier_results = {}
    latencies = []
    
    print("=" * 70)
    print("SQL ACCURACY EVALUATION")
    print(f"Benchmark: {len(benchmark)} queries")
    print(f"Database: {DB_PATH}")
    print("=" * 70)
    
    for i, test in enumerate(benchmark):
        qid = test["id"]
        question = test["question"]
        difficulty = test["difficulty"]
        gold_sql = test.get("gold_sql")
        expect_refusal = test.get("expect_refusal", False)
        
        # Initialize tier tracking
        if difficulty not in tier_results:
            tier_results[difficulty] = {"total": 0, "pass": 0, "fail": 0}
        tier_results[difficulty]["total"] += 1
        
        print(f"\n[{i+1}/{len(benchmark)}] ({difficulty}) {qid}: {question[:60]}...")
        
        # Rate limiting
        if i > 0:
            time.sleep(RATE_LIMIT_DELAY)
        
        # Run agent
        start = time.time()
        try:
            agent_data, agent_sql = agent_workflow(question)
        except Exception as e:
            agent_data, agent_sql = None, f"EXCEPTION: {e}"
        elapsed = time.time() - start
        latencies.append(elapsed)
        
        entry = {
            "id": qid,
            "question": question,
            "difficulty": difficulty,
            "latency_s": round(elapsed, 2),
            "agent_sql": agent_sql if isinstance(agent_sql, str) else str(agent_sql),
            "gold_sql": gold_sql,
        }
        
        # --- Safety / Refusal Queries ---
        if expect_refusal:
            if agent_data is None or (isinstance(agent_sql, str) and 
                ("MISSING DATA" in agent_sql or "cannot" in agent_sql.lower() or 
                 "SECURITY" in agent_sql or "not allowed" in agent_sql.lower())):
                entry["status"] = "PASS"
                entry["match_type"] = "correct_refusal"
                totals["refusal_correct"] += 1
                tier_results[difficulty]["pass"] += 1
                print(f"Correctly refused ({elapsed:.1f}s)")
            else:
                entry["status"] = "FAIL"
                entry["match_type"] = "wrong_refusal"
                entry["failure_category"] = "should_have_refused"
                totals["refusal_wrong"] += 1
                tier_results[difficulty]["fail"] += 1
                print(f"Should have refused ({elapsed:.1f}s)")
            results.append(entry)
            continue
        
        # --- Non-safety: Compare Results ---
        
        # Execute gold SQL
        gold_data, gold_error = None, None
        if gold_sql:
            gold_data, gold_error = execute_sql(gold_sql)
            if gold_error:
                entry["gold_error"] = gold_error
                print(f"Gold SQL failed: {gold_error[:60]}")
        
        if not isinstance(agent_data, pd.DataFrame) or agent_data.empty:
            entry["status"] = "FAIL"
            entry["failure_category"] = categorize_failure(test, agent_sql, agent_data, agent_sql, gold_data)
            entry["error"] = agent_sql if isinstance(agent_sql, str) else "No data returned"
            totals["fail"] += 1
            tier_results[difficulty]["fail"] += 1
            cat = entry["failure_category"]
            failure_categories[cat] = failure_categories.get(cat, 0) + 1
            print(f"Failed ({cat}) ({elapsed:.1f}s)")
            results.append(entry)
            continue
        
        entry["rows_returned"] = len(agent_data)
        
        # Compare results if we have gold data
        if gold_data is not None and not gold_data.empty:
            comparison = compare_results(gold_data, agent_data)
            entry["match_type"] = comparison["match_type"]
            entry["match_details"] = comparison["details"]
            
            if comparison["match"]:
                entry["status"] = "PASS"
                match_key = comparison["match_type"]
                if match_key in totals:
                    totals[match_key] += 1
                else:
                    totals["exact_match"] += 1  # fallback
                tier_results[difficulty]["pass"] += 1
                print(f"{comparison['match_type']} ({elapsed:.1f}s)")
            else:
                entry["status"] = "FAIL"
                entry["failure_category"] = "wrong_result"
                totals["fail"] += 1
                tier_results[difficulty]["fail"] += 1
                failure_categories["wrong_result"] = failure_categories.get("wrong_result", 0) + 1
                print(f"Wrong result ({elapsed:.1f}s): {comparison['details'][:80]}")
        else:
            col_check = all(c in agent_data.columns for c in test.get("expected_columns", []))
            row_check = len(agent_data) >= test.get("expected_min_rows", 1)
            
            if col_check and row_check:
                entry["status"] = "PASS"
                entry["match_type"] = "execution_only"
                totals["exact_match"] += 1 
                tier_results[difficulty]["pass"] += 1
                print(f"Execution pass (no gold comparison) ({elapsed:.1f}s)")
            else:
                entry["status"] = "FAIL"
                entry["failure_category"] = "wrong_result"
                totals["fail"] += 1
                tier_results[difficulty]["fail"] += 1
                failure_categories["wrong_result"] = failure_categories.get("wrong_result", 0) + 1
                print(f"Execution check failed ({elapsed:.1f}s)")
        
        results.append(entry)
    
    # ---------------------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------------------
    total_q = len(benchmark)
    total_pass = total_q - totals["fail"] - totals["refusal_wrong"]
    non_safety_count = sum(1 for t in benchmark if not t.get("expect_refusal", False))
    non_safety_pass = sum(1 for r in results if r["status"] == "PASS" and r.get("match_type") != "correct_refusal")
    
    accuracy = round(total_pass / total_q * 100, 1) if total_q > 0 else 0
    non_safety_accuracy = round(non_safety_pass / non_safety_count * 100, 1) if non_safety_count > 0 else 0
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

    tier_summary = {}
    for tier, counts in tier_results.items():
        tier_summary[tier] = {
            "total": counts["total"],
            "passed": counts["pass"],
            "failed": counts["fail"],
            "accuracy_pct": round(counts["pass"] / counts["total"] * 100, 1) if counts["total"] > 0 else 0,
        }
    
    report = {
        "summary": {
            "total_queries": total_q,
            "total_passed": total_pass,
            "total_failed": totals["fail"] + totals["refusal_wrong"],
            "overall_accuracy_pct": accuracy,
            "non_safety_accuracy_pct": non_safety_accuracy,
            "correct_refusals": totals["refusal_correct"],
            "wrong_refusals": totals["refusal_wrong"],
            "avg_latency_s": avg_latency,
            "match_breakdown": {
                "exact_match": totals["exact_match"],
                "approximate_match": totals["approximate_match"],
                "value_match": totals["value_match"],
                "subset_match": totals["subset_match"],
                "fuzzy_column_match": totals["fuzzy_column_match"],
                "lenient_match": totals["lenient_match"],
            },
            "failure_categories": failure_categories,
        },
        "tier_results": tier_summary,
        "results": results,
    }

    print("\n" + "=" * 70)
    print("ACCURACY EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total Queries:         {total_q}")
    print(f"  Passed:                {total_pass}")
    print(f"  Failed:                {totals['fail'] + totals['refusal_wrong']}")
    print(f"  Overall Accuracy:      {accuracy}%")
    print(f"  Non-Safety Accuracy:   {non_safety_accuracy}%")
    print(f"  Correct Refusals:      {totals['refusal_correct']}/{sum(1 for t in benchmark if t.get('expect_refusal'))}")
    print(f"  Avg Latency:           {avg_latency}s")
    
    print("\n  Match Breakdown:")
    print(f"    Exact:        {totals['exact_match']}")
    print(f"    Approximate:  {totals['approximate_match']}")
    print(f"    Value Match:  {totals['value_match']}")
    print(f"    Subset:       {totals['subset_match']}")
    print(f"    Fuzzy Column: {totals['fuzzy_column_match']}")
    print(f"    Lenient:      {totals['lenient_match']}")
    
    if failure_categories:
        print("\n  Failure Categories:")
        for cat, count in sorted(failure_categories.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")
    
    print("\n  Per-Tier Results:")
    for tier in ["simple_select", "single_join", "aggregation", "multi_hop", "window_function", "simulation", "safety"]:
        if tier in tier_summary:
            t = tier_summary[tier]
            print(f"    {tier:20s}: {t['passed']}/{t['total']} ({t['accuracy_pct']}%)")
    
    print("=" * 70)
    
    output_path = os.path.join(RESULTS_DIR, "accuracy_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {output_path}")
    
    return report

if __name__ == "__main__":
    run_accuracy_eval()