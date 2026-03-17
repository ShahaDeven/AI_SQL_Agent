"""
RETRIEVAL EVALUATION
====================
For each benchmark query, measures how well the hybrid retriever
(BM25 + semantic search) pulls relevant context.

Metrics:
  - Precision@k: Of the k retrieved examples, how many reference relevant tables?
  - Recall@k:    Of all relevant tables, how many were covered by retrieval?
  - MRR:         Mean Reciprocal Rank — how early does the first relevant result appear?

Usage:
    python eval/retrieval_eval.py

Results saved to eval/results/retrieval_report.json
"""

import os
import sys
import json
import re
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.retriever import get_few_shot_examples

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
K = 3 

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_benchmark():
    with open(BENCHMARK_PATH, "r") as f:
        return json.load(f)


def extract_tables_from_sql(sql: str) -> set:
    """Extract table names referenced in a SQL query."""
    if not sql:
        return set()
    
    sql_upper = sql.upper()
    
    # Known TPC-H tables
    known_tables = {"customer", "lineitem", "orders", "supplier", "nation", "region", "part", "partsupp"}
    
    found = set()
    for table in known_tables:
        # Match table name as a whole word (not part of column names)
        # Look for FROM/JOIN table patterns
        patterns = [
            rf'\bFROM\s+{table.upper()}\b',
            rf'\bJOIN\s+{table.upper()}\b',
            rf'\bFROM\s+{table.upper()}\s+',
            rf'\bJOIN\s+{table.upper()}\s+',
        ]
        for pattern in patterns:
            if re.search(pattern, sql_upper):
                found.add(table)
                break
    
    return found


def extract_columns_from_sql(sql: str) -> set:
    """Extract column names referenced in a SQL query."""
    if not sql:
        return set()
    
    # Find all word tokens that look like column names (contain underscore or known prefixes)
    tokens = re.findall(r'\b([a-z][a-z0-9_]*)\b', sql.lower())
    
    # Known column prefixes in TPC-H
    known_prefixes = {'c_', 'o_', 'l_', 's_', 'n_', 'r_', 'p_', 'ps_'}
    custom_columns = {'total_value', 'promo_reduction', 'churn_risk', 'total_value_forecast'}
    
    columns = set()
    for token in tokens:
        if any(token.startswith(p) for p in known_prefixes):
            columns.add(token)
        elif token in custom_columns:
            columns.add(token)
    
    return columns


def evaluate_retrieval(question: str, relevant_tables: list, relevant_columns: list, k: int = K) -> dict:
    """
    Run retrieval for a question and measure quality.
    
    Returns dict with precision, recall, retrieved_tables, retrieved_columns, etc.
    """
    # Get retrieved examples
    raw_examples = get_few_shot_examples(question, k=k)
    
    # Parse retrieved SQL from the formatted string
    retrieved_sqls = re.findall(r'SQL:\s*(.+?)(?:\n\n|$)', raw_examples, re.DOTALL)
    
    # Extract tables and columns from all retrieved examples
    retrieved_tables = set()
    retrieved_columns = set()
    per_example = []
    
    for i, sql in enumerate(retrieved_sqls):
        tables = extract_tables_from_sql(sql)
        columns = extract_columns_from_sql(sql)
        retrieved_tables.update(tables)
        retrieved_columns.update(columns)
        per_example.append({
            "rank": i + 1,
            "sql_fragment": sql[:100],
            "tables": sorted(tables),
            "columns": sorted(columns),
        })
    
    # Ground truth
    gold_tables = set(relevant_tables)
    gold_columns = set(relevant_columns)
    
    # --- Table-level metrics ---
    table_hits = retrieved_tables & gold_tables
    table_precision = len(table_hits) / len(retrieved_tables) if retrieved_tables else 0
    table_recall = len(table_hits) / len(gold_tables) if gold_tables else 1.0  # If no tables needed, recall is 100%
    
    # --- Column-level metrics ---
    column_hits = retrieved_columns & gold_columns
    column_precision = len(column_hits) / len(retrieved_columns) if retrieved_columns else 0
    column_recall = len(column_hits) / len(gold_columns) if gold_columns else 1.0
    
    # --- MRR (Mean Reciprocal Rank) ---
    # Rank of first retrieved example that contains at least one relevant table
    mrr = 0.0
    for ex in per_example:
        if set(ex["tables"]) & gold_tables:
            mrr = 1.0 / ex["rank"]
            break
    
    return {
        "table_precision": round(table_precision, 3),
        "table_recall": round(table_recall, 3),
        "column_precision": round(column_precision, 3),
        "column_recall": round(column_recall, 3),
        "mrr": round(mrr, 3),
        "retrieved_tables": sorted(retrieved_tables),
        "gold_tables": sorted(gold_tables),
        "table_hits": sorted(table_hits),
        "table_misses": sorted(gold_tables - retrieved_tables),
        "retrieved_columns": sorted(retrieved_columns),
        "gold_columns": sorted(gold_columns),
        "column_hits": sorted(column_hits),
        "per_example": per_example,
    }


def run_retrieval_eval():
    """Main evaluation loop."""
    benchmark = load_benchmark()
    
    # Skip safety queries (no retrieval needed)
    eval_queries = [q for q in benchmark if not q.get("expect_refusal", False)]
    
    results = []
    agg = {
        "table_precision": [], "table_recall": [],
        "column_precision": [], "column_recall": [],
        "mrr": [],
    }
    tier_agg = {}
    
    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print(f"Queries: {len(eval_queries)} (excluding safety)")
    print(f"k = {K}")
    print("=" * 70)
    
    for i, test in enumerate(eval_queries):
        qid = test["id"]
        question = test["question"]
        difficulty = test["difficulty"]
        relevant_tables = test.get("relevant_tables", [])
        relevant_columns = test.get("relevant_columns", [])
        
        print(f"\n[{i+1}/{len(eval_queries)}] ({difficulty}) {qid}: {question[:55]}...")
        
        try:
            result = evaluate_retrieval(question, relevant_tables, relevant_columns)
            result["id"] = qid
            result["question"] = question
            result["difficulty"] = difficulty
            
            # Aggregate
            for metric in ["table_precision", "table_recall", "column_precision", "column_recall", "mrr"]:
                agg[metric].append(result[metric])
            
            # Per-tier aggregate
            if difficulty not in tier_agg:
                tier_agg[difficulty] = {m: [] for m in agg.keys()}
            for metric in agg.keys():
                tier_agg[difficulty][metric].append(result[metric])
            
            print(f"  Table P={result['table_precision']:.2f} R={result['table_recall']:.2f} | "
                  f"Col P={result['column_precision']:.2f} R={result['column_recall']:.2f} | "
                  f"MRR={result['mrr']:.2f}")
            
            if result["table_misses"]:
                print(f"Missing tables: {result['table_misses']}")
                
        except Exception as e:
            print(f"Error: {e}")
            result = {
                "id": qid, "question": question, "difficulty": difficulty,
                "error": str(e),
                "table_precision": 0, "table_recall": 0,
                "column_precision": 0, "column_recall": 0, "mrr": 0,
            }
        
        results.append(result)
    
    # ---------------------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------------------
    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0
    
    summary = {
        "total_queries": len(eval_queries),
        "k": K,
        "overall": {
            "avg_table_precision": avg(agg["table_precision"]),
            "avg_table_recall": avg(agg["table_recall"]),
            "avg_column_precision": avg(agg["column_precision"]),
            "avg_column_recall": avg(agg["column_recall"]),
            "avg_mrr": avg(agg["mrr"]),
        },
        "per_tier": {},
    }
    
    for tier, metrics in tier_agg.items():
        summary["per_tier"][tier] = {
            "count": len(metrics["mrr"]),
            "avg_table_precision": avg(metrics["table_precision"]),
            "avg_table_recall": avg(metrics["table_recall"]),
            "avg_column_precision": avg(metrics["column_precision"]),
            "avg_column_recall": avg(metrics["column_recall"]),
            "avg_mrr": avg(metrics["mrr"]),
        }

    worst_recall = sorted(results, key=lambda x: x.get("table_recall", 0))[:5]
    
    report = {
        "summary": summary,
        "worst_recall_queries": [
            {"id": r["id"], "question": r["question"][:60], "table_recall": r.get("table_recall", 0), 
             "table_misses": r.get("table_misses", [])}
            for r in worst_recall
        ],
        "results": results,
    }

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 70)
    o = summary["overall"]
    print(f"  Queries Evaluated:    {len(eval_queries)}")
    print(f"  Retrieval k:          {K}")
    print("\n  Overall Averages:")
    print(f"    Table Precision@{K}:  {o['avg_table_precision']}")
    print(f"    Table Recall@{K}:     {o['avg_table_recall']}")
    print(f"    Column Precision@{K}: {o['avg_column_precision']}")
    print(f"    Column Recall@{K}:    {o['avg_column_recall']}")
    print(f"    MRR:                {o['avg_mrr']}")
    
    print("\n  Per-Tier Table Recall:")
    for tier in ["simple_select", "single_join", "aggregation", "multi_hop", "window_function", "simulation"]:
        if tier in summary["per_tier"]:
            t = summary["per_tier"][tier]
            print(f"    {tier:20s}: P={t['avg_table_precision']:.2f} R={t['avg_table_recall']:.2f} MRR={t['avg_mrr']:.2f} (n={t['count']})")
    
    print("\n  Worst Table Recall Queries:")
    for wq in worst_recall[:5]:
        print(f"    {wq.get('id', 'N/A'):12s}: recall={wq.get('table_recall', 0):.2f}  missing={wq.get('table_misses', [])}")
    
    print("=" * 70)

    output_path = os.path.join(RESULTS_DIR, "retrieval_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    run_retrieval_eval()