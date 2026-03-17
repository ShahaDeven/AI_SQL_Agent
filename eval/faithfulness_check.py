"""
FAITHFULNESS CHECK
==================
For each benchmark query, parses the agent's generated SQL and validates:
  1. Do all referenced tables exist in the actual schema?
  2. Do all referenced columns exist in those tables?
  3. Are joins on valid foreign key relationships?

Catches hallucinated tables, columns, and invalid joins.

Usage:
    python eval/faithfulness_check.py

Results saved to eval/results/faithfulness_report.json
"""

import os
import sys
import json
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import duckdb

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ACCURACY_REPORT = os.path.join(os.path.dirname(__file__), "results", "accuracy_report.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Auto-detect database
DEMO_DB = os.path.join(PROJECT_ROOT, "data", "sql_agent_demo.db")
FULL_DB = os.path.join(PROJECT_ROOT, "data", "supply_chain.db")
DB_PATH = DEMO_DB if os.path.exists(DEMO_DB) else FULL_DB


def load_schema() -> dict:
    """
    Load actual schema from DuckDB.
    Returns: {table_name: {col_name: col_type, ...}, ...}
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    schema = {}
    tables = ['customer', 'lineitem', 'orders', 'supplier', 'nation', 'part', 'region', 'partsupp']

    for table in tables:
        try:
            cols = con.execute(f"DESCRIBE {table}").fetchall()
            schema[table] = {col[0].lower(): col[1] for col in cols}
        except Exception:
            pass

    con.close()
    return schema


# Known valid foreign key relationships in TPC-H
VALID_JOINS = {
    ("customer", "nation"): ("c_nationkey", "n_nationkey"),
    ("supplier", "nation"): ("s_nationkey", "n_nationkey"),
    ("nation", "region"): ("n_regionkey", "r_regionkey"),
    ("orders", "customer"): ("o_custkey", "c_custkey"),
    ("lineitem", "orders"): ("l_orderkey", "o_orderkey"),
    ("lineitem", "supplier"): ("l_suppkey", "s_suppkey"),
    ("lineitem", "part"): ("l_partkey", "p_partkey"),
    ("partsupp", "supplier"): ("ps_suppkey", "s_suppkey"),
    ("partsupp", "part"): ("ps_partkey", "p_partkey"),
}

# Also add reverse direction
_reverse = {}
for (t1, t2), (c1, c2) in VALID_JOINS.items():
    _reverse[(t2, t1)] = (c2, c1)
VALID_JOINS.update(_reverse)


def extract_tables_from_sql(sql: str) -> set:
    """Extract table names from SQL using FROM/JOIN patterns."""
    known_tables = {"customer", "lineitem", "orders", "supplier", "nation", "region", "part", "partsupp"}
    sql_upper = sql.upper()
    found = set()

    for table in known_tables:
        patterns = [
            rf'\bFROM\s+{table.upper()}\b',
            rf'\bJOIN\s+{table.upper()}\b',
        ]
        for pattern in patterns:
            if re.search(pattern, sql_upper):
                found.add(table)
                break

    return found


def extract_columns_from_sql(sql: str) -> set:
    """Extract column references from SQL."""
    tokens = re.findall(r'\b([a-z][a-z0-9_]*)\b', sql.lower())

    known_prefixes = {'c_', 'o_', 'l_', 's_', 'n_', 'r_', 'p_', 'ps_'}
    custom_columns = {'total_value', 'promo_reduction', 'churn_risk', 'total_value_forecast'}
    sql_keywords = {
        'select', 'from', 'where', 'join', 'on', 'and', 'or', 'not', 'in', 'as',
        'group', 'by', 'order', 'having', 'limit', 'offset', 'union', 'all',
        'case', 'when', 'then', 'else', 'end', 'with', 'distinct', 'count',
        'sum', 'avg', 'min', 'max', 'round', 'extract', 'year', 'month',
        'quarter', 'day', 'desc', 'asc', 'null', 'is', 'between', 'like',
        'inner', 'left', 'right', 'outer', 'cross', 'full', 'exists',
        'over', 'partition', 'rows', 'range', 'preceding', 'following',
        'current', 'row', 'unbounded', 'lag', 'lead', 'rank', 'dense_rank',
        'row_number', 'percent_rank', 'ntile', 'first_value', 'last_value',
        'true', 'false', 'cast', 'coalesce', 'least', 'greatest',
        'strftime', 'date_trunc', 'date', 'int', 'integer', 'double',
        'varchar', 'float', 'boolean', 'text', 'numeric', 'real',
        'revenue', 'original', 'simulated', 'difference', 'pct_change',
        'scenario', 'baseline', 'increase', 'decrease',
    }

    columns = set()
    for token in tokens:
        if token in sql_keywords:
            continue
        if any(token.startswith(p) for p in known_prefixes):
            columns.add(token)
        elif token in custom_columns:
            columns.add(token)

    return columns


def extract_joins_from_sql(sql: str) -> list:
    """Extract JOIN conditions from SQL."""
    # Pattern: table1.col1 = table2.col2
    join_pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
    matches = re.findall(join_pattern, sql.lower())

    joins = []
    for alias1, col1, alias2, col2 in matches:
        joins.append({
            "alias1": alias1, "col1": col1,
            "alias2": alias2, "col2": col2,
        })
    return joins


def resolve_alias(sql: str, alias: str) -> str:
    """Try to resolve a table alias to the actual table name."""
    known_tables = {"customer", "lineitem", "orders", "supplier", "nation", "region", "part", "partsupp"}

    # Direct match
    if alias in known_tables:
        return alias

    # Look for "table alias" or "table AS alias" patterns
    patterns = [
        rf'\b(\w+)\s+AS\s+{re.escape(alias)}\b',
        rf'\b(\w+)\s+{re.escape(alias)}\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, sql, re.IGNORECASE)
        if match:
            candidate = match.group(1).lower()
            if candidate in known_tables:
                return candidate

    return alias


def check_faithfulness(sql: str, schema: dict) -> dict:
    """
    Validate SQL against actual schema.
    Returns: {faithful: bool, issues: [...], stats: {...}}
    """
    if not sql or not isinstance(sql, str):
        return {"faithful": False, "issues": ["No SQL provided"], "stats": {}}

    # Clean SQL
    clean_sql = sql.replace("```sql", "").replace("```", "").strip()

    issues = []

    # 1. Check tables
    referenced_tables = extract_tables_from_sql(clean_sql)
    real_tables = set(schema.keys())
    hallucinated_tables = referenced_tables - real_tables

    if hallucinated_tables:
        issues.append(f"Hallucinated tables: {sorted(hallucinated_tables)}")

    # 2. Check columns
    referenced_columns = extract_columns_from_sql(clean_sql)
    all_real_columns = set()
    for table_cols in schema.values():
        all_real_columns.update(table_cols.keys())

    hallucinated_columns = referenced_columns - all_real_columns
    if hallucinated_columns:
        issues.append(f"Hallucinated columns: {sorted(hallucinated_columns)}")

    # 3. Check column-table ownership (is this column in the right table?)
    wrong_table_columns = []
    for col in referenced_columns:
        if col in all_real_columns:
            # Find which table(s) this column belongs to
            valid_tables = {t for t, cols in schema.items() if col in cols}
            # Check if any referenced table contains this column
            if referenced_tables and not (referenced_tables & valid_tables):
                wrong_table_columns.append(f"{col} (belongs to {sorted(valid_tables)}, but query uses {sorted(referenced_tables)})")

    if wrong_table_columns:
        issues.append(f"Column-table mismatch: {wrong_table_columns}")

    # 4. Check join validity
    joins = extract_joins_from_sql(clean_sql)
    invalid_joins = []
    for join in joins:
        t1 = resolve_alias(clean_sql, join["alias1"])
        t2 = resolve_alias(clean_sql, join["alias2"])

        if t1 in real_tables and t2 in real_tables:
            pair = (t1, t2)
            if pair in VALID_JOINS:
                valid_col1, valid_col2 = VALID_JOINS[pair]
                if join["col1"] != valid_col1 or join["col2"] != valid_col2:
                    # Check reverse
                    if join["col1"] != valid_col2 or join["col2"] != valid_col1:
                        invalid_joins.append(f"{t1}.{join['col1']} = {t2}.{join['col2']} (expected: {valid_col1} = {valid_col2})")

    if invalid_joins:
        issues.append(f"Invalid joins: {invalid_joins}")

    stats = {
        "tables_referenced": len(referenced_tables),
        "tables_hallucinated": len(hallucinated_tables),
        "columns_referenced": len(referenced_columns),
        "columns_hallucinated": len(hallucinated_columns),
        "joins_found": len(joins),
        "joins_invalid": len(invalid_joins),
    }

    return {
        "faithful": len(issues) == 0,
        "issues": issues,
        "stats": stats,
        "referenced_tables": sorted(referenced_tables),
        "hallucinated_tables": sorted(hallucinated_tables),
        "referenced_columns": sorted(referenced_columns),
        "hallucinated_columns": sorted(hallucinated_columns),
    }


def run_faithfulness_check():
    """Main evaluation — reads accuracy_report.json and checks each generated SQL."""

    if not os.path.exists(ACCURACY_REPORT):
        print(f"ERROR: {ACCURACY_REPORT} not found. Run accuracy_eval.py first.")
        return

    with open(ACCURACY_REPORT, "r") as f:
        accuracy_data = json.load(f)

    schema = load_schema()

    print("=" * 70)
    print("FAITHFULNESS CHECK")
    print(f"Schema: {len(schema)} tables, {sum(len(v) for v in schema.values())} columns")
    print(f"Database: {DB_PATH}")
    print("=" * 70)

    results = []
    totals = {"faithful": 0, "unfaithful": 0, "skipped": 0}
    issue_counts = {"hallucinated_tables": 0, "hallucinated_columns": 0, "column_table_mismatch": 0, "invalid_joins": 0}
    tier_results = {}

    for r in accuracy_data["results"]:
        qid = r["id"]
        question = r["question"]
        difficulty = r["difficulty"]
        agent_sql = r.get("agent_sql", "")

        if difficulty not in tier_results:
            tier_results[difficulty] = {"total": 0, "faithful": 0}

        # Skip safety refusals (no SQL generated)
        if r.get("match_type") == "correct_refusal" or not agent_sql or "MISSING" in agent_sql or "SECURITY" in agent_sql:
            totals["skipped"] += 1
            continue

        tier_results[difficulty]["total"] += 1

        check = check_faithfulness(agent_sql, schema)
        check["id"] = qid
        check["question"] = question
        check["difficulty"] = difficulty

        if check["faithful"]:
            totals["faithful"] += 1
            tier_results[difficulty]["faithful"] += 1
            print(f"  ✅ {qid}: Faithful ({check['stats']['tables_referenced']} tables, {check['stats']['columns_referenced']} cols)")
        else:
            totals["unfaithful"] += 1
            print(f"  ❌ {qid}: {check['issues']}")

            # Count issue types
            for issue in check["issues"]:
                if "Hallucinated tables" in issue:
                    issue_counts["hallucinated_tables"] += 1
                if "Hallucinated columns" in issue:
                    issue_counts["hallucinated_columns"] += 1
                if "Column-table mismatch" in issue:
                    issue_counts["column_table_mismatch"] += 1
                if "Invalid joins" in issue:
                    issue_counts["invalid_joins"] += 1

        results.append(check)

    # Summary
    total_checked = totals["faithful"] + totals["unfaithful"]
    faithfulness_rate = round(totals["faithful"] / total_checked * 100, 1) if total_checked > 0 else 0

    tier_summary = {}
    for tier, counts in tier_results.items():
        if counts["total"] > 0:
            tier_summary[tier] = {
                "total": counts["total"],
                "faithful": counts["faithful"],
                "rate": round(counts["faithful"] / counts["total"] * 100, 1),
            }

    report = {
        "summary": {
            "total_checked": total_checked,
            "faithful": totals["faithful"],
            "unfaithful": totals["unfaithful"],
            "skipped": totals["skipped"],
            "faithfulness_rate_pct": faithfulness_rate,
            "issue_breakdown": issue_counts,
        },
        "tier_results": tier_summary,
        "results": results,
    }

    print("\n" + "=" * 70)
    print("FAITHFULNESS CHECK SUMMARY")
    print("=" * 70)
    print(f"  Queries Checked:      {total_checked}")
    print(f"  Faithful:             {totals['faithful']}")
    print(f"  Unfaithful:           {totals['unfaithful']}")
    print(f"  Skipped (refusals):   {totals['skipped']}")
    print(f"  Faithfulness Rate:    {faithfulness_rate}%")

    if any(v > 0 for v in issue_counts.values()):
        print("\n  Issue Breakdown:")
        for issue_type, count in issue_counts.items():
            if count > 0:
                print(f"    {issue_type}: {count}")

    print("\n  Per-Tier:")
    for tier in ["simple_select", "single_join", "aggregation", "multi_hop", "window_function", "simulation"]:
        if tier in tier_summary:
            t = tier_summary[tier]
            print(f"    {tier:20s}: {t['faithful']}/{t['total']} ({t['rate']}%)")

    # Show unfaithful queries
    unfaithful = [r for r in results if not r["faithful"]]
    if unfaithful:
        print("\n  Unfaithful Queries:")
        for r in unfaithful:
            print(f"    {r['id']}: {r['issues'][0][:80]}")

    print("=" * 70)

    output_path = os.path.join(RESULTS_DIR, "faithfulness_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {output_path}")

    return report


if __name__ == "__main__":
    run_faithfulness_check()