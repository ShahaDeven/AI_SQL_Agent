"""
SQLForge — house-style knowledge base for synthetic SFT data generation.

This module encodes everything Phase 0 taught us about *why* the base 7B failed,
so that the synthetic training data teaches the right conventions:

  1. Real categorical VALUES (churn_risk='HIGH_RISK', not 'High') — ground-truthed
     from data/sql_agent_demo.db, not assumed.
  2. The CANONICAL JOIN PATHS the benchmark grades on (revenue is attributed to the
     CUSTOMER's region: lineitem -> orders -> customer -> nation -> region, NOT the
     supplier's).
  3. The revenue formula on the modified schema: SUM(total_value * (1 - promo_reduction)).
  4. Output "house style" (which columns gold includes, rounding).

Everything here is shared by generation, labeling, and validation so the three
stages agree on one definition of "correct".
"""

import os
import duckdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "sql_agent_demo.db")

# Tables the agent exposes (mirrors src/agent_graph.get_schema).
TARGET_TABLES = ["customer", "lineitem", "orders", "supplier",
                 "nation", "part", "region", "partsupp"]

# ---------------------------------------------------------------------------
# Ground-truth categorical values (queried from the DB, not assumed).
# Used both to steer question generation and to validate that generated SQL
# uses real literals.
# ---------------------------------------------------------------------------
CATEGORICAL_VALUES = {
    "churn_risk": ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"],
    "r_name": ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"],
    "c_mktsegment": ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"],
    "o_orderpriority": ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"],
    "o_orderstatus": ["F", "O", "P"],
}

DATE_RANGE = ("1992-01-01", "1998-08-02")  # o_orderdate min/max

# Columns the modified schema REPLACES. The base model kept reaching for these.
# Revenue must be computed from total_value / promo_reduction instead.
DISCOURAGED_REVENUE_COLUMNS = ["l_extendedprice", "l_discount"]

REVENUE_FORMULA = "SUM(total_value * (1 - promo_reduction))"

# ---------------------------------------------------------------------------
# Canonical join paths — the single most important convention the eval grades.
# ---------------------------------------------------------------------------
HOUSE_STYLE_RULES = f"""HOUSE-STYLE CONVENTIONS (follow exactly):

1. REVENUE is always {REVENUE_FORMULA}. Never use l_extendedprice or l_discount
   for revenue — the schema was modified and those are superseded by total_value
   and promo_reduction.

2. CANONICAL JOIN PATHS (attribute facts to the CUSTOMER's geography, not the
   supplier's, unless the question is explicitly about suppliers):
   - lineitem revenue by region/nation:
       lineitem -> orders (l_orderkey=o_orderkey)
                -> customer (o_custkey=c_custkey)
                -> nation (c_nationkey=n_nationkey)
                -> region (n_regionkey=r_regionkey)
   - supplier-centric questions ("revenue by supplier", "top supplier per region"):
       lineitem -> supplier (l_suppkey=s_suppkey) -> nation -> region
   - "average discount" means AVG(promo_reduction).

3. GROUP BY the NAME (r_name, n_name, c_mktsegment), never the id.

4. CATEGORICAL LITERALS must match the data exactly (case-sensitive):
   churn_risk in {CATEGORICAL_VALUES['churn_risk']};
   r_name in {CATEGORICAL_VALUES['r_name']};
   c_mktsegment in {CATEGORICAL_VALUES['c_mktsegment']}.

5. Use DuckDB syntax. EXTRACT(YEAR FROM o_orderdate) for date parts (not strftime,
   not ::year). ROUND(x, 2) for percentages/growth rates.

6. Return ONLY the SQL. No markdown, no explanation."""

# ---------------------------------------------------------------------------
# Tiers — mirror eval/benchmark.json's difficulty labels. Weights over-sample
# the tiers where the 7B was weak in Phase 0 (agg/multi/window/simulation).
# ---------------------------------------------------------------------------
TIERS = {
    "simple_select": {
        "weight": 0.10,
        "desc": "single-table SELECT / COUNT / DISTINCT / ORDER BY / LIMIT, no joins.",
        "seeds": [
            "How many parts are in the catalog?",
            "List the distinct order statuses.",
        ],
    },
    "single_join": {
        "weight": 0.15,
        "desc": "exactly one join between two tables, optional GROUP BY.",
        "seeds": [
            "List each supplier alongside the region they operate in.",
            "How many line items belong to each order status?",
        ],
    },
    "aggregation": {
        "weight": 0.20,
        "desc": "multi-table joins with SUM/AVG/COUNT and GROUP BY; revenue questions.",
        "seeds": [
            "What is the total revenue for the AUTOMOBILE market segment?",
            "What is the average account balance of MEDIUM_RISK customers?",
        ],
    },
    "multi_hop": {
        "weight": 0.20,
        "desc": "nested/CTE reasoning: filters that depend on an aggregate, top-N-within, ratios.",
        "seeds": [
            "Which region has the lowest total revenue?",
            "List market segments whose average order value exceeds the overall average.",
        ],
    },
    "window_function": {
        "weight": 0.20,
        "desc": "window functions: RANK/ROW_NUMBER partitions, LAG growth, running totals, moving averages, percentiles.",
        "seeds": [
            "Rank suppliers by revenue within each nation.",
            "Show the month-over-month order count trend for 1996.",
        ],
    },
    "simulation": {
        "weight": 0.15,
        "desc": "what-if scenarios producing original vs simulated revenue side-by-side with difference and pct_change.",
        "seeds": [
            "What if we cut the discount by 20%? Show total revenue before and after.",
            "Simulate a 10% price increase and show revenue by market segment.",
        ],
    },
}


def get_schema(db_path: str = DEFAULT_DB_PATH) -> str:
    """Return the schema string exactly as src/agent_graph.get_schema builds it."""
    con = duckdb.connect(db_path, read_only=True)
    schema_str = "Database Schema (DuckDB):\n"
    for table in TARGET_TABLES:
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        col_list = [f"{col[0]} {col[1]}" for col in cols]
        schema_str += f"- {table}: {', '.join(col_list)}\n"
    con.close()
    return schema_str


def tier_allocation(total: int) -> dict:
    """Split a total pair count across tiers by weight (largest remainder rounding)."""
    raw = {t: total * cfg["weight"] for t, cfg in TIERS.items()}
    floors = {t: int(v) for t, v in raw.items()}
    remainder = total - sum(floors.values())
    # hand out the leftover to the largest fractional parts
    frac = sorted(TIERS, key=lambda t: raw[t] - floors[t], reverse=True)
    for t in frac[:remainder]:
        floors[t] += 1
    return floors


# ---------------------------------------------------------------------------
# Simulation output contract — matches the benchmark gold (and the agent's
# SIMULATION_BLOCK). Sim gold must be COLUMN-shaped, not row-per-scenario, or it
# will never match the eval's result sets.
# ---------------------------------------------------------------------------
SIMULATION_RULES = """SIMULATION OUTPUT FORMAT (this is a what-if question):
- Return original AND simulated revenue in ONE result set, as COLUMNS (never as
  separate rows / UNION ALL scenario labels).
- Required columns: the group column if grouped (e.g. r_name or c_mktsegment),
  original_value, simulated_value, difference, pct_change.
- Revenue is SUM(total_value * (1 - promo_reduction)); apply the price/discount
  change inside the simulated term only.
- If a scope is named (a region or segment), apply the change ONLY to matching rows
  with CASE WHEN and keep the other rows unchanged.
- difference = simulated_value - original_value
  pct_change = ROUND(((simulated_value - original_value) / original_value) * 100, 2)
- Shape: WITH original AS (...), simulated AS (...)
         SELECT ..., original_value, simulated_value, difference, pct_change
         FROM original JOIN simulated USING (group_col)  -- or CROSS JOIN if ungrouped"""

_SIM_TRIGGERS = ("what if", "simulate", "simulation", "sensitivity", "scenario",
                 " vs ", "before and after", "original", "simulated")


def is_simulation(question: str) -> bool:
    q = question.lower()
    return any(t in q for t in _SIM_TRIGGERS)
