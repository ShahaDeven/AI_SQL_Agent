"""
Phase 0 — result-set comparison utilities.

This module is a SELF-CONTAINED copy of the comparison logic in
`eval/accuracy_eval.py` (the same logic behind the reported 91.7% Claude number),
plus a robust SQL extractor and a DuckDB executor.

It deliberately has ZERO dependency on `src/agent_graph.py`, langchain, or any
API key, so it can run on a bare GPU pod that only has: pandas, duckdb.

Keeping the comparison identical to the production harness is what makes the
base-model number directly comparable to the existing agent numbers.
"""

import re
import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# SQL extraction + execution
# ---------------------------------------------------------------------------
def extract_sql(text: str) -> str:
    """
    Pull a single SQL statement out of a raw model completion.

    Handles: ```sql fences, leading prose ("Here is the query:"), and trailing
    explanation after the statement. Returns the statement (with trailing ';').
    """
    if not text:
        return ""
    t = text.strip()

    # 1. Prefer the content of the first fenced code block if present.
    if "```" in t:
        m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
        if m:
            t = m.group(1).strip()
        else:
            t = t.replace("```sql", "").replace("```", "").strip()

    # 2. Drop any leading prose before the first WITH/SELECT.
    m = re.search(r"(?is)\b(with|select)\b", t)
    if m:
        t = t[m.start():]

    # 3. Cut trailing explanation after the first statement terminator.
    if ";" in t:
        t = t[: t.index(";") + 1]

    return t.strip()


def execute_sql(sql: str, db_path: str) -> tuple:
    """Execute SQL read-only on DuckDB. Returns (DataFrame, error_str_or_None)."""
    try:
        clean = sql.replace("```sql", "").replace("```", "").strip()
        if not clean:
            return None, "empty SQL"
        con = duckdb.connect(db_path, read_only=True)
        df = con.execute(clean).fetchdf()
        con.close()
        return df, None
    except Exception as e:  # noqa: BLE001 - we want the message string
        return None, str(e)


# ---------------------------------------------------------------------------
# Normalisation + comparison  (verbatim from eval/accuracy_eval.py)
# ---------------------------------------------------------------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame for comparison."""
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().all():
                    df[col] = converted
            except Exception:
                pass

    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].round(2)

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
        name = name.lower().strip().replace("_", "").replace(" ", "")
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
    """Compare gold and agent result sets. Returns {match, match_type, details}."""
    gold_norm = normalize_df(gold_df)
    agent_norm = normalize_df(agent_df)

    # 1. Exact match
    if set(gold_norm.columns) == set(agent_norm.columns):
        agent_reordered = agent_norm[gold_norm.columns]
        if gold_norm.equals(agent_reordered):
            return {"match": True, "match_type": "exact", "details": "Exact match"}

    # 2. Fuzzy column name match
    if len(gold_norm) == len(agent_norm) and fuzzy_column_match(
        list(gold_norm.columns), list(agent_norm.columns)
    ):
        try:
            if pd.DataFrame(gold_norm.values).round(2).equals(
                pd.DataFrame(agent_norm.values).round(2)
            ):
                return {
                    "match": True,
                    "match_type": "fuzzy_column_match",
                    "details": f"Same values, fuzzy column match (gold: {list(gold_norm.columns)}, agent: {list(agent_norm.columns)})",
                }
        except Exception:
            pass

    # 3. Value match (same values, different column names)
    if len(gold_norm) == len(agent_norm) and len(gold_norm.columns) == len(agent_norm.columns):
        try:
            if pd.DataFrame(gold_norm.values).round(2).equals(
                pd.DataFrame(agent_norm.values).round(2)
            ):
                return {
                    "match": True,
                    "match_type": "value_match",
                    "details": "Same values, different column names",
                }
        except Exception:
            pass

    # 4. Row count match with key column overlap + numeric proximity (within 1%)
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
                        all_close = True
                        for gi, ai in zip(range(len(gold_nums.columns)), range(len(agent_nums.columns))):
                            gs = gold_nums.iloc[:, gi].sum()
                            as_ = agent_nums.iloc[:, ai].sum()
                            if gs != 0 and abs(gs - as_) / abs(gs) > 0.01:
                                all_close = False
                                break
                        if all_close:
                            return {
                                "match": True,
                                "match_type": "approximate",
                                "details": "Same groups, all numeric columns within 1%",
                            }

    # 5. Subset match
    if len(gold_norm) == len(agent_norm):
        common_cols = set(gold_norm.columns) & set(agent_norm.columns)
        if common_cols and len(common_cols) >= len(gold_norm.columns) * 0.5:
            gold_sub = normalize_df(gold_norm[sorted(common_cols)])
            agent_sub = normalize_df(agent_norm[sorted(common_cols)])
            if gold_sub.equals(agent_sub):
                return {
                    "match": True,
                    "match_type": "subset_match",
                    "details": f"Common columns match: {common_cols}",
                }

    # 6. Lenient: same shape, numeric within 5%
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
                return {
                    "match": True,
                    "match_type": "lenient_match",
                    "details": "Same shape, numeric values within 5%",
                }

    details = (
        f"Mismatch: gold has {len(gold_norm)} rows x {len(gold_norm.columns)} cols "
        f"({list(gold_norm.columns)}), agent has {len(agent_norm)} rows x "
        f"{len(agent_norm.columns)} cols ({list(agent_norm.columns)})"
    )
    return {"match": False, "match_type": "mismatch", "details": details}
