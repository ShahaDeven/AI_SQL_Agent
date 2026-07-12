"""
SQLForge — the quality gate for synthetic SFT pairs.

Two independent checks; a pair must pass BOTH to enter the training set:

  1. execute_and_validate() — the pair's SQL runs read-only on the DuckDB TPC-H DB,
     within a wall-clock timeout, and returns a non-empty, non-degenerate result.
  2. house_style_check() — static checks that the SQL follows the conventions the
     eval grades (real categorical literals, revenue via total_value/promo_reduction,
     no superseded columns).

Keeping this gate strict is what separates a rigorous distillation set from a
"the teacher said so" dump.
"""

import re
import threading

import duckdb
import pandas as pd

from schema_context import CATEGORICAL_VALUES, DISCOURAGED_REVENUE_COLUMNS


# ---------------------------------------------------------------------------
# 1. Execution validation
# ---------------------------------------------------------------------------
def clean_sql(sql: str) -> str:
    """Strip markdown fences / trailing semicolons noise."""
    return sql.replace("```sql", "").replace("```", "").strip()


def execute_and_validate(
    sql: str,
    db_path: str,
    timeout: float = 5.0,
    max_rows: int = 50_000,
) -> tuple:
    """
    Execute SQL read-only with a wall-clock timeout.

    Returns (ok: bool, df: DataFrame | None, reason: str).
    Rejects: exec errors, timeouts, empty results, all-null single cells, and
    result sets larger than max_rows (a likely cartesian / missing join).
    """
    sql = clean_sql(sql)
    if not sql:
        return False, None, "empty_sql"

    con = duckdb.connect(db_path, read_only=True)
    # Watchdog: DuckDB has no statement_timeout, so interrupt from another thread.
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        df = con.execute(sql).fetchdf()
    except duckdb.InterruptException:
        return False, None, f"timeout_>{timeout}s"
    except Exception as e:  # noqa: BLE001 — we want the message text
        return False, None, f"exec_error: {str(e)[:200]}"
    finally:
        timer.cancel()
        con.close()

    if df is None or df.empty:
        return False, None, "empty_result"
    if len(df) > max_rows:
        return False, None, f"too_many_rows: {len(df)} (> {max_rows}, likely cartesian)"
    # Degenerate single-cell null (e.g. SUM over an empty filter).
    if df.shape == (1, 1) and pd.isna(df.iloc[0, 0]):
        return False, None, "single_null_cell"

    return True, df, "ok"


# ---------------------------------------------------------------------------
# 2. House-style static checks
# ---------------------------------------------------------------------------
_CHURN_LIT = re.compile(r"churn_risk\s*(?:=|IN|in)\s*\(?\s*'([^']*)'", re.IGNORECASE)
_RNAME_LIT = re.compile(r"r_name\s*(?:=|IN|in)\s*\(?\s*'([^']*)'", re.IGNORECASE)
_SEG_LIT = re.compile(r"c_mktsegment\s*(?:=|IN|in)\s*\(?\s*'([^']*)'", re.IGNORECASE)


def house_style_check(sql: str) -> tuple:
    """
    Static convention checks. Returns (ok: bool, issues: list[str]).

    These catch the exact mistakes the base 7B made: wrong categorical casing,
    and computing revenue from the superseded l_extendedprice/l_discount columns.
    """
    issues = []
    low = sql.lower()

    # Superseded revenue columns.
    for col in DISCOURAGED_REVENUE_COLUMNS:
        if col in low:
            issues.append(f"uses_superseded_column:{col}")

    # Categorical literals must match the data exactly (case-sensitive).
    for pattern, key in ((_CHURN_LIT, "churn_risk"),
                         (_RNAME_LIT, "r_name"),
                         (_SEG_LIT, "c_mktsegment")):
        for value in pattern.findall(sql):
            if value not in CATEGORICAL_VALUES[key]:
                issues.append(f"bad_literal:{key}='{value}'")

    return (len(issues) == 0), issues


def validate_pair(sql: str, db_path: str, enforce_house_style: bool = True) -> dict:
    """
    Full gate for one (question, sql) pair.

    Returns a dict: {ok, reason, rows, cols, house_issues}.
    """
    ok, df, reason = execute_and_validate(sql, db_path)
    if not ok:
        return {"ok": False, "reason": reason, "rows": 0, "cols": 0, "house_issues": []}

    hs_ok, issues = house_style_check(sql)
    if enforce_house_style and not hs_ok:
        return {
            "ok": False,
            "reason": "house_style:" + ",".join(issues),
            "rows": len(df),
            "cols": df.shape[1],
            "house_issues": issues,
        }

    return {
        "ok": True,
        "reason": "ok",
        "rows": len(df),
        "cols": df.shape[1],
        "house_issues": issues,  # may be non-empty if enforce_house_style=False
    }
