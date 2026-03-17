"""
EVALUATION & TEST SUITE for AI SQL Agent
=========================================
Covers:
  1. SQL Safety Checker (Unit Tests)
  2. Retriever Quality (Relevance Tests)
  3. Agent Workflow (Integration Tests)
  4. Cache Behavior (Unit Tests)

Run:  pytest tests/test_agent.py -v
"""

import os
import sys
import json
import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup – make sure we can import from src/
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.agent_graph import check_sql_safety, run_query, get_schema, get_cached_sql, save_to_cache
from src.retriever import get_few_shot_examples, _load_documents


# ====================================================================
# 1. SQL SAFETY CHECKER  – the most critical security boundary
# ====================================================================
class TestSQLSafety:
    """Verify that dangerous SQL is blocked and safe SQL is allowed."""

    # --- Should PASS (safe queries) ---
    @pytest.mark.parametrize("sql", [
        "SELECT * FROM customer LIMIT 10",
        "SELECT c_name, SUM(total_value) FROM lineitem GROUP BY c_name",
        "WITH cte AS (SELECT * FROM orders) SELECT * FROM cte",
        "SELECT COUNT(*) FROM nation WHERE n_name = 'FRANCE'",
        "SELECT DISTINCT r_name FROM region",
    ])
    def test_safe_queries_pass(self, sql):
        """Safe SELECT statements should not raise."""
        # Should complete without raising
        check_sql_safety(sql)

    # --- Should FAIL (dangerous queries) ---
    @pytest.mark.parametrize("sql,keyword", [
        ("DROP TABLE customer", "DROP"),
        ("DELETE FROM orders WHERE 1=1", "DELETE"),
        ("INSERT INTO customer VALUES (1, 'hack')", "INSERT"),
        ("UPDATE supplier SET s_name = 'evil'", "UPDATE"),
        ("ALTER TABLE lineitem ADD COLUMN malware TEXT", "ALTER"),
        ("TRUNCATE TABLE nation", "TRUNCATE"),
    ])
    def test_dangerous_queries_blocked(self, sql, keyword):
        """DDL/DML statements must raise ValueError."""
        with pytest.raises(ValueError, match="SECURITY ALERT"):
            check_sql_safety(sql)

    def test_mixed_safe_and_dangerous(self):
        """A SELECT followed by a DROP should still be caught."""
        # Note: sqlparse may parse these as two statements
        sql = "SELECT * FROM customer; DROP TABLE customer;"
        with pytest.raises(ValueError):
            check_sql_safety(sql)


# ====================================================================
# 2. QUERY EXECUTION – run_query guardrails
# ====================================================================
class TestQueryExecution:
    """Verify run_query returns correct types and catches errors."""

    def test_valid_query_returns_dataframe(self):
        df, error = run_query("SELECT * FROM customer LIMIT 5")
        assert error is None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_invalid_sql_returns_error(self):
        df, error = run_query("SELECT * FROM nonexistent_table_xyz")
        assert df is None
        assert error is not None

    def test_drop_blocked_at_execution(self):
        """run_query has its own DROP/DELETE check as a second layer."""
        df, error = run_query("DROP TABLE customer")
        assert df is None
        assert "Security Error" in error or "Read-Only" in error

    def test_empty_result_is_valid(self):
        df, error = run_query("SELECT * FROM customer WHERE c_name = 'NONEXISTENT_NAME_XYZ'")
        assert error is None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ====================================================================
# 3. SCHEMA RETRIEVAL
# ====================================================================
class TestSchema:
    """Verify the schema function returns expected tables and columns."""

    def test_schema_contains_target_tables(self):
        schema = get_schema()
        for table in ["customer", "lineitem", "orders", "supplier", "nation", "part"]:
            assert table in schema, f"Table '{table}' missing from schema"

    def test_schema_is_nonempty_string(self):
        schema = get_schema()
        assert isinstance(schema, str)
        assert len(schema) > 100  # should be a substantial string


# ====================================================================
# 4. RETRIEVER QUALITY – does hybrid search return relevant examples?
# ====================================================================
class TestRetriever:
    """Verify the hybrid retriever returns topically relevant results."""

    def test_documents_load_successfully(self):
        docs = _load_documents()
        assert len(docs) > 0, "No documents loaded from JSON"
        assert all(hasattr(d, "page_content") for d in docs)
        assert all("sql_query" in d.metadata for d in docs)

    def test_retriever_returns_results(self):
        examples = get_few_shot_examples("total revenue by region")
        assert len(examples) > 0, "Retriever returned empty string"
        assert "SQL:" in examples, "Retriever output missing SQL examples"

    def test_revenue_query_retrieves_revenue_examples(self):
        """A revenue question should retrieve examples mentioning revenue-related terms."""
        examples = get_few_shot_examples("What is the total revenue per region?")
        examples_lower = examples.lower()
        # At least one of these terms should appear in retrieved examples
        revenue_terms = ["revenue", "total_value", "sum", "region"]
        matches = [t for t in revenue_terms if t in examples_lower]
        assert len(matches) >= 2, (
            f"Revenue query should retrieve relevant examples. "
            f"Found terms: {matches}. Full output:\n{examples}"
        )

    def test_supplier_query_retrieves_supplier_examples(self):
        """A supplier question should retrieve supplier-related examples."""
        examples = get_few_shot_examples("Which supplier has the most parts?")
        examples_lower = examples.lower()
        supplier_terms = ["supplier", "s_name", "part", "partsupp", "count"]
        matches = [t for t in supplier_terms if t in examples_lower]
        assert len(matches) >= 1, (
            f"Supplier query should retrieve relevant examples. "
            f"Found terms: {matches}. Full output:\n{examples}"
        )


# ====================================================================
# 5. CACHE BEHAVIOR
# ====================================================================
class TestCache:
    """Verify caching reads and writes correctly."""

    TEMP_CACHE = os.path.join(PROJECT_ROOT, "test_sql_cache_tmp.json")

    def setup_method(self):
        """Point cache to a temp file so we don't pollute the real one."""
        import src.agent_graph as ag
        self._original_cache = ag.CACHE_FILE
        ag.CACHE_FILE = self.TEMP_CACHE
        # Also update the module-level reference used by the functions
        if os.path.exists(self.TEMP_CACHE):
            os.remove(self.TEMP_CACHE)

    def teardown_method(self):
        import src.agent_graph as ag
        ag.CACHE_FILE = self._original_cache
        if os.path.exists(self.TEMP_CACHE):
            os.remove(self.TEMP_CACHE)

    def test_cache_miss_returns_none(self):
        result = get_cached_sql("never seen this question before xyz")
        assert result is None

    def test_save_and_retrieve(self):
        question = "test question for cache"
        sql = "SELECT 1"
        save_to_cache(question, sql)
        result = get_cached_sql(question)
        assert result == sql

    def test_cache_file_is_valid_json(self):
        save_to_cache("q1", "SELECT 1")
        save_to_cache("q2", "SELECT 2")
        with open(self.TEMP_CACHE, "r") as f:
            data = json.load(f)
        assert "q1" in data
        assert "q2" in data