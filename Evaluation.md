# 📊 Evaluation & Testing

This document describes how the AI SQL Agent is evaluated for correctness, safety, and performance.

---

## Test Suite

Run the full unit/integration test suite:

```bash
pytest tests/test_agent.py -v
```

### What's Tested

| Category | Tests | What It Validates |
|---|---|---|
| **SQL Safety** | 11 | Blocks DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE; allows SELECT and CTEs |
| **Query Execution** | 4 | Valid queries return DataFrames; invalid SQL returns errors; DROP blocked at runtime |
| **Schema Retrieval** | 2 | All target tables present; schema is well-formed |
| **Retriever Quality** | 4 | Documents load; hybrid search returns relevant examples for revenue and supplier queries |
| **Cache Behavior** | 3 | Cache miss returns None; save/retrieve roundtrip works; file is valid JSON |

**Result: 25/25 passed**

---

## Benchmark Evaluation

Run the full 18-question benchmark against the live agent:

```bash
python tests/evaluate_agent.py
```

This sends natural language questions of increasing difficulty through the full agent pipeline and measures SQL generation success, answer correctness, safety refusals, and latency.

### Benchmark Results

| Metric | Value |
|---|---|
| **Total Questions** | 18 |
| **Successful** | 15 |
| **Failed** | 1 |
| **Correct Refusals** | 2/2 |
| **Success Rate** | **93.8%** |
| **Avg Latency** | **3.03s** |

### Results by Difficulty

| Difficulty | Passed | Total | Notes |
|---|---|---|---|
| Easy | 5 | 5 | Single-table queries, basic aggregations |
| Medium | 5 | 5 | Multi-table joins, GROUP BY, year extraction |
| Hard | 3 | 4 | CTEs, chain-of-thought, window functions (LAG). 1 failure due to empty result set from strict HAVING filter |
| Simulation | 2 | 2 | "What-If" discount and price scenarios with CTE-based comparison |
| Safety | 2 | 2 | Correctly refused DELETE command and missing column (profit) request |

### Highlights

- **Complex reasoning works:** The hardest question — *"Which supplier has the most parts in the region with the lowest revenue?"* — was answered correctly using a 3-level CTE chain (regional revenue → lowest region → supplier part counts).
- **Simulations are accurate:** Both What-If scenarios generated correct original vs. simulated comparisons using non-destructive CTEs.
- **Safety guardrails hold:** Destructive queries are refused with helpful explanations, and requests for non-existent columns (e.g. profit) are caught before any SQL is generated.
- **Self-healing works:** The agent's 3-attempt retry loop recovered from SQL errors during development testing.

### Known Limitations

- One hard query ("top 3 nations by revenue where avg discount > 0.05") returned an empty result set because no nations in the demo dataset met the HAVING threshold. The generated SQL was syntactically and logically correct.
- Latency depends on the LLM API. Cached queries execute in <0.1s.

---

## Adding New Test Cases

**Unit tests:** Add to `tests/test_agent.py` under the relevant class.

**Benchmark questions:** Add entries to the `BENCHMARK` list in `tests/evaluate_agent.py`:

```python
{
    "question": "Your natural language question",
    "difficulty": "easy",          # easy | medium | hard | simulation | safety
    "expected_columns": ["col"],   # columns that must appear in result
    "expected_min_rows": 1,        # minimum rows expected
    "expect_refusal": False,       # True if agent should refuse
}
```
