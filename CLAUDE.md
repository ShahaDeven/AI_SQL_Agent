# AI SQL Agent — CLAUDE.md

## Project Overview

A production-grade natural language to SQL agent with a Streamlit web UI. Users ask questions in plain English; the agent generates and executes safe, read-only SQL against a DuckDB supply chain database.

---

## Project Structure

```
AI_SQL_Agent/
├── app.py                    # Streamlit UI (chat, sidebar tabs: Schema/Scenarios/Metrics)
├── requirements.txt          # Python dependencies (15 packages)
├── Dockerfile                # python:3.11-slim, exposes 8501
├── docker-compose.yml        # One-command deployment
├── docker-entrypoint.sh      # Auto-setup: generates DB + ChromaDB if missing
├── sql_cache.json            # LRU cache: question → SQL (JSON file)
├── check_models.py           # Utility to check available LLM models
│
├── src/
│   ├── agent_graph.py        # Core agent: LLM selection, self-healing loop, SQL safety
│   ├── retriever.py          # Hybrid BM25 + ChromaDB (50/50) retriever
│   ├── clarifier.py          # Rule-based ambiguity detection (regex, zero API cost)
│   ├── metrics.py            # Per-query observability: latency, tokens, cache hits
│   └── suppress_telemetry.py # Suppresses ChromaDB/HuggingFace telemetry noise
│
├── data/
│   ├── sql_examples.json     # Few-shot examples for retriever
│   ├── chroma_db/            # Vector store (local persistence)
│   ├── sql_agent_demo.db     # Demo DB — TPC-H SF=0.1 (~10MB)
│   └── supply_chain.db       # Full DB — TPC-H SF=1 (~1GB)
│
├── eval/
│   ├── benchmark.json        # 60 labeled queries across 7 tiers
│   ├── accuracy_eval.py      # Result-set comparison evaluator
│   ├── retrieval_eval.py     # Precision@k, Recall@k, MRR
│   ├── faithfulness_check.py # Schema validation (tables, columns, joins)
│   └── results/              # JSON evaluation reports
│
├── tests/
│   └── test_agent.py         # 25 unit & integration tests (pytest)
│
└── scripts/
    ├── demo_db.py            # Generate demo DB (TPC-H SF=0.1)
    ├── generate_data.py      # Generate full DB (SF=1)
    └── explore_db.py         # Schema inspection utility
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Anthropic Claude Sonnet 4 (auto-detects Google Gemini 2.5 Flash, OpenAI GPT-4o Mini) |
| **Orchestration** | LangChain 0.2.11 (workflow, chat history) |
| **Database** | DuckDB (in-process OLAP, read-only enforced) |
| **Frontend** | Streamlit (chat UI, sidebar tabs, auto-visualization) |
| **Vector Store** | ChromaDB (local persistence) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (CPU-only) |
| **Retrieval** | LangChain EnsembleRetriever (BM25 + semantic, 50/50) |
| **SQL Parsing** | sqlparse (statement-type + keyword validation) |
| **Testing** | pytest |
| **Containerization** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions → GHCR |
| **Language** | Python 3.11+ |

---

## How to Run

### Local (with Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env  # or create .env manually
# Add: ANTHROPIC_API_KEY="sk-ant-..."

# 3. Generate demo database (first time only)
python scripts/demo_db.py

# 4. Run the app
streamlit run app.py
# Opens at http://localhost:8501
```

### Docker (recommended)

```bash
# One command — auto-generates DB and ChromaDB if missing
docker-compose up
# Opens at http://localhost:8501
```

### Run Tests

```bash
pytest tests/
```

### Run Evaluation Suite

```bash
python eval/accuracy_eval.py
python eval/retrieval_eval.py
python eval/faithfulness_check.py
# Results saved to eval/results/
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required for Claude Sonnet 4 (primary LLM) |
| `GOOGLE_API_KEY` | Optional — fallback to Gemini 2.5 Flash |
| `OPENAI_API_KEY` | Optional — fallback to GPT-4o Mini |
| `ANONYMIZED_TELEMETRY` | Set to `False` to suppress ChromaDB telemetry |

LLM selection is automatic: checks Anthropic → Google → OpenAI in order.

---

## Coding Conventions

### Naming

- **Functions**: `snake_case` — `agent_workflow`, `check_sql_safety`, `run_query`
- **Classes**: `PascalCase` — `ClarificationResult`, `MetricsTracker`, `QueryMetrics`
- **Constants**: `UPPER_SNAKE_CASE` — `DB_PATH`, `CACHE_FILE`, `BLOCKED_KEYWORDS`
- **Private/internal**: underscore prefix — `_current`, `_timers`, `_load_documents`
- **Dataclasses**: `@dataclass` decorator with typed fields

### Type Hints

Always annotate function signatures and dataclass fields:
```python
def agent_workflow(user_question: str, chat_history: list = None) -> tuple:
    ...
```

### Error Handling

- Use try-except with specific error categories (NETWORK ERROR, SECURITY ALERT)
- Return `(result, error)` tuples from query functions
- Graceful fallback if optional services (cache, retriever) fail

### SQL Safety (dual-layer — never bypass)

1. `check_sql_safety()` in `src/agent_graph.py` — statement-type + keyword parsing
2. `run_query()` — DROP/DELETE check at execution time
- Blocked keywords: `DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE, GRANT, REVOKE`
- DuckDB is always opened in read-only mode

### Metrics Tracking Pattern

```python
metrics.begin_query(question)
metrics.start_timer("retriever")
# ... do work ...
metrics.stop_timer("retriever")
metrics.end_query(success=True)
```

### Caching Pattern

- JSON file (`sql_cache.json`) maps question strings → generated SQL
- Cache is checked before LLM call; invalidated on error
- Do not bypass the cache — it's part of the latency budget

### Self-Healing Loop

`agent_graph.py` retries failed SQL generation up to 3 times, feeding the error message back to the LLM. Do not increase this limit without benchmarking.

### Telemetry Suppression

`src/suppress_telemetry.py` must be imported before ChromaDB or HuggingFace in any new entry point to prevent noise.

### Streamlit Caching

Use `@st.cache_resource` for expensive shared objects (retriever, LLM client). Use `st.session_state` for per-session data (chat history, metrics, scenario comparisons).

---

## Architecture Summary

```
User Input
    ↓
Clarifier (regex ambiguity detection — zero API cost)
    ↓
Retriever (BM25 + ChromaDB hybrid → few-shot examples)
    ↓
LLM (Claude / Gemini / GPT-4o — generates SQL)
    ↓
SQL Safety Check (sqlparse + blocklist)
    ↓
DuckDB (read-only execution)
    ↓
Result + Auto-visualization (Streamlit)
```

For what-if/simulation queries, CTEs are injected into the SQL to modify parameters without touching the database.

---

## Key Performance Characteristics

- Avg latency: ~0.67s/query
- Table Recall@3: 99.1%
- Overall accuracy: 91.7% on 60-query benchmark
- Safety refusal rate: 100% on blocked queries

## Off Limits
- Never modify the eval/ benchmarks — they are ground truth
- Never open DuckDB in write mode
- Don't increase the self-healing retry limit beyond 3