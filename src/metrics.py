"""
Agent Metrics Tracker
======================
Lightweight observability for the AI SQL Agent pipeline.

Tracks per-query:
  - Total latency & per-stage latency (retriever, LLM, DB execution)
  - Cache hit/miss
  - LLM retry count & self-healing recovery
  - Token usage estimates
  - Errors and safety refusals
  - Simulation detection

Session-level aggregates:
  - Total queries, success rate, cache hit rate
  - Average latency, total estimated tokens
"""

import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class QueryMetrics:
    """Metrics for a single agent query."""
    question: str = ""
    timestamp: str = ""

    # Timing (seconds)
    total_latency: float = 0.0
    retriever_latency: float = 0.0
    llm_latency: float = 0.0
    db_latency: float = 0.0

    # Cache
    cache_hit: bool = False

    # LLM
    llm_attempts: int = 0
    self_healed: bool = False  # True if failed on attempt 1 but recovered
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0

    # Result
    success: bool = False
    rows_returned: int = 0
    error: Optional[str] = None

    # Flags
    is_simulation: bool = False
    safety_refused: bool = False
    generated_sql: str = ""


class MetricsTracker:
    """
    Session-level metrics aggregator.
    Stores all query metrics and computes summaries.
    """

    def __init__(self):
        self.queries: list[QueryMetrics] = []
        self._current: Optional[QueryMetrics] = None
        self._timers: dict[str, float] = {}

    # ---- Timer helpers ----
    def start_timer(self, name: str):
        """Start a named timer."""
        self._timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed seconds."""
        if name in self._timers:
            elapsed = time.time() - self._timers[name]
            del self._timers[name]
            return round(elapsed, 4)
        return 0.0

    # ---- Query lifecycle ----
    def begin_query(self, question: str):
        """Start tracking a new query."""
        self._current = QueryMetrics(
            question=question,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.start_timer("total")

    def end_query(self):
        """Finalize and store the current query metrics."""
        if self._current:
            self._current.total_latency = self.stop_timer("total")
            self.queries.append(self._current)
            result = self._current
            self._current = None
            return result
        return None

    # ---- Per-stage recording ----
    def record_cache_hit(self):
        if self._current:
            self._current.cache_hit = True

    def record_retriever_time(self, elapsed: float):
        if self._current:
            self._current.retriever_latency = round(elapsed, 4)

    def record_llm_call(self, elapsed: float, attempt: int, prompt_chars: int = 0, completion_chars: int = 0):
        if self._current:
            self._current.llm_latency += round(elapsed, 4)
            self._current.llm_attempts = attempt
            self._current.estimated_prompt_tokens += prompt_chars // 4
            self._current.estimated_completion_tokens += completion_chars // 4
            if attempt > 1:
                self._current.self_healed = True

    def record_db_execution(self, elapsed: float, rows: int = 0):
        if self._current:
            self._current.db_latency = round(elapsed, 4)
            self._current.rows_returned = rows

    def record_success(self, sql: str):
        if self._current:
            self._current.success = True
            self._current.generated_sql = sql

    def record_error(self, error: str):
        if self._current:
            self._current.success = False
            self._current.error = error

    def record_simulation(self):
        if self._current:
            self._current.is_simulation = True

    def record_safety_refusal(self):
        if self._current:
            self._current.safety_refused = True

    # ---- Aggregated summaries ----
    def get_summary(self) -> dict:
        """Return session-level aggregate metrics."""
        if not self.queries:
            return {
                "total_queries": 0,
                "success_rate": 0.0,
                "cache_hit_rate": 0.0,
                "avg_latency": 0.0,
                "total_estimated_tokens": 0,
                "self_healing_recoveries": 0,
                "safety_refusals": 0,
                "simulations_run": 0,
            }

        total = len(self.queries)
        successes = sum(1 for q in self.queries if q.success)
        cache_hits = sum(1 for q in self.queries if q.cache_hit)
        healed = sum(1 for q in self.queries if q.self_healed)
        refused = sum(1 for q in self.queries if q.safety_refused)
        simulations = sum(1 for q in self.queries if q.is_simulation)
        total_tokens = sum(q.estimated_prompt_tokens + q.estimated_completion_tokens for q in self.queries)
        avg_latency = sum(q.total_latency for q in self.queries) / total

        return {
            "total_queries": total,
            "success_rate": round(successes / total * 100, 1),
            "cache_hit_rate": round(cache_hits / total * 100, 1),
            "avg_latency": round(avg_latency, 2),
            "total_estimated_tokens": total_tokens,
            "self_healing_recoveries": healed,
            "safety_refusals": refused,
            "simulations_run": simulations,
        }

    def get_recent_queries(self, n: int = 10) -> list[dict]:
        """Return the last N query metrics as dicts."""
        return [asdict(q) for q in self.queries[-n:]]

    def get_latency_breakdown(self) -> dict:
        """Average latency per pipeline stage."""
        if not self.queries:
            return {"retriever": 0, "llm": 0, "db": 0, "other": 0}

        n = len(self.queries)
        avg_ret = sum(q.retriever_latency for q in self.queries) / n
        avg_llm = sum(q.llm_latency for q in self.queries) / n
        avg_db = sum(q.db_latency for q in self.queries) / n
        avg_total = sum(q.total_latency for q in self.queries) / n
        avg_other = max(0, avg_total - avg_ret - avg_llm - avg_db)

        return {
            "retriever": round(avg_ret, 3),
            "llm": round(avg_llm, 3),
            "db": round(avg_db, 3),
            "other": round(avg_other, 3),
        }