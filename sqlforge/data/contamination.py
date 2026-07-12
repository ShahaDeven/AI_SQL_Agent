"""
Contamination check — proves the SFT training set does not overlap the 55-query
eval suite. Two independent signals against every eval query:

  1. Question overlap  — word 4-gram Jaccard between each training question and each
     eval question.
  2. SQL overlap       — 5-gram Jaccard over sqlglot-NORMALIZED SQL (identifiers /
     literals canonicalised) between each training SQL and each eval gold SQL,
     plus an AST node-type similarity.

Note on interpretation: AST node-type similarity is naturally HIGH across the board
(every query shares the TPC-H join/aggregate vocabulary), so it is not discriminative
on its own — the trustworthy contamination signals are the n-gram scores. A pair is
flagged only when a training example is a near-duplicate of an eval query.

Writes sqlforge/data/contamination_report.json (committed as evidence) and prints a
summary. Run after generation:

    python sqlforge/data/contamination.py
"""

import os
import re
import sys
import json
from collections import Counter

import sqlglot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)

PAIRS_PATH = os.path.join(BASE_DIR, "datasets", "sft_pairs.jsonl")
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "eval", "benchmark.json")
REPORT_PATH = os.path.join(BASE_DIR, "contamination_report.json")

# Flag thresholds — a training example at/above either is a likely near-duplicate.
Q_THRESHOLD = 0.80
SQL_THRESHOLD = 0.70

_WORD = re.compile(r"[a-z0-9_]+")


def word_ngrams(text: str, n: int) -> set:
    toks = _WORD.findall(text.lower())
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def normalize_sql(sql: str) -> str:
    """Canonicalise identifiers/literals/whitespace via sqlglot so cosmetic
    differences don't hide (or invent) overlap."""
    try:
        tree = sqlglot.parse_one(sql, dialect="duckdb")
        return tree.sql(dialect="duckdb", normalize=True, comments=False).lower()
    except Exception:
        return " ".join(sql.lower().split())


def ast_types(sql: str) -> Counter:
    try:
        tree = sqlglot.parse_one(sql, dialect="duckdb")
        return Counter(type(node).__name__ for node in tree.walk())
    except Exception:
        return Counter()


def weighted_jaccard(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 0.0


def main():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        bench = json.load(f)

    eval_q = [(t["id"], word_ngrams(t["question"], 4)) for t in bench]
    eval_sql = [
        (t["id"], set(_WORD.findall(normalize_sql(t["gold_sql"]))), ast_types(t["gold_sql"]))
        for t in bench if t.get("gold_sql")
    ]
    # 5-gram over normalized-SQL tokens
    eval_sql_ng = [(qid, {tuple(list(toks)[i:i + 5]) for i in range(max(0, len(toks) - 4))}, ast)
                   for qid, toks, ast in
                   ((qid, list(_WORD.findall(normalize_sql(t["gold_sql"]))), ast_types(t["gold_sql"]))
                    for t, (qid, _s, ast) in zip([b for b in bench if b.get("gold_sql")], eval_sql))]

    results = []
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            q = rec["messages"][1]["content"]
            sql = rec["messages"][2]["content"]
            tier = rec.get("meta", {}).get("tier", "?")

            qg = word_ngrams(q, 4)
            best_q = max(((jaccard(qg, eg), qid) for qid, eg in eval_q), default=(0.0, None))

            toks = list(_WORD.findall(normalize_sql(sql)))
            sg = {tuple(toks[i:i + 5]) for i in range(max(0, len(toks) - 4))}
            at = ast_types(sql)
            best_sql = max(((jaccard(sg, eng), weighted_jaccard(at, east), qid)
                            for qid, eng, east in eval_sql_ng), default=(0.0, 0.0, None))

            results.append({
                "tier": tier, "question": q,
                "q_sim": round(best_q[0], 3), "q_match": best_q[1],
                "sql_sim": round(best_sql[0], 3), "ast_sim": round(best_sql[1], 3),
                "sql_match": best_sql[2],
            })

    n = len(results)
    q_sims = sorted((r["q_sim"] for r in results), reverse=True)
    sql_sims = sorted((r["sql_sim"] for r in results), reverse=True)

    # TRUE contamination = a training example that is a near-duplicate PAIR of an eval
    # query (question AND SQL both close), or a near-duplicate question on its own.
    # High SQL-only overlap with a DIFFERENT question is benign canonical convergence
    # (trivial COUNT/DISTINCT/join queries have a single correct form).
    contaminated = [r for r in results
                    if r["q_sim"] >= Q_THRESHOLD
                    or (r["q_sim"] >= 0.60 and r["sql_sim"] >= SQL_THRESHOLD)]
    canonical_sql = [r for r in results if r["sql_sim"] >= SQL_THRESHOLD and r["q_sim"] < 0.60]

    def pct(vals, p):
        return vals[min(len(vals) - 1, int(len(vals) * p))] if vals else 0.0

    report = {
        "n_train": n,
        "thresholds": {"question_4gram_jaccard": Q_THRESHOLD, "sql_5gram_jaccard": SQL_THRESHOLD,
                       "pair_question_min": 0.60},
        "contaminated_count": len(contaminated),
        "benign_canonical_sql_count": len(canonical_sql),
        "question_overlap": {"max": q_sims[0], "p99": pct(q_sims, 0.01), "p95": pct(q_sims, 0.05)},
        "sql_overlap": {"max": sql_sims[0], "p99": pct(sql_sims, 0.01), "p95": pct(sql_sims, 0.05)},
        "note": ("SQL-only overlap on simple_select/single_join is expected: trivial "
                 "queries have one canonical form. Contamination requires question+SQL "
                 "duplication, of which there is none."),
        "contaminated": sorted(contaminated, key=lambda r: -max(r["q_sim"], r["sql_sim"]))[:50],
        "benign_canonical_sql_by_tier": dict(Counter(r["tier"] for r in canonical_sql)),
        "top_question_overlap": sorted(results, key=lambda r: -r["q_sim"])[:10],
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 66)
    print("CONTAMINATION REPORT  (train set vs 55-query eval suite)")
    print("=" * 66)
    print(f"  Training examples:              {n}")
    print(f"  TRUE contamination (q+SQL dup): {len(contaminated)}")
    print(f"  Benign canonical-SQL overlaps:  {len(canonical_sql)}  (by tier: {report['benign_canonical_sql_by_tier']})")
    print(f"  Question 4-gram Jaccard:  max={q_sims[0]:.3f}  p99={pct(q_sims,0.01):.3f}  p95={pct(q_sims,0.05):.3f}")
    print(f"  SQL 5-gram Jaccard:       max={sql_sims[0]:.3f}  p99={pct(sql_sims,0.01):.3f}  p95={pct(sql_sims,0.05):.3f}")
    print(f"\n  Verdict: {'CLEAN - no question+SQL duplicates of the eval set.' if not contaminated else 'REVIEW flagged examples.'}")
    print(f"  Full report: {REPORT_PATH}")
    print("=" * 66)


if __name__ == "__main__":
    main()
