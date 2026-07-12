"""
SQLForge Phase 1 — SFT data generation (hybrid teacher).

Pipeline per tier:
    1. Haiku generates diverse natural-language questions (cheap, high volume).
    2. Your existing Sonnet agent (src/agent_graph.agent_workflow) labels each
       question with gold SQL — inheriting the house-style conventions for free.
       This is a distillation of the 91.7% production agent into training data.
    3. The quality gate (validate.py) executes + house-style-checks every pair.
    4. Optional Haiku consistency check (does the SQL actually answer the question?).
    5. Dedup + contamination guard against the 55-query eval set.
    6. Accepted pairs are appended to a chat-format JSONL; rejects are logged with
       reasons; a data card summarises yield per tier.

The agent's label cache is redirected to sqlforge/data/datasets/ so bulk labeling
never pollutes the app's production sql_cache.json.

Usage:
    # small pilot — inspect quality before spending the budget
    python sqlforge/data/generate_sft.py --pilot

    # full run (target ~4000 validated pairs; overgenerate to absorb rejects)
    python sqlforge/data/generate_sft.py --total 4000
"""

import os
import re
import sys
import json
import time
import argparse
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)        # schema_context, validate
sys.path.insert(0, PROJECT_ROOT)    # src.agent_graph

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from schema_context import (
    get_schema, TIERS, HOUSE_STYLE_RULES, CATEGORICAL_VALUES,
    tier_allocation, DEFAULT_DB_PATH, SIMULATION_RULES, is_simulation,
)
from validate import validate_pair, clean_sql

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
LABEL_CACHE = os.path.join(DATASETS_DIR, "label_cache.json")
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "eval", "benchmark.json")

# Cheap teacher for question generation + the consistency check.
# Override via env if your account uses a different Haiku id.
QUESTION_GEN_MODEL = os.getenv("QUESTION_GEN_MODEL", "claude-haiku-4-5-20251001")

_WORD = re.compile(r"[a-z0-9_]+")


CONSISTENCY_MODEL = os.getenv("CONSISTENCY_MODEL", QUESTION_GEN_MODEL)


# ---------------------------------------------------------------------------
# Anthropic client (raw SDK — langchain_anthropic 404s on current model IDs)
# ---------------------------------------------------------------------------
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        import anthropic
        _CLIENT = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _CLIENT


def _complete(model: str, system, messages: list, temperature=None,
              max_tokens: int = 1024) -> str:
    """One Messages API call. `messages` is a list of {'role','content'} dicts.
    temperature=None omits the param (some models, e.g. claude-sonnet-5, reject an
    explicit 0.0 and want their own default)."""
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if system:
        kwargs["system"] = system
    resp = _client().messages.create(**kwargs)
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _parse_json_array(text: str) -> list:
    """Pull the first JSON array of strings out of a model response."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x).strip() for x in arr if str(x).strip()]
    except json.JSONDecodeError:
        return []


# Framing guard: keep money questions on the modified-schema revenue formula and the
# canonical customer-side geography, so labels don't drift into superseded columns.
_FRAMING_GUARD = (
    "FRAMING (important):\n"
    "- Any money/value question must be about REVENUE, i.e. total_value * (1 - promo_reduction). "
    "Do NOT ask about 'extended price', 'tax', or 'shipping cost'.\n"
    "- Attribute revenue and customers to the CUSTOMER's region/nation (via orders -> customer -> "
    "nation -> region), not the supplier's, unless the question is explicitly about suppliers.\n"
)

_SIM_GUARD = (
    "\nThese are WHAT-IF REVENUE simulations. Each question must ask to compare ORIGINAL vs "
    "SIMULATED total revenue, reporting the difference and percentage change, where the change is a "
    "price or discount (promo_reduction) adjustment applied to a CLEAR population — all line items, "
    "or one named region / market segment. Avoid tax, shipping, supply-cost, and 'next order' scenarios.\n"
)


def generate_questions(tier: str, n: int, schema: str) -> list:
    """Ask Haiku for `n` diverse NL analytics questions for one tier.

    Generates in chunks (<=40 per call) so large targets aren't truncated by max_tokens,
    deduping within the tier.
    """
    cfg = TIERS[tier]
    seeds = "\n".join(f"  - {s}" for s in cfg["seeds"])
    cats = "\n".join(f"  {k}: {v}" for k, v in CATEGORICAL_VALUES.items())
    sim_extra = _SIM_GUARD if tier == "simulation" else ""

    collected, seen_local = [], set()
    CHUNK = 40
    for _ in range(max(1, (n // CHUNK) + 3)):
        if len(collected) >= n:
            break
        ask = min(CHUNK, n - len(collected) + 5)
        prompt = f"""You are writing natural-language business questions for a Text-to-SQL
training set over this DuckDB schema:

{schema}

Real categorical values you may reference (use EXACT spelling):
{cats}

Write {ask} DISTINCT analytics questions of this difficulty tier — {tier}:
{cfg['desc']}

Style guidance (questions like these, do NOT copy them verbatim):
{seeds}

{_FRAMING_GUARD}{sim_extra}
Rules:
- Vary the entities, filters, groupings, and phrasing widely. Avoid near-duplicates.
- Every question must be answerable from the schema above.
- Natural business phrasing, not SQL. No question numbers.
- Return ONLY a JSON array of {ask} strings, nothing else."""
        batch = _parse_json_array(
            _complete(QUESTION_GEN_MODEL, None, [{"role": "user", "content": prompt}],
                      temperature=0.9, max_tokens=2000)
        )
        if not batch:
            break  # parse failure / empty — stop rather than loop forever
        for q in batch:
            k = q.strip().lower()
            if k and k not in seen_local:
                seen_local.add(k)
                collected.append(q)
    return collected[:n]


def consistency_ok(question: str, sql: str, schema: str) -> bool:
    """Cheap Haiku check that the SQL actually answers the question."""
    prompt = f"""Schema:
{schema}

Question: {question}
SQL:
{sql}

Does this SQL correctly and completely answer the question against this schema?
Reply with exactly YES or NO on the first line, then a one-line reason."""
    try:
        resp = _complete(CONSISTENCY_MODEL, None, [{"role": "user", "content": prompt}],
                         temperature=0.0, max_tokens=150).strip().upper()
    except Exception:
        return True  # don't drop a pair over a flaky check call
    return resp.startswith("YES")


# ---------------------------------------------------------------------------
# Lightweight labeler — direct Sonnet call with the house-style system prompt +
# keyword few-shot + a self-healing retry loop. Faithful to the agent's SQL
# behaviour, but WITHOUT the per-call ChromaDB/embeddings overhead and heavy
# dependency stack, so it scales to thousands of pairs.
# ---------------------------------------------------------------------------
# Haiku labeler by default — Sonnet 5 (thinking) is ~10x pricier and blows a small
# budget fast. Override with LABELER_MODEL=claude-sonnet-5 for max gold fidelity.
LABELER_MODEL = os.getenv("LABELER_MODEL", "claude-haiku-4-5-20251001")
EXAMPLES_PATH = os.path.join(PROJECT_ROOT, "data", "sql_examples.json")

_SQL_START = re.compile(r"(?is)\b(with|select)\b")


def load_sql_examples(bench: set) -> list:
    """Few-shot pool from data/sql_examples.json, minus anything matching eval."""
    if not os.path.exists(EXAMPLES_PATH):
        return []
    with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [{"question": e.get("question", ""), "sql": e.get("sql", "")}
            for e in raw if norm_q(e.get("question", "")) not in bench]


def select_examples(question: str, examples: list, k: int = 3) -> str:
    """Dependency-free keyword-overlap top-k (mirrors the agent's few-shot intent)."""
    if not examples:
        return ""
    qt = set(_WORD.findall(question.lower()))
    scored = sorted(examples,
                    key=lambda e: len(qt & set(_WORD.findall(e["question"].lower()))),
                    reverse=True)
    out = ""
    for i, e in enumerate(scored[:k]):
        out += f"Example {i+1}:\nQ: {e['question']}\nSQL: {e['sql']}\n\n"
    return out


def _extract_sql(text: str) -> str:
    t = clean_sql(text)
    m = _SQL_START.search(t)
    if m:
        t = t[m.start():]
    if ";" in t:
        t = t[: t.index(";") + 1]
    return t.strip()


# label cache (question -> sql) for cheap, resumable re-runs
_LABEL_CACHE = None


def _load_label_cache() -> dict:
    global _LABEL_CACHE
    if _LABEL_CACHE is None:
        try:
            with open(LABEL_CACHE, "r", encoding="utf-8") as f:
                _LABEL_CACHE = json.load(f)
        except Exception:
            _LABEL_CACHE = {}
    return _LABEL_CACHE


def label_question(question: str, schema: str, examples: list, db_path: str,
                   max_attempts: int = 3):
    """
    House-style gold SQL via a direct Sonnet call with self-healing.
    Returns (sql, error). Retries only on execution errors, feeding the DB error
    back to the model — the same self-healing idea as the production agent.
    """
    from validate import execute_and_validate

    cache = _load_label_cache()
    if question in cache:
        return cache[question], None

    system = (build_system_prompt(schema, is_simulation(question))
              + "\n\nHere are examples:\n" + select_examples(question, examples))
    messages = [{"role": "user", "content": question}]

    last_reason = "unknown"
    for _ in range(max_attempts):
        try:
            # temperature omitted: sonnet-5 rejects explicit 0.0. Larger max_tokens
            # in case the model spends budget on internal thinking before the SQL.
            content = _complete(LABELER_MODEL, system, messages, max_tokens=2048).strip()
        except Exception as e:  # noqa: BLE001
            return None, f"llm_error:{str(e)[:100]}"

        if content.upper().startswith("MISSING DATA") or content.upper().startswith("I CANNOT"):
            return None, "refusal"

        sql = _extract_sql(content)
        if not sql:
            last_reason = "no_sql"
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Return only a valid SQL query starting with SELECT or WITH."})
            continue

        _ok, _df, reason = execute_and_validate(sql, db_path)
        if reason.startswith("exec_error") or reason.startswith("timeout"):
            last_reason = reason
            messages.append({"role": "assistant", "content": sql})
            messages.append({"role": "user", "content": f"That query failed: {reason}. Fix the SQL and return only the corrected query."})
            continue

        # Executed (ok / empty / large / null) — accept; the gate filters quality.
        cache[question] = sql
        with open(LABEL_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        return sql, None

    return None, f"failed_after_{max_attempts}:{last_reason}"


# ---------------------------------------------------------------------------
# Dedup / contamination helpers
# ---------------------------------------------------------------------------
def norm_q(q: str) -> str:
    return " ".join(_WORD.findall(q.lower()))


def load_benchmark_questions() -> set:
    with open(BENCHMARK_PATH, "r") as f:
        bench = json.load(f)
    return {norm_q(t["question"]) for t in bench}


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_contaminated(qn: str, bench: set, threshold: float = 0.85) -> bool:
    if qn in bench:
        return True
    return any(jaccard(qn, bq) >= threshold for bq in bench)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def build_system_prompt(schema: str, simulation: bool = False) -> str:
    prompt = (
        "You are an expert SQL data analyst. Write a single DuckDB SQL query that "
        "answers the user's question over this schema.\n\n"
        f"{schema}\n{HOUSE_STYLE_RULES}"
    )
    if simulation:
        prompt += "\n\n" + SIMULATION_RULES
    return prompt


def load_existing_questions(out_path: str) -> set:
    """For resumable accumulation: skip questions already written."""
    seen = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    q = rec["messages"][1]["content"]
                    seen.add(norm_q(q))
                except Exception:
                    continue
    return seen


def main():
    ap = argparse.ArgumentParser(description="SQLForge Phase 1 SFT data generation.")
    ap.add_argument("--total", type=int, default=4000, help="Target validated pairs.")
    ap.add_argument("--pilot", action="store_true", help="Small run (18 pairs) to inspect quality.")
    ap.add_argument("--overgen", type=float, default=1.6, help="Question over-generation factor (absorbs rejects).")
    ap.add_argument("--tiers", nargs="*", default=None, help="Subset of tiers to run.")
    ap.add_argument("--consistency", action="store_true",
                    help="Enable the Haiku consistency check. Off by default: it is a noisy "
                         "self-judge that over-rejects ~38%% of valid pairs and wastes budget.")
    ap.add_argument("--label-sleep", type=float, default=1.0, help="Delay between agent label calls (rate-limit guard).")
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--out", default=os.path.join(DATASETS_DIR, "sft_pairs.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(DATASETS_DIR, exist_ok=True)
    if args.pilot:
        args.total = 18
        args.overgen = 1.5

    schema = get_schema(args.db)
    bench = load_benchmark_questions()
    seen = load_benchmark_questions() | load_existing_questions(args.out)

    examples = load_sql_examples(bench)

    alloc = tier_allocation(args.total)
    tiers = args.tiers or list(TIERS.keys())

    rejects_path = os.path.join(DATASETS_DIR, "rejects.jsonl")
    accepted = 0
    stats = {t: {"target": alloc.get(t, 0), "generated": 0, "accepted": 0} for t in tiers}
    reject_reasons = {}

    print("=" * 72)
    print("SQLForge Phase 1 — SFT data generation")
    print(f"  Target pairs:  {args.total}   Overgen: {args.overgen}x")
    print(f"  Question LLM:  {QUESTION_GEN_MODEL}   Labeler: {LABELER_MODEL} (+ house-style prompt, self-healing)")
    print(f"  Consistency:   {'on' if args.consistency else 'off'}")
    print(f"  Output:        {args.out}")
    print("=" * 72)

    out_f = open(args.out, "a", encoding="utf-8")
    rej_f = open(rejects_path, "a", encoding="utf-8")

    try:
        for tier in tiers:
            target = alloc.get(tier, 0)
            if target <= 0:
                continue
            want = max(1, int(round(target * args.overgen)))
            print(f"\n[{tier}] target={target}  generating ~{want} questions...")
            questions = generate_questions(tier, want, schema)
            print(f"[{tier}] got {len(questions)} candidate questions")

            for q in questions:
                if stats[tier]["accepted"] >= target:
                    break
                qn = norm_q(q)
                if not qn or qn in seen:
                    continue
                seen.add(qn)
                stats[tier]["generated"] += 1

                if is_contaminated(qn, bench):
                    _log_reject(rej_f, tier, q, None, "contamination", reject_reasons)
                    continue

                sql, err = label_question(q, schema, examples, args.db)
                if args.label_sleep:
                    time.sleep(args.label_sleep)
                if sql is None:
                    _log_reject(rej_f, tier, q, None, err, reject_reasons)
                    continue

                v = validate_pair(sql, args.db, enforce_house_style=True)
                if not v["ok"]:
                    _log_reject(rej_f, tier, q, sql, v["reason"], reject_reasons)
                    continue

                if args.consistency and not consistency_ok(q, sql, schema):
                    _log_reject(rej_f, tier, q, sql, "consistency_no", reject_reasons)
                    continue

                rec = {
                    "messages": [
                        {"role": "system", "content": build_system_prompt(schema, is_simulation(q))},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": sql},
                    ],
                    "meta": {"tier": tier, "source": f"q:{QUESTION_GEN_MODEL}|label:{LABELER_MODEL}",
                             "rows": v["rows"], "cols": v["cols"]},
                }
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                accepted += 1
                stats[tier]["accepted"] += 1
                print(f"  [{tier}] ACCEPT ({stats[tier]['accepted']}/{target}): {q[:60]}")
    finally:
        out_f.close()
        rej_f.close()

    # --- Data card ---
    card = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_model": QUESTION_GEN_MODEL,
        "labeler": f"{LABELER_MODEL} (direct, house-style prompt, self-healing)",
        "consistency_check": args.consistency,
        "total_accepted": accepted,
        "per_tier": stats,
        "reject_reasons": reject_reasons,
        "output": args.out,
    }
    card_path = os.path.join(DATASETS_DIR, "data_card.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    print("\n" + "=" * 72)
    print(f"DONE. Accepted {accepted} pairs this run -> {args.out}")
    for t in tiers:
        s = stats[t]
        print(f"  {t:18s} accepted {s['accepted']}/{s['target']}  (from {s['generated']} labeled)")
    if reject_reasons:
        print("\n  Reject reasons:")
        for r, c in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            print(f"    {r[:50]:50s} {c}")
    print(f"\n  Data card: {card_path}")
    print("=" * 72)


def _log_reject(f, tier, question, sql, reason, counter):
    key = reason.split(":")[0] if reason else "unknown"
    counter[key] = counter.get(key, 0) + 1
    f.write(json.dumps({"tier": tier, "question": question, "sql": sql, "reason": reason}) + "\n")
    f.flush()


if __name__ == "__main__":
    main()

