# Phase 0 — Base-Model Baseline (go/no-go gate)

**One question:** does base `Qwen2.5-Coder` already crush TPC-H (≥ 85%) on your
55-query suite? If the **7B** does, the 7B fine-tune has no headroom — pivot the
headline gains to the **1.5B** model and make the synthetic training data harder
(spec §9). Measure this **before generating any data**.

## What "55-query suite" means here
`eval/benchmark.json` has 60 queries. Phase 0 uses the **55 non-safety** ones
(10 simple + 10 join + 10 aggregation + 10 multi-hop + 10 window + 5 simulation).
The 5 `safety` refusal queries are excluded by default (a base model doesn't
refuse). Add `--include-safety` if you want them.

## How it's measured (kept comparable to your 91.7%)
- **Same prompt** as your production agent (`src/agent_graph.py`): schema DDL +
  the modified-schema/revenue rules + the simulation-mode block for simulation
  queries. Few-shot is **off by default** (opt in with `--fewshot 3`).
- **Single-shot, temp 0.** No self-healing retry loop — that's scaffolding; the
  honest baseline floor is one generation. Matches the Phase 7 eval protocol.
- **Same result-set comparison** as `eval/accuracy_eval.py` (copied verbatim into
  `eval_compare.py`), so the base number is directly comparable to the agent's.
- **DuckDB only.** No Postgres needed in Phase 0 (that's for GRPO reward exec later).

## Files
| File | Purpose |
|---|---|
| `run_baseline.py` | The harness: prompt → generate → execute → compare → report. |
| `eval_compare.py` | Self-contained SQL extraction + DuckDB exec + result comparison (no API keys, no langchain). |
| `setup_pod.sh` | One-shot RunPod setup (installs vllm/duckdb/pandas, builds the DB, self-tests). |
| `results/` | JSON reports, one per model/config. |

---

## Run it

### 0. Locally, before renting the pod (no GPU) — validate the harness
```bash
python phase0/run_baseline.py --self-test
```
Feeds gold SQL through the pipeline; the 54 gold-backed queries should all PASS
(sim_05 has no gold, so it's the only one that can't be self-tested).

### 1. On the pod
Spin up the RTX A5000, SSH in, `cd` to the repo, then **inside tmux**:
```bash
bash phase0/setup_pod.sh
```
This installs deps, generates `data/sql_agent_demo.db` (TPC-H SF=0.1) if missing,
and runs the self-test.

### 2. The decisive runs
```bash
# 7B — the number the whole project hinges on
python phase0/run_baseline.py --model Qwen/Qwen2.5-Coder-7B-Instruct

# 1.5B — the guaranteed-headroom fallback model
python phase0/run_baseline.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct
```
Optional few-shot comparison (helps, and mirrors that your real agent retrieves
examples):
```bash
python phase0/run_baseline.py --model Qwen/Qwen2.5-Coder-7B-Instruct --fewshot 3
```

Each run prints an execution-accuracy %, per-tier breakdown, valid-SQL %,
throughput, and a **VERDICT** line, and writes a full JSON report to `results/`.

---

## Reading the result
- **7B ≥ 85%** → low headroom on 7B. Pivot headline to 1.5B + harder data.
- **7B < 85%** → headroom exists, the SFT→DPO story on 7B holds.
- Either way, the **1.5B** number is your GRPO baseline (bigger gap = better RL demo).

## Useful flags
| Flag | Default | Notes |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | HF id (vllm) or served name (openai). |
| `--backend` | `vllm` | `vllm` (in-process, batched), `openai` (HTTP endpoint), `selftest`. |
| `--fewshot` | `0` | Keyword-selected examples from `data/sql_examples.json` (contamination-guarded). |
| `--temperature` | `0.0` | |
| `--include-safety` | off | Add the 5 refusal queries → 60-query run. |
| `--limit N` | off | First N queries only (quick smoke). |
| `--gpu-mem` | `0.90` | vLLM `gpu_memory_utilization`. Lower if you OOM. |

## Notes
- vLLM serves one model at a time; run the two commands sequentially (each loads
  in ~1 min). Both fit comfortably on 24GB in bf16.
- Everything stays on `/workspace` (set `HF_HOME=/workspace/hf_cache`) so a dead
  pod loses nothing. Run inside `tmux`.
- The `openai` backend lets you point at any OpenAI-compatible endpoint (a vLLM
  server, or later the hosted API baselines in the success-criteria table).
