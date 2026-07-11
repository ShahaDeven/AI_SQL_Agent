#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 0 pod setup — RunPod RTX A5000 (24GB), PyTorch 2.x / CUDA 12.x template
# ---------------------------------------------------------------------------
# Minimal footprint: Phase 0 only needs to SERVE a base model and EXECUTE SQL
# on DuckDB. No Postgres, no langchain, no API keys. Postgres/TRL/etc. come in
# later phases.
#
# Run once, inside tmux, from the repo root on the pod:
#     bash phase0/setup_pod.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# Everything on the persistent volume so a dead pod loses nothing.
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

echo ">>> Installing Phase 0 deps (vllm pulls a compatible torch)..."
pip install --upgrade pip
pip install "vllm>=0.6.3" duckdb pandas

echo ">>> Verifying GPU is visible..."
nvidia-smi || { echo "!! nvidia-smi failed — no GPU visible"; exit 1; }

echo ">>> Generating the DuckDB demo DB (TPC-H SF=0.1) if missing..."
if [ ! -f data/sql_agent_demo.db ]; then
  python scripts/demo_db.py
else
  echo "    data/sql_agent_demo.db already present — skipping."
fi

echo ">>> Sanity-checking the eval harness (self-test, no model load)..."
python phase0/run_baseline.py --self-test

echo ""
echo "=========================================================================="
echo " Setup complete. Now run the baselines (inside tmux):"
echo ""
echo "   python phase0/run_baseline.py --model Qwen/Qwen2.5-Coder-7B-Instruct"
echo "   python phase0/run_baseline.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct"
echo ""
echo " Optionally add  --fewshot 3  for a few-shot number too."
echo "=========================================================================="
