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

echo ">>> Installing Phase 0 deps..."
# IMPORTANT: pin vLLM. The latest vLLM ships a torch built against a CUDA newer
# than many RunPod host drivers support (e.g. driver caps at CUDA 12.8 -> "12080"),
# which fails with "NVIDIA driver on your system is too old". vllm==0.6.6.post1
# pulls torch 2.5.1 (cu124), which runs on a 12.8 driver and supports the A5000.
pip install --upgrade pip
pip install "vllm==0.6.6.post1" "transformers==4.47.1" duckdb pandas
# transformers is pinned too: RunPod base images ship a bleeding-edge transformers
# whose tokenizer API breaks vllm 0.6.6 ("Qwen2Tokenizer has no attribute
# all_special_tokens_extended"). 4.47.1 matches this vllm release.

echo ">>> Verifying torch sees the GPU..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0))"

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
