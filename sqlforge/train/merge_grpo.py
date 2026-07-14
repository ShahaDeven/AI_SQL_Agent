"""
SQLForge Phase 4 — reconstruct the true GRPO model (base + SFT + GRPO) as a merged model.

WHY THIS EXISTS:
grpo.py warm-starts by merging the SFT adapter INTO the base weights in memory, then trains
a NEW LoRA on top of that SFT-merged base. So the saved GRPO adapter is only meaningful when
applied to SFT-merged weights — but the eval harness loads the RAW base + adapter, where the
SFT weights don't exist. Evaluating the GRPO adapter against raw base therefore scores at
BASE level (~32.7%), silently discarding everything SFT taught.

This script rebuilds the real stack and writes a single standalone model:

    raw base  ->  + SFT adapter (merge)  ->  + GRPO LoRA (merge)  ->  full merged model

Evaluate the output with --model (NOT --adapter), since the weights are already folded in:

    python phase0/run_baseline.py --model /workspace/sqlforge-grpo-1.5b-merged \
        --out /workspace/eval_1p5b_grpo.json

CPU-friendly (no GPU needed; loads in fp32/bf16 on CPU and writes ~3.1GB for the 1.5B).

Run:
    python sqlforge/train/merge_grpo.py \
        --grpo-adapter /workspace/sqlforge-grpo-1.5b/checkpoint-100 \
        --out /workspace/sqlforge-grpo-1.5b-merged
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    help="Raw base model the SFT adapter was trained on.")
    ap.add_argument("--sft-adapter", default="/workspace/sqlforge-sft-1.5b",
                    help="The SFT LoRA that grpo.py warm-started from (must match!).")
    ap.add_argument("--grpo-adapter", required=True,
                    help="GRPO LoRA dir, e.g. /workspace/sqlforge-grpo-1.5b/checkpoint-100")
    ap.add_argument("--out", required=True, help="Where to write the merged model.")
    args = ap.parse_args()

    print(f"1/3  base            : {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    # Order matters: replicate exactly what grpo.py did in memory — SFT first, then GRPO.
    print(f"2/3  merge SFT       : {args.sft_adapter}")
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()

    print(f"3/3  merge GRPO      : {args.grpo_adapter}")
    model = PeftModel.from_pretrained(model, args.grpo_adapter)
    model = model.merge_and_unload()

    print(f"     saving merged   : {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    # Tokenizer from the base (the adapters don't change the vocab).
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)

    print("\nDone. Evaluate with --model (weights are merged; do NOT pass --adapter):")
    print(f"  python phase0/run_baseline.py --model {args.out} --out /workspace/eval_1p5b_grpo.json")


if __name__ == "__main__":
    main()
