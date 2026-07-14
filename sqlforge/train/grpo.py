"""
SQLForge Phase 4 — GRPO with a verifiable execution reward (TRL GRPOTrainer + PEFT).

RL on the small Qwen2.5-Coder-1.5B: the 1.5B has huge headroom (base ~32.7% on the
suite), so unlike DPO on the already-saturated 7B, RL has room to actually MOVE the
number. The reward is not a learned reward model — it is the SAME execution-grounded
grader behind the 74.5% eval: generate SQL, run it read-only on DuckDB, compare the
result set to gold. Correct -> 1.0, valid-but-wrong -> partial, invalid/empty -> 0.0.

  base 1.5B --(reuse SFT pipeline)--> SFT 1.5B --(this script)--> GRPO 1.5B

The warm start (init_adapter) matters: from the raw base the 1.5B only emits valid SQL
~44% of the time, so most reward groups would be all-zero (no advantage signal). Warm-
starting from a quick 1.5B SFT gives dense enough rewards for GRPO to learn. Set
init_adapter: null in the config to run pure-RL-from-base instead.

Generation uses HF generate (use_vllm: false) on purpose — keeping vLLM out of the
training env avoids the torch/vLLM version conflict that broke Phase 0. A 1.5B is small
enough that HF generation is acceptable.

NEEDS ITS OWN VENV (trl>=0.14 for GRPOTrainer; the .venv-train trl==0.12.2 has no GRPO):
    python -m venv .venv-grpo && source .venv-grpo/bin/activate
    pip install --no-cache-dir -r sqlforge/requirements-grpo.txt   # do NOT touch torch

Run (pod, inside tmux, in .venv-grpo):
    python sqlforge/train/grpo.py --config sqlforge/configs/grpo.yaml
    # smoke test first (a handful of prompts, loss/reward should be sane):
    python sqlforge/train/grpo.py --config sqlforge/configs/grpo.yaml --limit 16 --max-steps 5
"""

import os
import sys
import argparse

import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sqlforge", "data"))  # validate, schema_context
sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase0"))            # eval_compare

from eval_compare import extract_sql, execute_sql, compare_results   # noqa: E402
from validate import execute_and_validate                           # noqa: E402
from schema_context import DEFAULT_DB_PATH                           # noqa: E402


# ---------------------------------------------------------------------------
# Verifiable execution reward — identical grounding to the 74.5% eval harness.
# ---------------------------------------------------------------------------
_GOLD_CACHE = {}   # gold SQL string -> (DataFrame | None); gold is fixed, so cache it.


def _gold_df(gold_sql: str, db_path: str):
    if gold_sql not in _GOLD_CACHE:
        df, err = execute_sql(gold_sql, db_path)
        _GOLD_CACHE[gold_sql] = None if err else df
    return _GOLD_CACHE[gold_sql]


def _completion_text(comp) -> str:
    """GRPO gives conversational completions as [{'role','content'}]; standard as str."""
    if isinstance(comp, list):
        return comp[0]["content"] if comp else ""
    return comp or ""


def make_reward(db_path: str, partial: float):
    """Build the reward fn. TRL repeats dataset columns per generation, so `gold`
    aligns 1:1 with `completions`."""

    def reward_exec(completions, gold, **kwargs):
        rewards = []
        for comp, gold_sql in zip(completions, gold):
            sql = extract_sql(_completion_text(comp))
            if not sql:
                rewards.append(0.0)
                continue
            ok, df, _reason = execute_and_validate(sql, db_path)   # timeout + row-cap watchdog
            if not ok:
                rewards.append(0.0)                                # didn't run / empty / cartesian
                continue
            gdf = _gold_df(gold_sql, db_path)
            if gdf is None:
                rewards.append(partial)                            # valid SQL, gold ungradeable
                continue
            rewards.append(1.0 if compare_results(gdf, df)["match"] else partial)
        return rewards

    reward_exec.__name__ = "exec_match"
    return reward_exec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sqlforge/configs/grpo.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Use only the first N prompts (smoke test).")
    ap.add_argument("--max-steps", type=int, default=0, help="Override max_steps (smoke test).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])
    db_path = cfg.get("db_path") or DEFAULT_DB_PATH

    # --- tokenizer ---
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- model (bf16). Optionally warm-start from a merged SFT adapter. ---
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    init_adapter = cfg.get("init_adapter")
    if init_adapter:
        # Fold the SFT adapter into the base weights so GRPO's new LoRA starts from the
        # SFT policy (and the frozen reference = SFT, which is what we want as the KL anchor).
        print(f"Warm-starting from SFT adapter: {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter)
        model = model.merge_and_unload()
        model.config.use_cache = False

    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    )

    # --- dataset: prompt = [system, user] messages (conversational); gold = SQL string ---
    ds = load_dataset("json", data_files=cfg["train_file"])["train"]

    def to_grpo(ex):
        return {"prompt": [ex["messages"][0], ex["messages"][1]],
                "gold": ex["messages"][2]["content"]}

    ds = ds.map(to_grpo, remove_columns=ds.column_names, desc="build GRPO prompts")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    grpo_config = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg.get("temperature", 0.9),
        beta=cfg.get("beta", 0.04),                       # KL coefficient to the frozen ref
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=args.max_steps or cfg.get("max_steps", -1),
        bf16=True,
        optim=cfg.get("optim", "adamw_torch"),
        use_vllm=cfg.get("use_vllm", False),              # HF generate: no vLLM in this env
        logging_steps=cfg.get("logging_steps", 5),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 2),
        seed=cfg.get("seed", 42),
        report_to="wandb",
        run_name=cfg["run_name"],
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=ds,
        reward_funcs=[make_reward(db_path, cfg.get("partial_reward", 0.2))],
        peft_config=lora,
        processing_class=tok,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(cfg["output_dir"])          # raw LoRA (needed for --resume)
    tok.save_pretrained(cfg["output_dir"])
    print(f"\nGRPO LoRA adapter saved to {cfg['output_dir']}")

    # CRITICAL: when warm-starting, the LoRA above was trained on top of the SFT-MERGED base,
    # so it is meaningless against the raw base — evaluating it with `--adapter` silently
    # scores at BASE level. Write a standalone merged model (base+SFT+GRPO) that evaluates
    # correctly with `--model`. (Same reconstruction as sqlforge/train/merge_grpo.py.)
    if init_adapter:
        merged_dir = cfg["output_dir"].rstrip("/") + "-merged"
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"Merged model (base+SFT+GRPO) saved to {merged_dir}")
        print(f"EVALUATE THIS ONE with --model {merged_dir}  (do NOT pass --adapter)")


if __name__ == "__main__":
    main()
