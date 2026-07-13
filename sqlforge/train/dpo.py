"""
SQLForge Phase 3 — DPO on top of the SFT adapter (bf16 LoRA, TRL DPOTrainer).

Adapter-as-reference trick (spec §5): one base model in memory with two LoRA adapters
loaded from the SFT checkpoint — "policy" (trainable, DPO-updated) and "reference"
(frozen = the SFT model). This keeps the reference correct (= SFT, not raw base) and
memory light, and the saved "policy" adapter is a single LoRA on base Qwen, so it
evaluates with the same harness as SFT (phase0/run_baseline.py --adapter ...).

Run (pod, in .venv-train, inside tmux):
    python sqlforge/train/dpo.py --config sqlforge/configs/dpo.yaml
"""

import os
import argparse

import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOConfig, DPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sqlforge/configs/dpo.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])

    # --- tokenizer ---
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- base + two adapters (policy trainable, reference frozen = SFT) ---
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, cfg["sft_adapter"],
                                      is_trainable=True, adapter_name="policy")
    model.load_adapter(cfg["sft_adapter"], adapter_name="reference")

    # --- dataset: render the prompt messages to a single string; chosen/rejected are SQL ---
    ds = load_dataset("json", data_files=cfg["dpo_file"])["train"]

    def fmt(ex):
        prompt = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "chosen": ex["chosen"], "rejected": ex["rejected"]}

    ds = ds.map(fmt, remove_columns=ds.column_names, desc="format DPO pairs")

    dpo_config = DPOConfig(
        output_dir=cfg["output_dir"],
        beta=cfg["beta"],
        max_prompt_length=cfg["max_prompt_length"],
        max_length=cfg["max_length"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=cfg["num_train_epochs"],
        bf16=True,
        optim=cfg.get("optim", "adamw_torch"),
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        report_to="wandb",
        run_name=cfg["run_name"],
        # adapter-as-reference: reference log-probs from the frozen SFT adapter
        model_adapter_name="policy",
        ref_adapter_name="reference",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,               # reference comes from the "reference" adapter
        args=dpo_config,
        train_dataset=ds,
        processing_class=tok,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # Save ONLY the DPO-updated policy adapter -> single LoRA on base Qwen.
    model.set_adapter("policy")
    model.save_pretrained(cfg["output_dir"], selected_adapters=["policy"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"\nDone. DPO adapter saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
