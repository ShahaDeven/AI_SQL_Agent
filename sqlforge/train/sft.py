"""
SQLForge Phase 2 — SFT with bf16 LoRA (TRL SFTTrainer + PEFT).

Trains Qwen2.5-Coder-7B-Instruct on the house-style SFT pairs. Key choices:
  - bf16 LoRA (no 4-bit): the A40's 48GB fits the bf16 base + adapters, so we skip
    quantization and its quality loss.
  - Completion-only loss: we tokenize with the chat template and mask the prompt
    (system schema + user question) to -100, so loss is computed ONLY on the SQL
    tokens. This is done explicitly (not via a chat-template magic flag) so it is
    robust across TRL versions.
  - Checkpoints every `save_steps` to /workspace so a dead pod loses <30 min;
    resume with --resume.

Run inside tmux, on the pod, in the .venv-train environment:
    wandb login
    python sqlforge/train/sft.py --config sqlforge/configs/sft.yaml
    # sanity gate first (overfit a tiny slice — loss should crater):
    python sqlforge/train/sft.py --config sqlforge/configs/sft.yaml --limit 50 --epochs 15
"""

import os
import argparse

import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sqlforge/configs/sft.yaml")
    ap.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
    ap.add_argument("--limit", type=int, default=0, help="Train on only the first N examples (smoke test).")
    ap.add_argument("--epochs", type=float, default=0, help="Override num_train_epochs (smoke test).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])
    seq = cfg["max_seq_len"]

    # --- tokenizer ---
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- model (bf16, no quantization) ---
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if cfg.get("use_flash_attn") else "sdpa",
    )
    model.config.use_cache = False  # required with gradient checkpointing

    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    )

    # --- tokenize + completion-only masking ---
    def preprocess(ex):
        full = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
        full_ids = tok(full, add_special_tokens=False, truncation=True, max_length=seq)["input_ids"]
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        p = min(len(prompt_ids), len(full_ids))          # mask the prompt span
        labels = [-100] * p + full_ids[p:]
        return {"input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels}

    ds = load_dataset("json", data_files={"train": cfg["train_file"], "val": cfg["val_file"]})
    if args.limit:
        ds["train"] = ds["train"].select(range(min(args.limit, len(ds["train"]))))
    ds = ds.map(preprocess, remove_columns=ds["train"].column_names, desc="tokenize+mask")

    collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True, return_tensors="pt")

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=args.epochs or cfg["num_train_epochs"],
        weight_decay=cfg.get("weight_decay", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        optim=cfg.get("optim", "adamw_torch"),
        bf16=True,
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        report_to="wandb",
        run_name=cfg["run_name"],
        max_seq_length=seq,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},  # dataset is already tokenized+masked
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=collator,
        peft_config=lora,
        processing_class=tok,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"\nDone. LoRA adapter + tokenizer saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
