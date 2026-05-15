"""
train_sft2.py — SFT Round 2 for Marine Hydrodynamics SLM.

Identical to train_sft.py except:
  - Trains on data/sft2_train.jsonl  (derived from grpo_train.jsonl, ~1985 records)
  - Validates on data/sft2_test.jsonl (derived from grpo_train.jsonl, ~221 records)
  - Saves adapter to ./sft2_model_output

This ensures the SFT base for GRPO Run 4 is trained ONLY on the GRPO training
pool, with zero overlap to grpo_eval.jsonl.
"""

import json
import argparse
from pathlib import Path
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Round 2 for Marine Hydrodynamics")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train_file", type=str, default="data/sft2_train.jsonl")
    parser.add_argument("--test_file", type=str, default="data/sft2_test.jsonl")
    parser.add_argument("--output_dir", type=str, default="./sft2_model_output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Initializing SFT-2 Pipeline")
    print(f"  Model     : {args.model_name_or_path}")
    print(f"  Train data: {args.train_file}")
    print(f"  Test data : {args.test_file}")
    print(f"  Output    : {args.output_dir}")
    print("=" * 60)

    print("Loading datasets...")
    dataset = load_dataset("json", data_files={
        "train": args.train_file,
        "test": args.test_file,
    })
    print(f"  Train: {len(dataset['train'])} records")
    print(f"  Test : {len(dataset['test'])} records")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in float32...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=False,
        fp16=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        optim="adamw_torch",
        max_grad_norm=1.0,
        warmup_steps=10,
        report_to="none",
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Trainer initialized. Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir} ...")
    try:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Model + tokenizer saved successfully!")
    except Exception as e:
        print(f"[WARN] Final save failed: {e}")
        print("  Epoch checkpoints are still available in the output_dir subdirectories.")


if __name__ == "__main__":
    main()
