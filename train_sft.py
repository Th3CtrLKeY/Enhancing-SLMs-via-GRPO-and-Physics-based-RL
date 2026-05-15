import json
import argparse
from pathlib import Path
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) for Marine Hydrodynamics")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Path to pretrained model")
    parser.add_argument("--train_file", type=str, default="data/sft_train.jsonl", help="Path to training data")
    parser.add_argument("--test_file", type=str, default="data/sft_test.jsonl", help="Path to test/validation data")
    parser.add_argument("--output_dir", type=str, default="./sft_model_output", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default 5e-5 for stability)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Initializing SFT Pipeline")
    print(f"  Model: {args.model_name_or_path}")
    print("=" * 60)

    # 1. Load Datasets
    print("Loading datasets...")
    dataset = load_dataset("json", data_files={"train": args.train_file, "test": args.test_file})
    
    # 2. Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # No special tokens needed - training data uses plain "Chain of Thought:" text
    # (Avoids BF16 NaN instability from never-seen custom tokens in 100% of samples)

    # 3. Model
    print("Loading base model in float32 (BF16 causes NaN via accumulated rounding errors)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    # No embedding resize needed - vocab size unchanged

    # 4. LoRA Setup
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,    # scaling=1.0 (was 2.0) - reduces gradient magnitude
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention only
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 5. SFTConfig (replaces TrainingArguments in TRL 1.0)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=False,             # Disabled: BF16 accumulated rounding causes NaN after ~25 steps
        fp16=False,             # float32 full precision training
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        optim="adamw_torch",
        max_grad_norm=1.0,
        warmup_steps=10,
        report_to="none",
    )

    # 6. SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Trainer initialized successfully.")
    
    # Start training!
    trainer.train()

    # Save the final adapter + tokenizer
    print(f"Saving model to {args.output_dir} ...")
    try:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Model + tokenizer saved successfully!")
    except Exception as e:
        print(f"[WARN] Final save failed: {e}")
        print("  Epoch checkpoints are still available in the output_dir subdirectories.")
        print(f"  Best checkpoint: {args.output_dir}/checkpoint-61")


if __name__ == "__main__":
    main()
