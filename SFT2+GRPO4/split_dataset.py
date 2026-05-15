"""
split_dataset.py — Stratified train/eval split of qa_dataset.jsonl

Splits qa_dataset.jsonl into:
  data/grpo_train.jsonl  (80% of each type)
  data/grpo_eval.jsonl   (20% of each type)

The split is STRATIFIED by question type (numerical / mcq / conceptual)
so every type is proportionally represented in both halves.

These two files serve as the single source of truth:
  - grpo_eval.jsonl  → baseline eval (Step 1) AND post-GRPO eval (Step 4)
  - grpo_train.jsonl → GRPO training data (Step 3)

Usage:
    python split_dataset.py                          # default 80/20
    python split_dataset.py --eval-ratio 0.15        # 85/15 split
    python split_dataset.py --seed 123               # different shuffle
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Stratified train/eval split of qa_dataset.jsonl")
    parser.add_argument("--input",      default="qa_dataset.jsonl",      help="Source JSONL file")
    parser.add_argument("--train-out",  default="data/grpo_train.jsonl",  help="Train output path")
    parser.add_argument("--eval-out",   default="data/grpo_eval.jsonl",   help="Eval output path")
    parser.add_argument("--eval-ratio", type=float, default=0.20,         help="Fraction held out for eval (default: 0.20)")
    parser.add_argument("--seed",       type=int,   default=42,           help="Random seed (default: 42)")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src.resolve()}")

    # ── Load ──────────────────────────────────────────────────────────────────
    records = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(records)} records from {src.name}")

    # ── Bucket by type ────────────────────────────────────────────────────────
    buckets: dict[str, list] = defaultdict(list)
    for rec in records:
        buckets[rec.get("type", "unknown")].append(rec)

    print(f"\nType distribution:")
    for t, items in sorted(buckets.items()):
        print(f"  {t:<14} {len(items):5d} records")

    # ── Stratified split ──────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    train_records, eval_records = [], []

    print(f"\nSplitting (eval ratio = {args.eval_ratio:.0%}, seed = {args.seed}):")
    for q_type, items in sorted(buckets.items()):
        rng.shuffle(items)
        n_eval  = max(1, round(len(items) * args.eval_ratio))
        n_train = len(items) - n_eval
        eval_records.extend(items[:n_eval])
        train_records.extend(items[n_eval:])
        print(f"  {q_type:<14}  train={n_train:4d}   eval={n_eval:4d}")

    # Shuffle the final lists so types are interleaved (important for GRPO batching)
    rng.shuffle(train_records)
    rng.shuffle(eval_records)

    # ── Write ─────────────────────────────────────────────────────────────────
    train_path = Path(args.train_out)
    eval_path  = Path(args.eval_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    train_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_records),
        encoding="utf-8"
    )
    eval_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_records),
        encoding="utf-8"
    )

    print(f"\nWrote {len(train_records):4d} records → {train_path}")
    print(f"Wrote {len(eval_records):4d} records → {eval_path}")

    # ── Sanity check: scorable eval items ────────────────────────────────────
    scorable = sum(1 for r in eval_records if r.get("type") in ("numerical", "mcq"))
    conceptual_eval = sum(1 for r in eval_records if r.get("type") == "conceptual")
    print(f"\nEval set breakdown:")
    print(f"  Scorable (numerical + mcq) : {scorable}  ← physics_reward_function targets")
    print(f"  Conceptual (LLM-judge)     : {conceptual_eval}  ← Groq judge targets")
    print(f"  Total eval                 : {len(eval_records)}")


if __name__ == "__main__":
    main()
