"""
prepare_sft2_data.py — Convert grpo_train.jsonl (flat) → ChatML for SFT-2.

Uses ONLY data/grpo_train.jsonl as input so that:
  - SFT-2 trains on exactly the same data pool as GRPO Run 4
  - data/grpo_eval.jsonl remains completely untouched (zero leakage)

Output (inside data/):
  sft2_train.jsonl  (90% of grpo_train.jsonl, ChatML)
  sft2_test.jsonl   (10% of grpo_train.jsonl, ChatML)

ChatML format matches the original SFT pipeline (<think> tags, Final Answer:).
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "grpo_train.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "data"

SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in Marine Hydrodynamics "
    "and Ocean Engineering. Approach all questions methodically and provide "
    "step-by-step reasoning."
)


def format_to_chatml(record: dict) -> dict:
    question = record.get("question", "")
    q_type = record.get("type", "conceptual")

    if q_type == "mcq" and "options" in record:
        options = record["options"]
        if isinstance(options, dict):
            opts_text = "\n".join(f"{k}: {v}" for k, v in options.items())
        elif isinstance(options, list):
            opts_text = "\n".join(f"{i+1}: {v}" for i, v in enumerate(options))
        else:
            opts_text = str(options)
        question = f"{question}\n\nOptions:\n{opts_text}"

    cot = record.get("chain_of_thought", "")
    ans = record.get("answer", "")

    if cot:
        assistant_content = f"<think>\n{cot}\n</think>\n\nFinal Answer:\n{ans}"
        if q_type == "mcq" and "explanation" in record:
            assistant_content += f"\n\nExplanation: {record['explanation']}"
    else:
        assistant_content = ans

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": record.get("source", "unknown"),
        "type": q_type,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT-2 dataset from grpo_train.jsonl (ChatML format)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_train = OUTPUT_DIR / "sft2_train.jsonl"
    out_test = OUTPUT_DIR / "sft2_test.jsonl"

    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        return

    print("=" * 60)
    print("  SFT-2 Data Preparation Pipeline")
    print(f"  Input : {args.input}")
    print(f"  Split : {args.train_ratio:.0%} train / {1-args.train_ratio:.0%} test")
    print("=" * 60)

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    print(f"Loaded {len(records)} records.")

    chatml_records = [format_to_chatml(r) for r in records]

    # Stratified split by question type (mirrors split_dataset.py)
    buckets: dict[str, list] = defaultdict(list)
    for rec in chatml_records:
        buckets[rec["type"]].append(rec)

    rng = random.Random(args.seed)
    train_set, test_set = [], []

    print("\nStratified split:")
    for q_type, items in sorted(buckets.items()):
        rng.shuffle(items)
        n_test = max(1, round(len(items) * (1 - args.train_ratio)))
        n_train = len(items) - n_test
        test_set.extend(items[:n_test])
        train_set.extend(items[n_test:])
        print(f"  {q_type:<14}  train={n_train:4d}   test={n_test:4d}")

    rng.shuffle(train_set)
    rng.shuffle(test_set)

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    save_jsonl(train_set, out_train)
    save_jsonl(test_set, out_test)

    print(f"\nTrain Set: {len(train_set)} records -> {out_train}")
    print(f"Test Set : {len(test_set)} records -> {out_test}")
    print("=" * 60)


if __name__ == "__main__":
    main()
