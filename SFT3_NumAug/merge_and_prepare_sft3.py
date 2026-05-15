"""
merge_and_prepare_sft3.py — Merge existing + supplement datasets for SFT-3 / GRPO-6
=====================================================================================
Combines:
  1. data/grpo_train.jsonl      — existing 2,206 QA pairs (original training pool)
  2. numerical_qa_supplement.jsonl — new numerical-focused pairs from generate_numerical_qa.py

After merging and deduplication, produces four output files in data/:
  sft3_train.jsonl   — 90% of merged pool (ChatML, for SFT-3)
  sft3_test.jsonl    — 10% of merged pool (ChatML, for SFT-3 eval)
  grpo6_train.jsonl  — 80% of merged pool (flat JSONL, for GRPO-6 training)
  grpo6_eval.jsonl   — 20% of merged pool (flat JSONL, for GRPO-6 evaluation)

Deduplication strategy: two questions are considered duplicates if their
lowercased text shares a leading 80-character prefix, which catches near-identical
questions generated from the same source chunk in different runs.

Usage:
    python merge_and_prepare_sft3.py
    python merge_and_prepare_sft3.py --supplement path/to/other.jsonl
    python merge_and_prepare_sft3.py --no-dedup  # skip deduplication
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data"

DEFAULT_EXISTING   = DATA_DIR / "grpo_train.jsonl"
DEFAULT_SUPPLEMENT = SCRIPT_DIR / "numerical_qa_supplement.jsonl"

SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in Marine Hydrodynamics "
    "and Ocean Engineering. Approach all questions methodically and provide "
    "step-by-step reasoning."
)

MCQ_SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in Marine Hydrodynamics "
    "and Ocean Engineering. For multiple-choice questions, reason through "
    "each option carefully and respond with the letter of the correct answer."
)


# ── Validation ────────────────────────────────────────────────────────────────

LEAKAGE_PHRASES = [
    "according to the passage", "the passage", "the text mentions",
    "the excerpt", "based on the provided context", "mentioned in the text",
    "in figure", "chapter", "as stated above",
]
REFUSAL_PHRASES = [
    "cannot be calculated", "insufficient information", "not possible",
    "cannot determine", "not provided in the", "cannot be solved",
]


def is_valid(rec: dict) -> bool:
    """Returns True if the record passes quality checks."""
    q_type = rec.get("type", "")
    if q_type not in {"conceptual", "numerical", "mcq"}:
        return False

    question = str(rec.get("question", "")).strip()
    cot      = str(rec.get("chain_of_thought", "")).strip()
    answer   = str(rec.get("answer", "")).strip()

    if len(question.split()) < 5:
        return False
    if len(cot.split()) < 10:
        return False
    if not answer:
        return False

    q_lower   = question.lower()
    cot_lower = cot.lower()
    for phrase in LEAKAGE_PHRASES:
        if phrase in q_lower or phrase in cot_lower:
            return False

    if q_type == "numerical":
        ans_lower = answer.lower()
        for phrase in REFUSAL_PHRASES:
            if phrase in ans_lower or phrase in cot_lower:
                return False

    if q_type == "mcq":
        options = rec.get("options", {})
        if not isinstance(options, dict) or len(options) < 2:
            return False

    return True


# ── Deduplication ─────────────────────────────────────────────────────────────

def dedup_key(rec: dict) -> str:
    """80-char prefix of lowercased question as dedup key."""
    q = str(rec.get("question", "")).lower().strip()
    return q[:80]


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique: list[dict] = []
    dupes = 0
    for rec in records:
        key = dedup_key(rec)
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
            unique.append(rec)
    return unique, dupes


# ── ChatML Conversion ─────────────────────────────────────────────────────────

def to_chatml(record: dict) -> dict:
    q_type   = record.get("type", "conceptual")
    question = record.get("question", "")

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

    sys_prompt = MCQ_SYSTEM_PROMPT if q_type == "mcq" else SYSTEM_PROMPT

    return {
        "messages": [
            {"role": "system",    "content": sys_prompt},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": record.get("source", "unknown"),
        "type":   q_type,
    }


# ── Dataset Splitting ─────────────────────────────────────────────────────────

def stratified_split(
    records: list[dict],
    ratio: float,
    seed: int,
    type_key: str = "type",
) -> tuple[list[dict], list[dict]]:
    """Stratified split preserving question type proportions."""
    buckets: dict[str, list] = defaultdict(list)
    for rec in records:
        buckets[rec.get(type_key, "unknown")].append(rec)

    rng = random.Random(seed)
    train_set: list[dict] = []
    test_set:  list[dict] = []

    for q_type, items in sorted(buckets.items()):
        rng.shuffle(items)
        n_test  = max(1, round(len(items) * (1 - ratio)))
        n_train = len(items) - n_test
        test_set.extend(items[:n_test])
        train_set.extend(items[n_test:])
        print(f"  {q_type:<14}  train={n_train:5d}   held-out={n_test:4d}")

    rng.shuffle(train_set)
    rng.shuffle(test_set)
    return train_set, test_set


def save_jsonl(data: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):5d} records → {path.name}")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {i} in {path.name}: {e}")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge existing dataset + numerical supplement for SFT-3 / GRPO-6."
    )
    parser.add_argument("--existing",    type=Path, default=DEFAULT_EXISTING,
                        help="Path to existing flat JSONL (grpo_train.jsonl)")
    parser.add_argument("--supplement",  type=Path, default=DEFAULT_SUPPLEMENT,
                        help="Path to numerical supplement JSONL")
    parser.add_argument("--no-dedup",    action="store_true",
                        help="Skip deduplication step")
    parser.add_argument("--sft-ratio",   type=float, default=0.90,
                        help="Train fraction for SFT split (default 0.90)")
    parser.add_argument("--grpo-ratio",  type=float, default=0.80,
                        help="Train fraction for GRPO split (default 0.80)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print("  SFT-3 / GRPO-6 Dataset Merge & Preparation")
    print("=" * 65)

    # ── Load existing records ──────────────────────────────────────────────────
    if not args.existing.exists():
        print(f"[ERROR] Existing dataset not found: {args.existing}")
        return

    print(f"\n[1] Loading existing dataset: {args.existing.name}")
    existing = load_jsonl(args.existing)
    print(f"    Loaded {len(existing)} records")

    # ── Load supplement ────────────────────────────────────────────────────────
    supplement: list[dict] = []
    if args.supplement.exists():
        print(f"\n[2] Loading supplement: {args.supplement.name}")
        supplement = load_jsonl(args.supplement)
        print(f"    Loaded {len(supplement)} records")
    else:
        print(f"\n[2] Supplement not found: {args.supplement}")
        print("    Proceeding with existing dataset only.")

    # ── Validate supplement ────────────────────────────────────────────────────
    if supplement:
        print(f"\n[3] Validating supplement...")
        valid_supplement = [r for r in supplement if is_valid(r)]
        invalid_count = len(supplement) - len(valid_supplement)
        print(f"    Valid:   {len(valid_supplement)}")
        print(f"    Invalid: {invalid_count} (removed)")
        supplement = valid_supplement

        # Type breakdown
        type_counts: dict[str, int] = defaultdict(int)
        for r in supplement:
            type_counts[r.get("type", "unknown")] += 1
        print("    Type breakdown:")
        for t, n in sorted(type_counts.items()):
            print(f"      {t:<14} {n:5d}")
    else:
        print("\n[3] Skipping validation (no supplement).")

    # ── Merge ──────────────────────────────────────────────────────────────────
    print(f"\n[4] Merging...")
    all_records = existing + supplement
    print(f"    Combined: {len(all_records)} records "
          f"({len(existing)} existing + {len(supplement)} supplement)")

    # ── Deduplicate ────────────────────────────────────────────────────────────
    if not args.no_dedup:
        print(f"\n[5] Deduplicating...")
        all_records, dupes = deduplicate(all_records)
        print(f"    Removed {dupes} duplicates → {len(all_records)} unique records")
    else:
        print("\n[5] Skipping deduplication (--no-dedup).")

    print(f"\n    Final type distribution:")
    final_types: dict[str, int] = defaultdict(int)
    for r in all_records:
        final_types[r.get("type", "unknown")] += 1
    for t, n in sorted(final_types.items()):
        pct = 100 * n / len(all_records)
        print(f"      {t:<14} {n:5d}  ({pct:.1f}%)")

    # ── GRPO Split (flat JSONL, 80/20) ────────────────────────────────────────
    print(f"\n[6] GRPO-6 split ({args.grpo_ratio:.0%} train / "
          f"{1-args.grpo_ratio:.0%} eval, stratified):")
    grpo_train, grpo_eval = stratified_split(
        all_records, args.grpo_ratio, args.seed, type_key="type"
    )
    save_jsonl(grpo_train, DATA_DIR / "grpo6_train.jsonl")
    save_jsonl(grpo_eval,  DATA_DIR / "grpo6_eval.jsonl")

    # ── SFT Split (ChatML, 90/10, using grpo6_train pool only) ───────────────
    # SFT-3 trains only on the GRPO training pool to maintain zero leakage with
    # the GRPO evaluation set (same principle as SFT-2).
    print(f"\n[7] SFT-3 split ({args.sft_ratio:.0%} train / "
          f"{1-args.sft_ratio:.0%} test, from grpo6_train pool only):")
    sft_raw_train, sft_raw_test = stratified_split(
        grpo_train, args.sft_ratio, args.seed + 1, type_key="type"
    )

    # Convert to ChatML
    sft_train = [to_chatml(r) for r in sft_raw_train]
    sft_test  = [to_chatml(r) for r in sft_raw_test]
    save_jsonl(sft_train, DATA_DIR / "sft3_train.jsonl")
    save_jsonl(sft_test,  DATA_DIR / "sft3_test.jsonl")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"  Total merged (unique)  : {len(all_records)}")
    print(f"  grpo6_train.jsonl      : {len(grpo_train)}")
    print(f"  grpo6_eval.jsonl       : {len(grpo_eval)}")
    print(f"  sft3_train.jsonl       : {len(sft_train)} (ChatML)")
    print(f"  sft3_test.jsonl        : {len(sft_test)} (ChatML)")
    print("\nNext steps:")
    print("  1. python train_sft3.py          # train SFT-3")
    print("  2. python eval_baseline.py       # evaluate SFT-3 baseline")
    print("  3. python train_grpo.py          # GRPO-6 from SFT-3 adapter")
    print("  4. python eval_grpo.py           # evaluate GRPO-6")
    print("=" * 65)


if __name__ == "__main__":
    main()
