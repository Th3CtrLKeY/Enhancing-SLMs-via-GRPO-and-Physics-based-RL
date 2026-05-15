"""
validate_qa.py — Post-generation QA Dataset Validator
=======================================================
Reads qa_dataset.jsonl, prints statistics, performs strict sanity checks 
(including context leakage and refusal detection), and outputs a clean JSONL.

Usage:
    python ~/mtp/validate_qa.py
    python ~/mtp/validate_qa.py --input /path/to/qa_dataset.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "qa_dataset.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "qa_dataset_clean.jsonl"

BANNED_LEAKAGE_PHRASES = [
    "according to the passage", "the passage", "the text", "the excerpt",
    "the provided context", "mentioned in the text", "in figure", "chapter",
    "as stated above"
]

BANNED_REFUSAL_PHRASES = [
    "cannot be calculated", "insufficient information", "not possible",
    "cannot determine", "not provided in the", "cannot be solved"
]

def load_dataset(path: Path) -> tuple[list[dict], int]:
    records = []
    errors  = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {i} — JSON parse error: {e}")
                errors += 1
    return records, errors


def validate_record(rec: dict) -> list[str]:
    """Returns list of issues found. Empty list = valid."""
    issues = []
    
    # 1. Structural Checks
    for field in ("question", "answer", "type", "chain_of_thought"):
        if field not in rec:
            issues.append(f"missing field '{field}'")
        elif not str(rec[field]).strip():
            issues.append(f"empty field '{field}'")

    if "type" in rec and rec["type"] not in {"conceptual", "numerical", "mcq"}:
        issues.append(f"unknown type '{rec['type']}'")

    # 2. Strict MCQ Checks
    if rec.get("type") == "mcq":
        options = rec.get("options", {})
        if not isinstance(options, dict) or len(options) < 2:
            issues.append("MCQ missing valid options dict")
        if "explanation" not in rec or len(str(rec.get("explanation", "")).strip()) < 10:
            issues.append("MCQ missing explanation")

    # 3. Word Count Checks (Fixed logic)
    if "question" in rec and len(str(rec.get("question", "")).split()) < 5:
        issues.append("question too short (<5 words)")
        
    if "chain_of_thought" in rec and len(str(rec.get("chain_of_thought", "")).split()) < 10:
        issues.append("chain_of_thought too short (<10 words)")

    # 4. Context Leakage Detection
    q_text = str(rec.get("question", "")).lower()
    cot_text = str(rec.get("chain_of_thought", "")).lower()
    for phrase in BANNED_LEAKAGE_PHRASES:
        if phrase in q_text or phrase in cot_text:
            issues.append(f"context leakage detected ('{phrase}')")

    # 5. Numerical Refusal Detection
    if rec.get("type") == "numerical":
        ans_text = str(rec.get("answer", "")).lower()
        for phrase in BANNED_REFUSAL_PHRASES:
            if phrase in ans_text or phrase in cot_text:
                issues.append(f"model refusal detected ('{phrase}')")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate QA dataset JSONL file")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] File not found: {args.input}")
        return

    print("=" * 60)
    print(f"  QA Dataset Validator")
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────────
    records, parse_errors = load_dataset(args.input)
    print(f"\n  Total records loaded : {len(records)}")
    print(f"  JSON parse errors    : {parse_errors}")

    # ── Per-record validation ─────────────────────────────────────────────────
    invalid = []
    valid_records = []
    
    for i, rec in enumerate(records):
        issues = validate_record(rec)
        if issues:
            invalid.append((i + 1, issues))
        else:
            valid_records.append(rec)

    valid_count = len(valid_records)
    print(f"  Valid records        : {valid_count}")
    print(f"  Invalid records      : {len(invalid)}")

    if invalid:
        print("\n  Top 10 invalid records:")
        for line_num, issues in invalid[:10]:
            print(f"    Line {line_num:4d}: {'; '.join(issues)}")

   
    # ── Type distribution ─────────────────────────────────────────────────────
    type_counts = Counter(r.get("type", "unknown") for r in valid_records)
    print("\n  ── Clean Dataset Question Types ──")
    for qtype, count in sorted(type_counts.items()):
        pct = 100 * count / valid_count if valid_count else 0
        bar = "█" * int(pct / 2)
        print(f"    {qtype:12s}  {count:5d}  ({pct:5.1f}%)  {bar}")

    # ── Final Output ──────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in valid_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\n  ✅ Clean dataset saved to: {args.output}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if valid_count >= 5000:
        status = "✅ READY — sufficient for GRPO training"
    elif valid_count >= 1000:
        status = "⚠️  PARTIAL — consider generating more"
    else:
        status = "❌ INSUFFICIENT — run generate_qa.py on more chunks"

    print(f"  {status}")
    print("=" * 60)

if __name__ == "__main__":
    main()