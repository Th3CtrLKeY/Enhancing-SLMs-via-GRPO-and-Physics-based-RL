"""
prepare_chunks.py — Split grpo_eval.jsonl into uploadable chunks for Gemini
===========================================================================
Filters to numerical + MCQ questions only, then writes JSON chunk files
that can be uploaded directly to gemini.google.com.

Usage:
    python prepare_chunks.py
    python prepare_chunks.py --dataset ../SFT2+GRPO4/data/grpo_eval.jsonl
    python prepare_chunks.py --chunk-size 50

Output:
    chunks/chunk_01_of_N.json   — upload to Gemini one at a time
    chunks/ground_truth.json    — answers for scoring (kept local, NOT uploaded)
    chunks/manifest.json        — maps chunk files to question indices
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATASET = SCRIPT_DIR.parent / "SFT2+GRPO4" / "data" / "grpo_eval.jsonl"
CHUNKS_DIR = SCRIPT_DIR / "chunks"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {i}: {e}")
    return records


def build_question_entry(idx: int, record: dict) -> dict:
    """Build the question dict to send to Gemini (NO answer included)."""
    q_type = record.get("type", "")
    entry = {
        "index": idx,
        "type": q_type,
        "question": record.get("question", "").strip(),
    }
    if q_type == "mcq" and "options" in record:
        options = record["options"]
        if isinstance(options, dict):
            entry["options"] = options
        elif isinstance(options, list):
            entry["options"] = {chr(65 + i): v for i, v in enumerate(options)}
    return entry


def build_ground_truth_entry(idx: int, record: dict) -> dict:
    """Build the ground-truth entry for local scoring."""
    q_type = record.get("type", "")
    gt = {
        "index": idx,
        "type": q_type,
        "source": record.get("source", ""),
        "answer_raw": record.get("answer", "").strip(),
    }
    if q_type == "mcq":
        answer = record.get("answer", "").strip()
        # Extract just the letter if full text is given
        options = record.get("options", {})
        if isinstance(options, dict):
            gt["options"] = options
            # Try to resolve letter from full answer text
            for letter, text in options.items():
                if answer.upper() == letter.upper() or answer.strip().lower() == str(text).strip().lower():
                    gt["mcq_letter"] = letter.upper()
                    break
            if "mcq_letter" not in gt:
                gt["mcq_letter"] = answer[:1].upper() if answer else ""
        else:
            gt["mcq_letter"] = answer[:1].upper() if answer else ""
    return gt


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Gemini-uploadable question chunks from grpo_eval.jsonl"
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help=f"Path to eval JSONL. Default: {DEFAULT_DATASET}"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=80,
        help="Questions per chunk (default: 80). Gemini handles up to ~200 easily."
    )
    parser.add_argument(
        "--types", nargs="+", default=["numerical", "mcq"],
        help="Question types to include (default: numerical mcq)"
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        # Try alternate paths
        alternates = [
            SCRIPT_DIR.parent / "SFT2+GRPO4" / "data" / "grpo_eval.jsonl",
            SCRIPT_DIR.parent / "data" / "grpo_eval.jsonl",
            SCRIPT_DIR.parent / "GRPO5_LoRA_Expanded" / "data" / "grpo_eval.jsonl",
        ]
        for alt in alternates:
            if alt.exists():
                args.dataset = alt
                print(f"  Using dataset: {alt}")
                break
        else:
            print(f"[ERROR] Dataset not found: {args.dataset}")
            print("  Pass --dataset <path> explicitly.")
            return

    CHUNKS_DIR.mkdir(exist_ok=True)

    print(f"\nLoading: {args.dataset}")
    all_records = load_jsonl(args.dataset)
    print(f"  Total records: {len(all_records)}")

    # Filter to requested types
    filtered = [
        (i, r) for i, r in enumerate(all_records)
        if r.get("type", "") in args.types
    ]
    print(f"  After filter ({', '.join(args.types)}): {len(filtered)} records")

    # Type distribution
    type_counts: dict[str, int] = defaultdict(int)
    for _, r in filtered:
        type_counts[r.get("type", "unknown")] += 1
    for t, n in sorted(type_counts.items()):
        print(f"    {t}: {n}")

    # Build question and ground-truth lists
    questions = []
    ground_truth = []
    for orig_idx, record in filtered:
        q = build_question_entry(orig_idx, record)
        gt = build_ground_truth_entry(orig_idx, record)
        gt["original_record_index"] = orig_idx
        questions.append(q)
        ground_truth.append(gt)

    # Save ground truth (local only — do NOT upload)
    gt_path = CHUNKS_DIR / "ground_truth.json"
    gt_path.write_text(
        json.dumps(ground_truth, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n  Ground truth saved → {gt_path.name}  (DO NOT upload this)")

    # Split into chunks
    total = len(questions)
    n_chunks = (total + args.chunk_size - 1) // args.chunk_size
    chunk_files = []
    manifest = {"chunks": [], "total_questions": total, "dataset": str(args.dataset)}

    for ci in range(n_chunks):
        start = ci * args.chunk_size
        end = min(start + args.chunk_size, total)
        chunk_qs = questions[start:end]
        indices = [q["index"] for q in chunk_qs]

        fname = f"chunk_{ci+1:02d}_of_{n_chunks:02d}.json"
        fpath = CHUNKS_DIR / fname
        fpath.write_text(
            json.dumps(chunk_qs, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        type_dist = defaultdict(int)
        for q in chunk_qs:
            type_dist[q["type"]] += 1

        manifest["chunks"].append({
            "file": fname,
            "question_count": len(chunk_qs),
            "indices": [indices[0], indices[-1]],
            "type_distribution": dict(type_dist),
            "gemini_response_file": f"responses/response_{ci+1:02d}_of_{n_chunks:02d}.json",
        })
        chunk_files.append(fname)
        print(f"  Chunk {ci+1}/{n_chunks}: {len(chunk_qs)} questions  → {fname}")

    manifest_path = CHUNKS_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Create responses folder
    (CHUNKS_DIR / "responses").mkdir(exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  {n_chunks} chunk files ready in: {CHUNKS_DIR}")
    print(f"{'='*55}")
    print("""
Workflow for each chunk:
  1. Open gemini.google.com
  2. Paste the contents of gemini_system_prompt.txt into the prompt box
  3. Upload chunk_XX_of_YY.json
  4. Send — wait for Gemini to output the full JSON array
  5. Copy Gemini's entire response
  6. Save it to: chunks/responses/response_XX_of_YY.json
  7. Repeat for all chunks
  8. Run: python score_gemini.py
""")


if __name__ == "__main__":
    main()
