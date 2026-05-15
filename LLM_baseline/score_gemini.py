"""
score_gemini.py — Score Gemini's responses against ground truth
===============================================================
After you have:
  1. Fed each chunk to Gemini (gemini.google.com)
  2. Saved each response to chunks/responses/response_XX_of_YY.json

Run:
    python score_gemini.py

Output:
    gemini_results.json  — full per-question results + summary stats
    gemini_summary.txt   — human-readable summary for the thesis
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR   = Path(__file__).parent
CHUNKS_DIR   = SCRIPT_DIR / "chunks"
RESPONSES_DIR = CHUNKS_DIR / "responses"
GT_FILE      = CHUNKS_DIR / "ground_truth.json"
OUTPUT_JSON  = SCRIPT_DIR / "gemini_results.json"
OUTPUT_TXT   = SCRIPT_DIR / "gemini_summary.txt"

NUMERICAL_TOLERANCE = 0.05   # 5% relative error → correct


# ── Answer extraction ──────────────────────────────────────────────────────────

def extract_number(text: str) -> float | None:
    """Extract the first standalone number from a string."""
    if not text:
        return None
    # Remove commas used as thousand-separators
    text = text.replace(",", "")
    # Match numbers with optional sign, decimal, exponent
    matches = re.findall(
        r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
        text
    )
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


def score_numerical(pred_text: str, gt_text: str) -> tuple[bool, float | None, float | None]:
    """Returns (correct, pred_val, rel_error)."""
    pred_val = extract_number(pred_text)
    gt_val   = extract_number(gt_text)
    if pred_val is None or gt_val is None:
        return False, pred_val, None
    if gt_val == 0:
        correct = abs(pred_val) < 1e-9
        return correct, pred_val, float("inf") if not correct else 0.0
    rel_error = abs(pred_val - gt_val) / abs(gt_val)
    return rel_error <= NUMERICAL_TOLERANCE, pred_val, rel_error


def score_mcq(pred_text: str, gt_letter: str) -> bool:
    """Returns True if the predicted letter matches ground truth."""
    if not pred_text or not gt_letter:
        return False
    # Accept single letter, or "A." or "Answer: B" etc.
    pred_clean = pred_text.strip().upper()
    # Take only the first letter if multiple present
    letter_match = re.match(r"^([A-D])", pred_clean)
    if letter_match:
        return letter_match.group(1) == gt_letter.upper()
    return False


# ── Load helpers ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gemini_responses() -> dict[int, dict]:
    """Loads all response files and indexes by question index."""
    if not RESPONSES_DIR.exists():
        print(f"[ERROR] Responses folder not found: {RESPONSES_DIR}")
        sys.exit(1)

    response_files = sorted(RESPONSES_DIR.glob("response_*.json"))
    if not response_files:
        print(f"[ERROR] No response files found in {RESPONSES_DIR}")
        print("  Save Gemini's output as: chunks/responses/response_01_of_N.json")
        sys.exit(1)

    print(f"  Found {len(response_files)} response file(s):")
    all_responses: dict[int, dict] = {}

    for rf in response_files:
        print(f"    {rf.name}")
        try:
            raw = rf.read_text(encoding="utf-8").strip()
            # Strip accidental markdown fences if Gemini wrapped output
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            if not isinstance(data, list):
                print(f"      [WARN] Expected a JSON array, got {type(data).__name__}. Skipping.")
                continue
            for item in data:
                idx = item.get("index")
                if idx is not None:
                    all_responses[int(idx)] = item
        except json.JSONDecodeError as e:
            print(f"      [WARN] JSON parse error in {rf.name}: {e}")
            print("      Fix the file manually and re-run.")

    print(f"  Total responses parsed: {len(all_responses)}")
    return all_responses


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not GT_FILE.exists():
        print(f"[ERROR] Ground truth not found: {GT_FILE}")
        print("  Run prepare_chunks.py first.")
        sys.exit(1)

    print("=" * 55)
    print("  Gemini LLM Baseline Scorer")
    print("=" * 55)

    ground_truth: list[dict] = load_json(GT_FILE)
    print(f"\n  Ground truth: {len(ground_truth)} questions")

    print("\n  Loading Gemini responses...")
    responses = load_gemini_responses()

    # Score each question
    results = []
    stats: dict[str, dict] = {
        "numerical": {"n": 0, "correct": 0, "no_answer": 0, "near_miss": 0},
        "mcq":       {"n": 0, "correct": 0, "no_answer": 0},
    }
    rel_errors: list[float] = []

    for gt in ground_truth:
        idx    = gt["index"]
        q_type = gt["type"]
        resp   = responses.get(idx)

        result = {
            "index":        idx,
            "type":         q_type,
            "source":       gt.get("source", ""),
            "ground_truth": gt["answer_raw"],
            "gemini_answer": None,
            "gemini_reasoning": None,
            "correct": False,
            "rel_error": None,
            "pred_val": None,
            "missing_response": resp is None,
        }

        if resp is None:
            result["missing_response"] = True
            results.append(result)
            stats.get(q_type, {})
            if q_type in stats:
                stats[q_type]["n"] += 1
                stats[q_type]["no_answer"] += 1
            continue

        result["gemini_answer"]    = resp.get("final_answer", "")
        result["gemini_reasoning"] = resp.get("reasoning", "")

        if q_type == "numerical":
            stats["numerical"]["n"] += 1
            pred_text = str(resp.get("final_answer", ""))
            gt_text   = gt["answer_raw"]
            correct, pred_val, rel_error = score_numerical(pred_text, gt_text)
            result["correct"]   = correct
            result["pred_val"]  = pred_val
            result["rel_error"] = round(rel_error, 6) if rel_error is not None else None

            if pred_val is None:
                stats["numerical"]["no_answer"] += 1
            else:
                rel_errors.append(rel_error if rel_error is not None else float("inf"))
                if correct:
                    stats["numerical"]["correct"] += 1
                elif rel_error is not None and rel_error <= 0.20:
                    stats["numerical"]["near_miss"] += 1

        elif q_type == "mcq":
            stats["mcq"]["n"] += 1
            pred_text = str(resp.get("final_answer", ""))
            gt_letter = gt.get("mcq_letter", gt["answer_raw"][:1].upper())
            correct   = score_mcq(pred_text, gt_letter)
            result["correct"]     = correct
            result["gt_letter"]   = gt_letter
            result["pred_letter"] = pred_text.strip().upper()[:1]
            if not pred_text.strip():
                stats["mcq"]["no_answer"] += 1
            elif correct:
                stats["mcq"]["correct"] += 1

        results.append(result)

    # ── Summary stats ──────────────────────────────────────────────────────────
    num_n = stats["numerical"]["n"]
    num_c = stats["numerical"]["correct"]
    mcq_n = stats["mcq"]["n"]
    mcq_c = stats["mcq"]["correct"]

    num_acc = 100 * num_c / num_n if num_n else 0
    mcq_acc = 100 * mcq_c / mcq_n if mcq_n else 0

    # Rel-error distribution for failures
    fail_errors = [r["rel_error"] for r in results
                   if r["type"] == "numerical" and not r["correct"]
                   and r["rel_error"] is not None]
    err_buckets = {"<5%": 0, "5-20%": 0, "20-100%": 0, "100-1000%": 0, ">1000%": 0}
    for re in fail_errors:
        if re < 0.05:   err_buckets["<5%"] += 1
        elif re < 0.20: err_buckets["5-20%"] += 1
        elif re < 1.0:  err_buckets["20-100%"] += 1
        elif re < 10.0: err_buckets["100-1000%"] += 1
        else:           err_buckets[">1000%"] += 1

    missing_total = sum(1 for r in results if r["missing_response"])

    summary = {
        "model": "Gemini (gemini.google.com)",
        "dataset": str(GT_FILE),
        "n_questions_total": len(ground_truth),
        "n_responses_received": len(responses),
        "n_missing_responses": missing_total,
        "numerical": {
            "n": num_n,
            "correct": num_c,
            "accuracy_pct": round(num_acc, 2),
            "no_answer": stats["numerical"]["no_answer"],
            "near_miss_count": stats["numerical"]["near_miss"],
            "near_miss_pct": round(100 * stats["numerical"]["near_miss"] / num_n, 2) if num_n else 0,
        },
        "mcq": {
            "n": mcq_n,
            "correct": mcq_c,
            "accuracy_pct": round(mcq_acc, 2),
            "no_answer": stats["mcq"]["no_answer"],
        },
        "numerical_failure_rel_error_dist": err_buckets,
    }

    output = {"summary": summary, "per_question": results}
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Human-readable summary ─────────────────────────────────────────────────
    lines = [
        "=" * 55,
        "  Gemini LLM Baseline — Evaluation Summary",
        "=" * 55,
        "",
        f"  Dataset            : grpo_eval.jsonl (n={len(ground_truth)})",
        f"  Responses received : {len(responses)} / {len(ground_truth)}",
        f"  Missing responses  : {missing_total}",
        "",
        "  ── Numerical ──────────────────────────────",
        f"  Accuracy           : {num_acc:.2f}%  ({num_c}/{num_n})",
        f"  Near misses (≤20%) : {stats['numerical']['near_miss']} ({100*stats['numerical']['near_miss']/num_n:.1f}%)" if num_n else "",
        f"  No answer extracted: {stats['numerical']['no_answer']}",
        "",
        "  ── MCQ ────────────────────────────────────",
        f"  Accuracy           : {mcq_acc:.2f}%  ({mcq_c}/{mcq_n})",
        f"  No answer          : {stats['mcq']['no_answer']}",
        "",
        "  ── Numerical Failure Breakdown ────────────",
        f"  Rel-error <5%      : {err_buckets['<5%']}   (arithmetic slip)",
        f"  Rel-error 5-20%    : {err_buckets['5-20%']}   (close)",
        f"  Rel-error 20-100%  : {err_buckets['20-100%']}   (wrong formula/scale)",
        f"  Rel-error 100-1000%: {err_buckets['100-1000%']}  (order-of-magnitude)",
        f"  Rel-error >1000%   : {err_buckets['>1000%']}  (completely wrong)",
        "",
        "  ── Comparison with SLM ────────────────────",
        "  (Fill in manually after SLM eval)",
        f"  Gemini  Numerical  : {num_acc:.2f}%",
        f"  Gemini  MCQ        : {mcq_acc:.2f}%",
        "  SLM (GRPO Run 5b)  Numerical : 38.91%",
        "  SLM (GRPO Run 5b)  MCQ       : 80.00%",
        "  SLM (GRPO Run 4)   Numerical : 38.46%",
        "  SLM (GRPO Run 4)   MCQ       : 85.45%",
        "",
        f"  Full results → {OUTPUT_JSON.name}",
        "=" * 55,
    ]
    summary_text = "\n".join(lines)
    OUTPUT_TXT.write_text(summary_text, encoding="utf-8")
    # Use sys.stdout.buffer to avoid Windows cp1252 encoding errors
    sys.stdout.buffer.write(("\n" + summary_text + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
