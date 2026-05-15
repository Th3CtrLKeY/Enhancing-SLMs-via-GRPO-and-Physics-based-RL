"""
eval_baseline.py — Full test-set baseline evaluation for the Marine Hydrodynamics SLM.

Supports two dataset formats automatically:
  1. Flat format  — qa_dataset.jsonl / grpo_eval.jsonl
  2. ChatML format — sft_test.jsonl

Scoring per question type:
  - numerical  → physics_reward_function: 1.0 if within 5% tolerance, else 0.0
  - mcq        → physics_reward_function: 1.0 if correct letter, else 0.0
  - conceptual → LLM-as-judge (Groq) when --judge flag is set:
                   0.0 / 0.33 / 0.67 / 1.0  (rubric: 0–3 score, normalised)
                 Skipped (not counted) when --judge is not set.

Usage (on server):
    cd ~/mtp && source sft_env/bin/activate

    # Step 1: physics-only baseline (fast, no API calls)
    python eval_baseline.py --dataset data/grpo_eval.jsonl

    # Step 1 + conceptual scoring via LLM judge (slower, uses Groq API)
    python eval_baseline.py --dataset data/grpo_eval.jsonl --judge

    # Quick sanity check on 50 samples
    python eval_baseline.py --dataset data/grpo_eval.jsonl --max-samples 50

Output:
    eval_baseline_results.json (or --output path) — per-item detail includes:
      original_question, question_prompt, dataset_answer_raw,
      dataset_chain_of_thought (flat), reference_assistant (ChatML), completion
"""

import argparse
import json
import re
import sys
import time
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from physics_verifier import physics_reward_function

# ── Config ────────────────────────────────────────────────────────────────────
ADAPTER_PATH   = "./sft_model_output/checkpoint-183"
BASE_MODEL     = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE    = "./eval_baseline_results.json"
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in Marine Hydrodynamics and "
    "Ocean Engineering. Approach all questions methodically and provide "
    "step-by-step reasoning."
)

# For MCQ questions we override the system prompt to force a letter answer.
# This enables clean, deterministic letter-based scoring (Option B).
MCQ_SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in Marine Hydrodynamics and "
    "Ocean Engineering. For multiple choice questions, carefully analyze each "
    "option and explain your reasoning step by step. "
    "You MUST end your response with exactly this line:\n"
    "'Therefore, the answer is: X'\n"
    "where X is the single letter A, B, C, or D."
)


# ── Dataset parsing ───────────────────────────────────────────────────────────
def parse_record(rec: dict) -> dict:
    """
    Normalise a record from either dataset format into a dict used for inference
    and for saving review fields (original Q/A from the file vs prompt sent to the model).

    Keys:
      question_prompt      — full user message sent to the model (may include Options:)
      original_question    — question text only (no appended options block)
      q_type, source
      gt_answer            — normalised ground truth for physics_reward_function
      mcq_gt_letter, mcq_gt_text
      dataset_answer_raw   — flat: rec['answer'] as stored; ChatML: None
      dataset_chain_of_thought — flat: rec['chain_of_thought'] if any
      reference_assistant  — ChatML: full gold assistant message; flat: None
    """
    q_type = rec.get("type", "conceptual")
    source = rec.get("source", "unknown")
    mcq_gt_letter: str | None = None
    mcq_gt_text: str | None = None
    dataset_answer_raw: str | None = None
    dataset_chain_of_thought: str | None = rec.get("chain_of_thought")
    reference_assistant: str | None = None

    if "messages" in rec:
        # ── ChatML format ──────────────────────────────────────────────────
        original_question = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        gt_raw = next((m["content"] for m in rec["messages"] if m["role"] == "assistant"), "")
        reference_assistant = gt_raw
        question_prompt = original_question
        gt_answer = gt_raw if q_type == "conceptual" else _extract_gt_from_assistant_text(gt_raw, q_type)
    else:
        # ── Flat format ────────────────────────────────────────────────────
        original_question = rec.get("question", "")
        question_prompt = original_question
        raw_ans = rec.get("answer", "").strip()
        dataset_answer_raw = raw_ans

        if q_type == "conceptual":
            gt_answer = raw_ans
        elif q_type == "mcq":
            options = rec.get("options") or {}
            gt_answer = _normalise_flat_answer(raw_ans, q_type)
            mcq_gt_letter = gt_answer
            if isinstance(options, dict) and mcq_gt_letter and mcq_gt_letter in options:
                mcq_gt_text = options[mcq_gt_letter]

            if isinstance(options, dict) and options:
                opts_text = "\n".join(
                    f"{k}: {v}" for k, v in sorted(options.items())
                )
                question_prompt = f"{original_question}\n\nOptions:\n{opts_text}"
        else:
            gt_answer = _normalise_flat_answer(raw_ans, q_type)

    if q_type == "mcq" and mcq_gt_letter is None and gt_answer and re.match(
        r"^[A-D]$", str(gt_answer).strip(), re.IGNORECASE
    ):
        mcq_gt_letter = str(gt_answer).strip().upper()

    return {
        "question_prompt": question_prompt,
        "original_question": original_question,
        "q_type": q_type,
        "gt_answer": gt_answer,
        "source": source,
        "mcq_gt_letter": mcq_gt_letter,
        "mcq_gt_text": mcq_gt_text,
        "dataset_answer_raw": dataset_answer_raw,
        "dataset_chain_of_thought": dataset_chain_of_thought,
        "reference_assistant": reference_assistant,
    }


def _extract_gt_from_assistant_text(text: str, q_type: str) -> str | None:
    """
    Extract ground-truth from a ChatML assistant field (contains 'Final Answer:' block).

    - numerical  → extracts the number (or returns raw for symbolic GT)
    - mcq letter → returns the letter
    - mcq phrase → returns the raw phrase (physics_reward_function handles matching)
    - conceptual → returns the full assistant text (for LLM judge)
    """
    if q_type == "conceptual":
        return text  # full text used by llm_judge

    match = re.search(r"Final Answer[:\s]+(.+)", text, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip().splitlines()[0].strip()

    if q_type == "numerical":
        # Try to extract a number; if the GT is a formula, return raw for symbolic fallback
        num = re.search(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?", raw)
        return num.group(0) if num else raw or None

    if q_type == "mcq":
        # Single letter option
        letter = re.match(r"^([A-Da-d])[):\s.]?$", raw)
        if letter:
            return letter.group(1).upper()
        # "correct answer is X" pattern
        expl = re.search(r"correct answer is\s+([A-Da-d])", text, re.IGNORECASE)
        if expl:
            return expl.group(1).upper()
        # Non-letter GT (phrase) — return raw for phrase matching in verifier
        return raw or None

    return None


def _normalise_flat_answer(raw: str, q_type: str) -> str | None:
    """
    Normalise the 'answer' field from flat-format (qa_dataset.jsonl) records.

    - numerical  → extracts the leading number; returns raw if it's a formula
    - mcq letter → returns the letter
    - mcq phrase → returns the raw phrase (73 items in qa_dataset have phrase answers)
    - conceptual → returns the full answer string (for LLM judge)
    """
    raw = raw.strip()
    if not raw:
        return None

    if q_type == "conceptual":
        return raw  # full text for llm_judge

    if q_type == "numerical":
        # Handle "46.48 Newtons", "F_d = 46.48 N", "5 x 10^-8 m"
        # Try A × 10^B first
        sci = re.search(r"([+-]?\d+\.?\d*)\s*[×xX\*]\s*10\^?\s*([+-]?\d+)", raw)
        if sci:
            try:
                return str(float(sci.group(1)) * 10 ** int(sci.group(2)))
            except (ValueError, OverflowError):
                pass
        # Then plain / e-notation number
        num = re.search(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?", raw)
        if num:
            return num.group(0)
        # Formula-style GT (e.g. "Vg = Vp / 2") → return raw for symbolic fallback
        return raw

    if q_type == "mcq":
        # Exact single letter (most common: "A", "B", "C", "D")
        letter = re.match(r"^([A-Da-d])[):\s.]?$", raw)
        if letter:
            return letter.group(1).upper()
        # Embedded "correct answer is X"
        expl = re.search(r"correct answer is\s+([A-Da-d])", raw, re.IGNORECASE)
        if expl:
            return expl.group(1).upper()
        # Phrase answer (e.g. "Reynolds number", "Inviscid flow") — return raw
        return raw

    return None  # unknown type


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(dataset_path: str):
    print("=" * 70)
    print("  Marine Hydrodynamics SLM — Baseline Evaluation")
    print(f"  Adapter  : {ADAPTER_PATH}")
    print(f"  Base     : {BASE_MODEL}")
    print(f"  Dataset  : {dataset_path}")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    print("Tokenizer loaded.")

    # Use explicit device placement instead of device_map="auto" to avoid
    # a torch 2.11.0 + accelerate bug in get_balanced_memory.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,    # 'dtype' replaces deprecated 'torch_dtype'
    )
    model = model.to(device)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, device_map={"": device})
    model.eval()
    print(f"Model loaded on {device}.\n")
    return model, tok


# ── Inference ──────────────────────────────────────────────────────────────────
def infer(model, tok, question: str, system_prompt: str | None = None) -> str:
    """Run the model on a question string and return the completion."""
    messages = [
        {"role": "system",  "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user",    "content": question},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids  = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
        )
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for Marine Hydrodynamics SLM")
    parser.add_argument(
        "--dataset", default="data/grpo_eval.jsonl",
        help="Path to eval dataset (flat or ChatML JSONL). Default: data/grpo_eval.jsonl"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the number of records evaluated (for quick tests). Default: all"
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Enable LLM-as-judge scoring for conceptual questions via Groq API."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=OUTPUT_FILE,
        help=f"Path for JSON report (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()
    out_path = Path(args.output)

    # Import judge lazily so the script still works without Groq keys
    # when --judge is not set.
    judge_fn = None
    if args.judge:
        from llm_judge import judge_conceptual
        judge_fn = judge_conceptual
        print("[judge] LLM-as-judge ENABLED for conceptual questions.")

    # 1. Load dataset
    test_path = Path(args.dataset)
    if not test_path.exists():
        sys.exit(f"ERROR: dataset file not found: {test_path.resolve()}")

    records = [json.loads(l) for l in test_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.max_samples:
        records = records[:args.max_samples]
    print(f"Loaded {len(records)} records from {test_path.name}.\n")

    # 2. Load model
    model, tok = load_model(args.dataset)

    # 3. Evaluate
    results = []
    counts  = {
        "numerical":  {"total": 0, "reward_sum": 0.0, "pass": 0},
        "mcq":        {"total": 0, "reward_sum": 0.0, "pass": 0},
        "conceptual": {"total": 0, "reward_sum": 0.0, "scored": 0},
    }
    fail_log = []

    def review_fields(p: dict) -> dict:
        """Fields saved for offline review (original dataset vs model I/O)."""
        return {
            "original_question": p["original_question"],
            "question_prompt": p["question_prompt"],
            "dataset_answer_raw": p["dataset_answer_raw"],
            "dataset_chain_of_thought": p["dataset_chain_of_thought"],
            "reference_assistant": p["reference_assistant"],
        }

    for i, rec in enumerate(records):
        p = parse_record(rec)
        question = p["question_prompt"]
        q_type = p["q_type"]
        gt_answer = p["gt_answer"]
        source = p["source"]
        mcq_gt_letter = p["mcq_gt_letter"]
        mcq_gt_text = p["mcq_gt_text"]

        print(f"[{i+1:3d}/{len(records)}] type={q_type:<12} source={Path(source).name[:40]}")

        # --- Conceptual: LLM-judge or skip ---
        if q_type == "conceptual":
            counts["conceptual"]["total"] += 1

            if judge_fn is None:
                # --judge not set, skip
                results.append({
                    "index": i,
                    "type": q_type,
                    "source": source,
                    "reward": None,
                    "skipped": True,
                    "ground_truth_for_judge": p["gt_answer"],
                    **review_fields(p),
                })
                print("             → SKIPPED (use --judge to score conceptual)\n")
                continue

            # LLM-judge path
            t0 = time.time()
            completion = infer(model, tok, question)
            elapsed = time.time() - t0

            reference = gt_answer or ""

            reward = judge_fn(question, reference, completion)
            counts["conceptual"]["reward_sum"] += reward
            counts["conceptual"]["scored"] += 1

            print(f"             → JUDGE  reward={reward:.2f}  ({elapsed:.1f}s)")
            print(f"               completion tail: ...{completion[-80:]!r}\n")
            results.append({
                "index": i,
                "type": q_type,
                "source": source,
                "ground_truth_for_judge": reference,
                "completion": completion,
                "reward": reward,
                "passed": reward >= 0.67,
                "elapsed_s": round(elapsed, 2),
                "skipped": False,
                **review_fields(p),
            })
            continue

        # --- Generate completion ---
        # MCQ uses a forced-letter system prompt so we can do clean letter scoring
        sys_prompt = MCQ_SYSTEM_PROMPT if q_type == "mcq" else None
        t0 = time.time()
        completion = infer(model, tok, question, system_prompt=sys_prompt)
        elapsed = time.time() - t0

        # --- Score ---
        if gt_answer is None:
            reward = 0.0
            note = "GT extraction failed"
        else:
            gt_dict: dict = {"type": q_type, "answer": gt_answer}
            if q_type == "mcq" and rec.get("options"):
                gt_dict["options"] = rec["options"]
            reward = physics_reward_function(completion, gt_dict)
            note = f"gt={gt_answer!r}"

        passed = reward >= 1.0
        c = counts[q_type]
        c["total"] += 1
        c["reward_sum"] += reward
        if passed:
            c["pass"] += 1
        else:
            failure = {
                "index": i,
                "type": q_type,
                "gt": gt_answer,
                "completion_tail": completion[-300:],
                **review_fields(p),
            }
            if q_type == "mcq":
                failure["gt_letter"] = mcq_gt_letter
                failure["gt_text"] = mcq_gt_text
            fail_log.append(failure)

        status = "PASS" if passed else "FAIL"
        print(f"             → {status}  reward={reward:.2f}  gt={gt_answer!r}  ({elapsed:.1f}s)")
        print(f"               completion tail: ...{completion[-120:]!r}\n")

        results.append({
            "index": i,
            "type": q_type,
            "source": source,
            "ground_truth_scoring": gt_answer,
            "mcq_gt_letter": mcq_gt_letter if q_type == "mcq" else None,
            "mcq_gt_text": mcq_gt_text if q_type == "mcq" else None,
            "completion": completion,
            "reward": reward,
            "passed": passed,
            "elapsed_s": round(elapsed, 2),
            "skipped": False,
            **review_fields(p),
        })

    # 4. Aggregate
    print("\n" + "=" * 70)
    print("  BASELINE EVALUATION SUMMARY")
    print("=" * 70)
    type_stats = {}

    for q_type in ("numerical", "mcq"):
        c = counts[q_type]
        n = c["total"]
        if n == 0:
            continue
        acc  = c["pass"] / n * 100
        mean = c["reward_sum"] / n
        print(f"  {q_type.upper():<14} n={n:3d}  accuracy={acc:5.1f}%  mean_reward={mean:.3f}")
        type_stats[q_type] = {"n": n, "accuracy_pct": round(acc, 2), "mean_reward": round(mean, 4)}

    cc = counts["conceptual"]
    if cc["scored"] > 0:
        mean_c = cc["reward_sum"] / cc["scored"]
        print(f"  {'CONCEPTUAL':<14} n={cc['total']:3d}  scored={cc['scored']}  mean_judge_reward={mean_c:.3f}  (0=wrong → 1=correct)")
        type_stats["conceptual"] = {"n": cc["total"], "scored": cc["scored"], "mean_judge_reward": round(mean_c, 4)}
    else:
        print(f"  {'CONCEPTUAL':<14} n={cc['total']:3d}  (skipped — run with --judge to score)")
    print("=" * 70)

    # 5. Failure analysis
    if fail_log:
        print(f"\n  FAILURE LOG ({len(fail_log)} items)\n")
        for fl in fail_log[:10]:  # show at most 10
            print(f"  [{fl['index']}] {fl['type']}  gt={fl['gt']!r}")
            qprev = (fl.get("original_question") or fl.get("question_prompt") or "")[:120]
            print(f"    Q: {qprev}")
            print(f"    A-tail: {fl['completion_tail'][-200:]}\n")

    # 6. Save JSON report
    report = {
        "checkpoint":        ADAPTER_PATH,
        "base_model":        BASE_MODEL,
        "dataset":           args.dataset,
        "llm_judge_enabled": args.judge,
        "n_records":         len(records),
        "type_stats":        type_stats,
        "failures":          fail_log,
        "per_item":          results,
        "per_item_fields": {
            "original_question": "Question text from JSONL before appending MCQ Options block (flat).",
            "question_prompt": "Exact user message sent to the model (includes Options for flat MCQ).",
            "dataset_answer_raw": "Flat: rec['answer'] as stored. ChatML: null (see reference_assistant).",
            "dataset_chain_of_thought": "Flat: rec['chain_of_thought'] if present.",
            "reference_assistant": "ChatML: gold assistant message from dataset. Flat: null.",
            "completion": "Model-generated response.",
            "ground_truth_scoring": "Normalised value passed to physics_reward_function (numerical/mcq).",
            "ground_truth_for_judge": "Reference text for conceptual LLM judge.",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
