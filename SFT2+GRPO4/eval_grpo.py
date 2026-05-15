"""
eval_grpo.py — Evaluation script for the GRPO-trained Marine Hydrodynamics SLM.

Mirrors eval_baseline.py but loads a GRPO adapter instead of the SFT adapter.
Kept as a SEPARATE script so baseline (SFT) results are never overwritten and
comparisons remain clean.  Use --adapter_path to point to any GRPO run output.

Scoring is identical to eval_baseline.py:
  - numerical  → physics_reward_function (1.0 if within 5% tolerance)
  - mcq        → physics_reward_function (1.0 if correct letter/phrase)
  - conceptual → LLM-as-judge (Groq) when --judge flag is set, else skipped

Optional --baseline flag loads a previous eval_baseline*.json and appends a
  "delta_vs_baseline" field to every per-item result for easy comparison.

Usage (on server):
    cd ~/mtp && source sft_env/bin/activate

    # Physics-only eval (fast)
    python eval_grpo.py --dataset data/grpo_eval.jsonl

    # With conceptual judge
    python eval_grpo.py --dataset data/grpo_eval.jsonl --judge

    # Compare to baseline results
    python eval_grpo.py --dataset data/grpo_eval.jsonl \\
        --baseline eval_baseline_results.json \\
        --output eval_grpo_results.json
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

from eval_baseline import (
    MCQ_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    parse_record,
)
from physics_verifier import (
    physics_reward_function,
    extract_numerical_answer,
    verify_numerical,
)

# ── Config ────────────────────────────────────────────────────────────────────
GRPO_ADAPTER_PATH = "./grpo_run4_output"
BASE_MODEL        = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE       = "./eval_grpo_run4_results.json"
MAX_NEW_TOKENS    = 1024


# ── Graded numerical diagnostic (mirrors train_grpo._graded_numerical_reward) ─
def _graded_numerical_diagnostic(completion: str, gt_answer: str) -> dict:
    """
    Compute a graded reward + relative error for a numerical answer.
    Returns a dict with 'graded_reward' and 'rel_error' for diagnostic purposes.
    The primary metric remains binary (physics_reward_function); this is secondary.
    """
    pred_val = extract_numerical_answer(completion)
    try:
        gt_val = float(gt_answer)
    except (ValueError, TypeError):
        return {"graded_reward": 0.0, "rel_error": None, "pred_val": None}

    if pred_val is None:
        return {"graded_reward": 0.0, "rel_error": None, "pred_val": None}

    if verify_numerical(pred_val, gt_val, tolerance=0.05):
        rel_err = 0.0 if gt_val == 0 else abs(pred_val - gt_val) / abs(gt_val)
        return {"graded_reward": 1.0, "rel_error": round(rel_err, 6), "pred_val": pred_val}

    if gt_val == 0:
        return {"graded_reward": 0.0, "rel_error": None, "pred_val": pred_val}

    rel_err = abs(pred_val - gt_val) / abs(gt_val)
    if rel_err <= 0.10:
        grade = 0.7
    elif rel_err <= 0.25:
        grade = 0.4
    elif rel_err <= 0.50:
        grade = 0.15
    else:
        grade = 0.0
    return {"graded_reward": grade, "rel_error": round(rel_err, 6), "pred_val": pred_val}


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(adapter_path: str):
    print("=" * 70)
    print("  Marine Hydrodynamics SLM — GRPO Evaluation")
    print(f"  GRPO Adapter : {adapter_path}")
    print(f"  Base Model   : {BASE_MODEL}")
    print("=" * 70)

    # Prefer loading tokenizer from the adapter dir (may have custom pad token).
    tok_source = adapter_path if Path(adapter_path).exists() else BASE_MODEL
    tok = AutoTokenizer.from_pretrained(tok_source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("Tokenizer loaded.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )
    model = model.to(device)
    model = PeftModel.from_pretrained(model, adapter_path, device_map={"": device})
    model.eval()
    print(f"GRPO model loaded on {device}.\n")
    return model, tok


# ── Inference ─────────────────────────────────────────────────────────────────
def infer(model, tok, question: str, system_prompt: str | None = None,
          max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    messages = [
        {"role": "system",  "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user",    "content": question},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids  = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ── Baseline comparison helper ────────────────────────────────────────────────
def _load_baseline_index(baseline_path: Path) -> dict[int, float]:
    """
    Load a baseline results JSON and return a dict of {index → reward}.
    Used to compute per-item delta.
    """
    if not baseline_path.exists():
        print(f"[WARN] Baseline file not found: {baseline_path}. Skipping comparison.")
        return {}
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    per_item = data.get("per_item", [])
    return {
        item["index"]: item.get("reward")
        for item in per_item
        if item.get("reward") is not None
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO-trained Marine Hydrodynamics SLM."
    )
    parser.add_argument(
        "--dataset", default="data/grpo_eval.jsonl",
        help="Eval dataset path (same file used for eval_baseline). Default: data/grpo_eval.jsonl"
    )
    parser.add_argument(
        "--adapter_path", default=GRPO_ADAPTER_PATH,
        help=f"Path to GRPO adapter checkpoint. Default: {GRPO_ADAPTER_PATH}"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap number of records evaluated (for quick tests)."
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Enable LLM-as-judge scoring for conceptual questions via Groq API."
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Path to eval_baseline*.json to compute per-item delta_vs_baseline."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=OUTPUT_FILE,
        help=f"Path for JSON report. Default: {OUTPUT_FILE}"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
        help=f"Max tokens for generation. Default: {MAX_NEW_TOKENS}"
    )
    args = parser.parse_args()
    out_path = Path(args.output)

    judge_fn = None
    if args.judge:
        from llm_judge import judge_conceptual
        judge_fn = judge_conceptual
        print("[judge] LLM-as-judge ENABLED for conceptual questions.")

    # 1. Load dataset
    test_path = Path(args.dataset)
    if not test_path.exists():
        sys.exit(f"ERROR: dataset not found: {test_path.resolve()}")
    records = [json.loads(l) for l in test_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.max_samples:
        records = records[:args.max_samples]
    print(f"Loaded {len(records)} records from {test_path.name}.\n")

    # 2. Load baseline index (optional)
    baseline_index: dict[int, float] = {}
    if args.baseline:
        baseline_index = _load_baseline_index(Path(args.baseline))
        print(f"Loaded baseline rewards for {len(baseline_index)} items from {args.baseline}.\n")

    # 3. Load GRPO model
    model, tok = load_model(args.adapter_path)
    print(f"[INFO] Generation max_new_tokens = {args.max_new_tokens}")

    # 4. Evaluate
    results = []
    counts = {
        "numerical":  {"total": 0, "reward_sum": 0.0, "pass": 0,
                       "graded_sum": 0.0, "no_answer": 0, "near_miss": 0},
        "mcq":        {"total": 0, "reward_sum": 0.0, "pass": 0},
        "conceptual": {"total": 0, "reward_sum": 0.0, "scored": 0},
    }
    fail_log = []

    def review_fields(p: dict) -> dict:
        return {
            "original_question":         p["original_question"],
            "question_prompt":           p["question_prompt"],
            "dataset_answer_raw":        p["dataset_answer_raw"],
            "dataset_chain_of_thought":  p["dataset_chain_of_thought"],
            "reference_assistant":       p["reference_assistant"],
        }

    for i, rec in enumerate(records):
        p       = parse_record(rec)
        question     = p["question_prompt"]
        q_type       = p["q_type"]
        gt_answer    = p["gt_answer"]
        source       = p["source"]
        mcq_gt_letter = p["mcq_gt_letter"]
        mcq_gt_text   = p["mcq_gt_text"]

        print(f"[{i+1:3d}/{len(records)}] type={q_type:<12} source={Path(source).name[:40]}")

        # ── Conceptual ────────────────────────────────────────────────────────
        if q_type == "conceptual":
            counts["conceptual"]["total"] += 1

            if judge_fn is None:
                results.append({
                    "index": i, "type": q_type, "source": source,
                    "reward": None, "skipped": True,
                    "ground_truth_for_judge": p["gt_answer"],
                    **review_fields(p),
                })
                print("             → SKIPPED (use --judge to score conceptual)\n")
                continue

            t0 = time.time()
            completion = infer(model, tok, question,
                              max_new_tokens=args.max_new_tokens)
            elapsed = time.time() - t0
            reference = gt_answer or ""
            reward = judge_fn(question, reference, completion)
            counts["conceptual"]["reward_sum"] += reward
            counts["conceptual"]["scored"] += 1

            delta = (reward - baseline_index[i]) if i in baseline_index else None
            print(f"             → JUDGE  reward={reward:.2f}  delta={delta:+.2f if delta is not None else 'N/A'}  ({elapsed:.1f}s)")
            results.append({
                "index": i, "type": q_type, "source": source,
                "ground_truth_for_judge": reference,
                "completion": completion,
                "reward": reward,
                "passed": reward >= 0.67,
                "delta_vs_baseline": delta,
                "elapsed_s": round(elapsed, 2),
                "skipped": False,
                **review_fields(p),
            })
            continue

        # ── Numerical / MCQ ───────────────────────────────────────────────────
        sys_prompt = MCQ_SYSTEM_PROMPT if q_type == "mcq" else None
        t0 = time.time()
        completion = infer(model, tok, question, system_prompt=sys_prompt,
                          max_new_tokens=args.max_new_tokens)
        elapsed = time.time() - t0

        if gt_answer is None:
            reward = 0.0
            note   = "GT extraction failed"
        else:
            gt_dict: dict = {"type": q_type, "answer": gt_answer}
            if q_type == "mcq" and rec.get("options"):
                gt_dict["options"] = rec["options"]
            reward = physics_reward_function(completion, gt_dict)
            note   = f"gt={gt_answer!r}"

        # Graded diagnostic for numerical questions
        graded_diag: dict | None = None
        if q_type == "numerical" and gt_answer is not None:
            graded_diag = _graded_numerical_diagnostic(completion, str(gt_answer))
            counts["numerical"]["graded_sum"] += graded_diag["graded_reward"]
            if graded_diag["pred_val"] is None:
                counts["numerical"]["no_answer"] += 1
            elif not (reward >= 1.0) and graded_diag["graded_reward"] > 0:
                counts["numerical"]["near_miss"] += 1

        passed = reward >= 1.0
        c = counts[q_type]
        c["total"] += 1
        c["reward_sum"] += reward
        if passed:
            c["pass"] += 1
        else:
            failure = {
                "index": i, "type": q_type, "gt": gt_answer,
                "completion_tail": completion[-300:],
                **review_fields(p),
            }
            if q_type == "mcq":
                failure["gt_letter"] = mcq_gt_letter
                failure["gt_text"]   = mcq_gt_text
            if graded_diag:
                failure["graded_diag"] = graded_diag
            fail_log.append(failure)

        delta  = (reward - baseline_index[i]) if i in baseline_index else None
        status = "PASS" if passed else "FAIL"
        delta_str = f"{delta:+.2f}" if delta is not None else "N/A"
        graded_str = f"  graded={graded_diag['graded_reward']:.2f}" if graded_diag else ""
        print(f"             → {status}  reward={reward:.2f}{graded_str}  delta={delta_str}  gt={gt_answer!r}  ({elapsed:.1f}s)")
        print(f"               completion tail: ...{completion[-120:]!r}\n")

        result_item = {
            "index": i, "type": q_type, "source": source,
            "ground_truth_scoring": gt_answer,
            "mcq_gt_letter": mcq_gt_letter if q_type == "mcq" else None,
            "mcq_gt_text":   mcq_gt_text   if q_type == "mcq" else None,
            "completion": completion,
            "reward": reward,
            "passed": passed,
            "delta_vs_baseline": delta,
            "elapsed_s": round(elapsed, 2),
            "skipped": False,
            **review_fields(p),
        }
        if graded_diag:
            result_item["graded_reward"] = graded_diag["graded_reward"]
            result_item["rel_error"] = graded_diag["rel_error"]
            result_item["pred_val"] = graded_diag["pred_val"]
        results.append(result_item)

    # 5. Aggregate
    print("\n" + "=" * 70)
    print("  GRPO EVALUATION SUMMARY")
    print("=" * 70)
    type_stats = {}

    for q_type in ("numerical", "mcq"):
        c = counts[q_type]
        n = c["total"]
        if n == 0:
            continue
        acc  = c["pass"] / n * 100
        mean = c["reward_sum"] / n

        # Mean delta vs baseline
        deltas = [r["delta_vs_baseline"] for r in results
                  if r["type"] == q_type and r.get("delta_vs_baseline") is not None]
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        delta_str = f"  Δbaseline={mean_delta:+.3f}" if mean_delta is not None else ""

        print(f"  {q_type.upper():<14} n={n:3d}  accuracy={acc:5.1f}%  mean_reward={mean:.3f}{delta_str}")
        type_stats[q_type] = {
            "n": n,
            "accuracy_pct": round(acc, 2),
            "mean_reward": round(mean, 4),
            "mean_delta_vs_baseline": round(mean_delta, 4) if mean_delta is not None else None,
        }

        # Graded diagnostic for numerical questions
        if q_type == "numerical":
            graded_mean = c["graded_sum"] / n if n > 0 else 0.0
            no_ans = c["no_answer"]
            near = c["near_miss"]
            print(f"  {'  (graded)':<14}        mean_graded={graded_mean:.3f}  "
                  f"no_answer={no_ans}  near_miss={near} (failed binary but graded>0)")
            type_stats["numerical_diagnostic"] = {
                "mean_graded_reward": round(graded_mean, 4),
                "no_answer_extracted": no_ans,
                "near_miss_count": near,
                "near_miss_pct": round(near / n * 100, 2) if n > 0 else 0.0,
            }

    cc = counts["conceptual"]
    if cc["scored"] > 0:
        mean_c = cc["reward_sum"] / cc["scored"]
        c_deltas = [r["delta_vs_baseline"] for r in results
                    if r["type"] == "conceptual" and r.get("delta_vs_baseline") is not None]
        mean_c_delta = sum(c_deltas) / len(c_deltas) if c_deltas else None
        delta_str = f"  Δbaseline={mean_c_delta:+.3f}" if mean_c_delta is not None else ""
        print(f"  {'CONCEPTUAL':<14} n={cc['total']:3d}  scored={cc['scored']}  mean_judge_reward={mean_c:.3f}{delta_str}")
        type_stats["conceptual"] = {
            "n": cc["total"], "scored": cc["scored"],
            "mean_judge_reward": round(mean_c, 4),
            "mean_delta_vs_baseline": round(mean_c_delta, 4) if mean_c_delta is not None else None,
        }
    else:
        print(f"  {'CONCEPTUAL':<14} n={cc['total']:3d}  (skipped — run with --judge to score)")
    print("=" * 70)

    # 6. Failure log
    if fail_log:
        print(f"\n  FAILURE LOG ({len(fail_log)} items)\n")
        for fl in fail_log[:10]:
            print(f"  [{fl['index']}] {fl['type']}  gt={fl['gt']!r}")
            qprev = (fl.get("original_question") or fl.get("question_prompt") or "")[:120]
            print(f"    Q: {qprev}")
            print(f"    A-tail: {fl['completion_tail'][-200:]}\n")

    # 7. Save report
    report = {
        "grpo_adapter":      args.adapter_path,
        "base_model":        BASE_MODEL,
        "dataset":           args.dataset,
        "baseline_compared": args.baseline,
        "llm_judge_enabled": args.judge,
        "n_records":         len(records),
        "type_stats":        type_stats,
        "failures":          fail_log,
        "per_item":          results,
        "per_item_fields": {
            "delta_vs_baseline": (
                "reward(GRPO) − reward(baseline) for this item. "
                "Positive = GRPO improved, negative = regressed, null = no baseline data."
            ),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
