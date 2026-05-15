# GRPO Training Log — Runs 2 & 3
**Project:** Major Thesis Project (MTP) — Physics-Aware Small Language Model for Marine Hydrodynamics  
**Model:** Qwen2.5-3B-Instruct (SFT-adapted → GRPO fine-tuned)  
**Method:** Group Relative Policy Optimization (GRPO) via LoRA  
**Date:** April 10–13, 2026  
**Server:** `mtp-server` (10.71.9.8) — NVIDIA H100 NVL 95 GiB, Ubuntu 24.04  
**Previous:** See `GRPO_Training_Log_Run1.md` for Run 1 details and initial debugging.

---

## 1. Motivation

Run 1 achieved +23.53 pp numerical accuracy and +14.55 pp MCQ accuracy over the SFT baseline. However, several limitations were identified:

1. **Binary numerical reward** — A correct answer scored 1.0, everything else 0.0, wasting gradient signal on "almost correct" answers.
2. **No completion logging** — Model-generated rollouts during training were not saved, making debugging impossible.
3. **Format reward too dominant** — Weights summing to ~0.4 meant the model could score well by formatting alone.
4. **Single epoch** — Only one pass over the training data.
5. **Truncation** — `max_new_tokens` was not being applied due to TRL API incompatibility (discovered in Run 3).

---

## 2. Run 2 — Changes from Run 1

### 2.1 Graded Numerical Reward

Replaced the binary 0/1 correctness for numerical questions with a tiered partial-credit function:

```
≤5%  relative error → 1.0  (matches physics_reward_function)
≤10% relative error → 0.7
≤25% relative error → 0.4
≤50% relative error → 0.15
>50% or no number   → 0.0
```

Implemented as `_graded_numerical_reward()` in `train_grpo.py`, using `extract_numerical_answer` and `verify_numerical` from `physics_verifier.py`.

MCQ questions remained binary (correct letter = 1.0, wrong = 0.0).

### 2.2 Reduced Format Reward Weights

Halved the format reward weights so it caps at ~0.2 instead of ~0.4, ensuring the graded numerical reward dominates:

| Component | Run 1 | Run 2 |
|---|---|---|
| Reasoning section | 0.10 | 0.05 |
| Final Answer line | 0.15 | 0.08 |
| Reasonable length | 0.10 | 0.05 |
| Structured steps | 0.05 | 0.02 |
| **Max total** | **~0.40** | **~0.20** |

### 2.3 Completion Logger

Added a thread-safe JSONL logger (`completions_log.jsonl`) that records every rollout during training:

```json
{
  "call": 1234,
  "q_type": "numerical",
  "reward": 0.85,
  "correctness": 0.7,
  "format_reward": 0.15,
  "gt_answer": "3.14",
  "completion_tail": "...last 500 chars of completion..."
}
```

Functions: `_init_completion_log()`, `_log_completion()`, `_close_completion_log()`.

### 2.4 Intermediate Checkpoint Saving

Added `--save_steps 50` argument to save checkpoints every 50 optimizer steps (in addition to per-epoch saves). This proved critical when Run 2 was interrupted — checkpoint-2550 out of 2650 steps was recoverable.

### 2.5 Hyperparameter Changes

| Parameter | Run 1 | Run 2 |
|---|---|---|
| `output_dir` | `grpo_model_output` | `grpo_run2_output` |
| `num_train_epochs` | 1 | 2 |
| `learning_rate` | 1e-5 | 2e-5 |
| `max_new_tokens` | 768 | 512 (CLI override) |
| `num_generations` | 8 | 6 (CLI override for VRAM) |
| `save_steps` | epoch-only | 50 |

### 2.6 Run 2 Execution Issues

**Groq API rate limiting:** Initially launched with `--use_judge --include_conceptual`, but the Groq LLM-as-judge hit severe 429 rate limits, causing steps to take hundreds of seconds (estimated 137 hours total). Restarted without `--use_judge`.

**OOM from accidental second launch:** A second `train_grpo.py` invocation was accidentally started on the same GPU, causing `torch.OutOfMemoryError`. The primary process had already reached checkpoint-2550 (96.2% complete) and was evaluated from there.

### 2.7 Run 2 Results (evaluated from checkpoint-2550)

Run 2 was not separately documented in detail as Run 3 was initiated shortly after with further improvements. The intermediate checkpoint showed comparable performance to Run 1.

---

## 3. Run 3 — Changes from Run 2

### 3.1 Increased `max_new_tokens`

Set default `max_new_tokens` from 768 to 1024 in the script to combat truncation. However, this change did **not take effect** during training (see Section 4.1).

### 3.2 Evaluation Enhancements

**Graded diagnostic column:** Added `_graded_numerical_diagnostic()` to `eval_grpo.py`, which computes for every numerical question:
- `graded_reward` (0.0–1.0 partial credit, same tiers as training)
- `rel_error` (exact relative error)
- `pred_val` (extracted numerical prediction)

**Aggregate diagnostics:** Summary now includes:
- `mean_graded_reward` across all numerical questions
- `no_answer_extracted` count (model produced no parseable number)
- `near_miss_count` (failed binary but graded > 0)

**Configurable `--max-new-tokens` CLI argument** for eval, defaulting to 1024.

### 3.3 Run 3 Hyperparameters

| Parameter | Value |
|---|---|
| `output_dir` | `grpo_run3_output` |
| `num_train_epochs` | 2 |
| `learning_rate` | 2e-5 |
| `max_new_tokens` | 1024 (intended, see Section 4.1) |
| `num_generations` | 8 |
| `temperature` | 0.8 |
| `top_p` | 0.95 |
| `save_steps` | 50 |
| `include_conceptual` | Yes (but see Section 3.4) |

### 3.4 Conceptual Questions in Run 3

The `--include_conceptual` flag was used, but the completions log shows 0 conceptual entries (21,200 total: 7,072 MCQ + 14,128 numerical). The training dataset `grpo_train.jsonl` appears to contain no conceptual questions, or they were filtered at parsing.

---

## 4. Critical Discovery: `max_new_tokens` Not Applied

### 4.1 The Bug

The server's TRL version (1.0.0) does not accept `max_new_tokens` as a `GRPOConfig` parameter. The `_safe_config_kwargs` helper correctly detected and logged this:

```
[INFO] GRPOConfig dropped unsupported keys: ['max_new_tokens']
```

However, unlike `temperature` and `top_p` (which had fallback code to apply via `model.generation_config`), there was **no fallback for `max_new_tokens`**. The model defaulted to its built-in generation limit (~256 tokens).

### 4.2 Evidence of Truncation

**Training log metrics (Run 3):**
- Early training: `completions/clipped_ratio: 0.05–0.15`, `completions/max_length: ~240`
- Late training: `completions/clipped_ratio: 0.75–0.95`, `completions/max_length: 256`

The model learned to produce longer, more thorough answers but was hard-capped at 256 tokens. By end of training, 75–95% of completions were truncated.

**Impact on reward signal:** Truncated completions lack the "Final Answer:" line, so `extract_numerical_answer` finds nothing and returns 0. The model received false 0-reward for potentially correct answers that were cut off mid-calculation.

### 4.3 Fix for Run 4

Added a `model.generation_config` fallback for `max_new_tokens`, matching the existing pattern for temperature/top_p:

```python
if "max_new_tokens" not in safe_cfg:
    print(f"[INFO] Applying max_new_tokens={args.max_new_tokens} via model.generation_config (fallback).")
    model.generation_config.max_new_tokens = args.max_new_tokens

# Verification print
effective_max = getattr(model.generation_config, "max_new_tokens", None)
print(f"[INFO] Effective model.generation_config.max_new_tokens = {effective_max}")
```

---

## 5. Run 3 Training Metrics

### 5.1 Training Summary

- **Steps:** 5,300 / 5,300 (100% complete, no crash)
- **Runtime:** 11 hours 54 minutes (~8.08 s/step average)
- **Completions logged:** 21,200 (2,650 prompts × 8 generations)
- **Train loss (final):** -0.005328 (negative is normal for policy gradient)

### 5.2 Completions Log Analysis

| Category | Count | Rate |
|---|---|---|
| MCQ completions | 7,072 | 82.6% correct |
| Numerical completions | 14,128 | 15.6% exact correct |
| Numerical partial credit (5–50% err) | 1,481 | 10.5% |
| Numerical zero correctness | 10,446 | 73.9% |
| Conceptual | 0 | — |

### 5.3 Learning Progression (from completions log)

| Period | Mean Correctness | Exact Correct Rate |
|---|---|---|
| First 25% (all types) | 0.4063 | 38.7% |
| Last 25% (all types) | 0.4038 | 38.6% |
| First 25% MCQ | 80.1% | — |
| **Last 25% MCQ** | **85.3%** | **+5.2 pp** |
| First 25% numerical | 16.4% | — |
| Last 25% numerical | 15.5% | -0.9 pp (flat) |

MCQ improved during training. Numerical stayed flat due to the truncation issue corrupting the reward signal.

---

## 6. Run 3 Evaluation Results

Evaluated from final checkpoint (5300/5300 steps) with `max_new_tokens=1024` in eval.

### 6.1 Cross-Run Comparison

| Metric | SFT Baseline | GRPO Run 1 | GRPO Run 3 | Δ (Run 3 vs SFT) |
|---|---|---|---|---|
| **Numerical accuracy** | 14.48% | 38.01% | **39.37%** | **+24.89 pp** |
| **MCQ accuracy** | 60.91% | 75.45% | **85.45%** | **+24.54 pp** |

### 6.2 Numerical Graded Breakdown (n=221)

| Bucket | Count | % | Meaning |
|---|---|---|---|
| Exact (≤5% error) | 43 | 19.5% | Binary PASS |
| Near (≤10% error) | 0 | 0.0% | — |
| Close (≤25% error) | 5 | 2.3% | Reasonable |
| Far (≤50% error) | 21 | 9.5% | Right ballpark |
| Wrong (>50% or no match) | 152 | 68.8% | Completely off |
| No number extracted | 3 | 1.4% | Model didn't produce a number |

### 6.3 Analysis

- **MCQ is the standout:** 85.45% is the best across all runs, a +10 pp jump from Run 1 and +24.5 pp from SFT baseline.
- **Numerical marginal improvement:** 39.37% vs 38.01% in Run 1 (+1.36 pp). The truncation bug severely limited numerical learning.
- **Bimodal numerical failures:** 68.8% of failures are completely wrong (graded=0.0), with only 7.24% near-misses. The model either gets it right or is way off — no "almost correct" middle ground. Suggests conceptual formula errors, not arithmetic imprecision.
- **Truncation fixed in eval:** Only 3/221 numerical items hit the time limit (vs dozens in Run 2 eval at 512 tokens). Eval is now measuring true capability.

### 6.4 Near-Miss Examples

| Index | Ground Truth | Predicted | Rel Error | Graded |
|---|---|---|---|---|
| 3 | 0.552 | 0.335 | 39.3% | 0.15 |
| 124 | 0.25 | 0.2 | 20.0% | 0.40 |
| 238 | -4 | -3.0 | 25.0% | 0.40 |
| 252 | 0.015 | 0.01 | 33.3% | 0.15 |

---

## 7. Dataset Verification

Confirmed that training and evaluation datasets are disjoint:

```bash
$ wc -l ~/mtp/data/grpo_train.jsonl ~/mtp/data/grpo_eval.jsonl
   2206 grpo_train.jsonl
    551 grpo_eval.jsonl
   2757 total

$ comm -12 <(sort grpo_eval.jsonl) <(sort grpo_train.jsonl) | wc -l
0
```

Zero overlap between training and evaluation datasets.

---

## 8. Files Reference

| File | Description |
|---|---|
| `grpo_run2_output/` | Run 2 adapter + checkpoints (from checkpoint-2550) |
| `grpo_run2_output/completions_log.jsonl` | Run 2 rollout log (14,901 entries) |
| `grpo_run3_output/` | Run 3 final adapter |
| `grpo_run3_output/completions_log.jsonl` | Run 3 rollout log (21,200 entries) |
| `grpo_run3.log` | Run 3 training stdout/stderr |
| `eval_grpo_run3_results.json` | Run 3 evaluation results with graded diagnostics |
| `eval_grpo_results.json` | Run 1 evaluation results |
| `eval_baseline_results_run4.json` | SFT baseline for comparison |

---

## 9. Cumulative Progress Summary

| Stage | Numerical Acc | MCQ Acc | Key Contribution |
|---|---|---|---|
| SFT Baseline | 14.48% | 60.91% | Domain knowledge, format structure |
| GRPO Run 1 | 38.01% (+23.5) | 75.45% (+14.5) | RL reward signal, format reward for gradient |
| GRPO Run 3 | 39.37% (+0.9) | 85.45% (+10.0) | Extended training (2 epochs), refined rewards |

---

## 10. Planned: Run 4

### 10.1 Critical Fix

`max_new_tokens` fallback via `model.generation_config` — ensures the 1024 token limit actually takes effect during training rollouts.

### 10.2 Expected Impact

With truncation eliminated:
- `completions/clipped_ratio` should stay below 0.3 (vs 0.75–0.95 in Run 3)
- Numerical reward signal becomes accurate (no more false 0-rewards)
- The graded numerical reward can finally function as intended
- Numerical accuracy is expected to improve substantially

### 10.3 Hyperparameters

| Parameter | Value | Change |
|---|---|---|
| `output_dir` | `grpo_run4_output` | — |
| `max_new_tokens` | 1024 (with fallback fix) | **Now actually applied** |
| `num_train_epochs` | 2 | Same |
| `learning_rate` | 2e-5 | Same |
| `num_generations` | 8 | Same |

### 10.4 Launch Command

```bash
tmux new-session -d -s grpo4 \
  "cd ~/mtp && source sft_env/bin/activate && \
   python train_grpo.py --include_conceptual 2>&1 | tee grpo_run4.log"
```

### 10.5 Verification Checklist

In the early log output, confirm:
- `[INFO] Applying max_new_tokens=1024 via model.generation_config (fallback).`
- `[INFO] Effective model.generation_config.max_new_tokens = 1024`
- `completions/clipped_ratio` stays below 0.3 in the first few logged steps
