# SFT-2 + GRPO Run 4 — Training Log
**Project:** Major Thesis Project (MTP) — Physics-Aware Small Language Model for Marine Hydrodynamics  
**Model:** Qwen2.5-3B-Instruct (SFT-adapted → GRPO fine-tuned)  
**Method:** Supervised Fine-Tuning (SFT-2) + Group Relative Policy Optimization (GRPO Run 4) via LoRA  
**Date:** April 14–19, 2026  
**Servers:** `mtp-server` (10.71.9.8), `aigpu` (10.71.9.36) — NVIDIA H100 NVL 95 GiB  
**Previous:** See `GRPO_Training_Log_Runs2-3.md` for Runs 2–3 details and the planned Run 4 specification.

---

## 1. Data Pipeline Correction

### 1.1 Problem: Data Leakage in SFT-1

An audit of the data pipeline revealed that the original SFT-1 training data and the GRPO evaluation data were created from independent splits of different-sized pools:

- **SFT-1:** 976 train / 109 test, derived from the initial ~1,100 QA pairs (`qa_dataset.jsonl` early version), split 90/10 with `random.seed(42)`.
- **GRPO:** 2,206 train / 551 eval, derived from the full ~2,757 QA pairs, split 80/20 with stratified sampling by question type.

Because these splits were performed independently, some SFT-1 training examples were present in the GRPO evaluation set. This means the "SFT Baseline" accuracy figures reported in earlier documentation (14.48% numerical, 60.91% MCQ) were evaluated on partially-seen data, making them unreliable as a true baseline.

### 1.2 Solution: SFT-2 on GRPO Training Pool

To establish a leak-free baseline, SFT was re-run (SFT-2) using only data from `grpo_train.jsonl` (2,206 records). This guarantees zero overlap between SFT-2 training data and the GRPO evaluation set (`grpo_eval.jsonl`, 551 records).

---

## 2. SFT-2 Training

### 2.1 Folder Structure

A new self-contained folder `SFT2+GRPO4/` was created with all necessary scripts and data:

```
SFT2+GRPO4/
├── prepare_sft2_data.py    # Converts grpo_train.jsonl → ChatML sft2_train/test
├── train_sft2.py           # SFT-2 training script
├── eval_baseline.py        # Modified for SFT-2 adapter path
├── eval_grpo.py            # Modified for GRPO Run 4 adapter path
├── train_grpo.py           # Modified for Run 4 (max_new_tokens fix, SFT-2 adapter)
├── physics_verifier.py
├── llm_judge.py
├── split_dataset.py
├── qa_dataset.jsonl
└── data/
    ├── grpo_train.jsonl     # 2,206 records (GRPO training pool)
    ├── grpo_eval.jsonl      # 551 records (held-out evaluation)
    ├── sft2_train.jsonl     # 1,985 records (90% of grpo_train)
    └── sft2_test.jsonl      # 221 records (10% of grpo_train)
```

### 2.2 Data Preparation

`prepare_sft2_data.py` processes `data/grpo_train.jsonl` into ChatML format:

- Input: 2,206 records from `grpo_train.jsonl`
- Output: 1,985 training / 221 test records (90/10 split)
- Format: System prompt + user question + assistant response with `<think>` tags and "Final Answer:"
- The `<think>` tag format is preserved for consistency with SFT-1 and modern reasoning-trace conventions

### 2.3 Training Configuration

SFT-2 used identical hyperparameters to SFT-1:

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| LoRA rank | 16 |
| LoRA alpha | 16 (scaling = 1.0) |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Precision | Float32 |
| Optimizer | AdamW (torch) |

### 2.4 Execution

- **Server:** `mtp-server` (10.71.9.8)
- **Environment:** `sft_env` conda environment
- **Output:** `sft2_model_output/`
- Training completed successfully with no NaN gradient issues (Float32 training)

---

## 3. SFT-2 Evaluation

### 3.1 Setup

- **Evaluation set:** `data/grpo_eval.jsonl` (551 records) — completely disjoint from SFT-2 training data
- **Script:** `eval_baseline.py` with `ADAPTER_PATH = "./sft2_model_output"` and `MAX_NEW_TOKENS = 1024` (increased from original 512 to prevent truncation)

### 3.2 Server Migration

The initial evaluation attempt on `mtp-server` hit an OOM error due to other processes consuming GPU memory. All necessary files (scripts, data, SFT-2 adapter) were transferred to `aigpu` (10.71.9.36), where evaluation completed successfully.

### 3.3 Results: Leak-Free SFT-2 Baseline

| Metric | SFT-1 (leaky) | SFT-2 (clean) | Interpretation |
|---|---|---|---|
| Numerical accuracy | 14.48% | **27.60%** | SFT-2 is higher — SFT-1 was deflated, not inflated, by the overlap |
| MCQ accuracy | 60.91% | **71.82%** | +10.91 pp — likely due to 2x more training data (1,985 vs 976) |

The SFT-2 baseline is substantially stronger than SFT-1, attributable to two factors:
1. **More training data:** 1,985 records vs 976 (2x increase)
2. **No data leakage:** Clean evaluation on truly unseen data

### 3.4 Truncation Check

A separate evaluation was run with the original `MAX_NEW_TOKENS = 512` to check whether truncation affected results. The results were identical, confirming that the SFT model produces concise answers well within the 512-token limit. The increase to 1024 was retained as a safety measure.

---

## 4. GRPO Run 4 Training

### 4.1 Critical Changes from Run 3

| Change | Run 3 | Run 4 | Rationale |
|---|---|---|---|
| SFT adapter | `checkpoint-183` (SFT-1) | `sft2_model_output` (SFT-2) | Clean data pipeline, no leakage |
| `max_new_tokens` | 1024 (not applied) | 1024 (with fallback fix) | Ensures token limit takes effect |
| `num_generations` | 8 | 6 | Reduced for GPU memory fit |
| Output directory | `grpo_run3_output` | `grpo_run4_output` | — |

### 4.2 The `max_new_tokens` Fallback Fix

The critical fix in `train_grpo.py` for Run 4:

```python
if "max_new_tokens" not in safe_cfg:
    print(f"[INFO] Applying max_new_tokens={args.max_new_tokens} via model.generation_config (fallback).")
    model.generation_config.max_new_tokens = args.max_new_tokens
else:
    print(f"[INFO] max_new_tokens={args.max_new_tokens} applied via GRPOConfig (native).")

effective_max = getattr(model.generation_config, "max_new_tokens", None)
print(f"[INFO] Effective model.generation_config.max_new_tokens = {effective_max}")
```

This explicitly sets `model.generation_config.max_new_tokens` when `GRPOConfig` drops the parameter, ensuring the 1024-token limit is applied during generation rollouts.

### 4.3 Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 2 |
| Learning rate | 2e-5 |
| Num. generations (K) | 6 |
| Max new tokens | 1024 (with fallback) |
| Temperature | 0.8 |
| Top-p | 0.95 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Precision | Float32 |
| Save steps | 50 |

### 4.4 Execution

- **Server:** `aigpu` (10.71.9.36)
- **GPU:** GPU 0 (NVIDIA H100 NVL 95 GiB), using `CUDA_VISIBLE_DEVICES=0`
- **Environment:** `~/mtp/venv` virtual environment
- **Session:** `tmux` session `grpo4`
- **Log file:** `grpo_run4_training.log`

### 4.5 Training Summary

| Metric | Value |
|---|---|
| Total steps | 8,828 |
| Total runtime | 21h 03m 48s |
| Avg step time | 8.59 s/step |
| Final train loss | 0.002744 |
| Final epoch | 2.0 |
| GPU memory usage | ~14 GB (snapshot; LoRA + batch_size=1 is lightweight) |

### 4.6 Truncation Fix Verification

The primary goal of Run 4 was to verify the `max_new_tokens` fix:

| Metric | Run 3 (broken) | Run 4 (fixed) |
|---|---|---|
| `completions/clipped_ratio` (early) | 0.05–0.15 | **0** |
| `completions/clipped_ratio` (late) | 0.75–0.95 | **0–0.1** |
| `completions/max_length` | ~256 (hard cap) | ~150–230 (voluntary) |

**Conclusion:** The fix worked. Completions were not being truncated in Run 4. However, the model voluntarily generates short answers (~100–200 tokens), well within the 1024 limit. The truncation in Runs 1–3 was therefore only affecting a subset of longer completions, not the majority.

### 4.7 Training Dynamics

Key observations from the training log:

- **Reward:** Mean reward fluctuated between 0.25–0.76, with reward std typically 0.10–0.23
- **Zero-std fraction:** `frac_reward_zero_std` ranged 0–0.75, indicating some steps still had uniform rewards within the group
- **Gradient norm:** Non-zero on most steps (0.05–0.34 typical), confirming active learning
- **Entropy:** Gradually decreased from ~0.65 to ~0.50, indicating policy convergence
- **Completion length:** Stable at ~100–200 tokens throughout — no length drift

---

## 5. GRPO Run 4 Evaluation Results

### 5.1 Headline Metrics

| Metric | SFT-2 Baseline | GRPO Run 4 | Delta |
|---|---|---|---|
| Numerical accuracy | 27.60% | **38.46%** | **+10.86 pp** |
| MCQ accuracy | 71.82% | **85.45%** | **+13.63 pp** |

### 5.2 Numerical Diagnostics

| Metric | Value |
|---|---|
| Mean graded reward | 0.2072 |
| No answer extracted | 3 (1.4%) |
| Near misses (5–10% error) | 8 (3.62%) |

### 5.3 Numerical Accuracy Plateau

Numerical accuracy has converged to ~38–39% across three independent GRPO runs:

| Run | Numerical Acc | SFT Adapter | Truncation Fixed? |
|---|---|---|---|
| Run 1 | 38.01% | SFT-1 | No |
| Run 3 | 39.37% | SFT-1 | No |
| Run 4 | 38.46% | SFT-2 | **Yes** |

The consistency across runs with different SFT bases, different training durations, and with/without the truncation fix strongly suggests that **~38–39% is the ceiling achievable with the current reward function and LoRA configuration**. The bimodal error distribution (from Run 3 graded breakdown: 19.5% exact, 68.8% completely wrong) indicates the bottleneck is formula selection, not arithmetic precision or output truncation.

---

## 6. Cross-Run Comparison (All Stages)

| Stage | Num. Acc. | MCQ Acc. | Eval Set | Key Notes |
|---|---|---|---|---|
| SFT-1 (leaky) | 14.48% | 60.91% | grpo_eval (551) | Data overlap with SFT-1 training; unreliable baseline |
| SFT-2 (clean) | 27.60% | 71.82% | grpo_eval (551) | True leak-free baseline; 2x training data |
| GRPO Run 1 | 38.01% | 75.45% | grpo_eval (551) | From SFT-1; binary reward; 1 epoch |
| GRPO Run 3 | 39.37% | 85.45% | grpo_eval (551) | From SFT-1; graded reward; truncation bug |
| **GRPO Run 4** | **38.46%** | **85.45%** | grpo_eval (551) | From SFT-2; truncation fixed; clean pipeline |

---

## 7. Thesis Report Corrections

The following corrections were applied to `Report/main.tex` to resolve inconsistencies identified in the documentation audit:

1. **Dataset size:** Updated from "over 1,100" to "approximately 2,750" throughout (Abstract, Contributions, Methodology, Limitations). Added explanation of the multi-round generation process.
2. **`<think>` tag format:** Corrected the narrative. The report previously claimed `<think>` tags were "abandoned" — in reality, removing them was tested as a debugging hypothesis but did not fix NaN. The Float32 fix resolved the issue, and `<think>` tags were retained.
3. **GPU specification:** Changed "NVIDIA H100 80 GB SXM5" to "NVIDIA H100 NVL 95 GB" in the infrastructure table and inline references.
4. **SFT-2 section added:** New Section 3.7 ("SFT-2: Corrected Data Pipeline") describing the re-training and its motivation.
5. **Baseline accuracy references:** All SFT baseline figures updated from SFT-1 (14.48%/60.91%) to SFT-2 (27.60%/71.82%) throughout Chapters 4 and 6, with a footnote on the SFT-1 column in the cross-run table.
6. **Evaluation set size:** Corrected from 552 to 551.
7. **GRPO adapter merge:** Updated to reference both `checkpoint-183` (Runs 1–3) and `sft2_model_output` (Run 4+).
8. **Appendix training samples:** Updated to show SFT-1/SFT-2 counts (976/1,985).

---

## 8. Documentation Inconsistencies Audit

A comprehensive audit of all documentation files was performed, resulting in `Documentation_Inconsistencies.md` which catalogues 10 discrepancies:

1. Dataset size (1,100 vs 2,757)
2. `<think>` tag contradiction
3. GPU specification (80 GB vs 95 GiB)
4. SFT baseline data leakage
5. Phantom `qa_dataset_clean.jsonl` reference
6. Conceptual questions claimed in GRPO Run 3 but absent from completions log
7. Confusing file naming (`eval_baseline_results_run4.json` for an SFT baseline)
8. Stale adapter paths in older documentation
9. Minor record count discrepancy (551 vs 552)
10. Undocumented SFT-2 pipeline (now documented in this file)

---

## 9. Analysis and Next Steps

### 9.1 Key Insight: Numerical Accuracy Bottleneck

The numerical accuracy plateau at ~38–39% across three GRPO runs (with different SFT bases, different training configurations, and with/without the truncation fix) indicates that the current approach has reached its ceiling. The bimodal error distribution — the model either applies the correct formula (19.5% exact) or an entirely wrong one (68.8% completely wrong) — confirms that the bottleneck is **formula selection**, not:
- Arithmetic precision (near misses are rare at 3.62%)
- Output truncation (fixed in Run 4, no improvement)
- Training duration (2 epochs sufficient, diminishing returns)
- Data leakage (SFT-2 clean pipeline, same plateau)

### 9.2 Planned: Increase LoRA Capacity (Option C)

The next experiment aims to increase model capacity to improve formula learning:

- Increase LoRA rank from 16 to 32 or 64
- Re-add MLP target modules (gate_proj, up_proj, down_proj) — now safe with Float32 training
- This gives the model more trainable parameters to encode diverse formula patterns

This is the lowest-effort change with meaningful potential impact, as it requires only configuration changes and no new data generation.

### 9.3 Other Potential Directions

- **Option A:** Augment SFT data with more worked numerical examples (high impact, higher effort)
- **Option B:** Curriculum-based GRPO training (medium impact)
- **Option D:** Rejection sampling — mine correct completions from `completions_log.jsonl` for SFT (medium impact, low effort)
- **Option E:** Prompt engineering for explicit formula identification (low-medium impact, low effort)
