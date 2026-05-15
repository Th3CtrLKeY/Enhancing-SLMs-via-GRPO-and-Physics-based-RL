# GRPO Training Log — Run 1
**Project:** Major Thesis Project (MTP) — Physics-Aware Small Language Model for Marine Hydrodynamics  
**Model:** Qwen2.5-3B-Instruct (SFT-adapted → GRPO fine-tuned)  
**Method:** Group Relative Policy Optimization (GRPO) via LoRA  
**Date:** April 2026  
**Server:** `mtp-server` (10.71.9.8) — NVIDIA H100 80GB, Ubuntu 24.04  

---

## 1. Objective

After SFT training achieved 77.4% token accuracy but weak numerical correctness (~14.5% on eval), GRPO was introduced as a reinforcement learning stage to improve the model's ability to produce **correct numerical answers** and **accurate MCQ selections** by directly optimising against a physics-verifier reward signal.

---

## 2. Pipeline Overview

### 2.1 Training Script: `train_grpo.py`

**Purpose:** GRPO training using TRL's `GRPOTrainer`. Mirrors SFT setup (same base model, same LoRA target modules) but replaces supervised cross-entropy loss with a policy-gradient objective.

**Data flow:**
1. Load flat JSONL records from `data/grpo_train.jsonl`.
2. Parse via `eval_baseline.parse_record` → construct ChatML prompts.
3. For each prompt, generate K completions (rollouts) via sampling.
4. Score each completion with a reward function → compute group-relative advantages → update policy.

**Key reuse:**
- `eval_baseline.py`: `parse_record`, `SYSTEM_PROMPT`, `MCQ_SYSTEM_PROMPT`
- `physics_verifier.py`: `physics_reward_function` (correctness scoring)
- `llm_judge.py`: Optional conceptual question scoring via Groq API

### 2.2 Evaluation Script: `eval_grpo.py`

A dedicated evaluation script (separate from `eval_baseline.py`) that:
- Loads the GRPO adapter from `./grpo_model_output`
- Runs inference on `data/grpo_eval.jsonl`
- Computes per-item and aggregate metrics
- Optionally computes `delta_vs_baseline` against SFT evaluation results

---

## 3. Starting Point: SFT Adapter Merge

GRPO training does **not** start from the bare pretrained model. Instead:

1. Load `Qwen/Qwen2.5-3B-Instruct` base weights.
2. Load the SFT LoRA adapter from `./sft_model_output/checkpoint-183`.
3. **Merge** the SFT adapter into the base weights via `model.merge_and_unload()`.
4. Apply a **new** LoRA adapter (with identical architecture) for GRPO training.

This ensures GRPO builds on the domain knowledge from SFT rather than starting from scratch.

**CLI argument:** `--adapter_path ./sft_model_output/checkpoint-183` (default).

---

## 4. Reward Function Design

### 4.1 Correctness Reward (Binary)

For numerical and MCQ questions, `physics_reward_function` returns:
- **1.0** if the model's answer matches the ground truth (within 5% tolerance for numerical, exact letter match for MCQ)
- **0.0** otherwise

For conceptual questions (when `--use_judge` is enabled), the Groq LLM judge returns a score on a 0–3 scale, normalised to 0.0–1.0.

### 4.2 Format Reward (Non-Binary, 0.0–0.4)

**Problem:** Binary correctness rewards produce zero variance within GRPO groups when all K completions are either all correct or all wrong — yielding zero advantages and zero gradients.

**Solution:** A multi-component format reward was added to provide continuous, varied signal:

| Component | Weight | Criterion |
|---|---|---|
| Reasoning section | 0.10 | Has `<think>`, "Chain of Thought", or "step-by-step" |
| Final Answer line | 0.15 | Has "Final Answer:" (critical for verifier parsing) |
| Reasonable length | 0.10 | 50–2000 chars (partial credit if >2000) |
| Structured steps | 0.05 | Numbered steps, bullet points, or "Step N" |

**Total reward per completion:** `correctness (0 or 1) + format (0.0 to ~0.4)`

This ensures intra-group variance even when correctness is uniform, which is essential for GRPO to compute non-zero advantages.

---

## 5. GRPO Hyperparameters (Run 1)

| Parameter | Value | Notes |
|---|---|---|
| `per_device_train_batch_size` | 1 | Each step generates K completions per prompt |
| `gradient_accumulation_steps` | 4 | Effective batch = 4 prompts |
| `num_train_epochs` | 1 | Single pass over training data |
| `learning_rate` | 1e-5 | Lower than SFT (5e-5) to avoid destabilising merged policy |
| `optim` | `adamw_torch` | Matches SFT; no fused variant |
| `warmup_steps` | 10 | Same as SFT |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `max_new_tokens` | 768 | Rollout cap per completion |
| `num_generations` | 8 | Group size K per prompt |
| `generation_batch_size` | 8 | Must equal `num_generations` (TRL constraint) |
| `temperature` | 0.8 | Sampling diversity for rollouts |
| `top_p` | 0.95 | Nucleus sampling |
| `bf16` / `fp16` | False / False | Float32 (matching SFT to avoid NaN) |
| `save_strategy` | `epoch` | |

### 5.1 LoRA Configuration (identical to SFT)

```python
LoraConfig(
    r=16,
    lora_alpha=16,        # scaling = 1.0
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
```

---

## 6. Debugging: The Zero-Gradient Problem

### 6.1 Symptom

Initial GRPO runs showed `grad_norm = 0` for a high percentage of training steps. The model was not learning.

### 6.2 Root Cause

GRPO computes advantages by comparing rewards within a group of K completions for the same prompt. If all K completions receive the same reward (e.g., all 0.0 or all 1.0), the advantage for every completion is zero, producing zero gradients.

With binary correctness rewards and greedy/low-diversity sampling, this happened frequently.

### 6.3 Fixes Applied (Iteratively)

**Fix 1: Increased `num_generations` (4 → 8)**
- More completions per group increases the probability that at least some differ in correctness.
- Partial improvement, but not sufficient alone.

**Fix 2: Added `temperature=0.8` and `top_p=0.95`**
- Replaced near-greedy generation with proper sampling.
- Ensures the K completions within each group are meaningfully different.
- Implemented with a fallback mechanism: if `GRPOConfig` doesn't accept these parameters (older TRL versions), they are applied directly to `model.generation_config`.

**Fix 3: Non-binary format reward**
- The `_format_reward` function (Section 4.2) provides continuous reward variation even when all completions have the same correctness score.
- This was the most impactful fix — it guarantees non-zero intra-group variance.

### 6.4 TRL Version Tolerance

The `GRPOConfig` API varies across TRL versions. A `_safe_config_kwargs` helper was implemented to:
1. Inspect the signature of `GRPOConfig.__init__` at runtime.
2. Filter out any unsupported keyword arguments.
3. Log which keys were dropped.
4. Apply generation parameters via `model.generation_config` as a fallback.

---

## 7. Bug Fix: IndexError in `physics_verifier.py`

### 7.1 Problem

During GRPO rollouts, a model completion could contain "Final Answer:" followed by empty whitespace or a newline. The `extract_numerical_answer` and `extract_symbolic_expression` functions called:
```python
fa_block = fa_match.group(1).strip().splitlines()[0].strip()
```
When `splitlines()` returned an empty list, this caused an `IndexError`.

### 7.2 Fix

Added a guard for empty lists:
```python
lines = fa_match.group(1).strip().splitlines()
fa_block = lines[0].strip() if lines else ""
if fa_block:
    # ... existing parsing logic ...
```

Applied to both `extract_numerical_answer` and `extract_symbolic_expression`.

---

## 8. Training Metrics: Negative Loss

**Observation:** GRPO training produced negative loss values.

**Explanation:** This is normal and expected for policy gradient methods. The GRPO loss is a surrogate policy-gradient objective (clipped ratio × advantage), not a cross-entropy loss. When the policy already assigns high probability to high-advantage completions, the objective becomes negative. Negative loss indicates good alignment between the policy and the reward signal — it is not an error.

---

## 9. Evaluation Results (Run 1)

### 9.1 GRPO vs. SFT Baseline Comparison

| Metric | SFT Baseline | GRPO Run 1 | Delta |
|---|---|---|---|
| **Numerical accuracy** | 14.48% (n=221) | 38.01% (n=221) | **+23.53 pp** |
| **MCQ accuracy** | 60.91% (n=110) | 75.45% (n=110) | **+14.55 pp** |
| Numerical mean reward | 0.1448 | 0.3801 | +0.2353 |
| MCQ mean reward | 0.6091 | 0.7545 | +0.1455 |
| Conceptual | Skipped (no judge) | Skipped (no judge) | — |
| Total records evaluated | 552 | 552 | — |

### 9.2 Analysis

- **Numerical accuracy** nearly tripled — the primary objective of GRPO was to improve numerical correctness, and this shows strong progress. The `mean_delta_vs_baseline` of +0.2353 indicates broad improvement across items, not just a few easy questions.
- **MCQ accuracy** improved by ~15 percentage points, indicating the GRPO reward signal (correct letter extraction) was effective.
- **Failure analysis** revealed that numerical failures often involved incorrect formula application despite correct output structure (the format reward successfully encouraged structured reasoning), while MCQ failures were reasoning errors.

### 9.3 Files

| File | Description |
|---|---|
| `eval_grpo_results.json` | Full evaluation report with per-item results and deltas |
| `eval_baseline_results_run4.json` | SFT baseline results used for comparison |
| `grpo_model_output/` | GRPO LoRA adapter weights |
| `grpo_model_output/adapter_config.json` | PEFT configuration confirming LoRA settings |

---

## 10. Server Commands Reference

### 10.1 Run GRPO Training

```bash
cd ~/mtp && source sft_env/bin/activate

nohup python train_grpo.py \
    --train_file data/grpo_train.jsonl \
    --adapter_path ./sft_model_output/checkpoint-183 \
    --output_dir ./grpo_model_output \
    --num_generations 8 \
    --temperature 0.8 \
    --max_new_tokens 768 \
    > grpo_training.log 2>&1 &
```

### 10.2 Run GRPO Evaluation

```bash
cd ~/mtp && source sft_env/bin/activate

python eval_grpo.py \
    --dataset data/grpo_eval.jsonl \
    --baseline eval_baseline_results_run4.json \
    --output eval_grpo_results.json
```

### 10.3 Download Results (PowerShell)

```powershell
scp ai21na3ai42@mtp-server:~/mtp/eval_grpo_results.json "c:\Users\raghu\Desktop\IIT\Sem 10\MTP\eval_grpo_results.json"
```

---

## 11. Known Limitations & Next Steps

1. **Conceptual questions not evaluated** — run with `--judge` and Groq API keys to score.
2. **Single epoch** — training used only 1 epoch; additional epochs may improve convergence.
3. **Reward shaping** — the format reward weights are hand-tuned; they could be optimised.
4. **No completion logging during training** — model-generated completions are not saved during GRPO rollouts, making post-hoc analysis of training dynamics difficult.
5. **Numerical accuracy still at 38%** — significant room for improvement via better reward signals, more training, or curriculum-based approaches.
