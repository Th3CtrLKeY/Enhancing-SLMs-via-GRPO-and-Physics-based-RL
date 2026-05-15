# GRPO Runs 5 & 5b, SFT3 NumAug Setup, and Gemini LLM Baseline

**Date range:** April 20–22, 2026  
**Picks up from:** `SFT2_GRPO_Run4_Log.md` (which ended after GRPO Run 4, planning Option C)  
**Server:** `raghuveer@10.71.9.36` (`aigpu`), folder `~/mtp/GRPO5_LoRA_Expanded/`  
**Local folder:** `c:\Users\raghu\Desktop\IIT\Sem 10\MTP\`

---

## 1. GRPO Run 5 — Option C: Expanded LoRA (Broken Run)

### 1.1 Motivation

GRPO Run 4 confirmed a numerical accuracy plateau at ~38–39% despite fixing the truncation bug. Analysis of the failure distribution showed 95% of failures were formula-selection errors (rel_error > 20%), not arithmetic. The hypothesis was that the model lacked sufficient **representational capacity** to learn diverse formula mappings with r=16 attention-only LoRA.

**Option C changes:**
- LoRA rank: r=16 → r=32, lora_alpha=16 → 32
- Target modules: `[q_proj, k_proj, v_proj, o_proj]` → added `[gate_proj, up_proj, down_proj]` (MLP layers)
- Adapter parameters: ~3.7M → ~14.8M (approximately 4× increase)

### 1.2 Setup

Folder: `GRPO5_LoRA_Expanded/`  
Files copied from `SFT2+GRPO4/` with modifications to `train_grpo.py`:
- Default `--output_dir` → `./grpo_run5_output`
- LoRA config updated as above

### 1.3 Launch Issues

**First attempt (GPU 0):** OOM immediately after startup. `nvidia-smi` showed GPU 0 already carrying two `biswa_env` processes using ~22GB each, leaving insufficient free VRAM for the larger LoRA.

**Second attempt (GPU 1):** Launched with `CUDA_VISIBLE_DEVICES=1 --num_generations 4` (reduced from 8 to lower peak memory).

### 1.4 Training Execution

- Ran 3,976 steps over ~15 hours
- **Critical bug re-emerged:** `GRPOConfig dropped unsupported keys: ['max_new_tokens']` — the existing fallback set `model.generation_config.max_new_tokens = 1024`, but `GRPOTrainer` creates its own internal `GenerationConfig` at construction time and never reads the model's config. The trainer defaulted to `max_new_tokens=256`.
- `completions/clipped_ratio`: **0.88–0.97 throughout** (same as Run 3 — widespread truncation)
- `completions/max_length`: 256 consistently (confirming the bug)

### 1.5 Evaluation Results

| Metric | Value |
|---|---|
| Numerical accuracy | 37.56% (83/221) |
| MCQ accuracy | 81.82% (90/110) |
| Mean graded reward | 0.2552 |
| Near misses (≤20%) | 14 (6.33%) |
| No answer extracted | 3 |

**Conclusion:** Slight regression from Run 4 (38.46% / 85.45%). The expanded LoRA did not help because the training signal was corrupted by truncation — the model never received correct reward signals for longer numerical derivations.

---

## 2. GRPO Run 5b — Option C with Proper max_new_tokens Fix

### 2.1 Root Cause Analysis

The previous fallback:
```python
model.generation_config.max_new_tokens = args.max_new_tokens
```
was insufficient because `GRPOTrainer.__init__()` clones the generation config internally at construction time. The trainer's copy must be patched **after** `GRPOTrainer(...)` is called.

### 2.2 Fixes Applied (to `GRPO5_LoRA_Expanded/train_grpo.py`)

**Fix 1 — Trainer-level generation config patch:**
```python
for attr in ("generation_config", "_generation_config"):
    cfg_obj = getattr(trainer, attr, None)
    if cfg_obj is not None:
        cfg_obj.max_new_tokens = args.max_new_tokens
        cfg_obj.temperature = args.temperature
        cfg_obj.do_sample = True
        cfg_obj.top_p = args.top_p
```

**Fix 2 — `SaveMetricsCallback`:**
A new `TrainerCallback` subclass that accumulates per-step metrics and saves them to `<output_dir>/training_metrics.json` at every checkpoint and at training end. This enables post-hoc inspection of training dynamics without re-reading log files.

**Output directory:** `grpo_run5b_output/` (Run 5 results preserved in `grpo_run5_output/`)

### 2.3 Verification

A trial run confirmed the patch was applied:
```
[INFO] Patched trainer.generation_config: max_new_tokens=768, temperature=0.8, top_p=0.95
[INFO] trainer.generation_config.max_new_tokens = 768
```

### 2.4 Training Configuration

| Parameter | Value |
|---|---|
| Base adapter | `sft2_model_output` (same as Run 4) |
| LoRA rank (r) | 32 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down proj (7 modules) |
| max_new_tokens | 768 |
| num_generations | 4 |
| num_train_epochs | 2 |
| learning_rate | 2e-5 |
| GPU | GPU 1, CUDA_VISIBLE_DEVICES=1 |
| Runtime | ~20 hours |

### 2.5 Training Dynamics (Confirmed Healthy)

| Metric | Run 5 (broken) | Run 5b (fixed) |
|---|---|---|
| clipped_ratio | 0.88–0.97 | **0.0–0.075** |
| max_length | 256 (hard cap) | 210–419 (free) |
| mean_terminated_length | 30–140 | 135–307 |
| reward_std | 0.05–0.12 | 0.12–0.23 |

### 2.6 Evaluation Results

| Metric | Value |
|---|---|
| Numerical accuracy | **38.91%** (86/221) |
| MCQ accuracy | **80.00%** (88/110) |
| Mean graded reward | 0.2097 |
| Near misses (≤20%) | 16 (7.24%) |
| No answer extracted | 3 |
| Short completion tails | **0** (truncation fully resolved) |

### 2.7 Failure Breakdown (Numerical)

| Relative Error Range | Count | Interpretation |
|---|---|---|
| < 5% | 0 | No borderline arithmetic errors |
| 5–20% | 5 | Rounding/approximation issues |
| 20–100% | 52 | Wrong formula, missing coefficient |
| 100–1000% | 44 | Wrong equation class entirely |
| > 1000% | 30 | Completely wrong formula family |

**95% of failures are formula-selection errors.** The model picks the wrong equation from the start and then correctly executes it. This is not fixable by GRPO alone — it requires better training data.

### 2.8 Conclusion

Expanding LoRA capacity (Option C) confirmed as **ineffective** for breaking the numerical accuracy ceiling. With the truncation fix properly applied, Run 5b achieved 38.91% — nearly identical to Run 4's 38.46%. The ceiling is confirmed at ~39% with the current dataset.

---

## 3. Cross-Run Comparison Table

| Run | Base | LoRA r | max_new_tokens | Numerical % | MCQ % |
|---|---|---|---|---|---|
| SFT-2 baseline | — | — | 1024 (eval) | 27.60% | 71.82% |
| GRPO Run 1 | SFT-1 (checkpoint-183) | 16 (attn) | ❌ 256 | ~26% | ~65% |
| GRPO Run 2 | SFT-1 | 16 (attn) | ❌ 256 | ~26% | ~67% |
| GRPO Run 3 | SFT-1 | 16 (attn) | ❌ 256 | ~27% | ~68% |
| GRPO Run 4 | SFT-2 | 16 (attn) | ✅ 1024 | 38.46% | 85.45% |
| GRPO Run 5 | SFT-2 | 32 (attn+MLP) | ❌ 256 | 37.56% | 81.82% |
| **GRPO Run 5b** | SFT-2 | 32 (attn+MLP) | ✅ 768 | **38.91%** | **80.00%** |

---

## 4. SFT3 NumAug — Option A Setup (In Progress)

### 4.1 Motivation

With Option C (capacity expansion) confirmed ineffective, the last viable path before submission is **Option A: Augment SFT data with worked numerical examples**. The hypothesis is that the model lacks formula-selection knowledge because the original ~2,206 training examples contain too few detailed numerical derivations per formula family.

### 4.2 New Folder: `SFT3_NumAug/`

A self-contained experiment folder created locally with the following pipeline:

```
extracted_text/ (553 chunks)
    ↓ [generate_numerical_qa.py + numerical_qa_prompt_template.txt]
numerical_qa_supplement.jsonl (~2,000+ new pairs)
    ↓ [merge_and_prepare_sft3.py]
data/grpo_train.jsonl (2,206 existing) ──┘
    ↓
data/sft3_train.jsonl + data/sft3_test.jsonl   → train_sft3.py → sft3_model_output/
data/grpo6_train.jsonl + data/grpo6_eval.jsonl → train_grpo.py → grpo_run6_output/
```

### 4.3 Key New Files

| File | Purpose |
|---|---|
| `numerical_qa_prompt_template.txt` | Formula-first prompt: 4 numerical + 1 MCQ per chunk. Requires formula name → full equation → value substitution → arithmetic → answer with units. Emphasizes diversity across formula families. |
| `generate_numerical_qa.py` | Groq-based generator (adapted from `generate_qa.py`). Reads `extracted_text/` chunks, uses `llama-3.3-70b-versatile`. Supports `--resume`, `--limit`, `--dry-run`. |
| `merge_and_prepare_sft3.py` | Loads existing + supplement, validates (context leak / refusal checks), deduplicates (80-char prefix key), produces 4 output splits. |
| `train_sft3.py` | SFT-3 trainer pointing to `sft3_train.jsonl` / `sft3_model_output` |
| `train_grpo.py` | GRPO-6 trainer with all Run 5b fixes, pointing to `grpo_run6_output`, `sft3_model_output` adapter, `grpo6_train.jsonl` dataset |
| `eval_grpo.py` | Evaluates `grpo_run6_output` on `grpo6_eval.jsonl` → `eval_grpo_run6_results.json` |
| `eval_baseline.py` | Evaluates `sft3_model_output` on `grpo6_eval.jsonl` → `eval_sft3_baseline_results.json` |

### 4.4 Generation Run Status

Generation was run locally on Windows using 6 Groq API keys with `--resume` support:
- **553 total chunks** identified across Newman, Faltinsen, MIT OCW, and IIT exam papers
- Completed ~173 chunks in a first pass (~865 pairs) before session interruption
- Resumed for ~207 more chunks (~1,035 pairs total at last check — 981 confirmed lines in `numerical_qa_supplement.jsonl`)
- **Halted** when all 6 Groq API keys hit their daily TPD limit (100,000 tokens/key/day)
- Keys will reset after 24 hours; generation can resume with `python generate_numerical_qa.py --resume`
- Remaining: ~173/553 chunks (31%) still to be processed

### 4.5 Quality Assessment (Based on First 35 Records)

- **Structure compliance:** ~94% — formula name, equation, substitution, result consistently present
- **Physical errors:** ~4–5 per batch — e.g., added mass using ρV instead of ½ρV, sign errors in Bernoulli. These are expected in LLM-generated data; GRPO reward will penalize incorrect answers during training.
- **Repetition:** ~15–20% duplicate question types caught by deduplication (mainly Froude number and wave speed problems)
- **Formula diversity:** Good — covers Froude, dispersion relation, Morison equation, Bernoulli, added mass, buoyancy, wave energy

---

## 5. Gemini LLM Baseline

### 5.1 Motivation

To contextualize the SLM's performance, a comparison against a frontier general-purpose LLM (Gemini via gemini.google.com) was established using the same `grpo_eval.jsonl` evaluation set.

### 5.2 Infrastructure Created: `LLM_baseline/`

| File | Purpose |
|---|---|
| `gemini_system_prompt.txt` | System prompt forcing structured JSON output: `[{"index": N, "type": "...", "reasoning": "...", "final_answer": "..."}]`. Includes rules against markdown fences, requires units in final_answer. |
| `prepare_chunks.py` | Filters numerical + MCQ from eval set, splits into chunks of ~80 questions, saves uploadable JSON files. |
| `score_gemini.py` | Parses all Gemini response files, extracts numerical values and MCQ letters, scores against ground truth with 5% tolerance. |

### 5.3 Workflow

1. `python prepare_chunks.py --dataset grpo_eval.jsonl` → 5 chunk files in `chunks/`
2. For each chunk: open gemini.google.com, paste system prompt, upload chunk JSON, copy response
3. Save responses as `chunks/responses/response_XX_of_05.json`
4. `python score_gemini.py` → `gemini_results.json` + `gemini_summary.txt`

**Response file issues encountered:**
- `response_02`: Gemini prefixed its response with a Python code block before the JSON array — extracted by finding first `[`
- `response_04`: Unescaped LaTeX backslashes (`\c`, `\r`, `\V` etc.) violated JSON spec — fixed by doubling all non-standard backslashes
- A `_repair_responses.py` one-off script was written to auto-fix both issues

### 5.4 Results

| Model | Numerical Accuracy | MCQ Accuracy |
|---|---|---|
| SFT-2 baseline (SLM) | 27.60% | 71.82% |
| **Gemini (frontier LLM)** | **37.10%** | **78.18%** |
| GRPO Run 4 (SLM) | 38.46% | **85.45%** |
| **GRPO Run 5b (SLM)** | **38.91%** | 80.00% |

### 5.5 Key Findings

1. **The fine-tuned 3B SLM outperforms Gemini on both numerical accuracy and MCQ accuracy** on this domain-specific evaluation set. This is a strong thesis result.

2. **Identical failure pattern:** Gemini's numerical failure distribution is essentially identical to the SLM's:

| Rel-error range | Gemini failures | SLM (Run 5b) failures |
|---|---|---|
| < 5% | 0 | 0 |
| 5–20% | 10 | 5 |
| 20–100% | 52 | 52 |
| 100–1000% | 40 | 44 |
| > 1000% | 34 | 30 |

Both models fail on **formula selection**, not arithmetic — confirming the eval set genuinely tests domain-specific knowledge that is not easily acquired from general pre-training.

3. **Zero "no answer" failures for Gemini** — it always produced a structured response, whereas the SLM occasionally failed to generate a parseable final answer (3 cases).

4. **MCQ advantage for the SLM** — the SLM's MCQ accuracy (80–85%) consistently exceeds Gemini's (78.18%), attributable to domain-specific MCQ training data.

---

## 6. Key Technical Finding: The 39% Ceiling

After 5 GRPO runs with varying configurations, the numerical accuracy has converged to ~38–39%:

- Run 4: 38.46% (correct max_new_tokens, r=16, attention-only LoRA)
- Run 5b: 38.91% (correct max_new_tokens, r=32, attention+MLP LoRA)
- Gemini: 37.10% (frontier LLM, no fine-tuning)

The ceiling is **not caused by**:
- Output truncation (fixed in Run 4, no improvement)
- Model capacity (r=32 + MLP in Run 5b, no improvement)
- Data leakage (SFT-2 clean pipeline, same plateau)
- Training instability (confirmed healthy dynamics in Run 5b)

The ceiling **is caused by** the model's inability to select the correct formula for a given problem, which is a **knowledge representation problem** addressable only through data augmentation — which is what SFT3_NumAug targets.

---

## 7. Next Steps (Before Submission April 24)

Given the submission deadline, the following are prioritized:

1. **Resume numerical QA generation** (`python generate_numerical_qa.py --resume` after key reset) — complete the remaining ~173 chunks
2. **Run `merge_and_prepare_sft3.py`** → prepare SFT-3 + GRPO-6 splits
3. **Run SFT-3 on server** → `train_sft3.py`
4. **Run GRPO-6** → `train_grpo.py --max_new_tokens 768` (all fixes applied)
5. **Evaluate GRPO-6** and compare against Gemini baseline
6. **Update thesis report** (`Report/main.tex`) with:
   - GRPO Run 5b results
   - Gemini comparison table
   - SFT3_NumAug motivation and (if completed) results
