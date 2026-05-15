# Documentation Inconsistencies Report
**Date:** April 17, 2026  
**Scope:** All files in `documentation/` — cross-referenced against actual code and data files.

---

## 1. Dataset Size: ~1,100 vs ~2,757 Records

**The most significant inconsistency across the documentation.**

| Document | Claim |
|---|---|
| `SFT_Training_Log.md` §1 | "~1,100 QA pairs" |
| `SFT_Training_Log.md` §3.1 | "qa_dataset.jsonl (~1,100 records)" |
| `Project_Full_Context.md` §5 | "976 records (train) + 109 records (test)" = 1,085 total |
| `GRPO_Training_Log_Runs2-3.md` §7 | "2206 grpo_train + 551 grpo_eval = **2,757** total" |

**What actually happened:** The QA dataset (`qa_dataset.jsonl`) was expanded from ~1,100 to ~2,757 records between the SFT-1 and GRPO stages — likely through additional runs of `generate_qa.py` on more source chunks. However, **no document records when, why, or how the dataset grew by 2.5x**.

**Impact:** The SFT-1 model was trained on only ~1,085 samples (90% of the original ~1,100), while GRPO trained on 2,206 records. Comparisons between SFT-1 and GRPO implicitly benefit from the larger dataset in GRPO, but this is never called out.

---

## 2. `<think>` Tags vs "Chain of Thought:" Format

**Multiple documents contradict each other and the actual code.**

| Source | Claims |
|---|---|
| `SFT_Training_Log.md` §3.3 | "Note: `<think>` tags were removed in the final working version" |
| `SFT_Training_Log.md` §5.9 | "We stripped the `<think>` and `</think>` tags from all training samples, replacing them with plain text structure: `Chain of Thought:`" |
| `SFT_Training_Log.md` §6.4 | Shows final format as `"Chain of Thought:\n[step-by-step reasoning]\n\nFinal Answer:\n[answer]"` |
| `Project_Full_Context.md` §5 | "Converted to plain text `Chain of Thought: ...`" |
| **Actual `prepare_sft_data.py` line 37** | `assistant_content = f"<think>\n{cot}\n</think>\n\nFinal Answer:\n{ans}"` |
| **Actual `sft_train.jsonl` line 1** | Contains `<think>` tags |

**What actually happened:** Removing `<think>` tags was **Hypothesis 7** during the NaN debugging (§5.9), which did NOT fix the NaN. The actual fix was switching to float32 (§6.1). After that fix, the `<think>` tags were **kept** in the data (as confirmed by both the code and the actual training files). The SFT log incorrectly states the tags were removed in the "final working version."

**Impact:** Misleading for anyone reading the docs — they would believe the model was trained without `<think>` tags, when it actually was trained WITH them.

---

## 3. GPU Specifications Inconsistent Across Documents

| Document | GPU Claimed |
|---|---|
| `SFT_Training_Log.md` §2 | "NVIDIA H100 **80GB** SXM5" |
| `Project_Full_Context.md` §1 | "NVIDIA H100 **80GB** SXM5" |
| `GRPO_Training_Log_Run1.md` header | "NVIDIA H100 **80GB**" |
| `GRPO_Training_Log_Runs2-3.md` header | "NVIDIA H100 NVL **95 GiB**" |
| **Actual `nvidia-smi` output** | H100 NVL, **95,830 MiB** (~93.6 GiB) |

**What actually happened:** The mtp-server has H100 NVL GPUs with ~95 GiB VRAM (not 80 GB SXM5). The earlier docs incorrectly state 80GB. The Runs 2-3 doc corrected it to 95 GiB but didn't note the change.

---

## 4. SFT Baseline Accuracy Presented Without Data Leakage Caveat

| Document | SFT Baseline Numerical | SFT Baseline MCQ |
|---|---|---|
| `GRPO_Training_Log_Run1.md` §9.1 | 14.48% | 60.91% |
| `GRPO_Training_Log_Runs2-3.md` §6.1 | 14.48% | 60.91% |

These numbers are presented as definitive baselines. However, **the SFT-1 training data (`sft_train.jsonl`, ~976 records) was derived independently from `qa_dataset.jsonl` and likely overlaps with `grpo_eval.jsonl` (551 records)**. This was identified in conversation but the existing documents never add a caveat or footnote.

**Clean SFT-2 baseline (from `grpo_train.jsonl` only, zero overlap with eval):**
- Numerical: **27.6%** (vs 14.48% previously claimed)
- MCQ: **71.8%** (vs 60.91% previously claimed)

The SFT-2 baseline is higher because: (a) it's trained on 2x more data, and (b) the previous 14.48% was measured with a potentially leaky eval set (which paradoxically produced a LOWER number, possibly because the SFT-1 model was trained on too few samples to benefit from overlap).

---

## 5. `qa_dataset_clean.jsonl` — Referenced but Possibly Not Used

| Source | Reference |
|---|---|
| `Project_Full_Context.md` §4 | "Output: `~/mtp/qa_dataset_clean.jsonl`" |
| `SFT_Training_Log.md` §3.2 | "Valid records are written to `qa_dataset_clean.jsonl`" |

But both `prepare_sft_data.py` and `split_dataset.py` default to reading **`qa_dataset.jsonl`** (not the clean version). Either:
- `qa_dataset_clean.jsonl` was never created / is stale, or
- It was created but the downstream scripts were never updated to use it
- Or `qa_dataset.jsonl` itself was cleaned in-place

The pipeline documentation implies a cleaned file exists in the chain, but the actual scripts bypass it.

---

## 6. Conceptual Questions in GRPO Run 3 — Contradiction

| Source | Claim |
|---|---|
| `GRPO_Training_Log_Runs2-3.md` §3.3 | "`include_conceptual`: Yes" |
| `GRPO_Training_Log_Runs2-3.md` §3.4 | "The training dataset `grpo_train.jsonl` appears to contain no conceptual questions" |
| `GRPO_Training_Log_Runs2-3.md` §5.2 | "21,200 total: 7,072 MCQ + 14,128 numerical. Conceptual: 0" |

**Arithmetic check:** `grpo_eval.jsonl` has 221 conceptual out of ~551 records (40%). At 80/20 split, `grpo_train.jsonl` should have ~882 conceptual records. With 2 epochs and 8 generations: (884 + 440) × 2 × 8 = 21,184 for numerical+MCQ only, which matches the logged 21,200. This means **conceptual questions were NOT included despite the flag being set**.

Possible causes:
- The `--include_conceptual` flag was not actually passed in the Run 3 launch command
- Or there's a parsing issue in `_to_examples` that silently drops conceptual records

Either way, the doc is self-contradictory.

---

## 7. `eval_baseline_results_run4.json` — Confusing File Name

`GRPO_Training_Log_Run1.md` §9.3 references `eval_baseline_results_run4.json` as the SFT baseline comparison file. The name "run4" implies it's a GRPO Run 4 result, but it's actually the **SFT baseline** results file that was created before any GRPO run. The naming creates confusion about what the file contains.

---

## 8. `eval_grpo.py` Default Adapter Path — Stale Across Docs

`GRPO_Training_Log_Run1.md` §2.2 states the eval script "loads the GRPO adapter from `./grpo_model_output`", but the script's `GRPO_ADAPTER_PATH` default has been updated across runs:

| Run | Actual Default |
|---|---|
| Run 1 | `./grpo_model_output` |
| Run 2 | `./grpo_run2_output` |
| Run 3 | `./grpo_run3_output` |
| Run 4 | `./grpo_run4_output` |

The Run 1 doc is not wrong per se (it was correct at the time), but the docs don't note which version of the scripts they correspond to.

---

## 9. grpo_eval.jsonl Record Count: 551 vs 552

| Source | Count |
|---|---|
| `GRPO_Training_Log_Runs2-3.md` §7 | "551 grpo_eval.jsonl" (from `wc -l`) |
| `eval_sft2_baseline_results.json` | `"n_records": 552` |

**Likely cause:** `wc -l` counts newlines. If the file lacks a trailing newline, `wc -l` reports one fewer. The actual number of JSON records is 552.

---

## 10. SFT Training Data Format Not Documented Consistently

| Document | Implied Source for SFT |
|---|---|
| `SFT_Training_Log.md` | `qa_dataset.jsonl` → `prepare_sft_data.py` → `sft_train.jsonl` (1,085 records) |
| `Project_Full_Context.md` | Same pipeline |
| `GRPO_Training_Log_Runs2-3.md` §10 | Plans to re-run SFT on `grpo_train.jsonl` (2,206 records) |

The SFT-2 pipeline (which uses `grpo_train.jsonl` as input instead of the original `qa_dataset.jsonl`) is not documented in any existing file. This creates a gap: the actual SFT model now being used (sft2_model_output) has a different data origin than what the SFT log describes.

---

## Summary of Action Items

| # | Inconsistency | Severity | Suggested Fix |
|---|---|---|---|
| 1 | Dataset grew 1,100 → 2,757 undocumented | **High** | Add a section documenting the dataset expansion |
| 2 | `<think>` tags: docs say removed, code says present | **High** | Correct SFT_Training_Log §3.3 and §6.4 |
| 3 | GPU specs: 80GB vs 95 GiB | Medium | Standardise to actual specs |
| 4 | SFT baseline without leakage caveat | **High** | Add footnote to GRPO Run 1 and Runs 2-3 docs |
| 5 | `qa_dataset_clean.jsonl` phantom file | Medium | Clarify whether it exists and is used |
| 6 | Conceptual questions in Run 3 | Medium | Correct the claim or investigate the bug |
| 7 | Confusing "run4" baseline filename | Low | Rename or add clarifying note |
| 8 | Stale adapter paths in docs | Low | Add version note |
| 9 | 551 vs 552 record count | Low | Use 552 consistently |
| 10 | SFT-2 pipeline undocumented | **High** | Will be covered in Run 4 documentation |
