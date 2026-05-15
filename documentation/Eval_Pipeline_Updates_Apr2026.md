# Eval Pipeline Updates (Post-SFT) — April 2026

This document covers **everything implemented/changed after the last documented milestone** in `Project_Full_Context.md` (Phase 6: Verification & Analysis), focusing on **evaluation**, **dataset splitting**, **MCQ handling**, **physics verification robustness**, and **saving review-friendly outputs**.

---

## 1) New / Updated Evaluation Workflow

### 1.1 Baseline evaluator (`eval_baseline.py`)

**Purpose**
- Run baseline evaluation of the SFT LoRA adapter (`./sft_model_output/checkpoint-183`) on an eval split (typically `data/grpo_eval.jsonl`).
- Score **numerical + MCQ** automatically using `physics_verifier.py`.
- Optionally score **conceptual** via an external LLM judge (Groq) if enabled.

**Key updates**
- **Avoided** `device_map="auto"` flows for PEFT adapter loading due to a server-side runtime issue (seen as `TypeError: unhashable type: 'set'` in the stack involving `accelerate` / `PeftModel`).
  - The stable path used: load base model, move to explicit device, and apply the adapter with an explicit device mapping.
- **MCQ prompting improvements**
  - Added an MCQ system prompt path (`MCQ_SYSTEM_PROMPT`) so the model is forced to respond with an option letter.
  - For flat MCQ records, the `Options:` block is appended into the prompt so the model sees the choices.
- **Review-friendly JSON outputs**
  - Each `per_item` row includes the original dataset question and answer fields alongside model output so results are easy to audit later (see Section 4).
  - Failures now also preserve the same review fields (not just truncated previews).
- Added `--output / -o` to control where the JSON report is saved (default remains `eval_baseline_results.json`).

**Outputs**
- A JSON report containing:
  - `type_stats`: per-type accuracy/reward
  - `per_item`: detailed per-question records
  - `failures`: a subset of failing items for quick inspection
  - `per_item_fields`: a small schema dictionary documenting what each stored field means

---

## 2) Dataset Splitting for Evaluation

### 2.1 Split script (`split_dataset.py`)

**Purpose**
- Create train/eval splits from the main QA dataset, producing an eval file like:
  - `data/grpo_eval.jsonl`

**Why**
- Keeps a stable held-out evaluation set for measuring improvements across changes (prompting, verifier updates, RL/GRPO later).

---

## 3) LLM Judge for Conceptual Questions (Optional)

### 3.1 Judge script (`llm_judge.py`)

**Purpose**
- For **conceptual** questions (where strict numeric/MCQ verification is inappropriate), use a judge LLM (Groq) to compare model output with reference answers.

**Notes**
- Requires `.env` to expose Groq credentials (keys are rotated in other scripts).
- `eval_baseline.py --judge` enables conceptual evaluation; otherwise conceptual items are marked `skipped: true` and not included in accuracy.

---

## 4) Saving “Question + Answer + Model Output” for Later Review

The evaluator was updated so every result row keeps both:
- the **original dataset question** and answer/reference, and
- the **exact prompt sent to the model**, plus the model’s completion.

### 4.1 Fields saved per item

Each `per_item` record includes (depending on record type/format):
- `original_question`: question text as stored in the dataset (flat format)
- `question_prompt`: exact prompt sent to the model (flat MCQ includes appended `Options:` block)
- `dataset_answer_raw`: raw dataset answer (flat)
- `dataset_chain_of_thought`: raw chain-of-thought field (flat, if present)
- `reference_assistant`: reference assistant message (ChatML datasets, if present)
- `completion`: model-generated output (absent for skipped conceptual items when judge disabled)
- plus scoring-related fields such as `reward`, `passed`, `ground_truth_scoring` (or `ground_truth_for_judge`), and timing (`elapsed_s`)

This makes it possible to review outputs offline without re-opening the dataset file.

---

## 5) Physics Verifier & MCQ Robustness Updates (`physics_verifier.py`)

### 5.1 MCQ answer extraction hardened

**Problem**
- Naive patterns (e.g. checking if `'a' in text`) caused inflated MCQ scores.

**Fix**
- Added a forced-letter extractor that recognizes common formats:
  - `Answer: B`, `Correct option is (C)`, `Therefore: D`, `\boxed{A}`, etc.
- Evaluation now uses this forced letter rather than substring heuristics.

### 5.2 Numerical extraction broadened

**Problem**
- Model outputs vary: scientific notation, units, symbolic forms, multiple candidate numbers, etc.

**Fix**
- More tolerant parsing:
  - collects multiple numerical candidates from the output
  - handles scientific notation and unit-adjacent formatting more robustly
  - compares against GT with tolerance logic to reduce false negatives

---

## 6) Response Capture Utility

### 6.1 `ask_model_save_responses.py`

**Purpose**
- Run the model on a set of questions and save raw responses for manual inspection.

**Update**
- Imports `SYSTEM_PROMPT` / `MCQ_SYSTEM_PROMPT` from `eval_baseline.py` to keep prompting consistent across eval and ad-hoc querying.

---

## 7) Server Run Commands (Canonical)

### 7.1 Run evaluation on server

```bash
cd ~/mtp
source sft_env/bin/activate

# smoke run
python eval_baseline.py --dataset data/grpo_eval.jsonl --max-samples 5 --output eval_smoke.json

# full run
python eval_baseline.py --dataset data/grpo_eval.jsonl --output eval_baseline_results.json

# conceptual judge (optional; requires Groq keys)
python eval_baseline.py --dataset data/grpo_eval.jsonl --judge --output eval_with_judge.json
```

### 7.2 Copy results back to Windows (PowerShell)

Prefer specifying the full destination filename (avoids trailing-quote/backslash pitfalls):

```powershell
scp ai21na3ai42@mtp-server:~/mtp/eval_baseline_results.json "c:\Users\raghu\Desktop\IIT\Sem 10\MTP\eval_baseline_results.json"
```

Copy any other file similarly, e.g.:

```powershell
scp ai21na3ai42@mtp-server:~/mtp/eval_baseline_updated_script.json~ "c:\Users\raghu\Desktop\IIT\Sem 10\MTP\eval_baseline_updated_script.json"
```

---

## 8) Observed Results Snapshot (for reference)

From the updated eval output format (judge disabled):
- **MCQ** accuracy is materially higher than **numerical** accuracy.
- **Conceptual** items are skipped unless `--judge` is enabled.

This reinforced the project conclusion that the next major step should focus on **numerical correctness** (GRPO/RL reward shaping using the verifier).

