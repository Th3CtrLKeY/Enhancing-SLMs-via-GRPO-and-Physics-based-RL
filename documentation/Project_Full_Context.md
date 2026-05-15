# Project Full Context: Marine Hydrodynamics SLM
**Goal:** Fine-tuning a Small Language Model (SLM) for expert-level reasoning in Marine Hydrodynamics and Ocean Engineering.

---

## 1. Infrastructure & Environment
*   **Server Host:** `10.71.9.8` (alias: `mtp-server`)
*   **GPU:** NVIDIA H100 80GB SXM5
*   **OS:** Ubuntu 24.04 LTS (Kernel 6.8.0)
*   **Primary Directory:** `~/mtp/` (on server) | `c:\Users\raghu\Desktop\IIT\Sem 10\MTP` (local)
*   **Virtual Environment:** `~/mtp/sft_env` (Python 3.12)
*   **Critical Storage Note:** The `/home` partition is at 100% capacity. 
    *   **HF Cache:** `export HF_HOME="/tmp/ai21na3ai42_hf_cache"`
    *   **Training Output:** Models are saved to `~/mtp/sft_model_output/` (which is often symlinked or directed to `/tmp` during active runs to avoid OS Error 28).

---

## 2. Phase 1: Data Acquisition & Extraction
*   **Source Materials:**
    *   Newman, J.N. — *Marine Hydrodynamics* (Digital Text)
    *   Faltinsen, O.M. — *Sea Loads on Ships and Offshore Structures* (OCR Scanned)
    *   MIT OCW 2.20 & 2.29 Lecture Notes and Exams.
*   **Extraction Tools:**
    *   `extract_text.py`: Standard PDF text extraction for digital sources.
    *   `ocr_scanned_pdfs.py`: Tesseract/OCR-based extraction for scanned documents.
*   **Output:** Text chunks stored in `~/mtp/extracted_text/` (~220 chunks from Newman).

---

## 3. Phase 2: QA Dataset Generation
*   **Script:** `~/mtp/generate_qa.py`
*   **Model:** `llama-3.3-70b-versatile` via **Groq API**.
*   **Logic:**
    *   Reads `qa_prompt_template.txt`.
    *   Generates pairs of **Conceptual**, **Numerical**, and **MCQ** questions.
    *   Includes a **Chain of Thought (CoT)** for every answer.
*   **Technical Implementation:**
    *   **Key Rotation:** Robustly rotates through multiple GROQ API keys (`.env`) to handle `429 Rate Limit` errors.
    *   **Checkpointing:** Tracks progress in `qa_checkpoint.json` to allow resumption.
    *   **Extraction:** Uses regex to pull JSON blocks from model responses.
*   **Output:** `~/mtp/qa_dataset.jsonl`.

---

## 4. Phase 3: Validation & Cleaning
*   **Validation Script:** `~/mtp/validate_qa.py`
*   **Filter Criteria:**
    *   Removed "Refusal" samples (e.g., "I cannot answer this").
    *   Removed "Context Leakage" (e.g., "According to the passage").
    *   Ensured minimum word counts (Conceptual answer > 30 words, CoT > 20 words).
    *   Cleaned Unicode: Replaced the `∞` symbol with "infinity" to prevent BF16 numerical instability.
*   **Output:** `~/mtp/qa_dataset_clean.jsonl`.

---

## 5. Phase 4: SFT Data Preparation
*   **Script:** `~/mtp/prepare_sft_data.py`
*   **Format:** ChatML (Standard for `TRL` 1.0.0+ `SFTTrainer`).
*   **Reasoning Format:**
    *   *Initial:* Wrapped in `<think>...</think>` tags.
    *   *Final:* Converted to plain text `Chain of Thought: ... \n\n Final Answer: ...` after discovering that custom special tokens triggered NaN gradient spikes in BF16 during initial SFT.
*   **Dataset Split:**
    *   `data/sft_train.jsonl` (90%) — 976 records.
    *   `data/sft_test.jsonl` (10%) — 109 records.

---

## 6. Phase 5: Supervised Fine-Tuning (SFT)
*   **Script:** `~/mtp/train_sft.py`
*   **Model:** `Qwen/Qwen2.5-3B-Instruct`.
*   **Training Infrastructure:**
    *   Library: `trl` 1.0.0 (uses `SFTConfig` and `SFTTrainer`).
    *   Method: PEFT/LoRA.
*   **The NaN Debugging Journey:**
    *   Persistent `grad_norm: nan` occurred after ~50 steps with BF16.
    *   **Root Cause:** Cumulative rounding errors in BF16 across multiple LoRA matrices (14 targets originally) caused overflows.
    *   **Solution:** Switched to **Float32 (Full Precision)**.
*   **Final LoRA Hyperparameters:**
    *   `r=16`, `lora_alpha=16` (scaling = 1.0).
    *   Target Modules: `["q_proj", "k_proj", "v_proj", "o_proj"]` (Attention only).
*   **Final Metrics:**
    *   Train Loss: `0.9729` (after 3 epochs).
    *   Eval Accuracy: `77.43%`.
    *   Artifact: `~/mtp/sft_model_output/checkpoint-183`.

---

## 7. Phase 6: Verification & Analysis
*   **Verification Script:** `~/mtp/physics_verifier.py`
    *   Uses **SymPy** for symbolic verification of physics equations.
    *   Calculates numerical tolerance for final answers.
*   **Smoke Test Tool:** `~/mtp/smoke_test.py`
*   **Current Performance State:**
    *   **Conceptual:** Strong (Definitions for Froude, Reynolds, etc. are correct).
    *   **Logical:** Strong (MCQs resolved correctly).
    *   **Numerical:** Weak. Model still hallucinations formulas or fails arithmetic during the CoT. This confirms the need for the next phase: **GRPO (Reinforcement Learning)**.

---

## 8. Summary of File Locations (Server)

| Purpose | Path |
| :--- | :--- |
| **SFT Training Data** | `~/mtp/data/sft_train.jsonl` |
| **Cleaned QA Dataset** | `~/mtp/qa_dataset_clean.jsonl` |
| **SFT Trainer Script** | `~/mtp/train_sft.py` |
| **LoRA Checkpoint (Final)** | `~/mtp/sft_model_output/checkpoint-183/` |
| **Physics Verifier Logic** | `~/mtp/physics_verifier.py` |
| **Smoke Test Script** | `~/mtp/smoke_test.py` |
| **Extracted Text Chunks** | `~/mtp/extracted_text/` |

---

## 9. Next Steps for LLM/Agent
The SFT stage is concluded. The model is properly formatted but requires reinforcement on numerical truth.
1.  **RL Preparation:** Define the Reward Function using `physics_verifier.py`.
2.  **GRPO Pipeline:** Implement Group Relative Policy Optimization (GRPO) to penalize wrong numerical answers.
3.  **VLLM Integration:** Serve the model via VLLM for high-throughput RL rollouts.
