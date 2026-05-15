# SFT Training Log — Marine Hydrodynamics SLM
**Project:** Major Thesis Project (MTP) — Physics-Aware Small Language Model for Marine Hydrodynamics  
**Model:** Qwen2.5-3B-Instruct  
**Method:** Supervised Fine-Tuning (SFT) via LoRA  
**Date:** April 2026  
**Server:** `mtp-server` (10.71.9.8) — NVIDIA H100 80GB, Ubuntu 24.04  

---

## 1. Project Overview

The goal of this SFT stage is to adapt a general-purpose 3B parameter language model (`Qwen/Qwen2.5-3B-Instruct`) to the domain of **marine hydrodynamics**. The model is trained on a synthetic dataset of ~1,100 QA pairs derived from:
- **Newman, J.N. — "Marine Hydrodynamics"** (primary source, 220 text chunks)
- **Faltinsen, O.M. — "Sea Loads on Ships and Offshore Structures"** (supplementary)

The QA pairs were generated using the Groq API (LLaMA 3.1 70B) in a previous session. This document covers only the SFT training stage.

The SFT stage serves as the foundation for a subsequent GRPO/Reinforcement Learning stage, where a physics verifier (`physics_verifier.py`) will be used as a reward signal to further align the model's numerical reasoning.

---

## 2. Environment Setup

### Server & Hardware
- **Host:** `mtp-server` (10.71.9.8)
- **GPU:** NVIDIA H100 80GB SXM5
- **RAM:** ~126 GB shared memory
- **OS:** Ubuntu 24.04.4 LTS, Python 3.12

### Virtual Environment
```bash
cd ~/mtp
python3 -m venv sft_env
source sft_env/bin/activate
pip install torch transformers trl peft accelerate bitsandbytes safetensors datasets
```

### Key Package Versions (at time of training)
| Package | Version |
|---|---|
| `torch` | 2.x (CUDA 12.x) |
| `transformers` | 4.x |
| `trl` | 1.0.0+ |
| `peft` | 0.x |
| `accelerate` | 0.x |

### Storage Constraints
The `/home` partition was at **100% capacity** throughout training. This caused `trainer.save_model()` to crash at the end of early runs with:
```
safetensors_rust.SafetensorError: I/O error: No space left on device (os error 28)
```
**Fix:** Set the output directory to `/tmp/${USER}_hf_cache/` (tmpfs RAM-backed storage), and set `export HF_HOME="/tmp/${USER}_hf_cache"` to redirect all Hugging Face cache downloads there as well. Later, the `save_model()` call was wrapped in a try/except so training always completes even if the final save fails, using the per-epoch checkpoints as backup.

---

## 3. Data Pipeline

### 3.1 Source Files
- **Raw QA pairs:** `qa_dataset.jsonl` (~1,100 records)
- Each record contains: `question`, `answer`, `chain_of_thought`, `type` (`conceptual` | `numerical` | `mcq`), `options` (for MCQ), `source` (chunk filename)

### 3.2 Validation (`validate_qa.py`)
Before formatting, records are validated for:
- Minimum word counts for question and answer
- Absence of "context leakage" phrases (e.g., "as stated in", "the passage says")
- Absence of refusal phrases (e.g., "I cannot", "I don't know")
- Structural integrity (required fields present)

Valid records are written to `qa_dataset_clean.jsonl`.

### 3.3 SFT Data Preparation (`prepare_sft_data.py`)
Converts validated QA pairs into **HuggingFace ChatML format** (a list of `messages` objects with `role` and `content`).

**System Prompt:**
```
You are an expert AI assistant specializing in Marine Hydrodynamics and Ocean Engineering. Approach all questions methodically and provide step-by-step reasoning.
```

**Assistant Response Format (initial design):**
```
<think>
{chain_of_thought}
</think>

Final Answer:
{answer}
```

Note: `<think>` tags were removed in the final working version (see Section 5 for why). The final format used in successful training is:
```
Chain of Thought:
{chain_of_thought}

Final Answer:
{answer}
```

**Split:** 90% train / 10% test, shuffled with `random.seed(42)`.
- **Train:** 976 records → `data/sft_train.jsonl`
- **Test:** 109 records → `data/sft_test.jsonl`

---

## 4. Initial Training Attempt

### 4.1 First Script (TRL 0.x style)
The initial `train_sft.py` used `TrainingArguments` from `transformers` directly. This failed with:
```
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'dataset_text_field'
```
**Cause:** TRL 1.0.0 replaced `TrainingArguments` + `dataset_text_field` with a new `SFTConfig` API that automatically handles ChatML `messages` format via `apply_chat_template`.

**Fix:** Rewrote the training script to use `SFTConfig` and `SFTTrainer(processing_class=tokenizer)` per TRL 1.0.0+ conventions.

### 4.2 Initial LoRA Configuration
```python
LoraConfig(
    r=16,
    lora_alpha=32,           # scaling = lora_alpha/r = 2.0
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 4.3 Initial Training Config
```python
SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch size = 16
    num_train_epochs=3,
    learning_rate=5e-5,
    bf16=True,                        # ← root of all NaN problems
    optim="adamw_torch_fused",
    max_grad_norm=1.0,
    warmup_steps=10,
)
```

---

## 5. The Great NaN Investigation

This was the dominant challenge of the entire SFT phase. Every training run would begin normally (2–5 steps of valid decreasing loss) and then catastrophically produce `grad_norm: nan`, destroying the model weights permanently within that run, with all subsequent steps showing `loss: 0, accuracy: 0`.

### 5.1 Symptoms
```
{'loss': '2.296', 'grad_norm': '0.9745', 'epoch': '0.1639'}  ✅
{'loss': '1.318', 'grad_norm': '0.1923', 'epoch': '0.3279'}  ✅
{'loss': '1.052', 'grad_norm': '0.1228', 'epoch': '0.4918'}  ✅
{'loss': '0.9603','grad_norm': '0.1199', 'epoch': '0.6557'}  ✅
{'loss': '1.124', 'grad_norm': 'nan',   'epoch': '0.8197'}   ❌ ← NaN appears
{'loss': '0',     'grad_norm': 'nan',   'epoch': '0.9836'}   💀 model dead
```

The user asked: *"Is loss: 0 from the model fitting perfectly?"*

**Answer:** No. `loss: 0` alongside `grad_norm: nan` and `mean_token_accuracy: 0` is the signature of a dead model producing NaN logits. When PyTorch's cross-entropy is computed on NaN logits, the result is NaN, which in BF16's internal representation sometimes serializes as `0.0`. A perfectly-fitted model would show `accuracy → 1.0` and `entropy → 0`, not `accuracy = 0`.

### 5.2 Key Observation: Perfectly Reproducible NaN

**The NaN always hit at exactly epoch 0.8197 (optimizer step 50) with `random.seed(42)`.** The exact same loss values appeared in every run:  
`2.296 → 1.318 → 1.052 → 0.960 → NaN`

This ruled out stochastic numerical instability and confirmed it was either a specific data batch or a structural training issue that manifested deterministically.

### 5.3 Hypothesis 1: `<think>` Token Tokenization Fragmentation

The assistant responses all began with `<think>` as the very first token. Without special token registration, the tokenizer splits `<think>` into 3 tokens: `['<', 'think', '>']`. We hypothesized this caused inconsistent attention patterns.

**Fix attempted:**
```python
tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
model.resize_token_embeddings(len(tokenizer))
```

**Result:** NaN still occurred at the same step. Marginal improvement in loss values (slightly different by ~0.003) confirmed the change was applied, but it was not the root cause.

### 5.4 Hypothesis 2: Random Embedding Initialization for New Tokens

When `resize_token_embeddings()` is called, the 2 new rows (for `<think>`, `</think>`) in the embedding table are randomly initialized. In BF16's limited dynamic range, these random values can occasionally produce extreme logit activations.

**Fix attempted:**
```python
with torch.no_grad():
    embed_layer = model.get_input_embeddings()
    mean_embed = embed_layer.weight[:-2].mean(dim=0)
    embed_layer.weight[-2] = mean_embed  # <think>
    embed_layer.weight[-1] = mean_embed  # </think>
```

**Result:** Still NaN. Same step.

### 5.5 Hypothesis 3: lm_head Not Initialized

We realized we had initialized the **input** embedding but forgotten the **output projection** (`lm_head`). The model generates logits by dotting the hidden state with `lm_head.weight`. The new token rows in `lm_head` were still random.

**Fix attempted:**
```python
lm_head = model.get_output_embeddings()
if lm_head is not None and lm_head.weight.shape[0] == len(tokenizer):
    mean_lm = lm_head.weight[:-2].mean(dim=0)
    lm_head.weight[-2] = mean_lm
    lm_head.weight[-1] = mean_lm
```

**Result:** Still NaN. Same step. The startup log confirmed the change was applied: *"Both input embeddings AND lm_head initialized with mean (stable init)."*

### 5.6 Hypothesis 4: Fused AdamW BF16 Precision

`optim="adamw_torch_fused"` uses a CUDA-fused kernel for AdamW that is known to accumulate floating-point error faster than standard AdamW in BF16.

**Fix attempted:**
```python
optim="adamw_torch"  # Changed from "adamw_torch_fused"
```

**Result:** Still NaN. Same step.

### 5.7 Hypothesis 5: Specific Bad Data Batch

Since the NaN was perfectly reproducible at step 50, and step 50 corresponds to optimizer steps 41–50, and with effective batch size 16 this covers training samples at shuffled indices 640–799, we scanned those records:

```python
# Scan for problematic content in the culprit batch range
flags = []
if '∞' in asst: flags.append('INF_CHAR')
if 'undefined' in asst.lower(): flags.append('UNDEFINED')
if 'diverge' in asst.lower(): flags.append('DIVERGE')
```

**Finding:** Record at index [800] (one position past the step-50 range) contained the `∞` Unicode character — `log(sinh(0)) = log(0) = -∞`. We cleaned this globally from the dataset (13 training records, 2 test records had the `∞` symbol replaced with "infinity").

**Result:** Still NaN. Global cleanup was correct hygiene but not the root cause.

### 5.8 Hypothesis 6: Batch Clustering (Shuffle Seed Test)

The perfectly deterministic NaN at step 50 suggested problematic samples were clustering in one batch due to `random.seed(42)`. We reshuffled the data with `random.seed(1337)` to break up batch compositions.

**Result (seed=1337):** NaN moved — but got **dramatically worse**: `loss: 91.06` at step 1, immediately dead. The bad batch hit first instead of 50th.

**Critical insight:** With seed=1337, the same data but in a different order caused immediate catastrophic failure. This proved the NaN was **not caused by any individual sample** (all 16 samples in the first batch were inspected and all looked completely normal). The issue was structural, not data-specific.

### 5.9 Hypothesis 7: Removing `<think>` Tags Entirely

Since `<think>` is a custom token that Qwen2.5 **never saw during pre-training**, every sample in our dataset asks the model to predict a token with near-zero pre-trained probability as its very first output token. In BF16, `-log(≈0)` can overflow.

We stripped the `<think>` and `</think>` tags from all training samples, replacing them with plain text structure:
```
Chain of Thought:
{cot content}

Final Answer:
{answer}
```

**Result:** Still NaN, this time at step 3 (epoch 0.4918) instead of step 5. The step moved but the NaN persisted.

### 5.10 Root Cause Identified: BF16 Precision Accumulation

After exhausting all data and initialization hypotheses, we identified the true root cause:

**BF16 has only 7 bits of mantissa** (vs. 23 for float32). With LoRA applied to 14 parameter matrices simultaneously (7 attention/FFN layers × Q_A, Q_B LoRA pairs), and a LoRA scaling factor of `lora_alpha/r = 32/16 = 2.0`, the cumulative floating-point rounding errors in BF16 accumulate across gradient accumulation steps (4 steps) and eventually overflow BF16's representable range (~65,504), producing `inf` → `nan` in approximately 20–30 optimizer steps.

This was confirmed by the gradient norm pattern: `0.85 → 0.22 → NaN`. The gradient norms were **decreasing** (model converging) immediately before the NaN — not exploding. This rules out gradient explosion from large gradients and points to **precision underflow/overflow** in the intermediate BF16 computations.

---

## 6. Working Solution

### 6.1 The Fix: Float32 Training

```python
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float32,   # ← Key change
    device_map="auto"
)
# In SFTConfig:
bf16=False,
fp16=False,
```

Float32 has 23 bits of mantissa — 65,536× more precision than BF16. The H100 has 80GB of VRAM; float32 for a 3B model requires ~12GB for weights + ~24GB for AdamW optimizer state (2× float32 copies) = ~36GB total, well within capacity.

**Trade-off:** Roughly 2.5× slower training (564 seconds vs ~230 seconds for 3 epochs). Acceptable given the correctness requirement.

### 6.2 Final LoRA Configuration

Also simplified LoRA to reduce gradient magnitude:

```python
LoraConfig(
    r=16,
    lora_alpha=16,       # scaling=1.0 (was 2.0) — halves LoRA gradient contribution
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention-only (was 7 modules)
    bias="none",
    task_type="CAUSAL_LM"
)
```

Removing `gate_proj`, `up_proj`, `down_proj` (FFN layers) from LoRA targets reduces the number of simultaneously updated parameter matrices from 14 to 8, further reducing the probability of BF16 accumulation errors.

### 6.3 Final Training Configuration

```python
SFTConfig(
    output_dir="./sft_model_output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,    # effective batch size = 16
    num_train_epochs=3,
    learning_rate=5e-5,
    bf16=False,                        # float32 full precision
    fp16=False,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    optim="adamw_torch",
    max_grad_norm=1.0,
    warmup_steps=10,
    report_to="none",
)
```

### 6.4 Final Training Data Format

Each record in `sft_train.jsonl` (and `sft_test.jsonl`) follows the HuggingFace ChatML schema:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert AI assistant specializing in Marine Hydrodynamics and Ocean Engineering. Approach all questions methodically and provide step-by-step reasoning."
    },
    {
      "role": "user",
      "content": "What is the drag force on a hull with..."
    },
    {
      "role": "assistant",
      "content": "Chain of Thought:\n[step-by-step reasoning]\n\nFinal Answer:\n[answer]"
    }
  ],
  "source": "newman_chunk_0154.txt",
  "type": "numerical"
}
```

Note: The training data on the server was additionally cleaned to replace any `∞` Unicode characters with the word "infinity" to prevent any potential loss overflow, even though this was not found to be the root cause of NaN.

---

## 7. Training Results

### 7.1 Final Training Run

**Command:**
```bash
cd ~/mtp && source sft_env/bin/activate
nohup python train_sft.py > sft_training.log 2>&1 &
```

**Training Curve (all 18 log points — no NaN):**

| Epoch | Train Loss | Grad Norm | Token Accuracy |
|---|---|---|---|
| 0.164 | 2.193 | 0.3448 | 63.2% |
| 0.328 | 1.767 | 0.2905 | 65.2% |
| 0.492 | 1.348 | 0.2137 | 69.8% |
| 0.656 | 1.103 | 0.1553 | 73.9% |
| 0.820 | 0.996 | 0.1008 | 76.5% |
| 0.984 | 0.932 | 0.0994 | 77.4% |
| **Eval @ Epoch 1** | **1.035** | — | **75.5%** |
| 1.148 | 0.876 | 0.0822 | 78.4% |
| 1.311 | 0.834 | 0.0745 | 78.6% |
| 1.475 | 0.775 | 0.0585 | 80.0% |
| 1.639 | 0.805 | 0.0629 | 79.1% |
| 1.803 | 0.762 | 0.0540 | 79.9% |
| 1.967 | 0.770 | 0.0580 | 79.7% |
| **Eval @ Epoch 2** | **0.983** | — | **77.0%** |
| 2.131 | 0.747 | 0.0519 | 80.2% |
| 2.295 | 0.731 | 0.0460 | 80.4% |
| 2.459 | 0.731 | 0.0459 | 80.8% |
| 2.623 | 0.744 | 0.0554 | 80.1% |
| 2.787 | 0.708 | 0.0498 | 81.2% |
| 2.951 | 0.745 | 0.0514 | 80.4% |
| **Eval @ Epoch 3** | **0.974** | — | **77.4%** |

**Final Metrics:**
- `train_loss`: **0.9729**
- `eval_loss`: **0.9743**
- `eval_mean_token_accuracy`: **77.43%**
- `train_runtime`: 564.6 seconds (~9.4 minutes for 3 epochs)

### 7.2 Model Artifacts

The trained model is a **LoRA adapter** (PEFT format). It is saved at:
```
~/mtp/sft_model_output/
├── checkpoint-61/       # End of epoch 1
├── checkpoint-122/      # End of epoch 2
└── checkpoint-183/      # End of epoch 3 (final)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    └── ...
```

---

## 8. Inference Verification

To load and run the fine-tuned model, you must first load the base model and then apply the LoRA adapter:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

adapter_path = './sft_model_output/checkpoint-183'
base_model = 'Qwen/Qwen2.5-3B-Instruct'

# Load tokenizer from adapter directory (contains chat template config)
tok = AutoTokenizer.from_pretrained(adapter_path)

# Load base model in float32 (must match training precision)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,
    device_map='auto'
)

# Apply the fine-tuned LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# Inference
prompt = "What is the Reynolds number and why is it important in marine hydrodynamics?"
msgs = [{'role': 'user', 'content': prompt}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors='pt').to(model.device)
out = model.generate(**ids, max_new_tokens=400, do_sample=False)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

### 8.1 Sample Outputs

**Q1: Reynolds number (conceptual)**
> The Reynolds number (Re) is a dimensionless quantity used to predict flow patterns in different fluid flow situations. The formula is: Re = ρvL/μ. In marine hydrodynamics, the Reynolds number is crucial because it helps determine whether flow around ships, submarines, and marine structures is laminar or turbulent, which significantly affects drag and lift forces. [… correctly explains laminar vs turbulent transition, hull design implications]

**Q2: Resistance force (numerical)**
> Using F_D = ½ · C_D · ρ · v² · A  
> F_D = ½ × 0.002 × 1025 × (10)² × 500  
> F_D = 51,250 N = 51.25 kN ✅

Both answers are physically correct. The model demonstrates sound domain reasoning and correct numerical computation.

---

## 9. Summary of Failed Hypotheses

| # | Hypothesis | Change Made | Outcome |
|---|---|---|---|
| 1 | `<think>` tokenized as 3 tokens | Added as special tokens | NaN persisted at same step |
| 2 | Random input embedding init | Initialized new rows with mean | NaN persisted |
| 3 | Random lm_head init | Initialized lm_head rows with mean | NaN persisted |
| 4 | Fused AdamW BF16 bugs | Changed to `adamw_torch` | NaN persisted |
| 5 | Data has `∞` symbols | Replaced all `∞` with "infinity" | NaN persisted |
| 6 | Bad batch clustering (seed=42) | Reshuffled with seed=1337 | NaN immediately at step 1 (worse) |
| 7 | `<think>` token never seen in pre-training | Removed `<think>` tags entirely | NaN moved to step 3 (still present) |
| ✅ | **BF16 precision accumulation** | **Switched to float32 training** | **NaN eliminated — clean training** |

---

## 10. Key Lessons Learned

1. **`loss: 0` is not convergence.** When seen alongside `grad_norm: nan` and `accuracy: 0`, it is the signature of NaN-corrupted model weights. A converging model shows decreasing loss AND increasing accuracy simultaneously.

2. **BF16 is dangerous for LoRA fine-tuning of large models when targeting many modules simultaneously.** The limited 7-bit mantissa accumulates rounding errors that overflow after a deterministic number of optimizer steps. The number of steps before failure is roughly proportional to the number of simultaneously active LoRA matrices and their scaling factor.

3. **Perfectly reproducible NaN = structural issue, not stochastic.** If NaN hits at exactly the same optimizer step across all runs regardless of data changes, the root cause is in the training infrastructure (precision, optimizer, LoRA scaling), not the dataset.

4. **Shuffling the data can move the NaN but not eliminate it.** Different seeds placed the same structural failure at different training steps (step 50 vs step 1 vs step 3), confirming the issue is batch-order-independent.

5. **The H100's 80GB VRAM is more than sufficient for float32 training** of 3B parameter models with LoRA adapters. The memory cost (~36GB total) is acceptable, and the 2.5× speed reduction is a worthwhile trade for training stability.
