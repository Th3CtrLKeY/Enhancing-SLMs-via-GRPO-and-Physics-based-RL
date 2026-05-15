"""
find_bad_batch.py — Identifies the exact training batch causing NaN gradients.

Simulates the training data loader and runs a forward+backward pass on each
effective batch to find the first one that produces NaN gradients.
"""
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

# ── Config (must match train_sft.py) ─────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_FILE     = "data/sft_train.jsonl"
RANDOM_SEED    = 42
BATCH_SIZE     = 4   # per_device_train_batch_size
GRAD_ACCUM     = 4   # gradient_accumulation_steps
LOG_STEPS      = 10  # only need to test up to step ~50

# ── Load tokenizer ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})

# ── Load model in FP32 for reliable NaN detection ────────────────────────────
print("Loading model (FP32 for reliable NaN detection)...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="cpu")
model.resize_token_embeddings(len(tok))
with torch.no_grad():
    emb = model.get_input_embeddings()
    mean = emb.weight[:-2].mean(0)
    emb.weight[-2] = mean
    emb.weight[-1] = mean

# Apply minimal LoRA so the model matches training setup
lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
model.train()

# ── Load and shuffle data identically to training ─────────────────────────────
print("Loading & shuffling data with seed=42 (mirrors training)...")
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(l) for l in f if l.strip()]

random.seed(RANDOM_SEED)
random.shuffle(records)
print(f"Total records: {len(records)}")

# ── Tokenize a batch ──────────────────────────────────────────────────────────
def tokenize_batch(batch_records):
    texts = []
    for r in batch_records:
        msgs = r["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    # Labels = input_ids with padding set to -100
    labels = enc["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    enc["labels"] = labels
    return enc

# ── Find the first batch producing NaN ───────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
optimizer.zero_grad()

total_samples = len(records)
micro_step = 0
optimizer_step = 0
accum_count = 0
max_optimizer_steps = 65  # check entire epoch 1

print(f"\nScanning up to {max_optimizer_steps} optimizer steps...")
print(f"Each optimizer step = {GRAD_ACCUM} micro-batches of {BATCH_SIZE} samples each")

i = 0
bad_found = False

while optimizer_step < max_optimizer_steps and i < total_samples:
    # Get micro-batch
    micro_batch = records[i : i + BATCH_SIZE]
    micro_batch_indices = list(range(i, min(i + BATCH_SIZE, total_samples)))
    
    if len(micro_batch) == 0:
        break
    
    batch = tokenize_batch(micro_batch)
    
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss
    
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n[!] NaN/Inf LOSS detected!")
        print(f"    Optimizer step: {optimizer_step+1}, Micro-batch: {accum_count+1}/{GRAD_ACCUM}")
        print(f"    Sample indices in sft_train.jsonl: {micro_batch_indices}")
        for idx in micro_batch_indices:
            r = records[idx]
            source = r.get("source", "unknown")
            rtype = r.get("type", "unknown")
            lengths = [len(m["content"]) for m in r["messages"]]
            print(f"    - Record[{idx}] source={source} type={rtype} content_lens={lengths}")
            # Print assistant content snippet
            for m in r["messages"]:
                if m["role"] == "assistant":
                    print(f"      assistant[:200]: {m['content'][:200]!r}")
        bad_found = True
        break
    
    (loss / GRAD_ACCUM).backward()
    accum_count += 1
    
    # Check gradients for NaN after backward
    nan_grad = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
    if nan_grad:
        print(f"\n[!] NaN GRADIENT detected after backward!")
        print(f"    Optimizer step: {optimizer_step+1}, Micro-batch: {accum_count}/{GRAD_ACCUM}")
        print(f"    Loss value: {loss.item():.4f}")
        print(f"    Sample indices in sft_train.jsonl: {micro_batch_indices}")
        for idx in micro_batch_indices:
            r = records[idx]
            source = r.get("source", "unknown")
            rtype = r.get("type", "unknown")
            lengths = [len(m["content"]) for m in r["messages"]]
            print(f"    - Record[{idx}] source={source} type={rtype} content_lens={lengths}")
            for m in r["messages"]:
                if m["role"] == "assistant":
                    print(f"      assistant[:200]: {m['content'][:200]!r}")
        bad_found = True
        break
    
    if accum_count == GRAD_ACCUM:
        optimizer.step()
        optimizer.zero_grad()
        optimizer_step += 1
        accum_count = 0
        if optimizer_step % 5 == 0:
            print(f"  Step {optimizer_step}/{max_optimizer_steps} — loss={loss.item():.4f} — all good so far")
    
    i += BATCH_SIZE

if not bad_found:
    print("\n[OK] No NaN detected in the first epoch. Problem may be in the trainer setup.")

print("\nDone.")
