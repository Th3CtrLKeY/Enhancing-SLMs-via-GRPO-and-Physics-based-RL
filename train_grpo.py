"""
train_grpo.py — GRPO training for Marine Hydrodynamics SLM.

This script is designed to mirror the existing SFT setup (`train_sft.py`) but
replace supervised loss with Group Relative Policy Optimization (GRPO).

It reuses:
  - `eval_baseline.parse_record` for dataset normalization / prompt formatting
  - `physics_verifier.physics_reward_function` as the reward function for:
        numerical, mcq, symbolic/derivation
  - (optional) `llm_judge.judge_conceptual` for conceptual RLAIF

Expected dataset:
  - Flat JSONL (recommended): `data/grpo_train.jsonl` from `split_dataset.py`
    Each line has at least: {"question": ..., "type": ..., "answer": ...}
    MCQ items also include: {"options": {"A": "...", ...}}

Notes:
  - This file intentionally does NOT pin TRL API to a specific version in code.
    The GRPO trainer API has shifted across TRL releases; the import section
    below gives a clear error if `trl` is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import inspect

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import re as _re

from eval_baseline import MCQ_SYSTEM_PROMPT, SYSTEM_PROMPT, parse_record
from physics_verifier import (
    physics_reward_function,
    extract_numerical_answer,
    verify_numerical,
)


# ── Completion logger (saves rollouts for post-hoc analysis) ──────────────────

_log_lock = threading.Lock()
_log_file = None
_log_call_counter = 0


def _init_completion_log(output_dir: str) -> None:
    global _log_file
    path = Path(output_dir) / "completions_log.jsonl"
    _log_file = open(path, "a", encoding="utf-8")
    print(f"[LOG] Completion logger initialised → {path}")


def _log_completion(
    q_type: str,
    completion: str,
    reward: float,
    correctness: float,
    fmt_reward: float,
    gt: dict,
) -> None:
    global _log_call_counter
    if _log_file is None:
        return
    _log_call_counter += 1
    record = {
        "call": _log_call_counter,
        "q_type": q_type,
        "reward": round(reward, 4),
        "correctness": round(correctness, 4),
        "format_reward": round(fmt_reward, 4),
        "gt_answer": str(gt.get("answer", "")),
        "completion_tail": completion[-500:],
    }
    with _log_lock:
        _log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if _log_call_counter % 50 == 0:
            _log_file.flush()


def _close_completion_log() -> None:
    global _log_file
    if _log_file is not None:
        _log_file.flush()
        _log_file.close()
        _log_file = None


# ── Graded numerical reward (partial credit for "close" answers) ──────────────

def _graded_numerical_reward(completion: str, gt_answer: str) -> float:
    """
    Returns a reward in [0.0, 1.0] based on how close the predicted number is
    to the ground truth.  Replaces the hard binary 0/1 from physics_reward_function
    for numerical questions, giving GRPO richer gradient signal.

    Tiers:
      ≤5%  relative error → 1.0  (same as physics_reward_function)
      ≤10% relative error → 0.7
      ≤25% relative error → 0.4
      ≤50% relative error → 0.15
      >50% or no number   → 0.0
    """
    pred_val = extract_numerical_answer(completion)
    try:
        gt_val = float(gt_answer)
    except (ValueError, TypeError):
        return 0.0

    if pred_val is None:
        return 0.0

    if verify_numerical(pred_val, gt_val, tolerance=0.05):
        return 1.0

    if gt_val == 0:
        return 0.0

    rel_err = abs(pred_val - gt_val) / abs(gt_val)
    if rel_err <= 0.10:
        return 0.7
    if rel_err <= 0.25:
        return 0.4
    if rel_err <= 0.50:
        return 0.15
    return 0.0


# ── Format / structure reward (non-binary, provides GRPO signal) ──────────────

# Run 2: reduced weights (cap ~0.2 instead of ~0.4) so that the new graded
# numerical reward dominates and the model can't game format alone.
_FMT_W_THINK   = 0.05   # has <think>...</think> or "Chain of Thought" section
_FMT_W_FINAL   = 0.08   # has "Final Answer:" line (critical for verifier parsing)
_FMT_W_LENGTH  = 0.05   # reasonable length (not too short / not truncated)
_FMT_W_STEPS   = 0.02   # shows numbered steps or bullet reasoning


def _format_reward(completion: str, q_type: str) -> float:
    """
    Non-binary reward for output structure / format quality.

    Returns a float in [0.0, ~0.2].  Because completions naturally vary in how
    well they follow the expected format, this component almost always produces
    intra-group variance — exactly what GRPO needs to compute non-zero advantages.
    """
    score = 0.0

    # 1) Reasoning section
    has_think = bool(
        _re.search(r"<think>", completion, _re.IGNORECASE)
        or _re.search(r"chain of thought", completion, _re.IGNORECASE)
        or _re.search(r"step[- ]by[- ]step", completion, _re.IGNORECASE)
    )
    if has_think:
        score += _FMT_W_THINK

    # 2) Final Answer line (the verifier depends on this)
    has_final = bool(_re.search(r"Final Answer[:\s]", completion, _re.IGNORECASE))
    if has_final:
        score += _FMT_W_FINAL

    # For MCQ: also accept "Therefore, the answer is:" pattern
    if q_type == "mcq" and not has_final:
        if _re.search(r"therefore[,\s]+the answer is", completion, _re.IGNORECASE):
            score += _FMT_W_FINAL

    # 3) Reasonable length (not too short, not obviously truncated)
    n_chars = len(completion.strip())
    if 50 < n_chars < 2000:
        score += _FMT_W_LENGTH
    elif n_chars >= 2000:
        score += _FMT_W_LENGTH * 0.5  # partial credit if very long but not empty

    # 4) Shows structured steps (numbered list, bullet points, or "Step N")
    has_steps = bool(
        _re.search(r"\b(step\s+\d|1[\.\)]\s)", completion, _re.IGNORECASE)
        or _re.search(r"\n\s*[-•]\s", completion)
    )
    if has_steps:
        score += _FMT_W_STEPS

    return score


def _require_trl() -> tuple[Any, Any]:
    """
    Import GRPO objects from TRL with a helpful error message.
    Returns (GRPOTrainer, GRPOConfig).
    """
    try:
        # Newer TRL naming
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
        return GRPOTrainer, GRPOConfig
    except Exception as e_new:
        try:
            # Some versions may expose them under trl.trainer
            from trl.trainer.grpo_trainer import GRPOConfig, GRPOTrainer  # type: ignore
            return GRPOTrainer, GRPOConfig
        except Exception:
            raise RuntimeError(
                "TRL GRPO is not available in this environment.\n"
                "Install dependencies first (example):\n"
                "  python -m pip install -U trl transformers datasets peft accelerate\n"
                f"Original import errors: {type(e_new).__name__}: {e_new}"
            )


def _safe_config_kwargs(config_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter kwargs to only those supported by the installed TRL config class.
    This makes the script robust to TRL version differences.
    Prints which keys were dropped so silent failures are visible.
    """
    try:
        sig = inspect.signature(config_cls.__init__)
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        accepted = {k: v for k, v in kwargs.items() if k in allowed}
        dropped = set(kwargs.keys()) - set(accepted.keys())
        if dropped:
            print(f"[INFO] GRPOConfig dropped unsupported keys: {sorted(dropped)}")
            print(f"       (These will be applied via model.generation_config instead if applicable.)")
        return accepted
    except Exception:
        return {}


def _load_jsonl(path: Path, max_samples: int | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if max_samples is not None:
        records = records[:max_samples]
    return records


def _system_prompt_for_type(q_type: str) -> str:
    # Match eval_baseline protocol: MCQ uses forced-letter prompt.
    return MCQ_SYSTEM_PROMPT if q_type == "mcq" else SYSTEM_PROMPT


@dataclass
class Example:
    prompt: str
    raw_question: str
    q_type: str
    gt: dict[str, Any]
    system_prompt: str


def _to_examples(records: list[dict], include_conceptual: bool) -> list[Example]:
    out: list[Example] = []
    for rec in records:
        p = parse_record(rec)
        q_type = p["q_type"]
        if (q_type == "conceptual") and (not include_conceptual):
            continue

        gt: dict[str, Any] = {"type": q_type, "answer": p["gt_answer"]}
        if q_type == "mcq" and rec.get("options"):
            gt["options"] = rec["options"]

        out.append(
            Example(
                prompt=p["question_prompt"],
                raw_question=p["original_question"] or p["question_prompt"],
                q_type=q_type,
                gt=gt,
                system_prompt=_system_prompt_for_type(q_type),
            )
        )
    return out


def _format_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_dataset(examples: list[Example], tokenizer) -> Dataset:
    rows = []
    for ex in examples:
        rows.append(
            {
                # TRL will sample completions from this prompt string.
                "prompt": _format_prompt(tokenizer, ex.system_prompt, ex.prompt),
                # Keep metadata for reward computation.
                "raw_question": ex.raw_question,
                "q_type": ex.q_type,
                "gt": ex.gt,
            }
        )
    return Dataset.from_list(rows)


def _completion_to_text(completion: Any) -> str:
    """Normalize TRL completion (plain string or chat-style nested lists) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, dict):
            content = first.get("content", "")
            return content if isinstance(content, str) else str(content)
        return _completion_to_text(first)
    return str(completion)


def _reward_row_at(reward_kwargs: dict[str, Any], index: int) -> dict[str, Any]:
    def nth(key: str) -> Any:
        v = reward_kwargs.get(key)
        if isinstance(v, list) and index < len(v):
            return v[index]
        return None

    return {"gt": nth("gt"), "q_type": nth("q_type"), "raw_question": nth("raw_question")}


def _reward_fn_factory(
    include_conceptual: bool,
    conceptual_judge: Callable[[str, str, str], float] | None,
) -> Callable[..., list[float]]:
    """
    Reward callable for TRL GRPOTrainer (current API).

    The trainer invokes:
      reward_fn(prompts=..., completions=..., completion_ids=..., **reward_kwargs)
    where reward_kwargs lists dataset columns (e.g. raw_question, q_type, gt), one entry
    per completion (including repeated rows for num_generations).
    """

    def reward_fn(
        prompts: Any = None,
        completions: Any = None,
        completion_ids: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        for k in ("trainer_state", "log_extra", "log_metric", "environments"):
            kwargs.pop(k, None)
        if not completions:
            return []

        rewards: list[float] = []
        for i, completion in enumerate(completions):
            text = _completion_to_text(completion)
            row = _reward_row_at(kwargs, i)
            gt = row.get("gt") or {}
            q_type = (row.get("q_type") or gt.get("type") or "conceptual").strip()

            fmt = _format_reward(text, q_type)

            if q_type == "conceptual":
                if (not include_conceptual) or (conceptual_judge is None):
                    correctness = 0.0
                    total = fmt
                else:
                    question_ctx = row.get("raw_question", "") or ""
                    reference = str(gt.get("answer", "") or "")
                    correctness = float(conceptual_judge(question_ctx, reference, text))
                    total = correctness + fmt
                _log_completion(q_type, text, total, correctness, fmt, gt)
                rewards.append(total)
                continue

            if q_type == "numerical":
                correctness = _graded_numerical_reward(text, str(gt.get("answer", "")))
            else:
                correctness = float(physics_reward_function(text, gt))
            total = correctness + fmt
            _log_completion(q_type, text, total, correctness, fmt, gt)
            rewards.append(total)
        return rewards

    return reward_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training for Marine Hydrodynamics SLM")
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument(
        "--adapter_path",
        type=str,
        default="./sft_model_output/checkpoint-183",
        help=(
            "Path to the SFT LoRA adapter checkpoint. The adapter weights are merged "
            "into the base model before GRPO LoRA is applied, so GRPO starts from a "
            "domain-tuned policy rather than a bare pretrained model. "
            "Set to '' (empty string) to skip and start from the bare base model."
        ),
    )
    p.add_argument("--train_file", type=str, default="data/grpo_train.jsonl")
    p.add_argument("--output_dir", type=str, default="./grpo_run4_output")
    p.add_argument("--max_samples", type=int, default=None)

    # GRPO knobs (trainer defaults vary by TRL version; expose common ones)
    # Batch / accumulation
    # batch_size=1 intentional: each GRPO step generates K completions per prompt,
    # so effective memory usage is batch_size × num_generations.
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # Learning rate — Run 1 used 1e-5 (conservative). Training was stable, so
    # bumped to 2e-5 for Run 2 to accelerate learning while still well below SFT (5e-5).
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=2)

    # Optimiser — match SFT: plain AdamW (no fused/paged variant needed here).
    p.add_argument("--optim", type=str, default="adamw_torch")

    # Warmup — same absolute step count as SFT warmup (10 steps).
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help=(
            "Rollout cap per completion. Run 2 used 512 which caused 75%+ clipped_ratio. "
            "Bumped to 1024 for Run 3 to eliminate false 0-reward from truncation. "
            "Use 512–768 on smaller VRAM (<40 GiB free)."
        ),
    )
    # Increased default from 4→8: larger groups mean more reward diversity within
    # each group, reducing frac_reward_zero_std and producing stronger GRPO signal.
    p.add_argument("--num_generations", type=int, default=8, help="Group size K per prompt")

    # Rollout temperature — use >0 so the K completions differ from each other.
    # 0.8 is a safe starting point: diverse enough for signal, low enough for coherence.
    # 0.0 = greedy (all completions nearly identical → zero-advantage groups).
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for rollout generation. 0.0=greedy (bad for GRPO signal).",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling for rollouts. Pairs with --temperature.",
    )

    # What to train on
    p.add_argument(
        "--include_conceptual",
        action="store_true",
        help="Include conceptual questions (requires Groq keys if --use_judge).",
    )
    p.add_argument(
        "--use_judge",
        action="store_true",
        help="Use Groq LLM-as-judge reward for conceptual questions.",
    )

    # Saving — save intermediate checkpoints every N steps (in addition to per-epoch)
    p.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save a checkpoint every N optimizer steps. Set 0 to disable (epoch-only saves).",
    )

    # Repro
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    GRPOTrainer, GRPOConfig = _require_trl()

    train_path = Path(args.train_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _init_completion_log(str(out_dir))

    # 1) Load dataset records
    records = _load_jsonl(train_path, max_samples=args.max_samples)
    examples = _to_examples(records, include_conceptual=bool(args.include_conceptual))
    if not examples:
        raise RuntimeError(
            "No training examples after filtering.\n"
            "If your dataset is mostly conceptual, pass --include_conceptual (and optionally --use_judge)."
        )

    # 2) Tokenizer + model
    # Load tokenizer from SFT adapter dir when available (it may have a custom pad token).
    adapter_path = args.adapter_path.strip() if args.adapter_path else ""
    tok_source = adapter_path if adapter_path else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    # If an SFT adapter path is provided, merge it into the base weights so that
    # GRPO starts from the domain-tuned policy rather than the bare pretrained model.
    if adapter_path:
        import os
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(
                f"SFT adapter not found: {adapter_path!r}\n"
                "Pass --adapter_path '' to skip and start from the base model."
            )
        print(f"Loading SFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Merging SFT adapter weights into base model…")
        model = model.merge_and_unload()
        print("Merge complete. GRPO LoRA will be applied on top of merged weights.")

    # 3) PEFT (match SFT: attention projections only)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4) Build RL dataset
    ds = _build_dataset(examples, tokenizer)

    # 5) Optional conceptual judge
    judge_fn = None
    if args.use_judge:
        from llm_judge import judge_conceptual
        judge_fn = judge_conceptual
        # Fail early if keys are missing (judge_conceptual will throw on first call otherwise)
        if not os.getenv("GROQ_API_KEYS") and not (Path(__file__).parent / ".env").exists():
            raise RuntimeError(
                "--use_judge was set but Groq keys were not found.\n"
                "Set GROQ_API_KEYS in environment or add a .env file with GROQ_API_KEYS=key1,key2,..."
            )

    reward_fn = _reward_fn_factory(
        include_conceptual=bool(args.include_conceptual),
        conceptual_judge=judge_fn,
    )

    # 6) Trainer config
    # Keep config fields conservative; TRL will ignore unknown fields on some versions,
    # but to be safe we only pass widely-supported args.
    desired_cfg = {
        "output_dir": str(out_dir),
        # Batch / steps
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        # Optimiser (mirrors SFT)
        "learning_rate": args.learning_rate,
        "optim": args.optim,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": 1.0,
        # Logging / saving
        "logging_steps": 10,
        "save_strategy": "steps" if args.save_steps > 0 else "epoch",
        "save_steps": args.save_steps if args.save_steps > 0 else 500,
        "report_to": "none",
        "seed": args.seed,
        # Generation controls (not available on all TRL versions)
        "max_new_tokens": args.max_new_tokens,
        "num_generations": args.num_generations,
        # generation_batch_size must be divisible by num_generations.
        # Setting it equal ensures this holds regardless of the K chosen.
        "generation_batch_size": args.num_generations,
        # Sampling — temperature>0 diversifies rollouts within each group,
        # which is essential for non-zero GRPO advantages with binary rewards.
        "temperature": args.temperature,
        "top_p": args.top_p,
        # Precision — keep float32 like SFT (BF16 caused NaN in SFT training)
        "bf16": False,
        "fp16": False,
    }
    safe_cfg = _safe_config_kwargs(GRPOConfig, desired_cfg)
    grpo_args = GRPOConfig(**safe_cfg)

    # Fallback: if generation params were dropped from GRPOConfig (older TRL),
    # apply them directly on the model's generation_config so rollouts use them.
    if "max_new_tokens" not in safe_cfg:
        print(f"[INFO] Applying max_new_tokens={args.max_new_tokens} via model.generation_config (fallback).")
        model.generation_config.max_new_tokens = args.max_new_tokens
    else:
        print(f"[INFO] max_new_tokens={args.max_new_tokens} applied via GRPOConfig (native).")

    if "temperature" not in safe_cfg and args.temperature > 0:
        print(f"[INFO] Applying temperature={args.temperature} via model.generation_config (fallback).")
        model.generation_config.temperature = args.temperature
        model.generation_config.do_sample = True
    if "top_p" not in safe_cfg and args.temperature > 0:
        print(f"[INFO] Applying top_p={args.top_p} via model.generation_config (fallback).")
        model.generation_config.top_p = args.top_p

    # Verify generation_config has the right max_new_tokens before training starts
    effective_max = getattr(model.generation_config, "max_new_tokens", None)
    print(f"[INFO] Effective model.generation_config.max_new_tokens = {effective_max}")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        reward_funcs=reward_fn,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    _close_completion_log()
    print(f"[LOG] Completion log saved to: {out_dir / 'completions_log.jsonl'}")


if __name__ == "__main__":
    # Avoid torch dynamo surprises in some Windows setups
    torch.set_float32_matmul_precision("high")
    main()

