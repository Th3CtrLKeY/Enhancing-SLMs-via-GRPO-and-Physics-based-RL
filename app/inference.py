"""
Inference helpers for the Marine Hydrodynamics SLM chat demo.

Loads Qwen2.5-3B-Instruct + a PEFT LoRA adapter and generates ChatML replies
using the same system prompt pattern as eval_baseline.py / smoke_test.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER_PATH = str(REPO_ROOT / "grpo_run3_output")

SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in Marine Hydrodynamics and "
    "Ocean Engineering. Approach all questions methodically and provide "
    "step-by-step reasoning."
)

MCQ_SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in Marine Hydrodynamics and "
    "Ocean Engineering. For multiple choice questions, carefully analyze each "
    "option and explain your reasoning step by step. "
    "You MUST end your response with exactly this line:\n"
    "'Therefore, the answer is: X'\n"
    "where X is the letter of the correct option (A, B, C, or D)."
)


def _resolve_adapter_path(adapter_path: Optional[str] = None) -> str:
    path = adapter_path or os.environ.get("ADAPTER_PATH", DEFAULT_ADAPTER_PATH)
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {resolved}. "
            "Set ADAPTER_PATH or place weights under grpo_run3_output/."
        )
    return str(resolved)


def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # Prefer bf16 on modern GPUs; fall back to fp16 if unsupported.
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class HydroSLM:
    """Lazy-loadable wrapper around the fine-tuned hydrodynamics SLM."""

    def __init__(
        self,
        base_model: Optional[str] = None,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.25,
    ) -> None:
        self.base_model = base_model or os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)
        self.adapter_path = _resolve_adapter_path(adapter_path)
        self.max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", max_new_tokens))
        self.temperature = float(os.environ.get("TEMPERATURE", temperature))
        self.model = None
        self.tokenizer = None
        self.device: Optional[torch.device] = None

    def load(self) -> "HydroSLM":
        if self.model is not None:
            return self

        # Prefer adapter-local tokenizer if present; otherwise use the base model.
        adapter_tok = Path(self.adapter_path) / "tokenizer_config.json"
        tok_source = self.adapter_path if adapter_tok.exists() else self.base_model
        print(f"[inference] Loading tokenizer from {tok_source}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_source,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = _pick_dtype()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        print(f"[inference] Loading base model {self.base_model} ({dtype}) on {device}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        print(f"[inference] Applying LoRA adapter from {self.adapter_path}")
        model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()

        self.model = model
        print("[inference] Model ready.")
        return self

    def _system_prompt_for(self, question: str) -> str:
        q = question.lower()
        if "options:" in q or "\na:" in q or "\nb:" in q:
            return MCQ_SYSTEM_PROMPT
        return SYSTEM_PROMPT

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        *,
        history: Optional[list[tuple[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()

        assert self.model is not None and self.tokenizer is not None

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt or self._system_prompt_for(question)},
        ]
        if history:
            for user_msg, assistant_msg in history:
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": question})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        temp = self.temperature if temperature is None else temperature
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temp and temp > 0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temp,
                    "top_p": 0.9,
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_GLOBAL: Optional[HydroSLM] = None


def get_model(
    base_model: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> HydroSLM:
    """Return a process-wide singleton HydroSLM (loads on first use)."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = HydroSLM(base_model=base_model, adapter_path=adapter_path)
        _GLOBAL.load()
    return _GLOBAL
