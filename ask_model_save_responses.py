"""
ask_model_save_responses.py — Ask the local SLM nine questions (3 per type) and save replies.

Types: conceptual, numerical, MCQ (marine hydrodynamics).

Uses the same system prompts as eval_baseline.py so outputs match the benchmark
protocol (generic system for conceptual/numerical; forced-letter MCQ prompt).

Usage:
    cd <project>
    python ask_model_save_responses.py
    python ask_model_save_responses.py --output my_run.json

Requires the same environment as smoke_test.py (transformers, peft, CUDA optional).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_baseline import MCQ_SYSTEM_PROMPT, SYSTEM_PROMPT

# ── Config (match smoke_test.py) ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "model_qa_responses.json"
ADAPTER_PATH = "./sft_model_output/checkpoint-183"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 512

# Three questions per type: conceptual, numerical, MCQ
QUESTIONS: list[dict] = [
    # Conceptual
    {
        "id": "C1",
        "type": "conceptual",
        "question": (
            "What is the difference between laminar and turbulent flow, and how does "
            "the Reynolds number help predict which regime will occur in a marine setting?"
        ),
    },
    {
        "id": "C2",
        "type": "conceptual",
        "question": (
            "Explain the physical meaning of the Froude number and its significance "
            "in ship resistance prediction."
        ),
    },
    {
        "id": "C3",
        "type": "conceptual",
        "question": (
            "What is added mass in the context of marine hydrodynamics and why does it "
            "matter for vessel motion analysis?"
        ),
    },
    # Numerical
    {
        "id": "N1",
        "type": "numerical",
        "question": (
            "A submarine travels at 5 m/s underwater. Its hull has a characteristic length of 80 m. "
            "The kinematic viscosity of seawater is 1.19e-6 m²/s. "
            "Calculate the Reynolds number and state whether the boundary layer is likely laminar or turbulent."
        ),
    },
    {
        "id": "N2",
        "type": "numerical",
        "question": (
            "A ship has a waterplane area of 1200 m², a block coefficient of 0.75, "
            "a length of 150 m, a beam of 25 m, and a draft of 8 m. "
            "Calculate the displacement volume."
        ),
    },
    {
        "id": "N3",
        "type": "numerical",
        "question": (
            "A wave has a period of 8 seconds in deep water. "
            "Calculate the wave speed and wavelength. Use g = 9.81 m/s²."
        ),
    },
    # MCQ
    {
        "id": "M1",
        "type": "mcq",
        "question": (
            "Which of the following statements about the Froude number is CORRECT?\n\n"
            "Options:\n"
            "A: It is the ratio of inertial forces to gravitational forces and is dimensionless\n"
            "B: It is the ratio of viscous forces to inertial forces\n"
            "C: It has units of m/s\n"
            "D: It is only relevant for pipe flows, not free-surface flows"
        ),
    },
    {
        "id": "M2",
        "type": "mcq",
        "question": (
            "In potential flow theory, which condition must be satisfied at a solid boundary?\n\n"
            "Options:\n"
            "A: The tangential velocity must be zero (no-slip condition)\n"
            "B: The normal component of velocity must equal the normal velocity of the boundary\n"
            "C: The pressure must be equal to the ambient pressure\n"
            "D: The vorticity must be maximum at the boundary"
        ),
    },
    {
        "id": "M3",
        "type": "mcq",
        "question": (
            "For deep-water linear waves, how does wave group velocity relate to phase velocity?\n\n"
            "Options:\n"
            "A: Group velocity equals phase velocity\n"
            "B: Group velocity is half the phase velocity\n"
            "C: Group velocity is twice the phase velocity\n"
            "D: Group velocity is zero while phase velocity is finite"
        ),
    },
]


def load_model():
    tok = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model = model.to(device)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, device_map={"": device})
    model.eval()
    return model, tok


def system_prompt_for_type(qtype: str) -> str:
    """Match eval_baseline.py: MCQ uses forced-letter instructions; others use default."""
    if qtype == "mcq":
        return MCQ_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def run_inference(model, tok, question: str, system_prompt: str) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
        )
    response = tok.decode(out[0][ids.input_ids.shape[1] :], skip_special_tokens=True)
    return response.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 9 SLM questions and save JSON.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    out_path: Path = args.output

    print("Loading model…")
    model, tok = load_model()
    device = str(model.device) if hasattr(model, "device") else "unknown"

    records = []
    for item in QUESTIONS:
        qid = item["id"]
        qtype = item["type"]
        qtext = item["question"]
        sys_p = system_prompt_for_type(qtype)
        print(f"\n[{qid}] ({qtype}) …")
        answer = run_inference(model, tok, qtext, sys_p)
        records.append(
            {
                "id": qid,
                "type": qtype,
                "question": qtext,
                "response": answer,
            }
        )
        preview = answer[:200] + ("…" if len(answer) > 200 else "")
        print(f"Response preview: {preview!r}")

    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "adapter_path": ADAPTER_PATH,
            "base_model": BASE_MODEL,
            "max_new_tokens": MAX_NEW_TOKENS,
            "device": device,
            "num_questions": len(QUESTIONS),
            "types": ["conceptual", "numerical", "mcq"],
            "per_type": 3,
            "prompts": (
                "eval_baseline.SYSTEM_PROMPT (conceptual, numerical); "
                "eval_baseline.MCQ_SYSTEM_PROMPT (mcq)"
            ),
        },
        "results": records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(records)} responses to {out_path}")


if __name__ == "__main__":
    main()
