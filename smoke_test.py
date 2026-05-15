"""
smoke_test.py — Smoke test for the fine-tuned Marine Hydrodynamics SLM.

Tests 3 question types:
  - Conceptual (domain knowledge recall)
  - Numerical  (physics calculation — verified against known correct answer)
  - MCQ        (multiple choice with explanation)

Usage (on server):
    cd ~/mtp && source sft_env/bin/activate
    python smoke_test.py
"""

import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
ADAPTER_PATH = "./sft_model_output/checkpoint-183"
BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 512

# ── Test Cases ────────────────────────────────────────────────────────────────
TESTS = [
    # ── CONCEPTUAL ─────────────────────────────────────────────────────────────
    {
        "id": "C1",
        "type": "conceptual",
        "question": "What is the difference between laminar and turbulent flow, and how does the Reynolds number help predict which regime will occur in a marine setting?",
        "check": None,  # No numerical check — evaluate qualitatively
    },
    {
        "id": "C2",
        "type": "conceptual",
        "question": "Explain the physical meaning of the Froude number and its significance in ship resistance prediction.",
        "check": None,
    },
    {
        "id": "C3",
        "type": "conceptual",
        "question": "What is added mass in the context of marine hydrodynamics and why does it matter for vessel motion analysis?",
        "check": None,
    },

    # ── NUMERICAL ──────────────────────────────────────────────────────────────
    {
        "id": "N1",
        "type": "numerical",
        "question": (
            "A submarine travels at 5 m/s underwater. Its hull has a characteristic length of 80 m. "
            "The kinematic viscosity of seawater is 1.19e-6 m²/s. "
            "Calculate the Reynolds number and state whether the boundary layer is likely laminar or turbulent."
        ),
        # Re = v*L/nu = 5*80/(1.19e-6) = 336,134,454 ≈ 3.36e8 → fully turbulent
        "check": {
            "expected": 3.36e8,
            "tolerance_pct": 5.0,
            "extract_pattern": r"[\d\.]+\s*[×x\*]\s*10\^?8|3\.3[\d]*\s*[×x]\s*10\^?8|336[\d,]+",
            "label": "Reynolds number ≈ 3.36×10⁸"
        },
    },
    {
        "id": "N2",
        "type": "numerical",
        "question": (
            "A ship has a waterplane area of 1200 m², a block coefficient of 0.75, "
            "a length of 150 m, a beam of 25 m, and a draft of 8 m. "
            "Calculate the displacement volume."
        ),
        # Vol = Cb * L * B * T = 0.75 * 150 * 25 * 8 = 22,500 m³
        "check": {
            "expected": 22500,
            "tolerance_pct": 2.0,
            "extract_pattern": r"22[,\s]?500|2\.25\s*[×x]\s*10\^?4",
            "label": "Displacement volume = 22,500 m³"
        },
    },
    {
        "id": "N3",
        "type": "numerical",
        "question": (
            "A wave has a period of 8 seconds in deep water. "
            "Calculate the wave speed and wavelength. "
            "Use g = 9.81 m/s²."
        ),
        # c = g*T/(2*pi) = 9.81*8/(2*pi) = 12.49 m/s
        # λ = g*T²/(2*pi) = 9.81*64/(2*pi) = 99.93 m ≈ 100 m
        "check": {
            "expected": 99.93,
            "tolerance_pct": 5.0,
            "extract_pattern": r"99\.?\d*\s*m|100\.?\d*\s*m",
            "label": "Wavelength ≈ 99.9 m"
        },
    },

    # ── MCQ ────────────────────────────────────────────────────────────────────
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
        "check": {
            "correct_option": "A",
            "label": "Correct answer is A"
        },
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
        "check": {
            "correct_option": "B",
            "label": "Correct answer is B (no-penetration condition)"
        },
    },
]

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    print("=" * 65)
    print("  Marine Hydrodynamics SLM — Smoke Test")
    print(f"  Adapter: {ADAPTER_PATH}")
    print(f"  Base:    {BASE_MODEL}")
    print("=" * 65)

    tok = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    print("Tokenizer loaded.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )
    model = model.to(device)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, device_map={"": device})
    model.eval()
    print("Model loaded and ready.\n")
    return model, tok


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, tok, question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
        )
    response = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


# ── Numerical Verification ────────────────────────────────────────────────────
def verify_numerical(response: str, check: dict) -> tuple[bool, str]:
    pattern = check["extract_pattern"]
    matches = re.findall(pattern, response, re.IGNORECASE)
    if not matches:
        return False, "Could not extract numerical answer from response."

    # Try to parse first match as a float
    raw = matches[0].replace(",", "").replace(" ", "").replace("×10^8", "e8").replace("x10^8","e8")
    try:
        extracted = float(raw)
    except ValueError:
        return False, f"Could not parse '{matches[0]}' as a number."

    expected = check["expected"]
    tol_pct  = check["tolerance_pct"]
    pct_err  = abs(extracted - expected) / expected * 100
    passed   = pct_err <= tol_pct
    msg = f"Extracted ≈ {extracted:.4g}, Expected ≈ {expected:.4g}, Error = {pct_err:.1f}%"
    return passed, msg


# ── MCQ Verification ──────────────────────────────────────────────────────────
def verify_mcq(response: str, check: dict) -> tuple[bool, str]:
    correct = check["correct_option"]
    # Look for explicit answer patterns like "Answer: A", "The answer is A", "option A"
    patterns = [
        rf"\bAnswer[:\s]+{correct}\b",
        rf"\bcorrect answer is\s+{correct}\b",
        rf"\b{correct}\s+is correct\b",
        rf"\bOption\s+{correct}\b.*correct",
        rf"\(?\b{correct}\b\)?\s*[:\-–]",
    ]
    for pat in patterns:
        if re.search(pat, response, re.IGNORECASE):
            return True, f"Model correctly identified option {correct}."

    # Fallback: check if correct option letter appears more prominently than others
    if re.search(rf"\b{correct}\b", response):
        return None, f"Option {correct} mentioned but answer not explicitly stated — manual review needed."
    return False, f"Model did not select option {correct}."


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model, tok = load_model()

    results = {"passed": 0, "failed": 0, "warning": 0, "total": len(TESTS)}

    for test in TESTS:
        tid    = test["id"]
        qtype  = test["type"].upper()
        q      = test["question"]
        check  = test["check"]

        print(f"\n{'─'*65}")
        print(f"[{tid}] {qtype}")
        print(f"Q: {q}\n")

        response = run_inference(model, tok, q)
        print(f"A:\n{response}\n")

        # Verification
        if check is None:
            status = "MANUAL"
            print(f"  ⚪  MANUAL REVIEW — No automated check for conceptual questions.")
        elif qtype == "NUMERICAL":
            passed, msg = verify_numerical(response, check)
            if passed is True:
                status = "PASS"
                results["passed"] += 1
                print(f"  ✅  PASS — {check['label']}")
                print(f"       {msg}")
            else:
                status = "FAIL"
                results["failed"] += 1
                print(f"  ❌  FAIL — {check['label']}")
                print(f"       {msg}")
        elif qtype == "MCQ":
            passed, msg = verify_mcq(response, check)
            if passed is True:
                status = "PASS"
                results["passed"] += 1
                print(f"  ✅  PASS — {check['label']}")
                print(f"       {msg}")
            elif passed is None:
                status = "WARN"
                results["warning"] += 1
                print(f"  ⚠️   WARN — {check['label']}")
                print(f"       {msg}")
            else:
                status = "FAIL"
                results["failed"] += 1
                print(f"  ❌  FAIL — {check['label']}")
                print(f"       {msg}")

    # Summary
    print(f"\n{'='*65}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*65}")
    print(f"  Total tests   : {results['total']}")
    print(f"  Auto-checked  : {results['passed'] + results['failed'] + results['warning']}")
    print(f"  ✅ Passed      : {results['passed']}")
    print(f"  ❌ Failed      : {results['failed']}")
    print(f"  ⚠️  Warnings    : {results['warning']}")
    print(f"  ⚪ Manual review: {results['total'] - results['passed'] - results['failed'] - results['warning']}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
