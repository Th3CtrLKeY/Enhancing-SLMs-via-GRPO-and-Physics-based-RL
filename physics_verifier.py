"""
physics_verifier.py — Physics-based reward function for Marine Hydrodynamics SLM.

Used by:
  - eval_baseline.py  (scoring completions against ground truth)
  - GRPO training script (as the RL reward signal)

Handles three ground-truth formats:
  - numerical  : scalar float (exact or within tolerance)
  - mcq letter : single letter A/B/C/D
  - mcq phrase : short answer phrase (e.g. "Reynolds number", "Inviscid flow")
  - symbolic   : algebraic expression verified via SymPy
"""

import re
import sympy as sp
from typing import Any, Dict


# ── Numerical extraction ───────────────────────────────────────────────────────

def extract_numerical_answer(text: str) -> float | None:
    """
    Extracts the final numerical answer from an LLM completion.

    Priority order:
      1. "Final Answer:" block — scientific notation  (3.36e8 or 3.36 × 10^8)
      2. "Final Answer:" block — plain decimal         (46.48)
      3. Last standalone number anywhere in text       (fallback)
    """
    # ── Priority 1 & 2: explicit Final Answer block ───────────────────────────
    # Match everything after "Final Answer:" on the same or next line
    fa_match = re.search(r'Final Answer[:\s]+(.*)', text, re.IGNORECASE)
    if fa_match:
        lines = fa_match.group(1).strip().splitlines()
        fa_block = lines[0].strip() if lines else ""

        if fa_block:
            # a) A × 10^B notation  (e.g. "3.36 × 10^8", "5 x 10^-3", "3.36*10^8")
            sci_cross = re.search(
                r'([+-]?\d+\.?\d*)\s*[×xX\*]\s*10\^?\s*([+-]?\d+)', fa_block
            )
            if sci_cross:
                try:
                    return float(sci_cross.group(1)) * 10 ** int(sci_cross.group(2))
                except (ValueError, OverflowError):
                    pass

            # b) Standard / e-notation  (e.g. "3.36e8", "-0.4", "46.48 N")
            std_num = re.search(r'([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', fa_block)
            if std_num:
                try:
                    return float(std_num.group(1))
                except ValueError:
                    pass

    # ── Priority 3: last number in full text (fallback) ───────────────────────
    all_nums = re.findall(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', text)
    for num_str in reversed(all_nums):
        try:
            return float(num_str)
        except ValueError:
            continue

    return None


def collect_numerical_candidates(text: str) -> list[float]:
    """
    Collect every scalar that could be a numeric answer from free-form text.

    Used when there is no reliable `Final Answer:` line (e.g. smoke_test /
    ask_model_save_responses style).  Includes:
      - a × 10^b  (Unicode ×, x, *)
      - plain floats and scientific e-notation
    Order: ×10^ chunks first (so large magnitudes are not split), then all floats.
    """
    out: list[float] = []
    seen: set[float] = set()

    def add(v: float) -> None:
        if v != v:  # NaN
            return
        # Avoid huge duplicate lists from integer 10 in "10^8"
        if v not in seen:
            seen.add(v)
            out.append(v)

    for m in re.finditer(
        r"([+-]?\d+\.?\d*)\s*[×xX\*]\s*10\^?\s*([+-]?\d+)", text
    ):
        try:
            add(float(m.group(1)) * 10 ** int(m.group(2)))
        except (ValueError, OverflowError):
            pass

    for m in re.finditer(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?", text):
        try:
            add(float(m.group(0)))
        except ValueError:
            pass

    return out


def any_numerical_matches_gt(
    completion: str, gt_val: float, tolerance: float = 0.05
) -> bool:
    """True if any extracted candidate is within tolerance of gt_val."""
    for cand in collect_numerical_candidates(completion):
        if verify_numerical(cand, gt_val, tolerance=tolerance):
            return True
    return False


def verify_numerical(pred_val: float, true_val: float, tolerance: float = 0.05) -> bool:
    """Returns True if pred_val is within `tolerance` (relative) of true_val."""
    if true_val == 0:
        # Absolute tolerance for zero-valued targets
        return abs(pred_val) <= 1e-4
    return abs((pred_val - true_val) / true_val) <= tolerance


# ── Symbolic extraction & verification ────────────────────────────────────────

def extract_symbolic_expression(text: str) -> str | None:
    """
    Extracts a SymPy-parseable expression from the Final Answer block.
    E.g. 'Final Answer: F_d = 0.5 * rho * v**2 * C_d * A'  →  '0.5 * rho * v**2 * C_d * A'
    """
    match = re.search(r'Final Answer[:\s]+(.*)', text, re.IGNORECASE)
    if not match:
        return None
    lines = match.group(1).strip().splitlines()
    ans_str = lines[0].strip() if lines else ""
    if not ans_str:
        return None
    if '=' in ans_str:
        ans_str = ans_str.split('=', 1)[1].strip()
    ans_str = ans_str.replace('$', '').replace('^', '**')
    return ans_str or None


def verify_symbolic(pred_expr_str: str, true_expr_str: str) -> bool:
    """Uses SymPy to check mathematical equivalence of two expressions."""
    try:
        pred_expr = sp.sympify(pred_expr_str)
        true_expr = sp.sympify(true_expr_str)
        return sp.simplify(pred_expr - true_expr) == 0
    except Exception as e:
        print(f"[WARN] Symbolic parsing failed: {e}")
        return False


# ── MCQ helpers ────────────────────────────────────────────────────────────────

def _is_letter_gt(gt: str) -> bool:
    """True if gt is a single A–D option letter."""
    return bool(re.match(r'^[A-Da-d]$', gt.strip()))


def extract_forced_mcq_letter(completion: str) -> str | None:
    """
    Extract A/B/C/D from common MCQ answer styles (forced prompt or free-form).

    Covers:
      - "Therefore, the answer is: A"
      - "The correct answer is A" / "correct answer is A:"
      - "Final Answer: A" / "**Final Answer: A**"
      - LaTeX "\\boxed{A}"
      - Leading "A:" after "answer"
    """
    patterns = [
        r"\\boxed\{([A-Da-d])\}",
        r"therefore[,\s]+the answer is[:\s]+([A-Da-d])\b",
        r"the\s+correct\s+answer\s+is[:\s]+([A-Da-d])\b",
        r"correct\s+answer\s+is[:\s]+([A-Da-d])\b",
        r"the\s+answer\s+is[:\s]+([A-Da-d])\b",
        r"final\s+answer[:\s*]+([A-Da-d])\b",
        r"\banswer\s+is[:\s]+([A-Da-d])\b",
        r"\boption\s+([A-Da-d])\s+is\s+correct\b",
    ]
    for pat in patterns:
        m = re.search(pat, completion, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _extract_final_answer_section(completion: str) -> str:
    """Returns the text after the last 'Final Answer:' marker (lowercased)."""
    parts = re.split(r'final answer[:\s]+', completion, flags=re.IGNORECASE)
    return parts[-1].lower() if len(parts) > 1 else completion.lower()


def _score_mcq_letter(completion: str, gt_letter: str) -> float:
    """Score when GT is a standard A/B/C/D letter."""
    gt_upper = gt_letter.upper()
    fa_section = _extract_final_answer_section(completion)

    # 1. Explicit "Final Answer: X" letter match
    letter_match = re.search(r'^([A-Da-d])', fa_section.strip())
    if letter_match and letter_match.group(1).upper() == gt_upper:
        return 1.0

    # 2. "The answer is X" / "Option X" patterns
    patterns = [
        rf'\bthe (?:correct )?answer is\s+{gt_upper}\b',
        rf'\boption\s+{gt_upper}\b',
        rf'\b{gt_upper}\s+is correct\b',
        rf'\b{gt_upper}\s*[:\-–]',
    ]
    for pat in patterns:
        if re.search(pat, completion, re.IGNORECASE):
            return 1.0

    return 0.0


def _score_mcq_phrase(completion: str, gt_phrase: str) -> float:
    """
    Score when GT is a short phrase (e.g. 'Inviscid flow', 'Reynolds number').
    Uses normalised substring matching in the Final Answer section.
    """
    fa_section = _extract_final_answer_section(completion)

    # Normalise whitespace and punctuation for comparison
    def normalise(s: str) -> str:
        return re.sub(r'[\s\W]+', ' ', s.lower()).strip()

    gt_norm = normalise(gt_phrase)
    fa_norm = normalise(fa_section)

    # Exact normalised match
    if gt_norm in fa_norm:
        return 1.0

    # Keyword overlap: all meaningful words from GT appear in final answer
    stopwords = {'the', 'a', 'an', 'is', 'of', 'and', 'or', 'in', 'to', 'that', 'for'}
    gt_words = set(gt_norm.split()) - stopwords
    if gt_words and all(w in fa_norm for w in gt_words):
        return 1.0

    return 0.0


# ── Core reward function ───────────────────────────────────────────────────────

def physics_reward_function(completion: str, ground_truth: Dict[str, Any]) -> float:
    """
    Core reward function for GRPO training and baseline evaluation.

    Args:
        completion:   The model's generated text (full response).
        ground_truth: Dict with keys:
                        "type"   → "numerical" | "mcq" | "conceptual" | "symbolic"
                        "answer" → ground truth value (str)

    Returns:
        Float reward:
          1.0  → correct
          0.0  → incorrect
          (conceptual always returns 0.0 — use llm_judge.py for those)
    """
    q_type    = ground_truth.get("type", "conceptual")
    gt_answer = str(ground_truth.get("answer", "")).strip()

    MAX_REWARD = 1.0
    MIN_REWARD = 0.0

    # ── Numerical ──────────────────────────────────────────────────────────────
    if q_type == "numerical":
        pred_val = extract_numerical_answer(completion)

        try:
            gt_val = float(gt_answer)
            # Primary: single extracted prediction (prefers `Final Answer:` line)
            if pred_val is not None and verify_numerical(
                pred_val, gt_val, tolerance=0.05
            ):
                return MAX_REWARD
            # Secondary: free-form replies (no `Final Answer:`) — accept if ANY
            # scalar in the text matches GT within tolerance (see model_qa_responses).
            if any_numerical_matches_gt(completion, gt_val, tolerance=0.05):
                return MAX_REWARD
        except (ValueError, TypeError):
            # GT isn't a plain float — try symbolic equivalence as fallback
            pred_expr = extract_symbolic_expression(completion)
            if pred_expr and verify_symbolic(pred_expr, gt_answer):
                return MAX_REWARD

        return MIN_REWARD

    # ── Symbolic / derivation ──────────────────────────────────────────────────
    elif q_type in ("derivation", "symbolic") or "symbolic" in q_type:
        pred_expr = extract_symbolic_expression(completion)
        if pred_expr and verify_symbolic(pred_expr, gt_answer):
            return MAX_REWARD
        return MIN_REWARD

    # ── MCQ ────────────────────────────────────────────────────────────────────
    elif q_type == "mcq":
        if not gt_answer:
            return MIN_REWARD

        gt = gt_answer.strip()

        if _is_letter_gt(gt):
            gt_upper = gt.upper()

            # ── Option B: forced-letter format (preferred) ─────────────────
            # If the inference used the MCQ system prompt, the model will write
            # "Therefore, the answer is: X" — extract and compare directly.
            forced = extract_forced_mcq_letter(completion)
            if forced is not None:
                return MAX_REWARD if forced == gt_upper else MIN_REWARD

            # ── Fallback A: option-text matching ───────────────────────────
            # If options were passed, resolve the correct option text and check
            # if it appears in the model's explanation.
            options = ground_truth.get("options") or {}
            if isinstance(options, dict) and gt_upper in options:
                return _score_mcq_phrase(completion, options[gt_upper])

            # ── Fallback B: bare letter matching ───────────────────────────
            return _score_mcq_letter(completion, gt_upper)

        else:
            # Non-letter GT (phrase-style MCQ, e.g. "Reynolds number")
            return _score_mcq_phrase(completion, gt)

    # ── Conceptual ─────────────────────────────────────────────────────────────
    else:
        # Use llm_judge.judge_conceptual() for these — not handled here
        return MIN_REWARD


# ── Self-tests ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # Numerical — exact
        ("Chain of Thought:\nUsing F=ma.\n\nFinal Answer:\n46.48 N",
         {"type": "numerical", "answer": "46.14"},
         1.0, "numerical close (within 5%)"),

        # Numerical — scientific notation in completion
        ("Final Answer:\n3.36 × 10^8",
         {"type": "numerical", "answer": "336000000"},
         1.0, "numerical sci-notation cross"),

        # Numerical — e-notation
        ("Final Answer:\n3.36e8",
         {"type": "numerical", "answer": "3.36e8"},
         1.0, "numerical e-notation"),

        # Numerical — wrong answer
        ("Final Answer:\n100.0",
         {"type": "numerical", "answer": "46.14"},
         0.0, "numerical wrong"),

        # Numerical — no Final Answer block; correct value appears in prose
        (
            "Re = (1025 * 5 * 80) / 1.19e-6 so Re ≈ 3.45 × 10^8 (turbulent).",
            {"type": "numerical", "answer": "336000000"},
            1.0,
            "numerical free-form multi-candidate",
        ),

        # MCQ — natural phrasing (ask_model_save_responses style)
        (
            "The correct answer is A: The Froude number is dimensionless.\n",
            {"type": "mcq", "answer": "A"},
            1.0,
            "MCQ correct answer is A",
        ),

        # MCQ — LaTeX boxed
        (
            "Deep water: \\( v_g = v_p / 2 \\) so \\boxed{B}",
            {"type": "mcq", "answer": "B"},
            1.0,
            "MCQ boxed letter",
        ),

        # MCQ — forced letter correct
        ("The Froude number is the ratio of inertial to gravitational forces.\n\nTherefore, the answer is: A",
         {"type": "mcq", "answer": "A"},
         1.0, "MCQ forced-letter correct"),

        # MCQ — forced letter wrong
        ("Viscosity determines this.\n\nTherefore, the answer is: B",
         {"type": "mcq", "answer": "A"},
         0.0, "MCQ forced-letter wrong"),

        # MCQ — forced letter via options dict
        ("Viscous dissipation dominates.\n\nTherefore, the answer is: A",
         {"type": "mcq", "answer": "A",
          "options": {"A": "Viscous dissipation", "B": "Turbulent mixing"}},
         1.0, "MCQ forced-letter with options"),

        # MCQ — phrase correct (fallback, no forced letter)
        ("Chain of Thought:\nViscous forces dominate.\n\nFinal Answer:\nInviscid flow",
         {"type": "mcq", "answer": "Inviscid flow"},
         1.0, "MCQ phrase correct"),

        # MCQ — phrase wrong
        ("Final Answer:\nTurbulent flow",
         {"type": "mcq", "answer": "Inviscid flow"},
         0.0, "MCQ phrase wrong"),

        # Symbolic
        ("Final Answer: x**2 + 2*x + 1",
         {"type": "symbolic", "answer": "(x+1)**2"},
         1.0, "symbolic equivalent"),
    ]

    print("physics_verifier.py — self-tests")
    print("=" * 55)
    all_pass = True
    for completion, gt, expected, label in tests:
        score = physics_reward_function(completion, gt)
        status = "PASS" if abs(score - expected) < 1e-9 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}  [{label}]  score={score:.2f}  expected={expected:.2f}")

    print("=" * 55)
    print("All tests passed." if all_pass else "SOME TESTS FAILED.")
