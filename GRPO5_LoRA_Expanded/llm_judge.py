"""
llm_judge.py — LLM-as-judge for conceptual Marine Hydrodynamics answers.

Uses Groq API (llama-3.3-70b-versatile) to score model answers against
a reference answer on a 0–3 scale, normalised to a 0.0–1.0 reward.

Score rubric:
  3 → Correct & complete  : all key concepts present, physically accurate
  2 → Mostly correct      : main concept right, minor gaps or imprecision
  1 → Partially correct   : some relevant content but significant gaps/errors
  0 → Incorrect           : wrong concept, physically inaccurate, or off-topic

Reward mapping: reward = score / 3.0
  → 1.0  (perfect),  0.67 (good),  0.33 (partial),  0.0  (wrong)

This module is designed to be imported by eval_baseline.py and later by
the GRPO training script for RLAIF on conceptual questions.

Key features:
  - Reuses the same key-rotation pattern as generate_qa.py
  - Caches results in a local JSON file to avoid redundant API calls
  - Handles 429 rate limits with exponential backoff + key rotation
  - Returns 0.0 on any unrecoverable API failure (safe default)

Usage (standalone test):
    python llm_judge.py
"""

import json
import os
import re
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).parent
GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_MODEL     = "llama-3.3-70b-versatile"
JUDGE_MAX_TOKENS = 16     # We only need a single digit back
JUDGE_TEMPERATURE = 0.0   # Deterministic scoring
REQUEST_TIMEOUT = 30
MAX_RETRIES     = 5
RETRY_DELAY     = 5       # seconds base delay (doubles on each retry)

CACHE_FILE = SCRIPT_DIR / "llm_judge_cache.json"

# ── Judge prompt ──────────────────────────────────────────────────────────────
JUDGE_SYSTEM = (
    "You are an expert evaluator for Marine Hydrodynamics and Ocean Engineering. "
    "Your task is to score a student model's answer against a reference answer. "
    "Respond with ONLY a single integer: 0, 1, 2, or 3. No explanation."
)

JUDGE_USER_TEMPLATE = """\
Score the model answer against the reference on this rubric:
  3 = Correct & complete: all key concepts present, physically accurate
  2 = Mostly correct: main concept right, but minor gaps or imprecision
  1 = Partially correct: some relevant content but significant gaps or errors
  0 = Incorrect: wrong concept, physically inaccurate, or irrelevant

Question:
{question}

Reference Answer:
{reference}

Model Answer:
{model_answer}

Score (0, 1, 2, or 3):"""


# ── API key management (mirrors generate_qa.py) ───────────────────────────────
def _load_groq_keys() -> list[str]:
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")
    raw = os.getenv("GROQ_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


class _KeyPool:
    """Round-robin key pool with exhaustion tracking."""
    def __init__(self, keys: list[str]):
        self.keys      = keys
        self.idx       = 0
        self.exhausted = set()

    def current(self) -> str | None:
        for _ in range(len(self.keys)):
            k = self.keys[self.idx % len(self.keys)]
            if k not in self.exhausted:
                return k
            self.idx += 1
        return None

    def rotate(self):
        self.idx += 1

    def exhaust_current(self):
        k = self.current()
        if k:
            self.exhausted.add(k)
            print(f"  [judge] Key exhausted ({len(self.exhausted)}/{len(self.keys)} total).")
        self.rotate()


_pool: _KeyPool | None = None

def _get_pool() -> _KeyPool:
    global _pool
    if _pool is None:
        keys = _load_groq_keys()
        if not keys:
            raise RuntimeError(
                "No Groq API keys found. Set GROQ_API_KEYS in .env or environment."
            )
        _pool = _KeyPool(keys)
    return _pool


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _cache_key(question: str, reference: str, model_answer: str) -> str:
    """Simple deterministic cache key."""
    import hashlib
    blob = f"{question}|||{reference}|||{model_answer}"
    return hashlib.md5(blob.encode()).hexdigest()


# ── Core judge call ───────────────────────────────────────────────────────────
def _call_groq_judge(question: str, reference: str, model_answer: str) -> int | None:
    """
    Call Groq judge. Returns integer score 0–3, or None on failure.
    Handles 429 rate limits with backoff + key rotation.
    """
    pool = _get_pool()
    prompt = JUDGE_USER_TEMPLATE.format(
        question=question.strip(),
        reference=reference.strip(),
        model_answer=model_answer.strip(),
    )

    for attempt in range(MAX_RETRIES):
        key = pool.current()
        if key is None:
            print("  [judge] All API keys exhausted. Returning None.")
            return None

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       JUDGE_MODEL,
            "messages":    [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens":  JUDGE_MAX_TOKENS,
            "temperature": JUDGE_TEMPERATURE,
        }

        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"].strip()
                match = re.search(r"[0-3]", text)
                if match:
                    return int(match.group(0))
                print(f"  [judge] Unexpected response '{text}'. Defaulting to 0.")
                return 0

            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", RETRY_DELAY * (2 ** attempt)))
                print(f"  [judge] 429 rate limit. Rotating key, waiting {retry_after}s...")
                pool.rotate()
                time.sleep(retry_after)

            elif resp.status_code in (401, 403):
                print(f"  [judge] Auth error on key. Exhausting and rotating.")
                pool.exhaust_current()

            else:
                delay = RETRY_DELAY * (2 ** attempt)
                print(f"  [judge] HTTP {resp.status_code}. Retrying in {delay}s...")
                time.sleep(delay)

        except requests.exceptions.Timeout:
            delay = RETRY_DELAY * (2 ** attempt)
            print(f"  [judge] Request timed out. Retrying in {delay}s...")
            time.sleep(delay)

        except requests.exceptions.RequestException as e:
            delay = RETRY_DELAY * (2 ** attempt)
            print(f"  [judge] Request error: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    print("  [judge] Max retries exceeded. Returning None.")
    return None


# ── Public interface ──────────────────────────────────────────────────────────
def judge_conceptual(
    question: str,
    reference_answer: str,
    model_answer: str,
    use_cache: bool = True,
) -> float:
    """
    Score a conceptual answer using LLM-as-judge.

    Args:
        question:         The original question posed to the model.
        reference_answer: Ground-truth answer from the dataset.
        model_answer:     The model's generated completion.
        use_cache:        If True, cache results to avoid redundant API calls.

    Returns:
        Reward in [0.0, 0.33, 0.67, 1.0].
        Returns 0.0 on any unrecoverable failure (safe default for RL).
    """
    cache = _load_cache() if use_cache else {}
    ck    = _cache_key(question, reference_answer, model_answer)

    if ck in cache:
        return cache[ck]

    score = _call_groq_judge(question, reference_answer, model_answer)
    if score is None:
        reward = 0.0
    else:
        reward = round(score / 3.0, 4)

    if use_cache:
        cache[ck] = reward
        _save_cache(cache)

    return reward


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("LLM Judge — standalone test")
    print("=" * 55)

    tests = [
        {
            "label": "CORRECT (should score ~1.0)",
            "question": "What is the Reynolds number and why is it important in marine hydrodynamics?",
            "reference": (
                "The Reynolds number (Re = ρvL/μ) is a dimensionless ratio of inertial "
                "to viscous forces. It determines whether flow is laminar or turbulent, "
                "which directly affects drag on hulls and the design of marine structures."
            ),
            "model": (
                "Chain of Thought:\nRe = ρvL/μ, ratio of inertial to viscous forces.\n\n"
                "Final Answer:\nThe Reynolds number indicates whether flow is laminar "
                "or turbulent, critical for predicting hull drag and boundary layer behaviour."
            ),
        },
        {
            "label": "PARTIAL (should score ~0.33–0.67)",
            "question": "What is the Reynolds number and why is it important in marine hydrodynamics?",
            "reference": (
                "The Reynolds number (Re = ρvL/μ) is a dimensionless ratio of inertial "
                "to viscous forces. It determines whether flow is laminar or turbulent."
            ),
            "model": (
                "Final Answer:\nThe Reynolds number is a dimensionless number used in fluid mechanics."
            ),
        },
        {
            "label": "WRONG (should score 0.0)",
            "question": "What is the Froude number?",
            "reference": "The Froude number (Fr = v/√(gL)) is the ratio of inertial to gravitational forces.",
            "model": "Final Answer:\nThe Froude number is related to viscosity and pressure drop in pipes.",
        },
    ]

    for t in tests:
        reward = judge_conceptual(t["question"], t["reference"], t["model"], use_cache=False)
        print(f"\n{t['label']}")
        print(f"  Reward: {reward:.2f}")
