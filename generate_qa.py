"""
generate_qa.py — Marine Hydrodynamics QA Dataset Generator
============================================================
Reads all extracted text chunks from ~/mtp/extracted_text/,
calls the Groq API (llama-3.3-70b-versatile) for each chunk,
and writes qa_dataset.jsonl to ~/mtp/

Usage (on the server):
    export GROQ_KEYS="gsk_key1,gsk_key2,gsk_key3"  # Set your keys first!
    python ~/mtp/generate_qa.py                    # full run
    python ~/mtp/generate_qa.py --limit 10         # test on first 10 chunks
    python ~/mtp/generate_qa.py --dry-run          # print prompts, no API calls
    python ~/mtp/generate_qa.py --resume           # skip chunks already processed

Author: MTP project — auto-generated (Corrected)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR     = Path(__file__).parent
EXTRACTED_DIR  = SCRIPT_DIR / "extracted_text"
OUTPUT_FILE    = SCRIPT_DIR / "qa_dataset.jsonl"
CHECKPOINT_FILE= SCRIPT_DIR / "qa_checkpoint.json"
PROMPT_TEMPLATE= SCRIPT_DIR / "qa_prompt_template.txt"

VLLM_URL       = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME     = "llama-3.3-70b-versatile"
MAX_TOKENS     = 4096  
TEMPERATURE    = 0.3
REQUEST_TIMEOUT= 60   
RETRY_ATTEMPTS = 10   
RETRY_DELAY    = 5    
DELAY_BETWEEN_CHUNKS = 5  

# ── API Key Management ──────────────────────────────────────────────────────
def load_env_keys():
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip("'\"")
    # Load GROQ_API_KEYS which matches the variable inside .env
    raw_keys = os.getenv("GROQ_API_KEYS", "")
    return [k.strip() for k in raw_keys.split(",") if k.strip()]

GROQ_API_KEYS = load_env_keys()
if not GROQ_API_KEYS:
    print("[FATAL] No API keys found in .env or environment.")
    sys.exit(1)

EXHAUSTED_KEYS = set()
CURRENT_KEY_IDX = 0

def get_api_key():
    global CURRENT_KEY_IDX
    for _ in range(len(GROQ_API_KEYS)):
        idx = CURRENT_KEY_IDX % len(GROQ_API_KEYS)
        key = GROQ_API_KEYS[idx]
        if key not in EXHAUSTED_KEYS:
            return key
        CURRENT_KEY_IDX += 1
    return None

def rotate_api_key():
    global CURRENT_KEY_IDX
    CURRENT_KEY_IDX += 1
    return get_api_key()

def mark_key_exhausted(key):
    EXHAUSTED_KEYS.add(key)
    print(f"    [CRITICAL] Key #{GROQ_API_KEYS.index(key)+1} is exhausted (limit reached). Total exhausted: {len(EXHAUSTED_KEYS)}/{len(GROQ_API_KEYS)}")

CHUNK_DIRS = [
    ("textbooks/newman_chunks",   "Newman - Marine Hydrodynamics"),
    ("textbooks/faltinsen_chunks","Faltinsen - Sea Loads"),
    ("mit_ocw_2.20/lecture_notes","MIT OCW 2.20 Lecture Notes"),
    ("mit_ocw_2.20/problem_sets", "MIT OCW 2.20 Problem Sets"),
    ("mit_ocw_2.20/exams",        "MIT OCW 2.20 Exams"),
    ("mit_ocw_2.29",              "MIT OCW 2.29 Numerical Hydrodynamics"),
    ("exam_papers",               "IIT Exam Papers"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prompt_template() -> str:
    if not PROMPT_TEMPLATE.exists():
        print(f"[ERROR] Prompt template not found: {PROMPT_TEMPLATE}")
        sys.exit(1)
    return PROMPT_TEMPLATE.read_text(encoding="utf-8")


def collect_chunks() -> list[dict]:
    chunks = []
    seen_paths = set()

    for rel_dir, source_label in CHUNK_DIRS:
        dir_path = EXTRACTED_DIR / rel_dir
        if not dir_path.exists():
            print(f"  [SKIP] Directory not found: {dir_path}")
            continue

        txt_files = sorted(dir_path.rglob("*.txt"))
        if not txt_files:
            txt_files = sorted(dir_path.rglob("**/*.txt"))

        count = 0
        for f in txt_files:
            if str(f) in seen_paths:
                continue
            seen_paths.add(str(f))
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            if len(text.split()) < 50:          
                continue
            chunks.append({
                "path": str(f),
                "source": f"{source_label} — {f.name}",
                "text": text,
            })
            count += 1

        print(f"  Loaded {count:4d} chunks  ← {rel_dir}")

    print(f"\n  Total chunks to process: {len(chunks)}")
    return chunks


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return set(data.get("processed", []))
    return set()


def save_checkpoint(processed: set[str]):
    CHECKPOINT_FILE.write_text(
        json.dumps({"processed": sorted(processed)}, indent=2),
        encoding="utf-8"
    )


def extract_json_from_response(raw: str) -> list[dict] | None:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    # Greedy .* captures from the very first { to the very last }
    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    raw = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', raw)

    try:
        data = json.loads(raw, strict=False)
        if isinstance(data, dict) and "qa_pairs" in data:
            return data["qa_pairs"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError as e:
        print(f"    [DEBUG] JSON error at line {e.lineno}, col {e.colno}: {e.msg}")
        pass

    return None


def validate_qa_pair(pair: dict, idx: int) -> bool:
    """Strict validation for a single QA pair."""
    required = {"type", "question", "chain_of_thought", "answer"}
    if not required.issubset(pair.keys()):
        return False
    if pair["type"] not in {"conceptual", "numerical", "mcq"}:
        return False
    if not isinstance(pair.get("question"), str) or len(pair["question"].strip()) < 10:
        return False
    if not isinstance(pair.get("answer"), (str, int, float)) or len(str(pair["answer"]).strip()) < 1:
        return False
    if not isinstance(pair.get("chain_of_thought"), str) or len(pair["chain_of_thought"].strip()) < 5:
        return False
        
    # STRICT MCQ VALIDATION
    if pair["type"] == "mcq":
        if "options" not in pair or not isinstance(pair["options"], dict) or len(pair["options"]) < 2:
            return False
        if "explanation" not in pair or not isinstance(pair["explanation"], str) or len(pair["explanation"].strip()) < 10:
            return False
            
    return True


def process_chunk(
    chunk: dict,
    template: str,
    out_fh,
    dry_run: bool = False,
) -> int:
    words = chunk["text"].split()
    passage = " ".join(words[:1500]) if len(words) > 1500 else chunk["text"]
    
    # System/User message splitting for better instruction adherence
    if "PASSAGE:" in template:
        system_instruction, rest = template.split("PASSAGE:", 1)
        system_instruction = system_instruction.strip()
        user_message = "PASSAGE:\n" + rest.replace("{passage}", passage).replace("{source}", chunk["source"])
    else:
        system_instruction = "You are an expert professor of Marine Hydrodynamics and Naval Architecture."
        user_message = template.replace("{passage}", passage).replace("{source}", chunk["source"])

    current_temperature = TEMPERATURE

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": current_temperature,
            "response_format": {"type": "json_object"}, # This is now kept on ALL attempts
        }

        if dry_run:
            print("  [DRY-RUN] Would send System Prompt:\n", system_instruction[:150], "...")
            print("  [DRY-RUN] Would send User Prompt:\n", user_message[:150], "...")
            return 0

        api_key = get_api_key()
        if api_key is None:
            print("    [FATAL] No keys available. Exiting script to preserve checkpoint...")
            import sys
            sys.exit(1)

        try:
            resp = requests.post(
                VLLM_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
            )
            resp.raise_for_status()
            raw_response = resp.json()["choices"][0]["message"]["content"].strip()
            
            qa_pairs = extract_json_from_response(raw_response)
            if qa_pairs:
                written = 0
                for i, pair in enumerate(qa_pairs):
                    if not validate_qa_pair(pair, i):
                        continue
                    record = {
                        "question": str(pair.get("question", "")).strip(),
                        "chain_of_thought": str(pair.get("chain_of_thought", "")).strip(),
                        "answer": str(pair.get("answer", "")).strip(),
                        "type": pair["type"],
                        "source": chunk["source"],
                        "source_file": Path(chunk["path"]).name,
                    }
                    if pair["type"] == "mcq":
                        record["options"] = pair["options"]
                        record["explanation"] = pair["explanation"]

                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    written += 1
                
                # If valid pairs were returned but all failed validation, trigger retry
                if written == 0 and len(qa_pairs) > 0:
                    print(f"    [WARN] Attempt {attempt}/{RETRY_ATTEMPTS} generated items, but all failed strict validation.")
                    raise ValueError("Validation failed for all generated items")
                
                return written
            else:
                print(f"    [WARN] Attempt {attempt}/{RETRY_ATTEMPTS} JSON parse failed")
                time.sleep(DELAY_BETWEEN_CHUNKS)

        except (requests.exceptions.HTTPError, ValueError) as e:
            error_msg = ""
            if isinstance(e, requests.exceptions.HTTPError):
                try:
                    error_msg = resp.json().get("error", {}).get("message", "")
                except:
                    pass
                status_code = resp.status_code
            else:
                error_msg = str(e)
                status_code = 0
            
            print(f"    [WARN] Attempt {attempt}/{RETRY_ATTEMPTS} Error (Key #{ (CURRENT_KEY_IDX % len(GROQ_API_KEYS)) + 1 }): {error_msg or e}")
            
            if status_code == 429:
                match_ms = re.search(r"in ([\d\.]+)ms", error_msg)
                match_s  = re.search(r"in ([\d\.]+)s", error_msg)
                match_m  = re.search(r"in ([\d\.]+)m(?![s])", error_msg)
                
                wait_time = 0
                if match_m:
                    print(f"      Key #{GROQ_API_KEYS.index(api_key)+1} requested a long wait ({match_m.group(1)}m). Marking as exhausted.")
                    mark_key_exhausted(api_key)
                    wait_time = 0.1
                elif match_ms:
                    wait_time = (float(match_ms.group(1)) / 1000.0) + 0.5
                elif match_s:
                    wait_time = float(match_s.group(1)) + 1.0
                else:
                    wait_time = (RETRY_DELAY * attempt)

                new_key = rotate_api_key()
                if new_key is None:
                    print("    [FATAL] All API keys are exhausted. Exiting script to preserve checkpoint...")
                    import sys
                    sys.exit(1)

                print(f"      Rate limited. Key #{GROQ_API_KEYS.index(new_key)+1} active. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                if status_code == 400:
                    current_temperature = min(1.0, current_temperature + 0.1)
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"    [WARN] Attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            time.sleep(RETRY_DELAY)
            
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from marine hydro chunks")
    parser.add_argument("--dry-run",  action="store_true", help="Print prompts only, no API calls")
    parser.add_argument("--resume",   action="store_true", help="Skip already-processed chunks")
    parser.add_argument("--limit",    type=int, default=0,  help="Process only N chunks (0 = all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Marine Hydrodynamics QA Generator")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"  Mode   : {'dry-run' if args.dry_run else 'live'}")
    print("=" * 60)

    template = load_prompt_template()
    print("\nCollecting chunks...")
    chunks = collect_chunks()

    if args.limit > 0:
        chunks = chunks[:args.limit]
        print(f"  (limited to first {args.limit} chunks)")

    processed = set()
    if args.resume:
        processed = load_checkpoint()
        before = len(chunks)
        chunks = [c for c in chunks if c["path"] not in processed]
        print(f"  Resuming: skipping {before - len(chunks)} already-done chunks")

    if not chunks:
        print("Nothing to process. All chunks already done.")
        return

    total_pairs  = 0
    total_chunks = len(chunks)
    start_time   = time.time()

    write_mode = "a" if args.resume else "w"
    with open(OUTPUT_FILE, write_mode, encoding="utf-8") as out_fh:
        for idx, chunk in enumerate(chunks, start=1):
            fname = Path(chunk["path"]).name
            print(f"\n[{idx:4d}/{total_chunks}] {fname[:60]}")

            written = process_chunk(chunk, template, out_fh, dry_run=args.dry_run)
            
            if written == 0 and not args.dry_run:
                print(f"\n[ERROR] Failed to process chunk after all retries. Skipping.")
                print(f"        Last chunk path: {chunk['path']}")
                with open(SCRIPT_DIR / "failed_chunks.log", "a", encoding="utf-8") as flog:
                    flog.write(f"{chunk['path']}\n")

            total_pairs += written

            processed.add(chunk["path"])
            if not args.dry_run:
                save_checkpoint(processed)

            elapsed = time.time() - start_time
            avg_sec = elapsed / idx
            remaining = (total_chunks - idx) * avg_sec
            print(f"    ✓ {written} pairs  | total={total_pairs}  "
                  f"| ETA {remaining/60:.1f} min")

            if idx < total_chunks:
                time.sleep(DELAY_BETWEEN_CHUNKS)

    print("\n" + "=" * 60)
    print(f"  Done! {total_pairs} QA pairs written to:")
    print(f"  {OUTPUT_FILE}")
    elapsed_min = (time.time() - start_time) / 60
    print(f"  Time: {elapsed_min:.1f} min")
    print("=" * 60)
    print("\nNext step: python ~/mtp/validate_qa.py")

if __name__ == "__main__":
    main()