import json
import random
import time
import requests
import re
from pathlib import Path

# --- Config ---
DATASET_PATH = r"c:\Users\raghu\Desktop\IIT\Sem 10\MTP\qa_dataset.jsonl"
OUTPUT_FAILED = r"c:\Users\raghu\Desktop\IIT\Sem 10\MTP\logically_failed_samples.jsonl"
OUTPUT_PASSED = r"c:\Users\raghu\Desktop\IIT\Sem 10\MTP\logically_passed_samples.jsonl"

# Copying API Key stuff from generate_qa.py
def load_env_keys():
    import os
    env_file = Path(r"c:\Users\raghu\Desktop\IIT\Sem 10\MTP\.env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip("'\"")
    raw_keys = os.getenv("GROQ_API_KEYS", "")
    return [k.strip() for k in raw_keys.split(",") if k.strip()]

GROQ_API_KEYS = load_env_keys()
if not GROQ_API_KEYS:
    import sys
    print("[ERROR] No GROQ_API_KEYS found in environment or .env file.")
    sys.exit(1)
EXHAUSTED_KEYS = set()
CURRENT_KEY_IDX = 0
MODEL_NAME = "llama-3.3-70b-versatile"
VLLM_URL = "https://api.groq.com/openai/v1/chat/completions"

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

def validate_sample_with_llm(sample: dict) -> dict:
    prompt = f"""
You are an expert evaluator of a Marine Hydrodynamics QA dataset.
Your job is to check if the following QA pair is logically sound and structurally correct.

Here is the sample:
Type: {sample.get('type')}
Question: {sample.get('question')}
Chain of Thought / Explanation: {sample.get('chain_of_thought')} {sample.get('explanation', '')}
Answer: {sample.get('answer')}
Options (if MCQ): {sample.get('options', [])}

Evaluation Rules:
1. Context Leakage: If any part refers to "the passage", "the text", "the excerpt", or "provided context", it is INVALID.
2. If Type is 'numerical': It MUST contain actual mathematical calculations, formulas, or step-by-step numbers in the chain of thought or explanation. Just a final number is INVALID.
3. If Type is 'mcq': The answer MUST clearly indicate the correct option (A, B, C, or D), and an explanation MUST exist.
4. If Type is 'conceptual': The answer must be clear, coherent, and logically sound.

Output strictly as JSON in the following format:
{{
   "is_valid": true, // or false
   "reason": "Short string explaining exactly why it failed, or 'Looks good' if valid."
}}
"""
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0
    }

    for attempt in range(5):
        api_key = get_api_key()
        if not api_key:
            print("No API keys left!")
            return {"is_valid": False, "reason": "No API keys available"}
            
        try:
            resp = requests.post(
                VLLM_URL, 
                json=payload, 
                headers={"Authorization": f"Bearer {api_key}"}, 
                timeout=15
            )
            
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"]
                val = json.loads(raw)
                return val
            elif resp.status_code == 429:
                rotate_api_key()
                time.sleep(2)
            else:
                time.sleep(2)
                
        except json.JSONDecodeError:
            return {"is_valid": False, "reason": "LLM returned malformed JSON."}
        except requests.exceptions.Timeout:
             time.sleep(2)
        except Exception as e:
            time.sleep(2)
            
    return {"is_valid": False, "reason": "API execution failed repeatedly or timed out."}


def main():
    samples = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    s = json.loads(line)
                    s['_original_line'] = line_no
                    samples.append(s)
                except:
                    pass
                
    processed_lines = set()
    if Path(OUTPUT_FAILED).exists():
        with open(OUTPUT_FAILED, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: processed_lines.add(json.loads(line)['_original_line'])
                    except: pass
                    
    if Path(OUTPUT_PASSED).exists():
        with open(OUTPUT_PASSED, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: processed_lines.add(json.loads(line)['_original_line'])
                    except: pass

    to_process = [s for s in samples if s['_original_line'] not in processed_lines]
                    
    print(f"Loaded {len(samples)} total samples.")
    print(f"Already processed: {len(processed_lines)}")
    print(f"Remaining to process: {len(to_process)}")
    print("-" * 50)
    
    file_failed = open(OUTPUT_FAILED, 'a', encoding='utf-8')
    file_passed = open(OUTPUT_PASSED, 'a', encoding='utf-8')
    
    try:
        for i, s in enumerate(to_process, 1):
            print(f"Eval {i}/{len(to_process)} (Line {s.get('_original_line')}) ... ", end="", flush=True)
            result = validate_sample_with_llm(s)
            
            if not result.get("is_valid", False):
                reason = result.get('reason', 'Unknown reason')
                print(f"FAILED: {reason}")
                s['_llm_validation_reason'] = reason
                file_failed.write(json.dumps(s) + '\n')
                file_failed.flush()
            else:
                print(f"PASSED")
                s['_llm_validation_reason'] = 'PASSED'
                file_passed.write(json.dumps(s) + '\n')
                file_passed.flush()
                
            time.sleep(0.5) 
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nEvaluation crashed: {e}")
    finally:
        file_failed.close()
        file_passed.close()
        
    print("-" * 50)
    print("Evaluation run terminated or completed.")

if __name__ == '__main__':
    main()
