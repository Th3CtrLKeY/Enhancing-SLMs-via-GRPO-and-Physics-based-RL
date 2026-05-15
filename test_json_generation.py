import os
import requests
import json
import re
from pathlib import Path

# Load from .env
env_file = Path(r"c:\Users\raghu\Desktop\IIT\Sem 10\MTP\.env")
for line in env_file.read_text(encoding="utf-8").splitlines():
    if '=' in line:
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip().strip("'\"")

keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",")]
# Try to find a working key, #4 is known to be the first non-exhausted one
key = keys[3] if len(keys) > 3 else keys[-1]

prompt_template = Path(r"c:\Users\raghu\Desktop\IIT\Sem 10\MTP\qa_prompt_template.txt").read_text(encoding="utf-8")

if "PASSAGE:" in prompt_template:
    system_instruction, rest = prompt_template.split("PASSAGE:", 1)
    system_instruction = system_instruction.strip()
    user_message = "PASSAGE:\n" + rest.replace("{passage}", "Water has a density of 1000 kg/m3. A ship is 50m long.").replace("{source}", "Dummy Source")
else:
    system_instruction = ""
    user_message = prompt_template

payload = {
    "model": "llama-3.3-70b-versatile",
    "messages": [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ],
    "max_tokens": 4096,
    "temperature": 0.3,
    "response_format": {"type": "json_object"}
}

print("Testing prompt generation against Groq to debug JSON failure...")
resp = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    json=payload,
    headers={"Authorization": f"Bearer {key}"},
    timeout=60
)

print(f"HTTP Status: {resp.status_code}")
if resp.status_code == 200:
    raw = resp.json()["choices"][0]["message"]["content"]
    print("\n--- RAW TEXT RECEIVED FROM LLM ---")
    print(raw)
    print("----------------------------------\n")
    
    # Attempt parse identical to generate_qa.py
    raw2 = re.sub(r"```json\s*", "", raw)
    raw2 = re.sub(r"```\s*", "", raw2)
    raw2 = raw2.strip()
    match = re.search(r"(\{.*?\}|\[.*?\])", raw2, re.DOTALL)
    if match: raw2 = match.group(0)
    raw2 = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', raw2)
    
    try:
        data = json.loads(raw2, strict=False)
        print("SUCCESS! JSON parsed perfectly.")
    except json.JSONDecodeError as e:
        print(f"JSON ERROR IDENTIFIED: {e}")
        lines = raw2.splitlines()
        for i, l in enumerate(lines, 1):
            if i == e.lineno:
                print(f" >> Line {i}: {l}")
                print("            " + " " * (e.colno - 1) + "^")
            else:
                pass # Only print bad line to avoid spam
else:
    print(resp.text)
