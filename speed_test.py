import os
import sys
import time

import requests

url = "https://api.groq.com/openai/v1/chat/completions"
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("Set GROQ_API_KEY in the environment (do not hardcode API keys).", file=sys.stderr)
    sys.exit(1)

payload = {
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Write a detailed 5-paragraph essay about the history of the universe."}],
    "max_tokens": 500,
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

print("Starting generation with Groq API...")
t0 = time.time()
resp = requests.post(url, json=payload, headers=headers).json()
t1 = time.time()

if "error" in resp:
    print("Error:", resp["error"])
else:
    elapsed = t1 - t0
    tokens = resp.get("usage", {}).get("completion_tokens", 0)
    speed = tokens / elapsed if elapsed > 0 else 0

    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Tokens generated: {tokens}")
    print(f"Speed: {speed:.2f} tokens/sec")
