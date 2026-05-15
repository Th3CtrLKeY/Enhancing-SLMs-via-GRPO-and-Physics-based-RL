import json

with open('data/sft_train.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f if l.strip()]

# Show a sample that contains 'nan' or 'inf' to understand context
# This could be legitimate (e.g. "significant", "infinite", "organic", "financial")
# OR it could be actual NaN float values that snuck in

shown = 0
for i, r in enumerate(records):
    for m in r.get('messages', []):
        content = m.get('content', '')
        lower = content.lower()
        # Find where 'nan' or 'inf' appear
        for word in ['nan', 'inf']:
            idx = lower.find(word)
            while idx != -1:
                surrounding = content[max(0, idx-30):idx+30]
                print(f"Record {i} [{m['role']}] pos {idx}: ...{surrounding!r}...")
                idx = lower.find(word, idx+1)
        if shown > 3:
            break
    shown += 1
    if shown > 3:
        break

# Also check for actual float nan values in the JSON structure itself
print("\n\n=== Checking for actual float NaN in nested structures ===")
import math

def find_nan_floats(obj, path=""):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            print(f"  REAL NaN/Inf float found at path: {path} = {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_nan_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_nan_floats(v, f"{path}[{i}]")

for i, r in enumerate(records[:20]):
    find_nan_floats(r, f"record[{i}]")
