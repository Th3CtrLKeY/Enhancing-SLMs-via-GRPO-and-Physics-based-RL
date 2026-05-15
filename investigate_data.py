import json
import random

# ── 1. Check the SFT training data structure ──────────────────────────────────
with open('data/sft_train.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f if l.strip()]

print(f'Total records: {len(records)}')
print(f'Keys in first record: {list(records[0].keys())}')

# ── 2. Check message roles and content lengths ────────────────────────────────
print("\n--- First 3 records in detail ---")
for i, r in enumerate(records[:3]):
    msgs = r.get('messages', [])
    roles = [m['role'] for m in msgs]
    print(f'Record {i}: {len(msgs)} messages, roles={roles}')
    for m in msgs:
        print(f'  [{m["role"]}] len={len(m["content"])} chars | first 80: {repr(m["content"][:80])}')
    print()

# ── 3. Find outliers: very long sequences ─────────────────────────────────────
lengths = []
for r in records:
    total = sum(len(m['content']) for m in r.get('messages', []))
    lengths.append(total)

lengths.sort(reverse=True)
print(f"\n--- Sequence length stats (by char count) ---")
print(f"Max length   : {lengths[0]}")
print(f"Min length   : {lengths[-1]}")
print(f"Mean length  : {sum(lengths)//len(lengths)}")
print(f"Top 10 lengths: {lengths[:10]}")

# ── 4. Check for empty/None content ──────────────────────────────────────────
empty = 0
for i, r in enumerate(records):
    for m in r.get('messages', []):
        if not m.get('content'):
            print(f"  [EMPTY CONTENT] record {i}, role={m['role']}")
            empty += 1
print(f"\nEmpty/None content fields: {empty}")

# ── 5. Check for NaN/inf in data (numeric answers) ───────────────────────────
import math
nan_count = 0
for i, r in enumerate(records):
    for m in r.get('messages', []):
        content = m.get('content', '')
        if 'nan' in content.lower() or 'inf' in content.lower():
            print(f"  [WARNING] record {i} contains 'nan' or 'inf' in content")
            nan_count += 1
            break
print(f"\nRecords with 'nan'/'inf' in text content: {nan_count}")

# ── 6. Check that all records have system/user/assistant ─────────────────────
malformed = 0
for i, r in enumerate(records):
    roles = [m['role'] for m in r.get('messages', [])]
    if roles != ['system', 'user', 'assistant']:
        print(f"  [MALFORMED] record {i}: roles={roles}")
        malformed += 1
print(f"\nMalformed records (wrong role structure): {malformed}")

print("\n=== Investigation complete ===")
