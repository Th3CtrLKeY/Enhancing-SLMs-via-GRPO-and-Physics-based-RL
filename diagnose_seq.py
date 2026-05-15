import json
import statistics

with open('data/sft_train.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f if l.strip()]

print(f"Total records: {len(records)}")

# Check character lengths
char_lengths = []
for r in records:
    total = sum(len(m['content']) for m in r.get('messages', []))
    char_lengths.append(total)

char_lengths_sorted = sorted(char_lengths, reverse=True)
print("\n--- Character length distribution ---")
print(f"Max   : {char_lengths_sorted[0]}")
print(f"Min   : {char_lengths_sorted[-1]}")
print(f"Mean  : {int(statistics.mean(char_lengths))}")
print(f"Stdev : {int(statistics.stdev(char_lengths))}")
print(f"Median: {int(statistics.median(char_lengths))}")
print(f"Top 10: {char_lengths_sorted[:10]}")
print(f"Records > 4000 chars : {sum(1 for l in char_lengths if l > 4000)}")
print(f"Records > 8000 chars : {sum(1 for l in char_lengths if l > 8000)}")
print(f"Records > 16000 chars: {sum(1 for l in char_lengths if l > 16000)}")

# Check assistant content specifically
print("\n--- Assistant content length ---")
asst_lengths = []
for r in records:
    for m in r.get('messages', []):
        if m['role'] == 'assistant':
            asst_lengths.append(len(m['content']))

asst_lengths.sort(reverse=True)
print(f"Max   : {asst_lengths[0]}")
print(f"Min   : {asst_lengths[-1]}")
print(f"Mean  : {int(statistics.mean(asst_lengths))}")
print(f"Top 5 : {asst_lengths[:5]}")

# Show the 3 longest records
print("\n--- Top 3 longest records ---")
sorted_records = sorted(records, key=lambda r: sum(len(m['content']) for m in r.get('messages', [])), reverse=True)
for idx, r in enumerate(sorted_records[:3]):
    total_len = sum(len(m['content']) for m in r.get('messages', []))
    print(f"\nRecord #{idx+1} total={total_len} chars:")
    for m in r.get('messages', []):
        print(f"  [{m['role']}] {len(m['content'])} chars | start: {m['content'][:100]!r}")

# Check for special token contamination
print("\n--- Special token contamination ---")
special_tokens = ['<|im_start|>', '<|im_end|>', '<think>', '</think>'] 
for token in special_tokens:
    count = sum(1 for r in records for m in r.get('messages', []) if token in m.get('content', ''))
    print(f"  Records containing {token!r}: {count}")

# Check for empty assistant answers
print("\n--- Empty/very short assistant answers ---")
short = [(i, len(m['content'])) for i, r in enumerate(records) for m in r.get('messages', []) if m['role'] == 'assistant' and len(m['content']) < 10]
print(f"  Records with assistant content < 10 chars: {len(short)}")
for idx, length in short[:5]:
    print(f"  Record {idx}: length={length}")
