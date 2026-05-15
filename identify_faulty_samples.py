import json
import re
import os

dataset_path = r"c:\Users\raghu\Desktop\IIT\Sem 10\MTP\qa_dataset.jsonl"
output_path = r"c:\Users\raghu\Desktop\IIT\Sem 10\MTP\faulty_samples.jsonl"
faulty_samples = []
total_samples = 0

# Keywords indicating context leakage
# We use a regex to capture common ways LLMs refer back to the context
pattern = re.compile(r'\b(passage|excerpt|provided context|the text mentions|according to the text|based on the text)\b', re.IGNORECASE)

with open(dataset_path, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            total_samples += 1
            
            is_faulty = False
            faulty_fields = []
            
            # We check the primary textual fields where the model generates content
            for field in ['question', 'chain_of_thought', 'answer', 'explanation']:
                if field in data and data[field]:
                    # Need to make sure data[field] is a string. 'options' is a list, but we aren't checking it here.
                    if isinstance(data[field], str) and pattern.search(data[field]):
                        is_faulty = True
                        faulty_fields.append(field)
            
            if is_faulty:
                faulty_samples.append({
                    'line_no': line_idx + 1,
                    'fields': faulty_fields,
                    'sample': data
                })
        except json.JSONDecodeError:
            print(f"Error decoding JSON on line {line_idx + 1}")

print(f"--- Leakage Analysis Report ---")
print(f"Total samples processed: {total_samples}")
print(f"Faulty samples found: {len(faulty_samples)}")
if total_samples > 0:
    print(f"Percentage of faulty samples: {(len(faulty_samples)/total_samples)*100:.2f}%\n")

from collections import Counter
field_counts = Counter([field for sample in faulty_samples for field in sample['fields']])
print("Fields where context leakage was found:")
for field, count in field_counts.items():
    print(f"  - {field}: {count} occurrences")

if faulty_samples:
    print(f"\nExporting {len(faulty_samples)} faulty samples to {os.path.basename(output_path)}...")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for s in faulty_samples:
            # Create a combined object for easier review
            export_obj = {
                "_line_no": s["line_no"],
                "_faulty_fields": s["fields"]
            }
            export_obj.update(s["sample"])
            out_f.write(json.dumps(export_obj) + '\n')
    print("Done!")

    print("\n--- Snippet of 2 Faulty Samples ---")
    for i in range(min(2, len(faulty_samples))):
        s = faulty_samples[i]
        print(f"\n[Sample on Line {s['line_no']}]")
        print(f"Faulty fields: {s['fields']}")
        print(f"Question: {s['sample'].get('question', '')}")
        print(f"Chain of Thought: {s['sample'].get('chain_of_thought', '')}")
