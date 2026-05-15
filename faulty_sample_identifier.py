import json

# Define input and output files
input_file = r"C:\Users\raghu\Desktop\qa_dataset.jsonl" # Change this if your file is named differently (e.g., qa_dataset_part_1.jsonl)
output_file = "qa_old_cleaned.jsonl"

# Defect criteria lists
leakage_words = ["passage", "text", "figure", "chapter"]
refusal_words = [
    "cannot be calculated", "insufficient information", "not possible", 
    "cannot determine", "not provided", "cannot be solved", 
    "without additional information"
]

total_lines = 0
faulty_count = 0
clean_count = 0

print(f"Reading from {input_file} and cleaning...")

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for i, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            continue
            
        total_lines += 1
        
        try:
            data = json.loads(line)
            q_text = data.get("question", "").lower()
            cot_text = data.get("chain_of_thought", "").lower()
            ans_text = str(data.get("answer", "")).lower()
            q_type = data.get("type", "")
            
            is_faulty = False
            
            # 1. Check MCQ Explanations and Options
            if q_type == "mcq":
                explanation = str(data.get("explanation", "")).strip()
                # Check if explanation is missing or if options are missing/malformed
                options = data.get("options", {})
                # Note: checking if options is a dict/list with at least 2 items
                if len(explanation) < 5 or not options or len(options) < 2:
                    is_faulty = True
                    
            # 2. Check Context Leakage
            if any(word in q_text or word in cot_text for word in leakage_words):
                is_faulty = True
                
            # 3. Check Numerical Refusals
            if q_type == "numerical" and any(word in ans_text or word in cot_text for word in refusal_words):
                is_faulty = True
                
            # Write to new file if it passes all checks
            if not is_faulty:
                # ensure_ascii=False keeps math symbols and special characters intact
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                clean_count += 1
            else:
                faulty_count += 1
                
        except json.JSONDecodeError:
            # If the JSON itself is broken, we consider it a faulty line and drop it
            print(f"  [WARN] Skipping line {i}: Invalid JSON")
            faulty_count += 1

print("==================================================")
print(f"Processing Complete.")
print(f"Total records evaluated : {total_lines}")
print(f"Faulty records removed  : {faulty_count}")
print(f"Clean records saved     : {clean_count}")
print(f"Clean dataset created   : {output_file}")
print("==================================================")