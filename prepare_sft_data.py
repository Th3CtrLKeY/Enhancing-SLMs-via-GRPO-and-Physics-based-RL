import json
import random
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "qa_dataset.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "data"

SYSTEM_PROMPT = "You are an expert AI assistant specializing in Marine Hydrodynamics and Ocean Engineering. Approach all questions methodically and provide step-by-step reasoning."

def format_to_chatml(record: dict) -> dict:
    """Converts a standard record into HuggingFace ChatML format."""
    
    question = record.get("question", "")
    q_type = record.get("type", "conceptual")
    
    # If MCQ, format the question to include options
    if q_type == "mcq" and "options" in record:
        options = record["options"]
        if isinstance(options, dict):
            opts_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        elif isinstance(options, list):
            opts_text = "\n".join([f"{i+1}: {v}" for i, v in enumerate(options)])
        else:
            opts_text = str(options)
        question = f"{question}\n\nOptions:\n{opts_text}"

    # Extract CoT and Answer
    cot = record.get("chain_of_thought", "")
    ans = record.get("answer", "")
    
    # Format the Assistant's response to include <think> ... </think> tags
    # This matches DeepSeek's and many modern reasoning SLMs' expected SFT structure
    if cot:
        assistant_content = f"<think>\n{cot}\n</think>\n\nFinal Answer:\n{ans}"
        if q_type == "mcq" and "explanation" in record:
            assistant_content += f"\n\nExplanation: {record['explanation']}"
    else:
        assistant_content = ans
        
    chatml = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ],
        "source": record.get("source", "unknown"),
        "type": q_type
    }
    return chatml

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset in ChatML format.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train_ratio", type=float, default=0.90)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_train = OUTPUT_DIR / "sft_train.jsonl"
    out_test = OUTPUT_DIR / "sft_test.jsonl"

    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        return

    print("=" * 60)
    print("  SFT Data Preparation Pipeline")
    print(f"  Input: {args.input}")
    print("=" * 60)

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    print(f"Loaded {len(records)} logically passed samples.")

    # Convert to ChatML
    chatml_records = [format_to_chatml(r) for r in records]

    # Shuffle robustly with a set seed for reproducibility
    random.seed(42)
    random.shuffle(chatml_records)

    # Split
    split_idx = int(len(chatml_records) * args.train_ratio)
    train_set = chatml_records[:split_idx]
    test_set = chatml_records[split_idx:]

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

    save_jsonl(train_set, out_train)
    save_jsonl(test_set, out_test)

    print("\n✅ Dataset formatted and split successfully!")
    print(f"  Train Set: {len(train_set)} records -> {out_train}")
    print(f"  Test Set : {len(test_set)} records -> {out_test}")
    print("=" * 60)

if __name__ == "__main__":
    main()
